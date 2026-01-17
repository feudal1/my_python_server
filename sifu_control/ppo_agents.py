import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
from PIL import Image
from collections import deque, defaultdict
import logging
from pathlib import Path
from ultralytics import YOLO
import pyautogui
import json


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config 
CONFIG = load_config()


# 配置日志
def setup_logging():
    """设置日志配置"""
    # 创建logger实例
    logger = logging.getLogger('ppo_agent_logger')
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_ppo_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器，避免重复
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class ResidualBlock(nn.Module):
    """
    ResNet基本块
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet特征提取器
    """
    def __init__(self, input_channels=3, block_channels=[32, 64, 128]):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 使用配置文件中的图像尺寸
        height = CONFIG.get('IMAGE_HEIGHT', 480)
        width = CONFIG.get('IMAGE_WIDTH', 640)
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, block_channels[0], 
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        layers = []
        # 第一层
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        
        # 第二层 - 下采样
        layers.append(ResidualBlock(block_channels[0], block_channels[1], stride=2))
        layers.append(ResidualBlock(block_channels[1], block_channels[1]))
        
        # 第三层 - 下采样
        layers.append(ResidualBlock(block_channels[1], block_channels[2], stride=2))
        layers.append(ResidualBlock(block_channels[2], block_channels[2]))
        
        self.layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_features = nn.Linear(block_channels[2], 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_features(x)
        return x


class PolicyNetwork(nn.Module):
    """
    策略网络
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(3)
        feature_size = 256  # 根据上面的修改调整
        
        # Actor heads
        self.move_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, move_action_dim)
        )
        
        self.turn_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, turn_action_dim)
        )
        
        # Action parameter head
        self.action_param_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Value network
        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化logger
        self.logger = setup_logging()

    def forward(self, state, return_debug_info=False):
        features = self.feature_extractor(state)
        
        # Actor outputs
        move_logits = self.move_actor(features)
        turn_logits = self.turn_actor(features)
        
        # Action parameters
        action_params = torch.tanh(self.action_param_head(features))
        
        # Critic output
        value = self.critic(features)
        
        if return_debug_info:
            debug_info = {
                'move_logits': move_logits.detach().cpu().numpy(),
                'turn_logits': turn_logits.detach().cpu().numpy(),
                'action_params': action_params.detach().cpu().numpy(),
                'value': value.detach().cpu().numpy(),
                'features_shape': features.shape
            }
            
            # 打印模型输出信息
            self.logger.info(f"Move: {[round(float(x), 2) for x in debug_info['move_logits'][0]]}")
            self.logger.info(f"Turn: {[round(float(x), 2) for x in debug_info['turn_logits'][0]]}")
            self.logger.info(f"参数: {[round(float(x), 2) for x in debug_info['action_params'][0]]}")
            self.logger.info(f"价值: {round(float(debug_info['value'][0][0]), 2)}")
            
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value,
                debug_info
            )
        else:
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value
            )


class Memory:
    """
    智能体的记忆存储类
    """
    def __init__(self):
        self.states = []
        self.move_actions = []
        self.turn_actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_params = []

    def clear_memory(self):
        del self.states[:]
        del self.move_actions[:]
        del self.turn_actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_params[:]

    def append(self, state, move_action, turn_action, logprob, reward, is_terminal, action_param=None):
        self.states.append(state)
        self.move_actions.append(move_action)
        self.turn_actions.append(turn_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        if action_param is not None:
            self.action_params.append(action_param)


class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description=None):
        # 如果没有传入target_description，使用默认配置
        if target_description is None:
            target_description = CONFIG['TARGET_DESCRIPTION']
        
        from control_api_tool import ImprovedMovementController
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = CONFIG['ENV_MAX_STEPS']
        self.last_detection_result = None
        self.last_area = 0
        self.logger = setup_logging()
        
        self.yolo_model = self._load_yolo_model()
        self._warm_up_detection_model()
        
        # 成功条件阈值
        self.MIN_GATE_AREA = CONFIG['MIN_GATE_AREA']
        self.CENTER_THRESHOLD = CONFIG['CENTER_THRESHOLD']
        
        # 动作历史记录
        self.action_history = deque(maxlen=10)
        
        # 添加存储最近检测图像的队列
        self.recent_detection_images = deque(maxlen=5)
        
    def _load_yolo_model(self):
        """
        加载YOLO模型
        """
        current_dir = Path(__file__).parent
        model_path = current_dir.parent / "models" / "find_gate.pt"
        
        if not model_path.exists():
            self.logger.error(f"YOLO模型文件不存在: {model_path}")
            return None
        
        try:
            model = YOLO(str(model_path))
            self.logger.info(f"成功加载YOLO模型: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载YOLO模型失败: {e}")
            return None

    def _warm_up_detection_model(self):
        """
        预热检测模型
        """
        self.logger.info("正在预热检测模型，确保模型已加载...")
        try:
            dummy_image = self.capture_screen()
            if dummy_image is not None and dummy_image.size > 0:
                dummy_result = self.detect_target(dummy_image)
                self.logger.info("检测模型已预热完成")
            else:
                self.logger.warning("无法获取初始截图进行模型预热")
        except Exception as e:
            self.logger.warning(f"模型预热过程中出现错误: {e}")

    def capture_screen(self):
        """
        截取当前屏幕画面
        """
        try:
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            from computer_server.prtsc import capture_window_by_title
            result = capture_window_by_title("sifu", "sifu_window_capture.png")
            if result:
                screenshot = Image.open("sifu_window_capture.png")
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            self.logger.warning("截图功能不可用，使用模拟图片")
            # 使用配置文件中的图像尺寸
            return np.zeros((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 3), dtype=np.uint8)

    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []
        
        try:
            results = self.yolo_model.predict(
                source=image,
                conf=CONFIG['DETECTION_CONFIDENCE'],
                save=False,
                verbose=False
            )
            
            detections = []
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': class_name,
                        'score': conf,
                        'width': width,
                        'height': height
                    })
            
            # 保存本次检测的带框图像
            self._save_detection_image_with_bounding_boxes(image, detections, "recent_detection")
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []

    def _check_climb_conditions(self, detection_results):
        """
        检查climb检测结果是否满足条件
        """
        confidence_threshold = CONFIG.get('CLIMB_CONFIDENCE_THRESHOLD', 0.85)
        if not detection_results:
            return False
        
        # 筛选出所有climb检测结果
        climb_detections = [
            detection for detection in detection_results
            if (
                detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            )
        ]
        
        if not climb_detections:
            return False
        
        # 如果只有一个climb检测，要求其置信度超过阈值
        if len(climb_detections) == 1:
            return climb_detections[0]['score'] > confidence_threshold
        # 如果有多个climb检测，要求至少有一个置信度超过阈值
        else:
            return any(detection['score'] > confidence_threshold for detection in climb_detections)

    def _calculate_target_presence_reward(self, detection_results):
        """
        基于目标检测的奖励
        """
        if not detection_results:
            return -0.05  # 未检测到任何目标的惩罚
        
        # 检测到gate给予较高奖励
        gate_detected = any(
            'gate' in detection['label'].lower() or detection['label'].lower() == 'gate'
            for detection in detection_results
        )
        
        if gate_detected:
            return 0.1
        
        # 检测到其他目标的奖励
        return 0.02

    def _calculate_progress_reward(self, detection_results):
        """
        基于目标检测进展的奖励
        """
        if not detection_results:
            return -0.02
        else:
            return 0.02

    def _calculate_repetition_penalty(self, action_taken):
        """
        重复动作惩罚
        """
        if not action_taken or len(self.action_history) < 3:
            return 0.0
        
        recent_actions = list(self.action_history)[-3:]
        same_action_count = sum(1 for act in recent_actions if act == action_taken)
        
        if same_action_count >= 2:
            return -0.1
        elif same_action_count >= 1:
            return -0.05
            
        return 0.0

    def _get_max_detection_area(self, detection_results):
        """
        获取检测结果中的最大面积
        """
        if not detection_results:
            return 0
        return max([det['width'] * det['height'] for det in detection_results])

    def reset_to_origin(self):
        """
        重置到原点操作
        """
        self.logger.info("执行重置到原点操作...")
        
        # 按键操作序列
        pyautogui.press('esc')
        time.sleep(0.2)
        pyautogui.press('q')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        
        # 检测门是否存在
        gate_detected = False
        while not gate_detected:
            new_state = self.capture_screen()
            detection_results = self.detect_target(new_state)
            
            for detection in detection_results:
                if detection['label'].lower() == 'gate' or 'gate' in detection['label'].lower():
                    gate_detected = True
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(0.3)
                    pyautogui.press('enter')
                    break
            
            if not gate_detected:
                self.logger.info(f"未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.2)
               
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        
        # 重置环境内部状态
        self.step_count = 0
        self.last_area = 0
        self.last_detection_result = None
        self.action_history.clear()
        self.logger.info("重置到原点操作完成")

    def step(self, move_action, turn_action, move_forward_step=2, turn_angle=30):
        """
        执行动作并返回新的状态、奖励和是否结束
        """
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        self.logger.debug(f"执行动作: 移动-{move_action_names[move_action]}, 转头-{turn_action_names[turn_action]}, 步长: {move_forward_step}, 角度: {turn_angle}")
        
        # 执行动作前先检查当前状态
        pre_action_state = self.capture_screen()
        pre_action_detections = self.detect_target(pre_action_state)
        
        # 检查当前状态是否有符合条件的climb类别
        pre_climb_detected = self._check_climb_conditions(pre_action_detections)
        
        if pre_climb_detected:
            self.logger.info(f"动作执行前已检测到符合条件的climb类别，立即终止")
            
            # 保存带识别框的图片
           
            
            # 完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            speed_bonus = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8) * (self.max_steps - self.step_count)
            reward = base_completion_reward + speed_bonus
            
            new_area = self._get_max_detection_area(pre_action_detections)
            self.last_area = new_area
            self.last_detection_result = pre_action_detections
            self.step_count += 1
            
            self.logger.info(f"Step {self.step_count}, Area: {new_area:.2f}, Reward: {reward:.2f}, "
                f"Detected: climb (pre-action), Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}")
            
            # 关键：执行游戏重置操作
            self.reset_to_origin()
            
            return pre_action_state, reward, True, pre_action_detections
        
        # 执行移动动作
        if move_action == 0 and move_forward_step > 0:  # forward
            self.controller.move_forward(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 1 and move_forward_step > 0:  # backward
            self.controller.move_backward(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 2 and move_forward_step > 0:  # strafe_left
            self.controller.strafe_left(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 3 and move_forward_step > 0:  # strafe_right
            self.controller.strafe_right(duration=move_forward_step * CONFIG['forward_coe'])
        
        # 执行转头动作
        if turn_action == 0 and turn_angle > 0:  # turn_left
            self.controller.turn_left(turn_angle * CONFIG['turn_coe'], duration=1)
        elif turn_action == 1 and turn_angle > 0:  # turn_right
            self.controller.turn_right(turn_angle * CONFIG['turn_coe'], duration=1)

        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        

        # 计算奖励
        reward, new_area = self.calculate_reward(
            detection_results, 
            self.last_area,
            (move_action, turn_action)
        )
        
        # 更新状态
        self.last_area = new_area
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否检测到符合条件的climb类别
        climb_detected = self._check_climb_conditions(detection_results)

        done = climb_detected or self.step_count >= self.max_steps
        
        # 如果检测到符合条件的climb，给予额外奖励
        if climb_detected:
            # 保存带识别框的图片
            
            
            # 基础完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            
            # 快速完成奖励
            quick_completion_factor = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8)
            quick_completion_bonus = quick_completion_factor * (self.max_steps - self.step_count) 
            
            # 总奖励
            total_completion_bonus = base_completion_reward + quick_completion_bonus
            reward += total_completion_bonus
            self.logger.info(f"检测到符合条件的climb类别！步骤: {self.step_count}, 基础奖励: {base_completion_reward:.2f}, "
                            f"快速完成奖励: {quick_completion_bonus:.2f}, 总奖励: {total_completion_bonus:.2f}")
            
            # 关键：执行游戏重置操作
            self.reset_to_origin()

        # 输出每步得分
        self.logger.info(f"S {self.step_count}, A: {new_area:.2f}, R: {reward:.2f}")
        
        # 更新动作历史
        self.action_history.append((move_action, turn_action))
        
        # 修改：只有在达到最大步数时才重置，避免重复重置
        if done and not climb_detected:  # 如果不是因为climb而结束，也要重置
            self.logger.info(f"达到最大步数 {self.max_steps}")
            # 在episode结束时重置游戏环境
            self.reset_to_origin()
        
        return new_state, reward, done, detection_results 

    def calculate_reward(self, detection_results, last_area, action_taken):
        """
        计算综合奖励
        """
        # 目标检测奖励
        target_reward = self._calculate_target_presence_reward(detection_results)
        
        # 进展奖励
        progress_reward = self._calculate_progress_reward(detection_results)
        
        # 重复动作惩罚
        repetition_penalty = self._calculate_repetition_penalty(action_taken)
        
        # 总奖励
        total_reward = target_reward + progress_reward + repetition_penalty
        
        # 计算新区域面积
        new_area = self._get_max_detection_area(detection_results) if detection_results else 0
        
        return total_reward, new_area

    def _save_detection_image_with_bounding_boxes(self, image, detection_results, prefix="detection"):
        """
        保存带检测框的图像，并保留最近5个
        """
        try:
            # 复制图像以避免修改原始图像
            img_with_boxes = image.copy()
            
            # 绘制检测框
            for detection in detection_results:
                bbox = detection['bbox']
                label = detection['label']
                score = detection['score']
                
                # 转换边界框坐标为整数
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制矩形框
                color = (0, 255, 0)  # 绿色框
                thickness = 2
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签和置信度文本
                text = f"{label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_color = (255, 255, 255)  # 白色文字
                text_thickness = 1
                
                # 获取文本框的尺寸
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # 绘制文本背景
                cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # 在图像上绘制文本
                cv2.putText(img_with_boxes, text, (x1, y1 - 5), font, 
                        font_scale, text_color, text_thickness)
            
            # 生成文件名
            timestamp = int(time.time() * 1000)  # 使用毫秒时间戳
            filename = f"{prefix}_detection_{timestamp}.png"
            
            # 确定保存路径
            save_dir = Path(__file__).parent / "detection_images"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = save_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), img_with_boxes)
            
            if success:
                
                
                # 将文件路径添加到最近检测图像队列
                self.recent_detection_images.append(str(filepath))
            else:
                self.logger.error(f"保存检测结果图像失败: {filepath}")
                
        except Exception as e:
            self.logger.error(f"保存带检测框图像时出错: {e}")

    def get_recent_detection_images(self):
        """
        获取最近5个检测图像的路径
        """
        return list(self.recent_detection_images)

    def reset(self):
        """
        重置环境
        """
        self.logger.debug("重置环境")
        self.step_count = 0
        self.last_area = 0
        self.last_detection_result = None
        self.action_history.clear()
        initial_state = self.capture_screen()
        return initial_state

class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        
        # 添加logger
        self.logger = setup_logging()
        
        # Create policy networks
        self.policy = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            betas=self.betas
        )
        self.policy_old = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def act(self, state, memory, return_debug_info=False):
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        
        # 使用旧策略获取动作概率
        with torch.no_grad():
            if return_debug_info:
                move_probs, turn_probs, action_params, state_val, debug_info = self.policy_old(state_tensor.unsqueeze(0), return_debug_info=True)
            else:
                move_probs, turn_probs, action_params, state_val = self.policy_old(state_tensor.unsqueeze(0))
                
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            # 采样动作
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            logprob = move_logprob + turn_logprob
        
        # 处理动作参数
        move_forward_step_raw = action_params[0][0].item()  # [-1,1]范围
        turn_angle_raw = action_params[0][1].item()         # [-1,1]范围
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 将 [-1, 1] 映射到实际范围
        move_forward_step = ((move_forward_step_raw + 1.0) / 2.0) * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = ((turn_angle_raw + 1.0) / 2.0) * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN
        
        # 存储到记忆中
        memory.append(
            state_tensor,
            move_action.item(),
            turn_action.item(),
            logprob.item(),
            0,  # 奖励稍后更新
            False,  # 是否结束稍后更新
            [move_forward_step, turn_angle]  # 存储参数
        )
        
        if return_debug_info:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle, debug_info
        else:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def update(self, memory):
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 转换为张量 - 注意logprobs需要特殊处理
        old_states = torch.stack(memory.states).detach()
        old_actions_move = torch.LongTensor(memory.move_actions).detach()
        old_actions_turn = torch.LongTensor(memory.turn_actions).detach()
        
        # 修改这一行：将logprobs转换为tensor
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).detach()
        
        # 修改这一行：正确处理action_params
        if memory.action_params:
            # 将列表转换为tensor
            action_params_tensor = torch.tensor(memory.action_params, dtype=torch.float32)
            old_values = action_params_tensor.detach()
        else:
            old_values = None

        # PPO更新
        for _ in range(self.K_epochs):
            # 前向传播
            move_probs, turn_probs, action_params, state_vals = self.policy(old_states)

            # 计算对数概率
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            logprobs = move_dist.log_prob(old_actions_move) + turn_dist.log_prob(old_actions_turn)
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算优势
            advantages = rewards - state_vals.squeeze().detach()

            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = self.MseLoss(state_vals.squeeze(), rewards)

            # 熵损失
            move_entropy = move_dist.entropy().mean()
            turn_entropy = turn_dist.entropy().mean()
            entropy_loss = move_entropy + turn_entropy

            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        memory.clear_memory()  
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        expected_height = CONFIG.get('IMAGE_HEIGHT', 480)
        expected_width = CONFIG.get('IMAGE_WIDTH', 640)
        
        if state_rgb.shape[0] != expected_height or state_rgb.shape[1] != expected_width:
            state_rgb = cv2.resize(state_rgb, (expected_width, expected_height))
        
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        
        return state_tensor

    def save_checkpoint(self, filepath, episode, optimizer_state_dict=None):
        """
        保存模型检查点
        """
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': optimizer_state_dict or self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型检查点已保存: {filepath}")

    def load_checkpoint(self, filepath):
        """
        加载模型检查点
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            self.logger.info(f"模型检查点已加载，从第 {start_episode} 轮开始继续训练")
            return start_episode + 1
        else:
            self.logger.info(f"检查点文件不存在: {filepath}")
            return 0