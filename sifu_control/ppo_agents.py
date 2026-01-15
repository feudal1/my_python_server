import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
from PIL import Image
import base64
from collections import deque
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
    return config['config']

CONFIG = load_config()


# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_ppo_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class SimpleFeatureExtractor(nn.Module):
    """
    简单CNN特征提取器，适合目标搜索任务
    """
    def __init__(self, input_channels=3):
        super(SimpleFeatureExtractor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 计算卷积输出大小
        conv_out_size = self._get_conv_out_size(input_channels)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def _get_conv_out_size(self, input_channels):
        """计算卷积层输出大小"""
        x = torch.zeros(1, input_channels, CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'])
        x = self.conv_layers(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


class SimpleActorCritic(nn.Module):
    """
    简单的Actor-Critic网络
    """
    def __init__(self, input_channels=3, move_action_space=4, turn_action_space=2):
        super(SimpleActorCritic, self).__init__()
        
        self.feature_extractor = SimpleFeatureExtractor(input_channels)
        feature_size = 128  # 与fc层输出匹配
        
        # Actor头部
        self.move_actor = nn.Linear(feature_size, move_action_space)
        self.turn_actor = nn.Linear(feature_size, turn_action_space)
        
        # Critic头部
        self.critic = nn.Linear(feature_size, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        
        move_logits = self.move_actor(features)
        turn_logits = self.turn_actor(features)
        
        move_probs = F.softmax(move_logits, dim=-1)
        turn_probs = F.softmax(turn_logits, dim=-1)
        
        state_value = self.critic(features)
        
        return move_probs, turn_probs, state_value


class SimpleMemory:
    """
    简单的经验回放缓冲区
    """
    def __init__(self):
        self.states = []
        self.move_actions = []
        self.turn_actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.move_actions[:]
        del self.turn_actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class SimplePPOAgent:
    """
    简单PPO智能体
    """
    def __init__(self, lr=0.0003, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = SimpleActorCritic()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = SimpleActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        if len(memory.rewards) == 0:
            return
            
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 归一化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        if len(memory.states) == 0:
            return
            
        # 转换为张量
        old_states = torch.stack(memory.states).detach()
        old_move_actions = torch.tensor(memory.move_actions, dtype=torch.long)
        old_turn_actions = torch.tensor(memory.turn_actions, dtype=torch.long)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # K次更新
        for _ in range(self.k_epochs):
            # 前向传播
            move_probs, turn_probs, state_values = self.policy(old_states)
            
            # 创建分布
            move_dist = torch.distributions.Categorical(move_probs)
            turn_dist = torch.distributions.Categorical(turn_probs)
            
            # 计算新的对数概率
            new_move_logprobs = move_dist.log_prob(old_move_actions)
            new_turn_logprobs = turn_dist.log_prob(old_turn_actions)
            new_logprobs = new_move_logprobs + new_turn_logprobs
            
            # 计算比率 (pi_theta / pi_theta_old)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            
            # 计算优势
            advantages = rewards - state_values.detach().squeeze(-1)
            
            # PPO代理损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            critic_loss = self.MseLoss(state_values.squeeze(-1), rewards)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

    def act(self, state, memory):
        """
        根据当前策略选择动作
        """
        state = self._preprocess_state(state)
        state_batch = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            move_probs, turn_probs, _ = self.policy_old(state_batch)
            
            move_dist = torch.distributions.Categorical(move_probs)
            turn_dist = torch.distributions.Categorical(turn_probs)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            action_logprob = move_logprob + turn_logprob
        
        # 随机选择移动距离和转角
        move_forward_step = random.uniform(0.5, 2.0)  # 随机移动距离
        turn_angle = random.uniform(10, 45)  # 随机转角
        
        # 存储经验
        memory.states.append(state)
        memory.move_actions.append(move_action.item())
        memory.turn_actions.append(turn_action.item())
        memory.logprobs.append(action_logprob.item())
        
        return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        return state_tensor

    def save_model(self, filepath):
        """保存模型"""
        torch.save(self.policy.state_dict(), filepath)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            self.policy.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            self.policy_old.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            print(f"模型已加载: {filepath}")
        else:
            print(f"模型文件不存在: {filepath}")


class TargetSearchEnvironment:
    """
    目标搜索环境 - 使用全局配置
    """
    def __init__(self, target_description=None):
        # 如果没有传入target_description，使用默认配置
        if target_description is None:
            target_description = CONFIG['TARGET_DESCRIPTION']
        
        from control_api_tool import ImprovedMovementController
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = CONFIG['ENV_MAX_STEPS']  # 统一使用ENV_MAX_STEPS
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.last_area = 0
        self.logger = logging.getLogger(__name__)
        
        # 记录探索历史
        self.position_history = []
        self.max_history_length = CONFIG['POSITION_HISTORY_LENGTH']
        self.yolo_model = self._load_yolo_model()
        self._warm_up_detection_model()
        
        # 成功条件阈值
        self.MIN_GATE_AREA = CONFIG['MIN_GATE_AREA']
        self.CENTER_THRESHOLD = CONFIG['CENTER_THRESHOLD']

    def reset_to_origin(self):
        """
        重置到原点操作
        """
        self.logger.info("执行重置到原点操作")
        print("执行重置到原点操作...")
        
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
                print(f"未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.2)
               
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(1.0)
        print("重置到原点操作完成")
        self.logger.info("重置到原点操作完成")

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
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': class_name,
                        'score': conf,
                        'width': width,
                        'height': height
                    })
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []

    def calculate_reward(self, detection_results, prev_distance, action_taken=None, prev_area=None):
        """
        改进的奖励函数
        """
        reward = 0.0
        
        # 基于探索的奖励
        exploration_bonus = CONFIG['EXPLORATION_BONUS']
        
        if not detection_results or len(detection_results) == 0:
            reward = CONFIG['NO_DETECTION_PENALTY'] + exploration_bonus
            self.logger.debug(f"未检测到目标，奖励: {reward:.2f}")
            return reward, 0
        
        # 找到最近的检测框和最大面积
        min_distance = float('inf')
        max_area = 0
        
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            img_width = detection.get('img_width', CONFIG['IMAGE_WIDTH'])
            img_height = detection.get('img_height', CONFIG['IMAGE_HEIGHT'])
            
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
            
            area = detection['width'] * detection['height']
            if area > max_area:
                max_area = area

        # 基础奖励
        base_detection_reward = CONFIG['BASE_DETECTION_REWARD']
        reward += base_detection_reward
        self.logger.debug(f"检测到门，基础奖励: {base_detection_reward}")

        return reward, max_area

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
        
        # 检查当前状态是否有climb类别
        pre_climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in pre_action_detections
        )
        
        if pre_climb_detected:
            self.logger.info(f"动作执行前已检测到climb类别，立即终止")
            reward, new_area = self.calculate_reward(pre_action_detections, self.last_center_distance, (move_action, turn_action), self.last_area)
            speed_bonus = CONFIG['CLIMB_REWARD_BONUS'] / max(1, self.step_count + 1)
            reward += speed_bonus
            self.last_area = new_area
            self.last_detection_result = pre_action_detections
            self.step_count += 1
            
            print(f"Step {self.step_count}, Area: {new_area:.2f}, Reward: {reward:.2f}, "
                f"Detected: climb (pre-action), Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}")
            
            return pre_action_state, reward, True, pre_action_detections
        
        # 执行移动动作
        if move_action == 0 and move_forward_step > 0:  # forward
            self.controller.move_forward(duration=move_forward_step*10)
        elif move_action == 1 and move_forward_step > 0:  # backward
            self.controller.move_backward(duration=move_forward_step*10)
        elif move_action == 2 and move_forward_step > 0:  # strafe_left
            self.controller.strafe_left(duration=move_forward_step*10)
        elif move_action == 3 and move_forward_step > 0:  # strafe_right
            self.controller.strafe_right(duration=move_forward_step*10)
        
        # 执行转头动作
        if turn_action == 0 and turn_angle > 0:  # turn_left
            self.controller.turn_left(turn_angle*400, duration=turn_angle)
        elif turn_action == 1 and turn_angle > 0:  # turn_right
            self.controller.turn_right(turn_angle*400, duration=turn_angle)

        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励
        current_distance = self.last_center_distance
        current_area = self.last_area
        area = 0
        if detection_results:
            min_distance = float('inf')
            max_area = 0
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                detection['img_width'] = new_state.shape[1]
                detection['img_height'] = new_state.shape[0]
                
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                
                area = detection['width'] * detection['height']
                if area > max_area:
                    max_area = area
            current_distance = min_distance
            current_area = max_area

        reward, new_area = self.calculate_reward(detection_results, self.last_center_distance, (move_action, turn_action), current_area)
        self.last_center_distance = current_distance
        self.last_area = new_area
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否检测到climb类别
        climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in detection_results
        )

        done = climb_detected or self.step_count >= self.max_steps
        
        # 如果检测到climb，给予额外奖励
        if climb_detected:
            speed_bonus = CONFIG['CLIMB_REWARD_BONUS'] / (self.step_count) 
            reward += speed_bonus
            self.logger.info(f"检测到climb类别！速度奖励: {speed_bonus:.2f}")

        # 输出每步得分
        print(f"Step {self.step_count}, Area: {current_area:.2f}, Reward: {reward:.2f}, "
            f" Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, "
            f"Move Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
        
        # 更新位置历史
        state_feature = len(detection_results) if detection_results else 0
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        if done:
            if climb_detected:
                self.logger.info(f"在第 {self.step_count} 步检测到 climb 类别")
            else:
                self.logger.info(f"达到最大步数 {self.max_steps}")
        
        return new_state, reward, done, detection_results
    
    def reset(self):
        """
        重置环境
        """
        self.logger.debug("重置环境")
        self.step_count = 0
        self.last_center_distance = float('inf')
        self.last_area = 0
        self.last_detection_result = None
        self.position_history = []
        initial_state = self.capture_screen()
        return initial_state


class MediumFeatureExtractor(nn.Module):
    """
    中等规模CNN特征提取器，适度增加参数量
    """
    def __init__(self, input_channels=3):
        super(MediumFeatureExtractor, self).__init__()
        
        # 中等规模CNN特征提取器
        self.conv_layers = nn.Sequential(
            # 第一层 - 适度减少通道数和步长
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四层 - 增加一层提高特征表达能力
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 展平


class MediumActorCritic(nn.Module):
    """
    中等规模Actor-Critic网络，改进版 - 分离的actor/critic路径
    """
    def __init__(self, input_channels=3, move_action_space=4, turn_action_space=2, image_height=480, image_width=640):
        super(MediumActorCritic, self).__init__()
        
        # 使用中等规模特征提取器
        self.feature_extractor = MediumFeatureExtractor(input_channels)
        
        conv_out_size = 64  # 64 * 1 * 1
        
        # 分离的actor和critic全连接层，避免参数共享
        self.actor_shared = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.critic_shared = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 动作头
        self.move_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, move_action_space)
        )
        self.turn_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, turn_action_space)
        )
        
        # 独立的参数预测头
        self.param_head = nn.Sequential(
            nn.Linear(64, 64),  # 增加中间层
            nn.ReLU(),
            nn.Dropout(0.2),    # 增加dropout防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),  # 增加中间层
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        actor_features = self.actor_shared(features)
        critic_features = self.critic_shared(features)
        
        move_action_probs = F.softmax(self.move_action_head(actor_features), dim=-1)
        turn_action_probs = F.softmax(self.turn_action_head(actor_features), dim=-1)
        action_params = torch.sigmoid(self.param_head(actor_features))  # 使用actor特征
        state_value = self.value_head(critic_features)  # 使用critic特征
        
        return move_action_probs, turn_action_probs, action_params, state_value


class Memory:
    """
    存储智能体的经验数据
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


class PPOAgent:
    """
    PPO智能体 - 使用全局配置
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        
        input_channels = 3
        height, width = 480, 640
        # 使用改进的网络架构
        self.policy = MediumActorCritic(input_channels, move_action_dim, turn_action_dim, height, width)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-4)
        self.policy_old = MediumActorCritic(input_channels, move_action_dim, turn_action_dim, height, width)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        if len(memory.rewards) == 0:
            return
            
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        if len(memory.states) == 0:
            return
            
        # 适度批次大小以平衡速度和稳定性
        old_states = torch.stack(memory.states).detach()
        old_move_actions = torch.tensor(memory.move_actions, dtype=torch.long)
        old_turn_actions = torch.tensor(memory.turn_actions, dtype=torch.long)
        old_action_params = torch.tensor(memory.action_params if hasattr(memory, 'action_params') else [[0, 0]] * len(memory.rewards), dtype=torch.float32)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # 适度调整更新轮数和批次大小
        for epoch in range(self.K_epochs):
            batch_size = CONFIG['BATCH_SIZE']  # 适度增加批次大小
            for i in range(0, len(old_states), batch_size):
                batch_states = old_states[i:i+batch_size]
                batch_move_actions = old_move_actions[i:i+batch_size]
                batch_turn_actions = old_turn_actions[i:i+batch_size]
                batch_action_params = old_action_params[i:i+batch_size]
                batch_logprobs = old_logprobs[i:i+batch_size]
                batch_rewards = rewards[i:i+batch_size]
                
                move_action_probs, turn_action_probs, action_params, state_values = self.policy(batch_states)
                
                move_dist = torch.distributions.Categorical(move_action_probs)
                turn_dist = torch.distributions.Categorical(turn_action_probs)
                
                new_move_logprobs = move_dist.log_prob(batch_move_actions)
                new_turn_logprobs = turn_dist.log_prob(batch_turn_actions)
                new_logprobs = new_move_logprobs + new_turn_logprobs
                
                advantages = batch_rewards - state_values.detach().squeeze(-1)
                ratios = torch.exp(new_logprobs - batch_logprobs.detach())
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                param_loss = F.mse_loss(action_params, batch_action_params) if batch_action_params.numel() > 0 else 0
                critic_loss = self.MseLoss(state_values.squeeze(-1), batch_rewards)
                
                # 调整损失权重，增强对连续参数的学习
                loss = actor_loss + 0.5 * critic_loss + 0.5 * param_loss  # 提高param_loss权重
                
                self.optimizer.zero_grad()
                # 适度梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=CONFIG['GRADIENT_CLIP_NORM'])
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作（增加探索机制）
        """
        state = self._preprocess_state(state)
        
        # 添加批次维度进行推理
        state_batch = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            move_action_probs, turn_action_probs, action_params, state_value = self.policy_old(state_batch)
            
            # 添加噪声到连续参数，增加探索
            noise_scale = CONFIG['NOISE_SCALE']  # 可调参数
            noisy_params = action_params + torch.randn_like(action_params) * noise_scale
            noisy_params = torch.clamp(noisy_params, 0, 1)  # 确保在[0,1]范围内
            
            move_dist = torch.distributions.Categorical(move_action_probs)
            turn_dist = torch.distributions.Categorical(turn_action_probs)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算组合动作的对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            action_logprob = move_logprob + turn_logprob  # 总对数概率
        
        # 使用带噪声的参数
        scaled_params = noisy_params.squeeze(0)
        move_forward_step = scaled_params[0].item()  # 映射到[0, 1.5]范围
        turn_angle = scaled_params[1].item()  # 映射到[0, 40]范围
        
        # 存储状态、动作和对数概率
        # 不要存储带批次维度的状态，只存储原始状态
        memory.states.append(state)  # 不带批次维度
        memory.move_actions.append(move_action.item())  # 存储移动动作
        memory.turn_actions.append(turn_action.item())  # 存储转向动作
        memory.logprobs.append(action_logprob.item())  # 确保是标量值
        if not hasattr(memory, 'action_params'):
            memory.action_params = []
        memory.action_params.append(scaled_params.tolist())  # 存储动作参数
        
        return move_action.item(), turn_action.item(), move_forward_step, turn_angle
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        import cv2
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        
        # 保持原始尺寸
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
        print(f"模型检查点已保存: {filepath}")
    
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
            print(f"模型检查点已加载，从第 {start_episode} 轮开始继续训练")
            return start_episode + 1
        else:
            print(f"检查点文件不存在: {filepath}")
            return 0


# GRU相关组件
class GRUFeatureExtractor(nn.Module):
    """
    带有GRU的特征提取器，能够处理时序信息
    """
    def __init__(self, input_channels=3, sequence_length=4, hidden_size=128):
        super(GRUFeatureExtractor, self).__init__()
        
        self.sequence_length = sequence_length
        
        # CNN特征提取器
        self.conv_layers = nn.Sequential(
            # 第一层 - 适度减少通道数和步长
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=3, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四层 - 增加一层提高特征表达能力
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 计算CNN输出大小
        conv_out_size = 64  # 64 * 1 * 1
        
        # GRU层处理时序信息
        self.gru = nn.GRU(
            input_size=conv_out_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 输出投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, conv_out_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        输入x形状: (batch_size, sequence_length, channels, height, width)
        或者 (batch_size, channels, height, width) - 如果是单帧
        """
        if len(x.shape) == 5:
            # 处理序列输入
            batch_size, seq_len, channels, height, width = x.shape
            
            # 将序列维度合并到批次维度进行CNN处理
            x = x.view(batch_size * seq_len, channels, height, width)
            features = self.conv_layers(x)
            features = self.global_pool(features)
            features = features.view(batch_size, seq_len, -1)  # (batch, seq, features)
            
            # 通过GRU处理时序信息
            gru_output, _ = self.gru(features)
            # 取最后一个时间步的输出
            output = gru_output[:, -1, :]  # (batch, hidden_size)
            output = self.projection(output)
        else:
            # 处理单帧输入
            features = self.conv_layers(x)
            features = self.global_pool(features)
            output = features.view(x.size(0), -1)  # 展平
        
        return output


class GRUActorCritic(nn.Module):
    """
    带有GRU的Actor-Critic网络，能够利用历史信息
    """
    def __init__(self, input_channels=3, move_action_space=4, turn_action_space=2, 
                 image_height=480, image_width=640, sequence_length=4, hidden_size=128):
        super(GRUActorCritic, self).__init__()
        
        # 使用带有GRU的特征提取器
        self.feature_extractor = GRUFeatureExtractor(
            input_channels, sequence_length, hidden_size
        )
        
        conv_out_size = 64  # CNN输出特征维度
        gru_hidden_size = hidden_size  # GRU隐藏层维度
        
        # 分离的actor和critic全连接层
        self.actor_shared = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.critic_shared = nn.Sequential(
            nn.Linear(gru_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 动作头
        self.move_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, move_action_space)
        )
        self.turn_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, turn_action_space)
        )
        
        # 独立的参数预测头
        self.param_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        actor_features = self.actor_shared(features)
        critic_features = self.critic_shared(features)
        
        move_action_probs = F.softmax(self.move_action_head(actor_features), dim=-1)
        turn_action_probs = F.softmax(self.turn_action_head(actor_features), dim=-1)
        action_params = torch.sigmoid(self.param_head(actor_features))
        state_value = self.value_head(critic_features)
        
        return move_action_probs, turn_action_probs, action_params, state_value


class GRUMemory:
    """
    存储智能体的经验数据，支持序列数据
    """
    def __init__(self, sequence_length=4):
        self.sequence_length = sequence_length
        self.states = []
        self.move_actions = []
        self.turn_actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_params = []
        # 用于构建状态序列
        self.state_buffer = deque(maxlen=sequence_length)

    def push(self, state, move_action, turn_action, logprob, reward, is_terminal, action_param):
        """添加经验到缓冲区"""
        self.states.append(state)
        self.move_actions.append(move_action)
        self.turn_actions.append(turn_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.action_params.append(action_param)

    def get_state_sequence(self, index):
        """获取指定索引处的状态序列"""
        start_idx = max(0, index - self.sequence_length + 1)
        end_idx = index + 1
        
        # 获取状态序列
        sequence = list(self.states[start_idx:end_idx])
        
        # 如果序列长度不足，用重复第一个状态补足
        while len(sequence) < self.sequence_length:
            sequence.insert(0, sequence[0])
        
        return torch.stack(sequence)

    def clear_memory(self):
        del self.states[:]
        del self.move_actions[:]
        del self.turn_actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_params[:]
        self.state_buffer.clear()

    def get_sequences(self):
        """获取所有状态序列"""
        sequences = []
        for i in range(len(self.states)):
            seq = self.get_state_sequence(i)
            sequences.append(seq)
        return torch.stack(sequences)


class GRUPPOAgent:
    """
    使用GRU的PPO智能体 - 使用全局配置
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        self.sequence_length = config['SEQUENCE_LENGTH']
        
        input_channels = 3
        height, width = 480, 640
        
        # 使用GRU网络架构
        self.policy = GRUActorCritic(
            input_channels, move_action_dim, turn_action_dim, 
            height, width, self.sequence_length, config['HIDDEN_SIZE']
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-4
        )
        self.policy_old = GRUActorCritic(
            input_channels, move_action_dim, turn_action_dim, 
            height, width, self.sequence_length, config['HIDDEN_SIZE']
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # 用于在推理时维护状态序列
        self.state_history = deque(maxlen=self.sequence_length)

    def update(self, memory):
        if len(memory.rewards) == 0:
            return
            
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        if len(memory.states) == 0:
            return
            
        # 构建状态序列
        state_sequences = memory.get_sequences()
        old_move_actions = torch.tensor(memory.move_actions, dtype=torch.long)
        old_turn_actions = torch.tensor(memory.turn_actions, dtype=torch.long)
        old_action_params = torch.tensor(
            memory.action_params if hasattr(memory, 'action_params') else 
            [[0, 0]] * len(memory.rewards), dtype=torch.float32
        )
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # 训练更新
        for epoch in range(self.K_epochs):
            batch_size = CONFIG['BATCH_SIZE']
            for i in range(0, len(state_sequences), batch_size):
                batch_states = state_sequences[i:i+batch_size]
                batch_move_actions = old_move_actions[i:i+batch_size]
                batch_turn_actions = old_turn_actions[i:i+batch_size]
                batch_action_params = old_action_params[i:i+batch_size]
                batch_logprobs = old_logprobs[i:i+batch_size]
                batch_rewards = rewards[i:i+batch_size]
                
                move_action_probs, turn_action_probs, action_params, state_values = self.policy(batch_states)
                
                move_dist = torch.distributions.Categorical(move_action_probs)
                turn_dist = torch.distributions.Categorical(turn_action_probs)
                
                new_move_logprobs = move_dist.log_prob(batch_move_actions)
                new_turn_logprobs = turn_dist.log_prob(batch_turn_actions)
                new_logprobs = new_move_logprobs + new_turn_logprobs
                
                advantages = batch_rewards - state_values.detach().squeeze(-1)
                ratios = torch.exp(new_logprobs - batch_logprobs.detach())
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                param_loss = F.mse_loss(action_params, batch_action_params) if batch_action_params.numel() > 0 else 0
                critic_loss = self.MseLoss(state_values.squeeze(-1), batch_rewards)
                
                loss = actor_loss + 0.5 * critic_loss + 0.5 * param_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=CONFIG['GRADIENT_CLIP_NORM'])
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作，使用GRU处理历史状态
        """
        state = self._preprocess_state(state)
        
        # 将当前状态添加到历史记录
        self.state_history.append(state)
        
        # 构建状态序列
        if len(self.state_history) < self.sequence_length:
            # 如果历史记录不够，复制最早的状态填充
            while len(self.state_history) < self.sequence_length:
                self.state_history.appendleft(self.state_history[0])
        
        # 转换为张量
        state_seq = torch.stack(list(self.state_history)).unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            move_action_probs, turn_action_probs, action_params, state_value = self.policy_old(state_seq)
            
            # 添加噪声到连续参数，增加探索
            noise_scale = CONFIG['NOISE_SCALE']
            noisy_params = action_params + torch.randn_like(action_params) * noise_scale
            noisy_params = torch.clamp(noisy_params, 0, 1)
            
            move_dist = torch.distributions.Categorical(move_action_probs)
            turn_dist = torch.distributions.Categorical(turn_action_probs)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            action_logprob = move_logprob + turn_logprob
        
        # 使用带噪声的参数
        scaled_params = noisy_params.squeeze(0)
        move_forward_step = scaled_params[0].item()
        turn_angle = scaled_params[1].item()
        
        # 存储经验
        memory.push(
            state, move_action.item(), turn_action.item(), 
            action_logprob.item(), 0, False, scaled_params.tolist()
        )
        
        return move_action.item(), turn_action.item(), move_forward_step, turn_angle
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        import cv2
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
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
        print(f"模型检查点已保存: {filepath}")
    
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
            print(f"模型检查点已加载，从第 {start_episode} 轮开始继续训练")
            return start_episode + 1
        else:
            print(f"检查点文件不存在: {filepath}")
            return 0