"""
门搜索PPO实现
使用PPO算法在虚拟环境中搜索目标
"""

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
import pyautogui  # 添加pyautogui库用于按键操作


# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器w
from control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 50  # 增加最大步数，给更多探索机会
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.last_area = 0  # 新增：跟踪上一帧目标区域
        self.logger = logging.getLogger(__name__)
        
        # 记录探索历史，帮助判断是否在原地打转
        self.position_history = []
        self.max_history_length = 10
        self.yolo_model = self._load_yolo_model()
        # 预先初始化检测模型，确保在执行任何动作前模型已加载
        self._warm_up_detection_model()
        
        # 新增：定义成功条件的阈值
        self.MIN_GATE_AREA = 30000  # 根据实际情况调整，降低阈值以便更容易达成目标
        self.CENTER_THRESHOLD = 80  # 放宽居中要求
        

    def reset_to_origin(self):
        """
        重置到原点操作：按下ESC键，按Q键，按下回车键，检测门，若无门则等待1秒按回车再次检测，最后再按一次回车
        """
        self.logger.info("执行重置到原点操作")
        print("执行重置到原点操作...")
        
        # 按键操作序列
        pyautogui.press('esc')
        time.sleep(0.5)  # 等待0.5秒
        pyautogui.press('q')
        time.sleep(0.5)  # 等待0.5秒
        pyautogui.press('enter')
        time.sleep(0.5)  # 等待0.5秒
        pyautogui.press('enter')
        time.sleep(0.5)  # 等待0.5秒
        # 检测门是否存在
        gate_detected = False
        while not gate_detected:
            # 重新截图并检测门
            new_state = self.capture_screen()
            detection_results = self.detect_target(new_state)
            
            # 检查是否检测到了门
            for detection in detection_results:
                if detection['label'].lower() == 'gate' or 'gate' in detection['label'].lower():
                    gate_detected = True
                    time.sleep(0.5)
                    pyautogui.press('enter')
                    time.sleep(1)
                    pyautogui.press('enter')
                    break
            
            if not gate_detected:
                print("未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.5)  # 等待0.5秒
                        
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.5)  # 等待0.5秒
        pyautogui.press('enter')
        time.sleep(10)
        print("重置到原点操作完成")
        self.logger.info("重置到原点操作完成")

    def _load_yolo_model(self):
        """
        加载YOLO模型
        """
        # 获取项目根目录
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
        预热检测模型，确保在训练开始前模型已加载到内存
        """
        self.logger.info("正在预热检测模型，确保模型已加载...")
        try:
            # 在环境初始化时立即加载模型
            dummy_image = self.capture_screen()
            if dummy_image is not None and dummy_image.size > 0:
                # 对当前屏幕截图进行检测，触发模型加载
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
        # 使用现有的浏览器截图功能，修改为截取名称包含"sifu"的窗口
        try:
            from computer_server.prtsc import capture_window_by_title
            # 截取标题包含"sifu"的窗口
            result = capture_window_by_title("sifu", "sifu_window_capture.png")
            if result:
                # 读取保存的图片
                from PIL import Image
                screenshot = Image.open("sifu_window_capture.png")
                # 转换为numpy数组
                import numpy as np
                screenshot = np.array(screenshot)
                # 由于PIL读取的图像格式是RGB，而OpenCV使用BGR，所以需要转换
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            # 如果没有截图功能，模拟返回一张图片
            self.logger.warning("截图功能不可用，使用模拟图片")
            return np.zeros((480, 640, 3), dtype=np.uint8_)
    
    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []
        
        try:
            # 进行预测 - 降低置信度阈值以增加检测敏感性
            results = self.yolo_model.predict(
                source=image,
                conf=0.1,  # 从0.2降低到0.1，提高检测敏感性
                save=False,
                 verbose=False
            )
            
            # 获取检测结果
            detections = []
            result = results[0]  # 获取第一个结果
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标 [x1, y1, x2, y2]
                confs = result.boxes.conf.cpu().numpy()  # 获取置信度
                cls_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID
                
                # 获取类别名称（如果模型有类别名称）
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")
                    
                    # 计算边界框中心坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 计算边界框宽度和高度
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],  # 左上角和右下角坐标
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
    
    def calculate_exploration_bonus(self):
        """
        计算探索奖励，避免在原地打转
        简化版本：只检查连续相同状态的数量
        """
        if len(self.position_history) < 2:
            return 0.0
        
        # 检查最近的状态是否重复过多（可能表示在原地打转）
        recent_states = self.position_history[-5:] if len(self.position_history) >= 5 else self.position_history
        most_common_state = max(set(recent_states), key=recent_states.count)
        repetition_count = recent_states.count(most_common_state)
        
        # 如果最近5个状态中有4个以上是相同的，说明可能在原地打转
        if repetition_count >= 4:
            return -0.1  # 给予轻微惩罚，鼓励改变行为
        else:
            return 0.0   # 不给予奖励，专注于主要的奖励信号
    
    def calculate_reward(self, detection_results, prev_distance, action_taken=None, prev_area=None):
        """
        改进的奖励函数：弱化居中奖励，强化面积增长，移除首次检测的高额奖励
        """
        reward = 0.0
        
        if not detection_results or len(detection_results) == 0:
            # 没有检测到目标，给予轻微惩罚
            exploration_bonus = self.calculate_exploration_bonus()
            reward = -0.2 + exploration_bonus
            self.logger.debug(f"未检测到目标，奖励: {reward:.2f}")
            return reward, 0
        
        # 找到最近的检测框和最大面积
        min_distance = float('inf')
        max_area = 0
        
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            img_width = detection.get('img_width', 640)
            img_height = detection.get('img_height', 480)
            
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
            
            area = detection['width'] * detection['height']
            if area > max_area:
                max_area = area
        
        # 基于面积变化的奖励（主要奖励）
        area_reward = 0.0
        if prev_area is not None and prev_area > 0:
            area_ratio = max_area / prev_area
            if area_ratio > 1.05:  # 面积增大超过5%
                area_bonus = 2.0 * (area_ratio - 1.0)  # 调整面积奖励权重
                area_reward = max(area_bonus, 0.2)  # 确保最小正奖励
                self.logger.debug(f"目标面积增大 {area_ratio:.2f}倍，面积奖励: {area_reward:.2f}")
            elif area_ratio >= 0.95:  # 面积变化在5%以内
                area_reward = 0.1
                self.logger.debug(f"目标面积基本不变，小奖励: {area_reward:.2f}")
            else:  # 面积减少
                area_penalty = -0.5 * (1.0 - area_ratio)
                area_reward = area_penalty
                self.logger.debug(f"目标面积减小，惩罚: {area_reward:.2f}")
        else:
            # 第一次检测到目标时的奖励，降低
            area_reward = 0.3
            self.logger.debug(f"首次检测到目标，奖励: {area_reward:.2f}")
        
        # 基于距离的奖励（辅助）
        distance_reward = 0.0
        if prev_distance != float('inf'):
            if min_distance < prev_distance:
                distance_improve = (prev_distance - min_distance) / prev_distance
                distance_reward = 0.5 * distance_improve  # 降低距离奖励权重
                self.logger.debug(f"距离变近，距离奖励: {distance_reward:.2f}")
            else:
                distance_decrease = (min_distance - prev_distance) / prev_distance
                if area_ratio > 1.1:  # 面积增大超过10%
                    distance_reward = 0.1
                    self.logger.debug(f"面积增大但距离变远，小奖励: {distance_reward:.2f}")
                else:
                    distance_reward = -0.2 * distance_decrease
                    self.logger.debug(f"距离变远且面积无改善，惩罚: {distance_reward:.2f}")
        
        # 改进的居中奖励（不再是固定值）
        center_reward = 0.0
        if min_distance < 80:  # 居中奖励与面积相关
            center_bonus = min(0.5, (max_area / 20000))  # 面积越大，居中奖励越高，但上限较低
            center_reward = center_bonus
            self.logger.debug(f"居中奖励: {center_reward:.2f}")
        
        # 综合奖励
        reward = area_reward + distance_reward + center_reward
        
        # 鼓励朝着目标方向移动
        if action_taken is not None and prev_distance != float('inf'):
            if min_distance < prev_distance:
                reward += 0.1
        
        return reward, max_area
    
  
    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束
        动作空间: 0-forward, 1-backward, 2-turn_left, 3-turn_right, 4-strafe_left, 5-strafe_right
        """
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        self.logger.debug(f"执行动作: {action_names[action]}")
        
        # 动态调整动作步长
        current_area = self.last_area
        move_forward_step = 2
        turn_angle = 30
        
        # 如果目标已经比较大，使用更精细的动作
        if current_area > 20000:  # 目标已经比较近
            move_forward_step = 1  # 更小的移动步长
            turn_angle = 15       # 更小的转向角度
        
        # 执行动作
        if action == 0:  # forward
            self.controller.move_forward(move_forward_step)
        elif action == 1:  # backward
            self.controller.move_backward(max(1, move_forward_step // 2))
        elif action == 2:  # turn_left
            self.controller.turn_left(turn_angle)
        elif action == 3:  # turn_right
            self.controller.turn_right(turn_angle)
        elif action == 4:  # strafe_left
            self.controller.strafe_left(max(1, move_forward_step // 2))
        elif action == 5:  # strafe_right
            self.controller.strafe_right(max(1, move_forward_step // 2))
        
        time.sleep(0.2)  # 保持延迟
        
        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励 - 添加当前目标区域
        current_distance = self.last_center_distance
        current_area = self.last_area  # 获取上一帧的目标区域
        area = 0
        if detection_results:
            # 计算当前最近目标到中心的距离
            min_distance = float('inf')
            max_area = 0
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2  # 修复：应该是bbox[3]而不是bbox[2]
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                # 添加检测信息到detection字典
                detection['img_width'] = new_state.shape[1]
                detection['img_height'] = new_state.shape[0]
                
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                
                # 计算目标区域
                area = detection['width'] * detection['height']
                if area > max_area:
                    max_area = area
            current_distance = min_distance
            current_area = max_area

        reward, new_area = self.calculate_reward(detection_results, self.last_center_distance, action, current_area)
        self.last_center_distance = current_distance
        self.last_area = new_area  # 保存当前目标区域
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否成功找到门（新条件）
        gate_found_and_close = (
            detection_results 
            and current_distance < self.CENTER_THRESHOLD 
            and current_area > self.MIN_GATE_AREA
        )
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or gate_found_and_close
        
        # 如果成功找到门，给予额外奖励
        if gate_found_and_close:
            reward += 10.0  # 成功完成任务的大奖励
            self.logger.info(f"成功找到目标！额外奖励: 10.0, 当前面积: {current_area}")
        
        # 输出每步得分
        detected_targets = len(detection_results) if detection_results else 0
        print(f"Step {self.step_count}, Area: {current_area:.2f}, Reward: {reward:.2f}, "
              f"Targets Detected: {detected_targets}, Distance to Center: {current_distance:.2f}, "
              f"Done: {done}")
        
        # 更新位置历史，记录当前状态的特征（如检测结果数量）
        state_feature = len(detection_results)  # 这里用检测到的对象数量作为状态特征
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        if done:
            if gate_found_and_close:
                self.logger.info(f"在第 {self.step_count} 步成功找到目标，面积: {current_area}")
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
        self.last_area = 0  # 重置目标区域
        self.last_detection_result = None
        self.position_history = []  # 重置位置历史
        initial_state = self.capture_screen()
        return initial_state


class ActorCritic(nn.Module):
    """
    PPO使用的Actor-Critic网络
    """
    def __init__(self, input_channels=3, action_space=6, image_height=480, image_width=640):
        super(ActorCritic, self).__init__()
        
        # 卷积层处理图像输入
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 自适应平均池化层，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))  # 固定输出为10x10
        
        # 全连接层 - 输入大小是固定的，因为池化层输出固定尺寸
        conv_out_size = 64 * 10 * 10  # 64个通道 * 10 * 10 = 6400
        
        # Actor (策略网络) - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        # Critic (价值网络) - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # 确保输出尺寸一致
        x = x.view(x.size(0), -1)  # 展平
        
        # 分别计算动作概率和状态价值
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value


class Memory:
    """
    存储智能体的经验数据
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_dim, action_dim, lr=0.0003, betas=(0.9, 0.999), gamma=0.99, K_epochs=40, eps_clip=0.1):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        # 修正：不再使用state_dim参数中的高度和宽度，而是使用固定尺寸
        input_channels = 3
        height, width = 480, 640  # 固定网络输入尺寸
        self.policy = ActorCritic(input_channels, action_dim, height, width)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(input_channels, action_dim, height, width)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
    
    def update(self, memory):
        # 检查是否有经验数据
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
        
        # 检查内存中是否有足够的数据
        if len(memory.states) == 0 or len(memory.actions) == 0 or len(memory.logprobs) == 0:
            self.logger.warning("Memory is empty, skipping update")
            return
            
        # 处理状态张量 - 确保所有状态都是一致的形状
        try:
            # 将状态列表堆叠成批量张量
            # memory.states中的每个元素应该是一个形状为(C, H, W)的张量
            old_states = torch.stack(memory.states).detach()
            
            # 确保所有动作和对数概率张量形状一致
            old_actions_list = []
            old_logprobs_list = []
            
            for action, logprob in zip(memory.actions, memory.logprobs):
                # 确保动作是标量张量
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.long)
                elif action.dim() == 0:  # 如果已经是标量
                    pass
                elif action.dim() == 1 and action.numel() == 1:  # 如果是单元素一维张量
                    action = action.item()
                    action = torch.tensor(action, dtype=torch.long)
                
                # 确保对数概率是标量张量
                if not isinstance(logprob, torch.Tensor):
                    logprob = torch.tensor(logprob, dtype=torch.float32)
                elif logprob.dim() == 0:  # 如果已经是标量
                    pass
                elif logprob.dim() == 1 and logprob.numel() == 1:  # 如果是单元素一维张量
                    logprob = logprob.item()
                    logprob = torch.tensor(logprob, dtype=torch.float32)
                
                old_actions_list.append(action)
                old_logprobs_list.append(logprob)
            
            # 现在堆叠处理后的张量
            old_actions = torch.stack(old_actions_list).detach()
            old_logprobs = torch.stack(old_logprobs_list).detach()
        
        except RuntimeError as e:
            self.logger.error(f"Error stacking tensors: {e}")
            self.logger.error(f"State shapes: {[s.shape for s in memory.states[:5]]}")
            self.logger.error(f"Action shapes: {[a.shape if isinstance(a, torch.Tensor) else type(a) for a in memory.actions[:5]]}")
            self.logger.error(f"Logprob shapes: {[l.shape if isinstance(l, torch.Tensor) else type(l) for l in memory.logprobs[:5]]}")
            return
        
        # K epochs更新策略 - 减少K_epochs以提高稳定性
        for _ in range(self.K_epochs):
            # 计算优势
            logprobs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(logprobs)
            entropy = dist.entropy().mean()
            
            # 计算动作的对数概率
            new_logprobs = dist.log_prob(old_actions)
            
            # 计算优势
            advantages = rewards - state_values.detach().squeeze(-1)
            
            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            
            # 计算PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            critic_loss = self.MseLoss(state_values.squeeze(-1), rewards)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # 保持熵权重不变
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 限制梯度范数以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作
        """
        state = self._preprocess_state(state)
        
        # 添加批次维度进行推理
        state_batch = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            action_probs, state_value = self.policy_old(state_batch)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        # 存储状态、动作和对数概率
        # 不要存储带批次维度的状态，只存储原始状态
        memory.states.append(state)  # 不带批次维度
        memory.actions.append(action.item())  # 确保是标量值
        memory.logprobs.append(action_logprob.item())  # 确保是标量值
        
        return action.item()
    
    def evaluate(self, state):
        """
        评估状态-动作对的价值
        """
        action_probs, state_value = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        
        return dist, state_value.squeeze(-1)
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        
        # 统一调整图像尺寸为网络期望的大小
        import torch.nn.functional as F
        state_tensor = F.interpolate(
            state_tensor.unsqueeze(0),  # 添加批次维度
            size=(480, 640),           # 目标尺寸
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # 移除批次维度
        
        return state_tensor


def train_gate_search_ppo_agent(episodes=200, model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    训练门搜索PPO智能体
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练门搜索PPO智能体，目标: {target_description}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 初始化环境和智能体
    env = TargetSearchEnvironment(target_description)
    
    # 定义状态和动作维度 - 这里我们只需要动作维度，因为网络使用固定尺寸
    action_dim = 6  # 6种动作类型
    
    # 调整PPO超参数以适应新的奖励结构
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        action_dim,
        lr=0.0003,
        gamma=0.95,      # 降低gamma，让智能体更关注近期奖励
        K_epochs=30,     # 适当减少更新轮数
        eps_clip=0.1
    )
    memory = Memory()
    
    scores = deque(maxlen=100)
    total_rewards = deque(maxlen=100)
    final_areas = deque(maxlen=100)  # 记录最终门面积
    
    # 新增：累积多个episode的数据用于批量更新
    batch_memory = Memory()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0  # 记录本episode的最终门面积
        
        while not done:
            # 选择动作 - 这会在act方法中自动添加状态、动作和对数概率到memory
            action = ppo_agent.act(state, memory)
            
            # 执行动作
            next_state, reward, done, detection_results = env.step(action)
            
            # 存储奖励和终止标志到当前episode的memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 记录最终面积
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results)
            
            if done:
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Episode: {episode+1}/{episodes}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                # 在每个episode结束后执行重置到原点操作
                env.reset_to_origin()
                break

        # 将当前episode的数据添加到批量memory中
        batch_memory.states.extend(memory.states)
        batch_memory.actions.extend(memory.actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        
        # 清空当前episode的memory
        memory.clear_memory()
        
        # 每10个episode更新一次策略
        if (episode + 1) % 10 == 0:
            print(f"Updating PPO policy for episodes {(episode-9)} to {episode+1} (this may take a moment)...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"Policy updated in {update_end_time - update_start_time:.2f} seconds.")
            
            # 清空批量memory
            batch_memory.clear_memory()
        
        # 每10个episode保存一次模型
        if (episode + 1) % 10 == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            torch.save(ppo_agent.policy.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved at episode {episode+1}: {checkpoint_path}")

        # 每20轮打印一次平均分数
        if episode % 20 == 0 and episode > 0:
            avg_score = np.mean(scores)
            avg_total_reward = np.mean(total_rewards)
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, Average Total Reward: {avg_total_reward:.2f}, Average Final Area: {avg_final_area:.2f}")
    
    # 如果还有剩余的episode数据未更新，最后再更新一次
    if len(batch_memory.rewards) > 0:
        print(f"Updating final PPO policy with remaining episodes...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    logger.info("PPO训练完成！")
    
    # 保存最终模型
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"PPO模型已保存为 {model_path}")
    
    return ppo_agent


def evaluate_trained_ppo_agent(model_path="gate_search_ppo_model.pth", episodes=10, target_description="gate"):
    """
    评估已训练好的PPO模型性能
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估PPO模型: {model_path}")
    
    # 定义状态和动作维度
    action_dim = 6
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件 {model_path} 不存在，使用随机模型")
    
    evaluation_results = []
    
    for episode in range(episodes):
        # 初始化环境
        env = TargetSearchEnvironment(target_description)
        state = env.reset()
        memory = Memory()  # 创建空的记忆对象，仅用于act函数
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        while not done and step_count < 50:
            action = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据，因为我们只是在评估
            memory.clear_memory()
            
            # 打印动作
            action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - {action_names[action]}")
            
            state, reward, done, detection_results = env.step(action)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {target_description}")
                
            # 检查是否成功找到门
            current_distance = float('inf')
            current_area = 0
            if detection_results:
                for detection in detection_results:
                    bbox = detection['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    img_center_x = state.shape[1] / 2
                    img_center_y = state.shape[0] / 2
                    distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                    area = detection['width'] * detection['height']
                    if distance < current_distance:
                        current_distance = distance
                        current_area = area
            
            if detection_results and current_distance < env.CENTER_THRESHOLD and current_area > env.MIN_GATE_AREA:
                success = True
                logger.info(f"Episode {episode+1}: 成功找到目标！")
        
        result = {
            'episode': episode + 1,
            'steps': step_count,
            'total_reward': total_reward,
            'success': success,
            'final_area': current_area
        }
        evaluation_results.append(result)
        
        # 在每个episode结束后执行重置到原点操作
        env.reset_to_origin()
        
        logger.info(f"Episode {episode+1} 完成 - Steps: {step_count}, Total Reward: {total_reward}, Success: {success}, Final Area: {current_area}")
    
    # 计算总体统计信息
    successful_episodes = sum(1 for r in evaluation_results if r['success'])
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
    avg_final_area = np.mean([r['final_area'] for r in evaluation_results])
    
    stats = {
        'total_episodes': episodes,
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / episodes if episodes > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'avg_final_area': avg_final_area,
        'details': evaluation_results
    }
    
    logger.info(f"PPO评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{episodes})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}, 平均最终面积: {avg_final_area:.2f}")
    
    return stats


def load_and_test_ppo_agent(model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    加载已训练的PPO模型并进行测试
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载PPO模型并测试: {model_path}")
    
    # 定义动作维度
    action_dim = 6
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 初始化环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < 50:
        action = ppo_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        # 检查是否成功找到门
        current_distance = float('inf')
        current_area = 0
        if detection_results:
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = state.shape[1] / 2
                img_center_y = state.shape[0] / 2
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                area = detection['width'] * detection['height']
                if distance < current_distance:
                    current_distance = distance
                    current_area = area
        
        if done and detection_results and current_distance < env.CENTER_THRESHOLD and current_area > env.MIN_GATE_AREA:
            logger.info("成功找到目标！")
            return {"status": "success", "result": f"成功找到{target_description}！", "steps": step_count, "total_reward": total_reward}
        elif done:
            logger.info("未能找到目标")
            return {"status": "partial_success", "result": f"未能找到{target_description}", "steps": step_count, "total_reward": total_reward}
    
    # 执行重置到原点操作
    env.reset_to_origin()
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    logger.info(result["result"])
    return result


def find_gate_with_ppo(target_description="gate"):
    """
    使用PPO算法寻找门（训练+测试流程）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用PPO寻找目标: {target_description}")
    
    # 确保模型保存目录存在
    model_path = "./model/gate_search_ppo_model.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练PPO智能体
    trained_agent = train_gate_search_ppo_agent(target_description=target_description, model_path=model_path)
    
    # 初始化环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < 50:
        action = trained_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        # 检查是否成功找到门
        current_distance = float('inf')
        current_area = 0
        if detection_results:
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = state.shape[1] / 2
                img_center_y = state.shape[0] / 2
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                area = detection['width'] * detection['height']
                if distance < current_distance:
                    current_distance = distance
                    current_area = area
        
        if done and detection_results and current_distance < env.CENTER_THRESHOLD and current_area > env.MIN_GATE_AREA:
            logger.info("成功找到目标！")
            return "成功找到目标！"
        elif done:
            logger.info("未能找到目标")
            return "未能找到目标"
    
    # 执行重置到原点操作
    env.reset_to_origin()
    
    result = f"测试完成 - Steps: {step_count}, Total Reward: {total_reward}"
    logger.info(result)
    return result


def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    logger = logging.getLogger(__name__)
    
    tool_functions = {
        "find_gate_with_ppo": find_gate_with_ppo,
        "train_gate_search_ppo_agent": train_gate_search_ppo_agent,
        "evaluate_trained_ppo_agent": evaluate_trained_ppo_agent,
        "load_and_test_ppo_agent": load_and_test_ppo_agent
    }
    
    if tool_name in tool_functions:
        try:
            # 特殊处理带参数的函数
            if tool_name == "train_gate_search_ppo_agent":
                episodes = int(args[0]) if args else 200
                model_path = args[1] if len(args) > 1 else "gate_search_ppo_model.pth"
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](episodes, model_path, target_desc)
            elif tool_name == "evaluate_trained_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                episodes = int(args[1]) if len(args) > 1 else 10
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](model_path, episodes, target_desc)
            elif tool_name == "load_and_test_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                target_desc = args[1] if len(args) > 1 else "gate"
                result = tool_functions[tool_name](model_path, target_desc)
            elif tool_name == "find_gate_with_ppo":
                target_desc = args[0] if args else "gate"
                result = tool_functions[tool_name](target_desc)
            else:
                result = tool_functions[tool_name]()
            logger.info(f"PPO工具执行成功: {tool_name}")
            return {"status": "success", "result": str(result)}
        except Exception as e:
            logger.error(f"执行PPO工具时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 添加详细的错误追踪
            return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}
    else:
        logger.error(f"错误: 未知的PPO工具 '{tool_name}'")
        print(f"错误: 未知的PPO工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的PPO工具 '{tool_name}'"}


def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: train_gate_search_ppo_agent")
        print("2. 寻找门（包含训练）: find_gate_with_ppo")
        print("3. 评估已训练模型: evaluate_trained_ppo_agent")
        print("4. 加载并测试模型: load_and_test_ppo_agent")
        
        # 首先执行完整的训练过程，将模型保存到model文件夹
        print("\n=== 开始训练门搜索智能体 (20轮，模型将保存到model文件夹) ===")
        result = execute_ppo_tool("train_gate_search_ppo_agent", "200", "./model/gate_search_ppo_model.pth", "gate")
        print(f"\n训练结果: {result}")
        
        print("\n=== 使用训练好的模型进行测试 ===")
        test_result = execute_ppo_tool("load_and_test_ppo_agent", "./model/gate_search_ppo_model.pth", "gate")
        print(f"\n测试结果: {test_result}")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)


if __name__ == "__main__":
    main()