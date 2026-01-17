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
from collections import deque, defaultdict
import logging
from pathlib import Path
from ultralytics import YOLO
import pyautogui
import json
import hashlib


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
    logger = logging.getLogger('ppo_agent_logger')  # 添加logger名称
    
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
    ResNet基本块，适合您的应用场景
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


# 在 ppo_agents.py 中修改 ResNetFeatureExtractor
class ResNetFeatureExtractor(nn.Module):
    """
    增强版ResNet特征提取器 - 提高区分能力
    """
    def __init__(self, input_channels=3, block_channels=[32, 64, 128, 256]):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 使用配置文件中的图像尺寸
        height = CONFIG.get('IMAGE_HEIGHT', 480)
        width = CONFIG.get('IMAGE_WIDTH', 640)
        
        # 初始卷积层 - 增加感受野
        self.conv1 = nn.Conv2d(input_channels, block_channels[0], 
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层 - 增加层数和通道数
        layers = []
        # 第一层 - 保持原有
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        
        # 第二层 - 下采样
        layers.append(ResidualBlock(block_channels[0], block_channels[1], stride=2))
        layers.append(ResidualBlock(block_channels[1], block_channels[1]))
        
        # 第三层 - 下采样
        layers.append(ResidualBlock(block_channels[1], block_channels[2], stride=2))
        layers.append(ResidualBlock(block_channels[2], block_channels[2]))
        
        # 第四层 - 新增，进一步下采样
        layers.append(ResidualBlock(block_channels[2], block_channels[3], stride=2))
        layers.append(ResidualBlock(block_channels[3], block_channels[3]))
        
        self.layers = nn.Sequential(*layers)
        
        # 全局平均池化 - 增大输出维度
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 从(1,1)改为(2,2)，保留更多空间信息
        self.fc_features = nn.Linear(block_channels[3] * 4, 512)  # 从展平后映射到512维特征

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_features(x)  # 映射到更高维特征空间
        return x

class EnhancedGRUPolicyNetwork(nn.Module):
    """
    增强版带有GRU的策略网络 - 更强的表达能力
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=256):  # 增加隐藏层大小
        super(EnhancedGRUPolicyNetwork, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(3)
        feature_size = 512  # 根据上面的修改调整
        
        # 增加GRU复杂度
        self.gru = nn.GRU(
            feature_size, 
            hidden_size, 
            num_layers=3,  # 增加层数
            batch_first=True, 
            dropout=0.2,  # 增加dropout
            bidirectional=True
        )
        self.gru_output = nn.Linear(hidden_size * 2, hidden_size)
        
        # Actor heads - 增加复杂度
        self.move_actor = nn.Sequential(
            nn.Linear(hidden_size, 512),  # 增加维度
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, move_action_dim)
        )
        
        self.turn_actor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, turn_action_dim)
        )
        
        # Action parameter head - 增加非线性
        self.action_param_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),  # 使用SiLU激活
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Value network - 增强
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 初始化logger
        self.logger = setup_logging()
        
        # 使用更好的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """更好的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # 设置偏置让GRU更容易记住长期信息
                    n = param.size(0)
                    param.data[:n//4].fill_(1.0)

    def forward(self, states, hidden_state=None, return_debug_info=False):
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Flatten the states for feature extraction
        states_flat = states.view(-1, *states.shape[2:])
        features = self.feature_extractor(states_flat)
        
        # Reshape back to (batch, seq, feature)
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through GRU
        gru_out, hidden = self.gru(features, hidden_state)
        gru_out = self.gru_output(gru_out)
        
        # Use the last output from GRU
        last_output = gru_out[:, -1, :]
        
        # Actor outputs
        move_logits = self.move_actor(last_output)
        turn_logits = self.turn_actor(last_output)
        
        # Action parameters
        action_params = torch.tanh(self.action_param_head(last_output))
        
        # Critic output
        value = self.critic(last_output)
        
        if return_debug_info:
            debug_info = {
                'gru_last_output': last_output.detach().cpu().numpy(),
                'move_logits': move_logits.detach().cpu().numpy(),
                'turn_logits': turn_logits.detach().cpu().numpy(),
                'action_params': action_params.detach().cpu().numpy(),
                'value': value.detach().cpu().numpy(),
                'last_output_shape': last_output.shape,
                'features_shape': features.shape
            }
            
            # 打印模型输出信息
            self.logger.info(f"GRU: {[round(float(x), 2) for x in debug_info['gru_last_output'][0][:4]]}")
            self.logger.info(f"Move: {[round(float(x), 2) for x in debug_info['move_logits'][0][:10]]}")
            self.logger.info(f"Turn: {[round(float(x), 2) for x in debug_info['turn_logits'][0][:10]]}")
            self.logger.info(f"参数: {[round(float(x), 2) for x in debug_info['action_params'][0]]}")
            self.logger.info(f"价值: {round(float(debug_info['value'][0][0]), 2)}")
            
            if features.size(0) > 1:  # 如果批次大小大于1，才能比较特征差异
                feature_diff = torch.mean(torch.abs(features[0] - features[1])).item()
                self.logger.info(f"特征差异: {feature_diff:.4f}")
            
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value,
                hidden,
                debug_info
            )
        else:
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value,
                hidden
            )

class GRUMemory:
    """
    GRU智能体的记忆存储类
    """
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.states = deque(maxlen=sequence_length)
        self.move_actions = deque(maxlen=sequence_length)
        self.turn_actions = deque(maxlen=sequence_length)
        self.logprobs = deque(maxlen=sequence_length)
        self.rewards = deque(maxlen=sequence_length)
        self.is_terminals = deque(maxlen=sequence_length)
        self.action_params = deque(maxlen=sequence_length)

    def clear_memory(self):
        self.states.clear()
        self.move_actions.clear()
        self.turn_actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.action_params.clear()

    def append(self, state, move_action, turn_action, logprob, reward, is_terminal, action_param=None):
        self.states.append(state)
        self.move_actions.append(move_action)
        self.turn_actions.append(turn_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        if action_param is not None:
            self.action_params.append(action_param)


class EnhancedStuckDetector:
    """
    增强版卡死检测器 - 检测AI是否陷入无法前进的状态
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.movement_history = deque(maxlen=window_size)
        self.stuck_threshold = 3  # 连续多少次认为卡死
        self.position_history = deque(maxlen=10)  # 新增：位置历史
        self.visual_change_history = deque(maxlen=5)  # 新增：视觉变化历史
        self.action_history = deque(maxlen=8)  # 新增：动作历史

    def record_position(self, position_feature):
        """记录位置特征"""
        self.position_history.append(position_feature)

    def record_action(self, action):
        """记录动作"""
        self.action_history.append(action)

    def is_repeating_pattern(self):
        """检测重复模式"""
        if len(self.action_history) < 4:
            return False
        
        # 检查最近的动作是否重复
        recent_actions = list(self.action_history)[-4:]
        unique_actions = set(recent_actions)
        
        return len(unique_actions) <= 2  # 4个动作中有2个或更少唯一动作

    def record_movement(self):
        """记录有效移动"""
        self.movement_history.append(1)  # 1表示有效移动

    def record_stuck(self):
        """记录卡死状态"""
        self.movement_history.append(0)  # 0表示卡死

    def is_stuck(self):
        """改进的卡死判断"""
        if len(self.movement_history) < self.window_size:
            return False
        
        stuck_count = sum(1 for x in self.movement_history if x == 0)
        return stuck_count >= self.stuck_threshold or self.is_repeating_pattern()

    def get_stuck_ratio(self):
        """获取卡死比例"""
        if not self.movement_history:
            return 0.0
        return sum(1 for x in self.movement_history if x == 0) / len(self.movement_history)


class EnhancedTargetSearchEnvironment:
    """
    增强版目标搜索环境 - 使用全局配置，加强卡死检测
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
        self.logger = setup_logging()  # 使用统一的日志配置
        
        # 记录探索历史
        self.position_history = []
        self.max_history_length = CONFIG['POSITION_HISTORY_LENGTH']
        self.yolo_model = self._load_yolo_model()
        self._warm_up_detection_model()
        
        # 成功条件阈值
        self.MIN_GATE_AREA = CONFIG['MIN_GATE_AREA']
        self.CENTER_THRESHOLD = CONFIG['CENTER_THRESHOLD']
        
        # 新增：物理状态追踪
        self.previous_frame = None
        self.action_history = deque(maxlen=10)  # 记录最近10个动作
        self.visual_change_history = deque(maxlen=10)  # 记录最近10帧的视觉变化
        self.action_effectiveness = defaultdict(lambda: deque(maxlen=5))  # 动作有效性统计
        self.stuck_detector = EnhancedStuckDetector(window_size=5)  # 增强卡死检测器
        self.frame_change_threshold = 0.05  # 视觉变化阈值
        self.no_progress_steps = 0  # 连续无进展步数
        self.total_no_progress_steps = 0  # 总无进展步数
        self.state_difficulty_tracker = {
        'recent_states': deque(maxlen=10),
        'state_variance': 0.0,
        'state_diversity_score': 0.0
    }
    def capture_screen(self):
        """
        截取当前屏幕画面，并计算状态多样性
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
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            
            # 计算图像差异性
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 存储最近的状态用于多样性计算
            if len(self.state_difficulty_tracker['recent_states']) > 0:
                # 计算与之前状态的差异
                prev_state = self.state_difficulty_tracker['recent_states'][-1]
                state_diff = np.mean(np.abs(screenshot.astype(np.float32) - prev_state.astype(np.float32)))
                self.state_difficulty_tracker['state_variance'] = state_diff
            
            self.state_difficulty_tracker['recent_states'].append(screenshot.copy())
            
            return screenshot
        except ImportError:
            self.logger.warning("截图功能不可用，使用模拟图片")
            return np.zeros((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 3), dtype=np.uint8)
    def _extract_position_feature(self, frame, detections):
        """提取位置特征用于重复检测"""
        if frame is None:
            return (0, 0, 0)  # 默认特征
        
        # 提取图像哈希值和检测结果统计
        img_hash = hash(frame.tostring())
        detection_count = len(detections) if detections else 0
        total_area = sum(det['width'] * det['height'] for det in detections) if detections else 0
        
        return (img_hash, detection_count, total_area)

    def _calculate_visual_change_reward(self, action_taken):
        """
        基于视觉变化的奖励 - 检查动作是否产生了预期的视觉反馈
        """
        if self.previous_frame is None:
            return 0.0
            
        current_frame = self.capture_screen()
        if current_frame is None:
            return 0.0
            
        # 计算帧间差异
        frame_diff = self._calculate_frame_difference(self.previous_frame, current_frame)
        self.visual_change_history.append(frame_diff)
        
        if not action_taken:
            return 0.0
            
        move_action, turn_action = action_taken
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        # 预期的视觉变化量
        expected_change = self._get_expected_change_for_action(move_action, turn_action)
        
        # 如果视觉变化小于预期且大于阈值，给予奖励；否则给予惩罚
        if frame_diff > expected_change * 0.3:  # 有足够变化
            # 根据变化量给予适度奖励
            change_reward = min(frame_diff * 0.8, 0.15)  # 增加奖励幅度
            self.stuck_detector.record_movement()
            return change_reward
        else:
            # 视觉变化不足，可能卡死了
            self.stuck_detector.record_stuck()
            return -0.15  # 增加惩罚力度

    def _get_expected_change_for_action(self, move_action, turn_action):
        """
        根据执行的动作返回预期的视觉变化量
        """
        expected = 0.0
        
        # 移动动作通常产生较大的透视变化
        if move_action in [0, 1]:  # forward/backward
            expected += 0.25
        elif move_action in [2, 3]:  # strafe left/right
            expected += 0.20
            
        # 转头动作产生视角变化
        if turn_action in [0, 1]:  # turn left/right
            expected += 0.15
            
        return expected

    def _calculate_action_consistency_reward(self, action_taken):
        """
        奖励动作与视觉变化的一致性
        """
        if not action_taken or len(self.visual_change_history) < 2:
            return 0.0
            
        move_action, turn_action = action_taken
        recent_changes = list(self.visual_change_history)[-3:]  # 最近3次变化
        
        avg_change = np.mean(recent_changes) if recent_changes else 0.0
        
        # 如果执行了移动动作但视觉变化很小，说明可能无效
        if move_action in [0, 1, 2, 3] and avg_change < 0.03:
            return -0.05  # 移动无效惩罚
        elif turn_action in [0, 1] and avg_change < 0.015:
            return -0.04  # 转头无效惩罚
        elif avg_change > 0.12:  # 显著变化奖励
            return 0.03
            
        return 0.0

    def _calculate_stuck_penalty(self):
        """
        强化的卡死惩罚
        """
        if self.stuck_detector.is_stuck():
            self.total_no_progress_steps += 1
            # 根据连续卡死次数增加惩罚
            consecutive_stuck = min(self.total_no_progress_steps / 8, 2.0)
            base_penalty = CONFIG.get('STUCK_PENALTY', -0.25)  # 提高基础惩罚
            return base_penalty * consecutive_stuck
        return 0.0

    def _calculate_exploration_reward(self):
        """
        基于探索行为的奖励
        """
        if len(self.visual_change_history) < 5:
            return 0.0
            
        # 计算视觉变化的标准差，高变化表示探索行为
        changes = list(self.visual_change_history)
        variation = np.std(changes)
        
        if variation > 0.06:  # 高变化性奖励
            return 0.03
        elif variation < 0.008:  # 低变化性惩罚
            return -0.02
            
        return 0.0

    def _calculate_repetition_penalty(self, action_taken):
        """
        强化的重复动作惩罚
        """
        if not action_taken or len(self.action_history) < 4:  # 增加历史长度
            return 0.0
        
        recent_actions = list(self.action_history)[-4:]  # 检查更多历史
        same_action_count = sum(1 for act in recent_actions if act == action_taken)
        
        if same_action_count >= 3:  # 连续3次相同动作
            return -0.35  # 大幅提高惩罚
        elif same_action_count >= 2:  # 连续2次相同动作
            return -0.2  # 提高惩罚
            
        return 0.0

    def _calculate_progress_reward(self, detection_results):
        """
        长期进展奖励 - 基于目标检测的总体趋势
        """
        if not detection_results:
            self.no_progress_steps += 1
            return -0.02
        else:
            # 如果检测到目标，减少无进展计数
            self.no_progress_steps = max(0, self.no_progress_steps - 1)
            return 0.02

    def _calculate_target_presence_reward(self, detection_results):
        """
        最小化的存在奖励 - 仅当检测到目标时给予小奖励
        """
        if detection_results:
            # 检测到gate给予稍高奖励
            gate_detected = any(
                'gate' in detection['label'].lower() or detection['label'].lower() == 'gate'
                for detection in detection_results
            )
            if gate_detected:
                return 0.08
            else:
                return 0.02  # 检测到其他目标的小奖励
        return -0.02  # 未检测到任何目标的轻微惩罚

    def _calculate_frame_difference(self, frame1, frame2):
        """
        计算两帧之间的差异程度
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差值
        diff = cv2.absdiff(gray1, gray2)
        
        # 计算平均差异值
        mean_diff = np.mean(diff) / 255.0  # 归一化到[0,1]
        
        return mean_diff

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
        self.logger.info("执行重置到原点操作")
        self.logger.info( "执行重置到原点操作...")
        
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
                self.logger.info( f"未检测到门，等待1秒后按回车重新检测...")
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
        self.last_center_distance = float('inf')
        self.last_area = 0
        self.last_detection_result = None
        self.position_history = []
        self.action_history.clear()
        self.visual_change_history.clear()
        self.no_progress_steps = 0
        self.total_no_progress_steps = 0
        self.previous_frame = None
        self.stuck_detector = EnhancedStuckDetector(window_size=5)
        self.logger.info( "重置到原点操作完成")
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
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []

    def _check_climb_conditions(self, detection_results, confidence_threshold=0.9):
        """
        检查climb检测结果是否满足条件
        参数:
            detection_results: YOLO检测结果列表
            confidence_threshold: 置信度阈值，默认0.9
        返回:
            bool: 如果满足条件（单个检测且置信度>0.9，或者多个检测中至少一个置信度>0.9）则返回True
        """
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
            self._save_detection_image_with_bounding_boxes(pre_action_state, pre_action_detections, prefix="pre_action_climb_detected")
            
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
        
        # 记录执行动作前的帧
        pre_action_frame = pre_action_state.copy() if pre_action_state is not None else None
        
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
        
        # 检查执行动作后是否检测到符合条件的climb类别
        post_climb_detected = self._check_climb_conditions(detection_results)
        
        if post_climb_detected:
            # 保存带识别框的图片
            self._save_detection_image_with_bounding_boxes(new_state, detection_results, prefix="post_action_climb_detected")
        
        # 计算奖励 - 使用全新的奖励函数
        reward, new_area = self.calculate_reward(
            detection_results, 
            self.last_center_distance, 
            (move_action, turn_action), 
            self.last_area
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
            self._save_detection_image_with_bounding_boxes(new_state, detection_results, prefix="final_climb_detected")
            
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
        
        # 更新位置历史
        state_feature = len(detection_results) if detection_results else 0
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # 修改：只有在达到最大步数时才重置，避免重复重置
        if done and not climb_detected:  # 如果不是因为climb而结束，也要重置
            self.logger.info(f"达到最大步数 {self.max_steps}")
            # 在episode结束时重置游戏环境
            self.reset_to_origin()
        
        return new_state, reward, done, detection_results 
    def calculate_reward(self, detection_results, last_center_distance, action_taken, last_area):
        """
        计算综合奖励 - 需要补充完整实现
        """
        # 目标检测奖励
        target_reward = self._calculate_target_presence_reward(detection_results)
        
        # 进展奖励
        progress_reward = self._calculate_progress_reward(detection_results)
        
        # 卡死惩罚
        stuck_penalty = self._calculate_stuck_penalty()
        
        # 重复动作惩罚
        repetition_penalty = self._calculate_repetition_penalty(action_taken)
        
        # 视觉变化奖励
        visual_change_reward = self._calculate_visual_change_reward(action_taken)
        
        # 动作一致性奖励
        consistency_reward = self._calculate_action_consistency_reward(action_taken)
        
        # 探索奖励
        exploration_reward = self._calculate_exploration_reward()
        
        # 总奖励
        total_reward = (target_reward + progress_reward + stuck_penalty + 
                    repetition_penalty + visual_change_reward + 
                    consistency_reward + exploration_reward)
        
        # 计算新区域面积
        new_area = self._get_max_detection_area(detection_results) if detection_results else 0
        
        return total_reward, new_area
    def _save_detection_image_with_bounding_boxes(self, image, detection_results, prefix="detection"):
        """
        保存带检测框的图像
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
                self.logger.info(f"检测结果图像已保存: {filepath}")
            else:
                self.logger.error(f"保存检测结果图像失败: {filepath}")
                
        except Exception as e:
            self.logger.error(f"保存带检测框图像时出错: {e}")
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
        self.action_history.clear()
        self.visual_change_history.clear()
        self.no_progress_steps = 0
        self.total_no_progress_steps = 0
        self.previous_frame = None
        self.stuck_detector = EnhancedStuckDetector(window_size=5)
        initial_state = self.capture_screen()
        return initial_state


# 在 EnhancedGRUPPOAgent 类的 __init__ 方法中添加：
class EnhancedGRUPPOAgent:
    """
    增强版基于GRU的PPO智能体 - 增加收敛监控，改进卡死检测
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        self.sequence_length = config['SEQUENCE_LENGTH']
        self.hidden_size = config['HIDDEN_SIZE']
        
        # 添加logger
        self.logger = setup_logging()
        
        # Create policy networks
        self.policy = EnhancedGRUPolicyNetwork(
            state_dim, move_action_dim, turn_action_dim, self.hidden_size
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            betas=self.betas
        )
        self.policy_old = EnhancedGRUPolicyNetwork(
            state_dim, move_action_dim, turn_action_dim, self.hidden_size
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # State history for sequence input
        self.state_history = deque(maxlen=self.sequence_length)
        
        # 收敛监控相关变量
        self.convergence_monitor = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_successes': [],
            'loss_history': [],
            'learning_rates': [],
            'grad_norms': [],
            'entropy_history': [],
            'kl_divergence': []
        }
        
        # 早停相关
        self.best_avg_reward = float('-inf')
        self.no_improve_count = 0
        self.patience = config.get('EARLY_STOP_PATIENCE', 50)
        
        # 学习率调度器 - 移除verbose参数
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=10
                # 注意：这里移除了verbose参数，因为它在某些PyTorch版本中不受支持
            )
        except TypeError:
            # 如果仍然有问题，提供备选方案
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=10
            )
        
        # 自适应学习率因子
        self.adaptive_lr_factor = 1.0

    def act(self, state, memory, return_debug_info=False):
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        
        # 将当前状态添加到历史记录
        self.state_history.append(state_tensor)
        
        # 如果历史记录长度不足，用当前状态填充
        while len(self.state_history) < self.sequence_length:
            self.state_history.appendleft(state_tensor)
        
        # 转换为张量并添加批次维度
        state_seq = torch.stack(list(self.state_history)).unsqueeze(0)  # [batch=1, seq_len, channels, height, width]
        
        # 使用旧策略获取动作概率
        with torch.no_grad():
            if return_debug_info:
                move_probs, turn_probs, action_params, state_val, _, debug_info = self.policy_old(state_seq, return_debug_info=True)
            else:
                move_probs, turn_probs, action_params, state_val, _ = self.policy_old(state_seq)
                
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            # 采样动作
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            logprob = move_logprob + turn_logprob
        
        # 处理动作参数，采用更灵活的映射方式
        move_forward_step_raw = action_params[0][0].item()  # [-1,1]范围
        turn_angle_raw = action_params[0][1].item()         # [-1,1]范围
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 改进的探索策略：增加动态噪声
        exploration_rate = CONFIG.get('INITIAL_EXPLORATION_RATE', 1.0) * \
                        (CONFIG.get('EXPLORATION_DECAY_RATE', 0.999) ** len(self.state_history))
        
        exploration_noise = CONFIG.get('ACTION_EXPLORATION_NOISE', 0.4) * exploration_rate  # 增加噪声
        
        # 对原始输出添加更强的噪声
        move_raw_with_noise = move_forward_step_raw + np.random.normal(0, exploration_noise)
        turn_raw_with_noise = turn_angle_raw + np.random.normal(0, exploration_noise)
        
        # 确保在 [-1, 1] 范围内
        move_raw_clipped = np.clip(move_raw_with_noise, -1.0, 1.0)
        turn_raw_clipped = np.clip(turn_raw_with_noise, -1.0, 1.0)
        
        # 使用非线性映射，让边缘值更容易出现
        def enhanced_mapping(raw_value):
            # 将 [-1, 1] 映射到 [-3, 3] 以扩大sigmoid的作用范围
            expanded_value = raw_value * 3.0
            # 使用tanh作为非线性映射，这样边缘值更容易出现
            mapped_value = (np.tanh(expanded_value) + 1.0) / 2.0  # 映射到 [0, 1]
            return np.clip(mapped_value, 0.0, 1.0)
        
        move_forward_step_normalized = enhanced_mapping(move_raw_clipped)
        turn_angle_normalized = enhanced_mapping(turn_raw_clipped)
        
        # 应用线性变换到实际范围
        move_forward_step = move_forward_step_normalized * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = turn_angle_normalized * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN
        
        # 存储到记忆中
        memory.append(
            state_tensor,
            move_action.item(),
            turn_action.item(),
            logprob.item(),
            0,  # 奖励稍后更新
            False,  # 是否结束稍后更新
            [move_forward_step_normalized, turn_angle_normalized]  # 存储归一化的参数
        )
        
        if return_debug_info:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle, debug_info
        else:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def update(self, memory):
        if len(memory.states) == 0:
            return
        
        # 将长序列切分成多个固定长度的子序列进行训练
        batch_size = 32  # 或其他合适的批大小
        sequence_length = self.sequence_length
        
        # 将记忆切分成多个序列段
        all_states = list(memory.states)
        all_move_actions = list(memory.move_actions)
        all_turn_actions = list(memory.turn_actions)
        all_logprobs = list(memory.logprobs)
        all_rewards = list(memory.rewards)
        all_terminals = list(memory.is_terminals)
        
        if len(all_states) < sequence_length:
            # 如果序列太短，直接使用全部数据
            self._train_single_sequence(
                all_states, all_move_actions, all_turn_actions,
                all_logprobs, all_rewards, all_terminals
            )
            return
        
        # 滑动窗口切分序列
        for start_idx in range(0, len(all_states) - sequence_length + 1, sequence_length // 2):
            end_idx = min(start_idx + sequence_length, len(all_states))
            
            seq_states = all_states[start_idx:end_idx]
            seq_move_actions = all_move_actions[start_idx:end_idx]
            seq_turn_actions = all_turn_actions[start_idx:end_idx]
            seq_logprobs = all_logprobs[start_idx:end_idx]
            seq_rewards = all_rewards[start_idx:end_idx]
            seq_terminals = all_terminals[start_idx:end_idx]
            
            if len(seq_states) == sequence_length:
                self._train_single_sequence(
                    seq_states, seq_move_actions, seq_turn_actions,
                    seq_logprobs, seq_rewards, seq_terminals
                )

    def _train_single_sequence(self, states_list, move_actions, turn_actions, 
                            old_logprobs, rewards, terminals):
        """训练单个序列段"""
        # 构建状态序列
        states = torch.stack(states_list).unsqueeze(0)  # [1, seq_len, C, H, W]
        move_actions = torch.tensor(move_actions, dtype=torch.long)
        turn_actions = torch.tensor(turn_actions, dtype=torch.long)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        terminals = torch.tensor(terminals, dtype=torch.bool)

        # 计算折扣奖励
        discounted_rewards = []
        running_add = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(terminals)):
            if is_terminal:
                running_add = 0
            running_add = reward + (self.gamma * running_add)
            discounted_rewards.insert(0, running_add)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        if discounted_rewards.numel() > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # PPO更新循环
        for _ in range(self.K_epochs):
            # 前向传播
            move_probs, turn_probs, action_params, state_vals, _ = self.policy(states)
            
            # 修正张量形状处理 - 确保维度一致
            # state_vals形状应该是 [batch, seq_len, 1] -> [seq_len]
            if state_vals.dim() == 3:
                state_vals = state_vals.squeeze(0).squeeze(-1)  # [seq_len]
            elif state_vals.dim() == 2:
                state_vals = state_vals.squeeze(-1)  # [seq_len]
            elif state_vals.dim() == 1 and state_vals.size(0) != len(rewards):
                # 如果只剩一个值，需要扩展
                state_vals = state_vals.expand(len(rewards))
            
            # 确保state_vals和discounted_rewards维度一致
            if state_vals.size(0) != discounted_rewards.size(0):
                # 如果长度不匹配，调整到相同的长度
                min_len = min(state_vals.size(0), discounted_rewards.size(0))
                state_vals = state_vals[:min_len]
                discounted_rewards = discounted_rewards[:min_len]

            # 确保两个张量具有相同的形状
            assert state_vals.shape == discounted_rewards.shape, \
                f"state_vals shape {state_vals.shape} must match discounted_rewards shape {discounted_rewards.shape}"

            # 处理输出维度 - 修正形状处理逻辑
            # move_probs, turn_probs形状应该是 [batch, seq_len, action_dim]
            move_probs_squeezed = move_probs.squeeze(0)  # [seq_len, move_action_dim]
            turn_probs_squeezed = turn_probs.squeeze(0)  # [seq_len, turn_action_dim]

            # 检查张量形状并进行必要的修正
            if move_probs_squeezed.dim() == 1:
                # 如果只有1维，扩展为2维 [1, action_dim]
                move_probs_squeezed = move_probs_squeezed.unsqueeze(0).expand(len(move_actions), -1)
            if turn_probs_squeezed.dim() == 1:
                # 如果只有1维，扩展为2维 [1, action_dim] 
                turn_probs_squeezed = turn_probs_squeezed.unsqueeze(0).expand(len(turn_actions), -1)

            # 确保动作索引在有效范围内
            max_move_action = move_probs_squeezed.size(1) - 1
            max_turn_action = turn_probs_squeezed.size(1) - 1
            
            move_actions = torch.clamp(move_actions, 0, max_move_action)
            turn_actions = torch.clamp(turn_actions, 0, max_turn_action)

            # 计算对数概率 - 确保gather操作的维度正确
            move_logprobs = torch.log(move_probs_squeezed.gather(1, move_actions.unsqueeze(1)).squeeze(-1))
            turn_logprobs = torch.log(turn_probs_squeezed.gather(1, turn_actions.unsqueeze(1)).squeeze(-1))
            logprobs = move_logprobs + turn_logprobs

            # 计算比率和优势
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = discounted_rewards - state_vals.detach()

            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失 - 现在维度应该匹配
            critic_loss = self.MseLoss(state_vals, discounted_rewards)

            # 熵损失
            move_entropy = -(move_probs_squeezed * torch.log(move_probs_squeezed + 1e-10)).sum(dim=-1).mean()
            turn_entropy = -(turn_probs_squeezed * torch.log(turn_probs_squeezed + 1e-10)).sum(dim=-1).mean()
            entropy_bonus = move_entropy + turn_entropy

            # 总损失
            entropy_coeff = CONFIG.get('ENTROPY_COEFFICIENT', 0.02)
            loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy_bonus

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=CONFIG.get('GRADIENT_CLIP_NORM', 0.5))
            self.optimizer.step() 
    def check_convergence_status(self, episode_reward, episode_length, success_flag=False):
        """
        检查收敛状态并返回相关信息
        """
        self.convergence_monitor['episode_rewards'].append(episode_reward)
        self.convergence_monitor['episode_lengths'].append(episode_length)
        self.convergence_monitor['episode_successes'].append(success_flag)
        
        # 计算滑动窗口统计数据
        window_size = 20  # 可配置的滑动窗口大小
        recent_rewards = self.convergence_monitor['episode_rewards'][-window_size:]
        recent_lengths = self.convergence_monitor['episode_lengths'][-window_size:]
        recent_successes = self.convergence_monitor['episode_successes'][-window_size:]
        
        if len(recent_rewards) >= window_size:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)
            
            # 更新早停计数
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # 如果性能持续下降，降低学习率
            if len(recent_rewards) >= 10:
                recent_trend = np.polyfit(range(len(recent_rewards[-10:])), recent_rewards[-10:], 1)
                if recent_trend[0] < -0.05:  # 负斜率超过阈值，说明性能在下降
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.9, 1e-6)  # 最小学习率1e-6
            
            # 更新学习率调度器
            self.scheduler.step(avg_reward)
            
            return {
                'avg_recent_reward': avg_reward,
                'avg_recent_length': avg_length,
                'recent_success_rate': success_rate,
                'should_stop_early': self.no_improve_count >= self.patience,
                'current_patience': self.patience - self.no_improve_count,
                'current_learning_rate': self.optimizer.param_groups[0]['lr'],
                'is_improving': recent_trend[0] > 0 if len(recent_rewards) >= 10 else True
            }
        
        return {
            'avg_recent_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_recent_length': np.mean(recent_lengths) if recent_lengths else 0,
            'recent_success_rate': np.mean(recent_successes) if recent_successes else 0,
            'should_stop_early': False,
            'current_patience': self.patience,
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_improving': True
        }

    def get_convergence_report(self):
        """
        获取收敛报告
        """
        report = {}
        
        if self.convergence_monitor['loss_history']:
            recent_losses = self.convergence_monitor['loss_history'][-10:]
            report['recent_avg_total_loss'] = np.mean([l['total_loss'] for l in recent_losses])
            report['recent_avg_actor_loss'] = np.mean([l['actor_loss'] for l in recent_losses])
            report['recent_avg_critic_loss'] = np.mean([l['critic_loss'] for l in recent_losses])
            report['recent_avg_entropy'] = np.mean([l['entropy'] for l in recent_losses])
            report['recent_avg_grad_norm'] = np.mean([l['grad_norm'] for l in recent_losses])
        
        if self.convergence_monitor['learning_rates']:
            report['current_learning_rate'] = self.convergence_monitor['learning_rates'][-1]
            report['learning_rate_trend'] = 'decreasing' if len(self.convergence_monitor['learning_rates']) > 1 and \
                self.convergence_monitor['learning_rates'][-1] < self.convergence_monitor['learning_rates'][0] else 'stable'
        
        if len(self.convergence_monitor['episode_rewards']) >= 20:
            recent_rewards = self.convergence_monitor['episode_rewards'][-20:]
            reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
            report['reward_trend'] = reward_trend
            report['is_converging'] = reward_trend > 0  # 正斜率表示改善趋势
        
        report['total_episodes_trained'] = len(self.convergence_monitor['episode_rewards'])
        report['total_updates'] = len(self.convergence_monitor['loss_history'])
        report['success_rate'] = np.mean(self.convergence_monitor['episode_successes']) if self.convergence_monitor['episode_successes'] else 0
        
        return report

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