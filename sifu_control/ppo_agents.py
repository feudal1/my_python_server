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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        return x


class GRUPolicyNetwork(nn.Module):
    """
    带有GRU的策略网络 - 增强输出多样性
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=128):
        super(GRUPolicyNetwork, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(3)
        feature_size = 128  # 根据ResNetFeatureExtractor的输出调整
        
        # GRU层
        self.gru = nn.GRU(feature_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1) 
        
        # Actor heads
        self.move_actor = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, move_action_dim)
        )
        
        self.turn_actor = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, turn_action_dim)
        )
        
        # Action parameter prediction head - 使用不同的激活函数
        self.action_param_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(negative_slope=0.01),  # 使用LeakyReLU增加非线性
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 2),
            nn.Tanh()  # 保持tanh输出到[-1,1]范围
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states, hidden_state=None):
        batch_size = states.size(0)
        seq_len = states.size(1)
        
        # Flatten the states for feature extraction
        states_flat = states.view(-1, *states.shape[2:])
        features = self.feature_extractor(states_flat)
        
        # Reshape back to (batch, seq, feature)
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through GRU
        gru_out, hidden = self.gru(features, hidden_state)
        
        # Use the last output from GRU
        last_output = gru_out[:, -1, :]
        
        # Actor outputs
        move_logits = self.move_actor(last_output)
        turn_logits = self.turn_actor(last_output)
        
        # Action parameters - 使用tanh确保输出在[-1, 1]之间
        action_params = torch.tanh(self.action_param_head(last_output))
        
        # Critic output
        value = self.critic(last_output)
        
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


class GRUPPOAgent:
    """
    基于GRU的PPO智能体
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
        
        # Create policy networks
        self.policy = GRUPolicyNetwork(
            state_dim, move_action_dim, turn_action_dim, self.hidden_size
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            betas=self.betas
        )
        self.policy_old = GRUPolicyNetwork(
            state_dim, move_action_dim, turn_action_dim, self.hidden_size
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # State history for sequence input
        self.state_history = deque(maxlen=self.sequence_length)

    def act(self, state, memory):
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
        
        exploration_noise = CONFIG.get('ACTION_EXPLORATION_NOISE', 0.15) * exploration_rate
        
        # 对原始输出添加更强的噪声
        move_raw_with_noise = move_forward_step_raw + np.random.normal(0, exploration_noise)
        turn_raw_with_noise = turn_angle_raw + np.random.normal(0, exploration_noise)
        
        # 确保在 [-1, 1] 范围内
        move_raw_clipped = np.clip(move_raw_with_noise, -1.0, 1.0)
        turn_raw_clipped = np.clip(turn_raw_with_noise, -1.0, 1.0)
        
        # 使用非线性映射，让边缘值更容易出现
        # 将 [-1, 1] 映射到 [0, 1]，但使用sigmoid-like函数让极值更容易出现
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
        
        return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def update(self, memory):
        if len(memory.states) == 0:
            return
            
        # Convert memory to tensors
        states = torch.stack(list(memory.states)).unsqueeze(0)  # Add batch dimension
        move_actions = torch.tensor(list(memory.move_actions), dtype=torch.long)
        turn_actions = torch.tensor(list(memory.turn_actions), dtype=torch.long)
        old_logprobs = torch.tensor(list(memory.logprobs), dtype=torch.float)
        rewards = torch.tensor(list(memory.rewards), dtype=torch.float)
        terminals = torch.tensor(list(memory.is_terminals), dtype=torch.bool)

        # Compute discounted rewards
        discounted_rewards = []
        running_add = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(terminals)):
            if is_terminal:
                running_add = 0
            running_add = reward + (self.gamma * running_add)
            discounted_rewards.insert(0, running_add)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Optimize policy K epochs
        for _ in range(self.K_epochs):
            # Forward pass
            move_probs, turn_probs, action_params, state_vals, _ = self.policy(states)
            
            # Calculate action logprobs
            move_dists = Categorical(move_probs)
            turn_dists = Categorical(turn_probs)
            
            move_logprobs = move_dists.log_prob(move_actions)
            turn_logprobs = turn_dists.log_prob(turn_actions)
            logprobs = move_logprobs + turn_logprobs
            
            # Calculate entropy bonus to encourage exploration
            move_entropy = -(move_probs * torch.log(move_probs + 1e-10)).sum(dim=-1).mean()
            turn_entropy = -(turn_probs * torch.log(turn_probs + 1e-10)).sum(dim=-1).mean()
            entropy_bonus = move_entropy + turn_entropy
            
            # Calculate ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate advantages
            advantages = discounted_rewards - state_vals.squeeze(-1).detach()
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = self.MseLoss(state_vals.squeeze(-1), discounted_rewards)
            
            # Total loss - including entropy regularization
            entropy_coeff = CONFIG.get('ENTROPY_COEFFICIENT', 0.01)
            loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy_bonus
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=CONFIG['GRADIENT_CLIP_NORM'])
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

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


class StuckDetector:
    """
    卡死检测器 - 检测AI是否陷入无法前进的状态
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.movement_history = deque(maxlen=window_size)
        self.stuck_threshold = 3  # 连续多少次认为卡死

    def record_movement(self):
        """记录有效移动"""
        self.movement_history.append(1)  # 1表示有效移动

    def record_stuck(self):
        """记录卡死状态"""
        self.movement_history.append(0)  # 0表示卡死

    def is_stuck(self):
        """判断是否卡死"""
        if len(self.movement_history) < self.window_size:
            return False
        
        # 如果最近window_size次中大部分都是卡死状态
        stuck_count = sum(1 for x in self.movement_history if x == 0)
        return stuck_count >= self.stuck_threshold

    def get_stuck_ratio(self):
        """获取卡死比例"""
        if not self.movement_history:
            return 0.0
        return sum(1 for x in self.movement_history if x == 0) / len(self.movement_history)


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
        
        # 新增：物理状态追踪
        self.previous_frame = None
        self.action_history = deque(maxlen=10)  # 记录最近10个动作
        self.visual_change_history = deque(maxlen=10)  # 记录最近10帧的视觉变化
        self.action_effectiveness = defaultdict(lambda: deque(maxlen=5))  # 动作有效性统计
        self.stuck_detector = StuckDetector(window_size=5)  # 卡死检测器
        self.frame_change_threshold = 0.05  # 视觉变化阈值
        self.no_progress_steps = 0  # 连续无进展步数
        self.total_no_progress_steps = 0  # 总无进展步数

    def calculate_reward(self, detection_results, prev_distance, action_taken=None, prev_area=None):
        """
        完全新版的奖励函数 - 专注于物理移动的有效性
        """
        reward = 0.0
        
        # 1. 视觉变化分析奖励
        visual_change_reward = self._calculate_visual_change_reward(action_taken)
        reward += visual_change_reward
        
        # 2. 动作一致性奖励
        consistency_reward = self._calculate_action_consistency_reward(action_taken)
        reward += consistency_reward
        
        # 3. 卡死惩罚
        stuck_penalty = self._calculate_stuck_penalty()
        reward += stuck_penalty
        
        # 4. 探索奖励
        exploration_reward = self._calculate_exploration_reward()
        reward += exploration_reward
        
        # 5. 重复动作惩罚
        repetition_penalty = self._calculate_repetition_penalty(action_taken)
        reward += repetition_penalty
        
        # 6. 长期进展奖励
        progress_reward = self._calculate_progress_reward(detection_results)
        reward += progress_reward
        
        # 7. 目标检测奖励（最小化，主要作为辅助）
        target_reward = self._calculate_target_presence_reward(detection_results)
        reward += target_reward
        
        # 更新内部状态
        current_frame = self.capture_screen()
        if current_frame is not None:
            self.previous_frame = current_frame.copy()
        
        if action_taken:
            self.action_history.append(action_taken)
        
        return reward, self._get_max_detection_area(detection_results)

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
            change_reward = min(frame_diff * 0.5, 0.1)
            self.stuck_detector.record_movement()
            return change_reward
        else:
            # 视觉变化不足，可能卡死了
            self.stuck_detector.record_stuck()
            return -0.05  # 小惩罚

    def _get_expected_change_for_action(self, move_action, turn_action):
        """
        根据执行的动作返回预期的视觉变化量
        """
        expected = 0.0
        
        # 移动动作通常产生较大的透视变化
        if move_action in [0, 1]:  # forward/backward
            expected += 0.2
        elif move_action in [2, 3]:  # strafe left/right
            expected += 0.15
            
        # 转头动作产生视角变化
        if turn_action in [0, 1]:  # turn left/right
            expected += 0.1
            
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
        if move_action in [0, 1, 2, 3] and avg_change < 0.05:
            return -0.03  # 移动无效惩罚
        elif turn_action in [0, 1] and avg_change < 0.02:
            return -0.02  # 转头无效惩罚
        elif avg_change > 0.1:  # 显著变化奖励
            return 0.02
            
        return 0.0

    def _calculate_stuck_penalty(self):
        """
        计算卡死惩罚
        """
        if self.stuck_detector.is_stuck():
            self.total_no_progress_steps += 1
            return CONFIG.get('STUCK_PENALTY', -0.1) * min(self.total_no_progress_steps / 10, 1.0)
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
        
        if variation > 0.05:  # 高变化性奖励
            return 0.02
        elif variation < 0.01:  # 低变化性惩罚
            return -0.01
            
        return 0.0

    def _calculate_repetition_penalty(self, action_taken):
        """
        重复动作惩罚
        """
        if not action_taken or len(self.action_history) < 3:
            return 0.0
            
        recent_actions = list(self.action_history)[-3:]
        same_action_count = sum(1 for act in recent_actions if act == action_taken)
        
        if same_action_count >= 3:  # 连续3次相同动作
            return -0.05
        elif same_action_count >= 2:  # 连续2次相同动作
            return -0.02
            
        return 0.0

    def _calculate_progress_reward(self, detection_results):
        """
        长期进展奖励 - 基于目标检测的总体趋势
        """
        if not detection_results:
            self.no_progress_steps += 1
            return -0.01
        else:
            # 如果检测到目标，减少无进展计数
            self.no_progress_steps = max(0, self.no_progress_steps - 1)
            return 0.01

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
                return 0.05
            else:
                return 0.01  # 检测到其他目标的小奖励
        return -0.01  # 未检测到任何目标的轻微惩罚

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
        time.sleep(0.2)
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
            
            # 完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            speed_bonus = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8) * (self.max_steps - self.step_count)
            reward = base_completion_reward + speed_bonus
            
            new_area = self._get_max_detection_area(pre_action_detections)
            self.last_area = new_area
            self.last_detection_result = pre_action_detections
            self.step_count += 1
            
            print(f"Step {self.step_count}, Area: {new_area:.2f}, Reward: {reward:.2f}, "
                  f"Detected: climb (pre-action), Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}")
            
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
        
        # 检查是否检测到climb类别
        climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in detection_results
        )

        done = climb_detected or self.step_count >= self.max_steps
        
        # 如果检测到climb，给予额外奖励
        if climb_detected:
            # 基础完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            
            # 快速完成奖励
            quick_completion_factor = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8)
            quick_completion_bonus = quick_completion_factor * (self.max_steps - self.step_count) 
            
            # 总奖励
            total_completion_bonus = base_completion_reward + quick_completion_bonus
            reward += total_completion_bonus
            self.logger.info(f"检测到climb类别！步骤: {self.step_count}, 基础奖励: {base_completion_reward:.2f}, "
                            f"快速完成奖励: {quick_completion_bonus:.2f}, 总奖励: {total_completion_bonus:.2f}")

        # 输出每步得分
        print(f"Step {self.step_count}, Area: {new_area:.2f}, Reward: {reward:.3f}, "
              f"Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, "
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
        self.action_history.clear()
        self.visual_change_history.clear()
        self.no_progress_steps = 0
        self.total_no_progress_steps = 0
        self.previous_frame = None
        initial_state = self.capture_screen()
        return initial_state