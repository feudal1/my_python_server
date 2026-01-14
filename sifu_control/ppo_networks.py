import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
import glob
import re
from collections import deque
from torch.distributions import Categorical

# 中等规模CNN特征提取器
class MediumFeatureExtractor(nn.Module):
    """
    中等规模CNN特征提取器，适度增加参数量
    """
    def __init__(self, input_channels=3):
        super(MediumFeatureExtractor, self).__init__()
        
        # 中等规模CNN特征提取器
        self.conv_layers = nn.Sequential(
            # 第一层 - 适度减少通道数和步长
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=3, padding=2, bias=False),  # 适度步长减少尺寸
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
    中等规模Actor-Critic网络，适度增加参数量
    """
    def __init__(self, input_channels=3, move_action_space=4, turn_action_space=2, image_height=480, image_width=640):
        super(MediumActorCritic, self).__init__()
        
        # 使用中等规模特征提取器
        self.feature_extractor = MediumFeatureExtractor(input_channels)
        
        conv_out_size = 64  # 64 * 1 * 1
        
        # 中等规模的全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加轻微的dropout防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 分别输出移动和转向动作的概率分布
        self.move_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, move_action_space)  # 4种移动动作
        )
        self.turn_action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, turn_action_space)  # 2种转向动作
        )
        self.param_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 步长和角度
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 价值估计
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        shared_features = self.shared_fc(features)  # [batch_size, 64]
        
        move_action_probs = F.softmax(self.move_action_head(shared_features), dim=-1)
        turn_action_probs = F.softmax(self.turn_action_head(shared_features), dim=-1)
        action_params = torch.sigmoid(self.param_head(shared_features))
        state_value = self.value_head(shared_features)
        
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
    PPO智能体 - 使用中等规模网络
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, lr=0.0005, betas=(0.9, 0.999), 
                 gamma=0.98, K_epochs=4, eps_clip=0.2):  # 适度调整参数
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs  # 适度增加更新轮数
        self.eps_clip = eps_clip
        
        input_channels = 3
        height, width = 480, 640
        # 使用中等规模网络
        self.policy = MediumActorCritic(input_channels, move_action_dim, turn_action_dim, height, width)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, weight_decay=1e-5)  # 适度权重衰减
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
            batch_size = 16  # 适度增加批次大小
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
                
                # 适度调整损失权重
                loss = actor_loss + 0.5 * critic_loss + 0.2 * param_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # 适度梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作
        """
        state = self._preprocess_state(state)
        
        # 添加批次维度进行推理
        state_batch = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            move_action_probs, turn_action_probs, action_params, state_value = self.policy_old(state_batch)
            move_dist = torch.distributions.Categorical(move_action_probs)
            turn_dist = torch.distributions.Categorical(turn_action_probs)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算组合动作的对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            action_logprob = move_logprob + turn_logprob  # 总对数概率
        
        # 将连续参数转换为实际的步长和角度值
        # 将[0,1]范围映射到实际范围
        scaled_params = action_params.squeeze(0)  # 移除批次维度
        move_forward_step = 1.5 * scaled_params[0].item()  # 映射到[0, 4]范围
        turn_angle = 40 * scaled_params[1].item()  # 映射到[0, 40]范围
        
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


def find_latest_checkpoint(model_path):
    """
    查找最新的检查点文件
    """
    model_dir = os.path.dirname(model_path)
    model_base_name = os.path.basename(model_path).replace('.pth', '')
    
    # 查找所有相关的检查点文件
    checkpoint_pattern = re.compile(rf'{re.escape(model_base_name)}_checkpoint_ep_(\d+)\.pth$')
    found_checkpoints = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            match = checkpoint_pattern.match(file)
            if match:
                full_path = os.path.join(model_dir, file)
                epoch_num = int(match.group(1))
                found_checkpoints.append((full_path, epoch_num))
    
    if found_checkpoints:
        # 返回具有最高epoch编号的检查点
        latest_checkpoint = max(found_checkpoints, key=lambda x: x[1])
        return latest_checkpoint[0]
    
    return None


def train_gate_search_ppo_agent_medium(episodes=50, model_path="gate_search_ppo_model_medium.pth", target_description="gate"):
    """
    中等规模门搜索PPO智能体训练 - 默认查找旧检查点并继续训练
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练中等规模门搜索PPO智能体，目标: {target_description}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 导入并初始化环境
    from sifu_control.gate_find_ppo import TargetSearchEnvironment
    env = TargetSearchEnvironment(target_description)
    
    move_action_dim = 4  # 4种移动动作: forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # 2种转向动作: turn_left, turn_right
    
    # 使用中等规模的超参数
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim,
        lr=0.0005,       # 适度学习率
        gamma=0.98,      # 较高的折扣因子
        K_epochs=4,      # 适度更新轮数
        eps_clip=0.2     # 适度PPO裁剪
    )
    
    start_episode = 0
    # 自动查找最新的检查点文件
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint:
        # 从检查点加载
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        logger.info(f"从检查点继续训练: {latest_checkpoint}, 从第 {start_episode} 轮开始")
    elif os.path.exists(model_path):
        # 检查点不存在，但主模型文件存在，从主模型加载
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"从预训练模型加载: {model_path}")
        except Exception as e:
            logger.warning(f"加载主模型失败: {e}")
    
    memory = Memory()
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = Memory()
    
    # 如果从检查点开始，调整总训练轮数
    total_training_episodes = start_episode + episodes
    
    for episode in range(start_episode, total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 环境交互
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
                turn_action_names = ["turn_left", "turn_right"]
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                env.reset_to_origin()
                break

        # 累积批量数据
        batch_memory.states.extend(memory.states)
        batch_memory.move_actions.extend(memory.move_actions)
        batch_memory.turn_actions.extend(memory.turn_actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        if hasattr(memory, 'action_params'):
            batch_memory.action_params.extend(memory.action_params)
        
        memory.clear_memory()
        
        # 每5个episode更新一次
        if (episode + 1) % 5 == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-4)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            ppo_agent.save_checkpoint(checkpoint_path, episode, ppo_agent.optimizer.state_dict())
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.2f}, Avg Reward: {avg_total_reward:.2f}")
    
    # 最终更新
    if len(batch_memory.rewards) > 0:
        print("更新最终策略...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    logger.info("PPO训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"PPO模型已保存为 {model_path}")
    
    return ppo_agent


def continue_training_ppo_agent(model_path, additional_episodes=20, target_description="gate"):
    """
    基于现有模型继续训练
    """
    logger = logging.getLogger(__name__)
    logger.info(f"基于现有模型继续训练: {model_path}, 额外训练 {additional_episodes} 轮")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 导入并初始化环境
    from sifu_control.gate_find_ppo import TargetSearchEnvironment
    env = TargetSearchEnvironment(target_description)
    
    move_action_dim = 4  # 4种移动动作: forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # 2种转向动作: turn_left, turn_right
    
    # 创建PPO智能体
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim,
        lr=0.0005,       # 适度学习率
        gamma=0.98,      # 较高的折扣因子
        K_epochs=4,      # 适度更新轮数
        eps_clip=0.2     # 适度PPO裁剪
    )
    
    # 首先尝试从检查点加载，如果找不到检查点，则从主模型加载
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint:
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
    else:
        # 尝试从主模型文件加载
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"从主模型文件加载: {model_path}")
            start_episode = 0
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return {"status": "error", "message": f"加载模型失败: {e}"}
    
    memory = Memory()
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = Memory()
    
    total_training_episodes = start_episode + additional_episodes
    
    logger.info(f"从第 {start_episode} 轮开始，继续训练 {additional_episodes} 轮，总共到第 {total_training_episodes} 轮")
    
    for episode in range(start_episode, total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 环境交互
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
                turn_action_names = ["turn_left", "turn_right"]
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                env.reset_to_origin()
                break

        # 累积批量数据
        batch_memory.states.extend(memory.states)
        batch_memory.move_actions.extend(memory.move_actions)
        batch_memory.turn_actions.extend(memory.turn_actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        if hasattr(memory, 'action_params'):
            batch_memory.action_params.extend(memory.action_params)
        
        memory.clear_memory()
        
        # 每5个episode更新一次
        if (episode + 1) % 5 == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-4)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            ppo_agent.save_checkpoint(checkpoint_path, episode, ppo_agent.optimizer.state_dict())
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.2f}, Avg Reward: {avg_total_reward:.2f}")
    
    # 最终更新
    if len(batch_memory.rewards) > 0:
        print("更新最终策略...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    logger.info("继续训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"更新后的PPO模型已保存为 {model_path}")
    
    return {"status": "success", "message": f"继续训练完成，共训练了 {additional_episodes} 轮", "final_episode": total_training_episodes}


def evaluate_trained_ppo_agent_medium(model_path="gate_search_ppo_model_medium.pth", episodes=5, target_description="gate"):
    """
    评估中等规模已训练好的PPO模型性能
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估中等规模PPO模型: {model_path}")
    
    # 定义状态和动作维度
    move_action_dim = 4
    turn_action_dim = 2
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), move_action_dim, turn_action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件 {model_path} 不存在，使用随机模型")
    
    evaluation_results = []
    
    # 从外部导入环境
    from sifu_control.gate_find_ppo import TargetSearchEnvironment
    
    for episode in range(episodes):
        # 初始化环境
        env = TargetSearchEnvironment(target_description)
        state = env.reset()
        memory = Memory()  # 创建空的记忆对象，仅用于act函数
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        while not done and step_count < 50:  # 适度增加最大步数
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据，因为我们只是在评估
            memory.clear_memory()
            
            # 打印动作
            move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["turn_left", "turn_right"]
            
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                       f"Step: {move_forward_step}, Turn Angle: {turn_angle}")
            
            state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
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


def load_and_test_ppo_agent_medium(model_path="gate_search_ppo_model_medium.pth", target_description="gate"):
    """
    加载并测试中等规模已训练的PPO模型
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载中等规模PPO模型并测试: {model_path}")
    
    # 定义动作维度
    move_action_dim = 4
    turn_action_dim = 2
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), move_action_dim, turn_action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 从外部导入环境
    from sifu_control.gate_find_ppo import TargetSearchEnvironment
    
    # 初始化环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < 50:  # 适度增加最大步数
        move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        logger.info(f"Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                   f"Step: {move_forward_step}, Turn Angle: {turn_angle}")
        
        state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
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


def find_gate_with_ppo_medium(target_description="gate"):
    """
    使用中等规模PPO算法寻找门（训练+测试流程）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用中等规模PPO寻找目标: {target_description}")
    
    # 确保模型保存目录存在
    model_path = "./model/gate_search_ppo_model_medium.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练PPO智能体
    trained_agent = train_gate_search_ppo_agent_medium(
        episodes=30,  # 训练episodes数量
        target_description=target_description, 
        model_path=model_path
    )
    
    # 测试训练好的智能体
    test_result = load_and_test_ppo_agent_medium(model_path, target_description)
    
    return test_result