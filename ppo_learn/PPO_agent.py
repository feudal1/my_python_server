
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络（Actor）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络（Critic）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=80, entropy_coeff=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.old_policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.old_policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action.item(), action_probs[0][action].item()
    
    def update(self, memory):
        # 计算回报值
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        # 归一化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # 转换为张量
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device).detach()
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(self.device).detach()
        
        # 更新策略K次
        for _ in range(self.K_epochs):
            # 获取当前策略下的动作概率和状态值
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            state_values = torch.squeeze(state_values)
            
            # 计算比率 (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算损失函数
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # PPO损失 = -min(surr1, surr2) + MSE Loss (value function)
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards)
            
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]