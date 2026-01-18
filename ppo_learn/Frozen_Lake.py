
import gym
import numpy as np
import torch
from PPO_agent import PPOAgent, Memory

def train_frozen_lake():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    
    # 将状态维度转换为连续值（用于神经网络）
    # 对于离散状态，我们将其转换为one-hot编码
    state_dim_continuous = state_dim
    
    # 创建PPO智能体
    agent = PPOAgent(state_dim_continuous, action_dim)
    
    # 训练参数
    num_episodes = 1000
    max_timesteps = 100
    update_timestep = 2000  # 每2000步更新一次
    
    memory = Memory()
    running_reward = 0
    avg_length = 0
    
    for i_episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # 处理新版本gym返回元组的情况
            state = state[0]
        
        # 将离散状态转换为one-hot编码
        state_onehot = np.zeros(state_dim_continuous)
        state_onehot[state] = 1
        
        episode_reward = 0
        
        for t in range(max_timesteps):
            # 选择动作
            action, logprob = agent.select_action(state_onehot)
            
            # 执行动作
            next_state, reward, done, trunc, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state, reward, done, trunc, _ = next_state, reward, done or trunc, trunc, _
            
            # 将离散状态转换为one-hot编码
            next_state_onehot = np.zeros(state_dim_continuous)
            next_state_onehot[next_state] = 1
            
            # 存储经验
            memory.states.append(torch.FloatTensor(state_onehot))
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            
            # 更新状态
            state_onehot = next_state_onehot
            state = next_state
            episode_reward += reward
            
            # 如果回合结束则跳出
            if done or trunc:
                break
        
        running_reward += episode_reward
        avg_length += t + 1
        
        # 更新网络
        if i_episode % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()
        
        # 打印平均奖励
        if i_episode % 100 == 0:
            avg_reward = running_reward / 100
            print(f'Episode {i_episode}, average reward: {avg_reward:.2f}')
            running_reward = 0
            avg_length = 0

if __name__ == "__main__":
    train_frozen_lake()