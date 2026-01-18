
import gym
import numpy as np
import torch
from PPO_agent import PPOAgent, Memory

def main():
    """
    主训练函数 - 使用PPO算法训练Frozen Lake环境
    """
    print("开始训练PPO智能体...")
    
    # 创建环境
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    state_dim = env.observation_space.n  # 16个状态 (4x4网格)
    action_dim = env.action_space.n     # 4个动作 (上、右、下、左)
    
    # 将离散状态转换为连续表示（one-hot编码）
    state_dim_continuous = state_dim
    
    # 创建PPO智能体
    agent = PPOAgent(state_dim_continuous, action_dim, 
                     lr=0.0003, 
                     gamma=0.99, 
                     eps_clip=0.2, 
                     K_epochs=80,
                     entropy_coeff=0.01)
    
    # 训练参数
    num_episodes = 1000
    max_timesteps = 100
    update_timestep = 2000  # 每2000步更新一次策略
    
    memory = Memory()
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):  # 处理新版本gym返回元组的情况
            state = state[0]
        
        # 将离散状态转换为one-hot编码
        state_onehot = np.zeros(state_dim_continuous)
        state_onehot[state] = 1
        
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # 选择动作
            action, logprob = agent.select_action(state_onehot)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 给予到达目标的奖励，否则给予小的负奖励
            if terminated and reward == 0:  # 到达洞中
                reward = -1
            elif terminated and reward > 0:  # 到达目标
                reward = 1
            
            # 将离散状态转换为one-hot编码
            next_state_onehot = np.zeros(state_dim_continuous)
            next_state_onehot[next_state] = 1
            
            # 存储经验到记忆中
            memory.states.append(torch.FloatTensor(state_onehot))
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            
            # 更新状态
            state_onehot = next_state_onehot
            state = next_state
            episode_reward += reward
            
            # 如果回合结束则跳出
            if done:
                break
        
        running_reward += episode_reward
        
        # 更新网络
        if timestep % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()
        
        # 打印平均奖励
        if i_episode % 100 == 0:
            avg_reward = running_reward / 100
            print(f'Episode {i_episode}, last 100 episodes average reward: {avg_reward:.2f}')
            running_reward = 0

    print("训练完成！")
    
    # 测试训练好的模型
    test_model(env, agent, state_dim_continuous)

def test_model(env, agent, state_dim_continuous, n_tests=10):
    """
    测试训练好的模型
    """
    print("\n开始测试训练好的模型...")
    total_rewards = []
    
    for i in range(n_tests):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        state_onehot = np.zeros(state_dim_continuous)
        state_onehot[state] = 1
        
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作（不带随机性）
            with torch.no_grad():
                action_probs, _ = agent.old_policy(torch.FloatTensor(state_onehot).unsqueeze(0))
                action = torch.argmax(action_probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 显示环境（可选）
            # env.render()
            
            state = next_state
            state_onehot = np.zeros(state_dim_continuous)
            state_onehot[state] = 1
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    avg_test_reward = sum(total_rewards) / len(total_rewards)
    print(f'测试结果 - 平均奖励: {avg_test_reward:.2f}, 成功次数: {sum(total_rewards)} / {n_tests}')

if __name__ == "__main__":
    main()