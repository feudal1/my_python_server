import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import glob
import re
from collections import deque
from torch.distributions import Categorical
import json
from pathlib import Path


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config 

CONFIG = load_config()


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


def create_environment_and_agent():
    """
    创建环境和智能体
    """
    from ppo_agents import EnhancedTargetSearchEnvironment
    from ppo_agents import EnhancedGRUPPOAgent
    from ppo_agents import GRUMemory
    
    env = EnhancedTargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    move_action_dim = 4  # forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # turn_left, turn_right
    
    ppo_agent = EnhancedGRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    return env, ppo_agent


def load_model(ppo_agent, model_path):
    """
    加载模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def initialize_model(model_path, load_existing=True):
    """
    初始化模型，包括加载已存在的模型或创建新模型
    """
    env, ppo_agent = create_environment_and_agent()
    
    start_episode = 0
    
    if load_existing:
        # 首先查找检查点文件
        latest_checkpoint = find_latest_checkpoint(model_path)
        
        if latest_checkpoint:
            # 如果找到检查点，优先加载检查点
            start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        elif os.path.exists(model_path):
            # 没有检查点但主模型存在，加载主模型
            if not load_model(ppo_agent, model_path):
                raise Exception(f"加载模型失败: {model_path}")
            start_episode = 0
    
    return env, ppo_agent, start_episode


def run_episode(env, ppo_agent, episode_num, total_episodes, training_mode=True, print_debug=False):
    """
    运行单个episode，确保在结束时重置环境
    """
    state = env.reset()  # 重置环境
    total_reward = 0
    step_count = 0
    done = False
    final_area = 0
    success_flag = False
    
    # 为每个episode创建独立的记忆
    from ppo_agents import GRUMemory
    episode_memory = GRUMemory(sequence_length=CONFIG['SEQUENCE_LENGTH'])
    ppo_agent.state_history.clear()
    
    # 定义动作名称映射
    move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
    turn_action_names = ["turn_left", "turn_right"]
    
    # 添加最大步数限制，防止无限循环
    max_steps = CONFIG.get('MAX_STEPS_PER_EPISODE', 1000)
    
    while not done and step_count < max_steps:
        if training_mode:
            # 训练模式：使用act方法
            if print_debug:
                move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(
                    state, episode_memory, return_debug_info=True)
                
                # 打印调试信息
                if 'move_probs' in debug_info:
                    print(f"move_probs shape: {debug_info['move_probs'].shape}")
                    print(f"turn_probs shape: {debug_info['turn_probs'].shape}")
                    print(f"move_action: {move_action}, turn_action: {turn_action}")
            else:
                move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(
                    state, episode_memory)
        else:
            # 评估模式：使用确定性动作
            move_action, turn_action, move_forward_step, turn_angle = get_deterministic_action(ppo_agent, state)
        
        # 执行环境步骤
        next_state, reward, done, detection_results = env.step(
            move_action, turn_action, move_forward_step, turn_angle)
        
        # 更新episode_memory中的奖励和终端状态
        if len(episode_memory.rewards) > 0:
            episode_memory.rewards[-1] = reward
            episode_memory.is_terminals[-1] = done
        
        # 更新状态
        state = next_state
        total_reward += reward
        step_count += 1
        
        # 记录最终检测面积
        if detection_results:
            final_area = max(d['width'] * d['height'] for d in detection_results 
                           if 'width' in d and 'height' in d)
    
    # 检查是否成功找到目标
    if detection_results:
        climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in detection_results
        )
        success_flag = climb_detected
    
    # 在训练模式下，需要更新智能体
    if training_mode and len(episode_memory.rewards) > 0:
        # 更新智能体
        ppo_agent.update(episode_memory)
    
    # 确保在episode结束时重置环境
    # 注意：env.reset()已经在env.step()中被调用了，所以这里不需要再次调用
    # env.reset()
    
    return {
        'total_reward': total_reward,
        'step_count': step_count,
        'final_area': final_area,
        'success_flag': success_flag,
        'detection_results': detection_results
    }

def get_deterministic_action(ppo_agent, state):
    """
    在评估模式下获取确定性动作
    """
    state_tensor = ppo_agent._preprocess_state(state)
    
    # 将当前状态添加到历史记录
    ppo_agent.state_history.append(state_tensor)
    
    # 如果历史记录长度不足，用当前状态填充
    while len(ppo_agent.state_history) < ppo_agent.sequence_length:
        ppo_agent.state_history.appendleft(state_tensor)
    
    # 转换为张量并添加批次维度
    state_seq = torch.stack(list(ppo_agent.state_history)).unsqueeze(0)
    
    # 使用旧策略获取动作概率（但不采样，而是选择最高概率的动作）
    with torch.no_grad():
        move_probs, turn_probs, action_params, state_val, _ = ppo_agent.policy_old(state_seq)
        
        # 确保处理正确的张量维度
        move_probs_squeezed = move_probs.squeeze() if move_probs.dim() > 1 else move_probs
        turn_probs_squeezed = turn_probs.squeeze() if turn_probs.dim() > 1 else turn_probs
        
        # 选择最高概率的动作
        move_action = torch.argmax(move_probs_squeezed, dim=-1).item()
        turn_action = torch.argmax(turn_probs_squeezed, dim=-1).item()
        
        # 获取动作参数
        move_forward_step_raw = action_params[0][0].item()
        turn_angle_raw = action_params[0][1].item()
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 映射到实际范围
        move_forward_step = (torch.tanh(torch.tensor(move_forward_step_raw)) + 1) / 2 * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = (torch.tanh(torch.tensor(turn_angle_raw)) + 1) / 2 * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN

    return move_action, turn_action, move_forward_step, turn_angle


# 在 perform_training_loop 函数中添加状态多样性监控
def perform_training_loop(env, ppo_agent, start_episode, total_episodes):
    """
    执行训练循环 - 增加状态多样性监控和检查点保存
    """
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    # 训练统计变量
    training_stats = {
        'episode_count': 0,
        'successful_episodes': 0,
        'average_reward_history': [],
        'action_diversity': [],  # 新增：动作多样性
        'state_diversity': [],   # 新增：状态多样性
        'entropy_history': []    # 新增：策略熵
    }
    
    print(f"开始训练循环，从第 {start_episode} 轮到第 {total_episodes} 轮")
    loop_start_time = time.time()
    
    for episode in range(start_episode, total_episodes):
        # 记录episode开始时的状态
        episode_states = []
        episode_actions = []
        
        print_debug = True  # 每50轮打印一次详细信息
        result = run_episode(env, ppo_agent, episode, total_episodes, training_mode=True, print_debug=print_debug)
        
        # 更新统计数据
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        final_areas.append(result['final_area'])
        training_stats['episode_count'] += 1
        
        if result['success_flag']:
            training_stats['successful_episodes'] += 1
        
        # 使用收敛监控
        convergence_info = ppo_agent.check_convergence_status(
            result['total_reward'], result['step_count'], result['success_flag'])
        
        # 每10轮保存一次检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = f"{CONFIG['MODEL_PATH'].rsplit('.', 1)[0]}_checkpoint_ep_{episode + 1}.pth"
            ppo_agent.save_checkpoint(checkpoint_path, episode + 1)
            print(f"检查点已保存: {checkpoint_path}")
        print(f"训练进度: {episode+1}/{total_episodes}")
        # 每25轮打印一次收敛报告
        if episode %10 == 0:
            current_time = time.time()
            elapsed_time = current_time - loop_start_time
            
            avg_reward = np.mean(total_rewards) if total_rewards else 0
            success_rate = training_stats['successful_episodes'] / training_stats['episode_count'] if training_stats['episode_count'] > 0 else 0
            
            print(f"Episode数 {episode}: 平均奖励: {avg_reward:.3f}, "
                  f"成功率: {success_rate:.3f}, "
                  f"总奖励: {result['total_reward']:.3f}, "
                  f"步数: {result['step_count']}, "
                  f"成功: {result['success_flag']}")
            print(f"收敛监控 - 学习率: {convergence_info['current_learning_rate']:.6f}")
            print(f"当前训练耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        # 检查早停条件
        if convergence_info['should_stop_early']:
            current_time = time.time()
            elapsed_time = current_time - loop_start_time
            print(f"早停触发: 连续{ppo_agent.patience}轮无改善，总耗时: {elapsed_time:.2f} 秒")
            break
    
    return training_stats
def continue_training_gru_ppo_agent(model_path=None):
    """
    基于现有GRU模型继续训练 - 使用全局配置，带收敛监控
    """
    config = CONFIG
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    print(f"基于现有GRU模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    # 开始计时
    start_time = time.time()
    
    # 初始化模型
    env, ppo_agent, start_episode = initialize_model(model_path, load_existing=True)
    
    total_training_episodes = start_episode + config['EPISODES']
    print(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, start_episode, total_training_episodes)
    
    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"GRU继续训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"更新后的GRU PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮，耗时 {training_duration:.2f} 秒", 
        "final_episode": total_training_episodes,
        "training_stats": {
            "successful_episodes": training_stats['successful_episodes'],
            "total_episodes": training_stats['episode_count']
        },
        "training_duration": training_duration
    }


def train_new_gru_ppo_agent(model_path=None):
    """
    从头开始训练GRU PPO智能体
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    print(f"开始从头训练GRU PPO智能体: {model_path}")
    
    # 开始计时
    start_time = time.time()
    
    # 初始化模型（不加载现有模型）
    env, ppo_agent, start_episode = initialize_model(model_path, load_existing=False)
    
    total_training_episodes = CONFIG['EPISODES']
    print(f"从第 {start_episode} 轮开始，训练 {CONFIG['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, start_episode, total_training_episodes)
    
    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"GRU从头训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"新训练的GRU PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"从头训练完成，共训练了 {CONFIG['EPISODES']} 轮，耗时 {training_duration:.2f} 秒", 
        "final_episode": total_training_episodes,
        "training_stats": {
            "successful_episodes": training_stats['successful_episodes'],
            "total_episodes": training_stats['episode_count']
        },
        "training_duration": training_duration
    }


def evaluate_trained_gru_ppo_agent(model_path=None):
    """
    评估已训练的GRU PPO模型
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    print(f"评估已训练的GRU PPO模型: {model_path}")
    
    # 开始计时
    start_time = time.time()
    
    # 创建环境和智能体
    env, ppo_agent = create_environment_and_agent()
    
    # 加载训练好的模型
    if not load_model(ppo_agent, model_path):
        return {"status": "error", "message": f"模型加载失败"}
    
    # 设置为评估模式
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    
    evaluation_episodes = CONFIG.get('EVALUATION_EPISODES', 10)
    scores = []
    total_rewards = []
    success_count = 0
    
    print(f"开始评估，共 {evaluation_episodes} 个episode")
    
    for episode in range(evaluation_episodes):
        # 每隔几轮打印一次调试信息
        print_debug =True  # 每2轮打印一次调试信息
        result = run_episode(env, ppo_agent, episode, evaluation_episodes, training_mode=False, print_debug=print_debug)
        
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        
        if result['success_flag']:
            success_count += 1
        
        print(f"Eval Ep: {episode+1}/{evaluation_episodes}, Steps: {result['step_count']}, "
              f"Reward: {result['total_reward']:.3f}, Success: {result['success_flag']}")
    
    # 计算评估指标
    avg_score = np.mean(scores) if scores else 0
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    success_rate = success_count / evaluation_episodes if evaluation_episodes > 0 else 0
    
    evaluation_result = {
        'avg_score': avg_score,
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'total_evaluated_episodes': evaluation_episodes,
        'success_count': success_count
    }
    
    # 结束计时
    end_time = time.time()
    evaluation_duration = end_time - start_time
    
    print(f"\n=== 评估结果 ===")
    print(f"平均步数: {avg_score:.3f}")
    print(f"平均奖励: {avg_reward:.3f}")
    print(f"成功率: {success_rate:.3f}")
    print(f"成功次数: {success_count}/{evaluation_episodes}")
    print(f"评估耗时: {evaluation_duration:.2f} 秒 ({evaluation_duration/60:.2f} 分钟)")
    
    print("模型评估完成！")
    
    return {
        "status": "success",
        "message": f"模型评估完成，耗时 {evaluation_duration:.2f} 秒",
        "evaluation_result": evaluation_result,
        "evaluation_duration": evaluation_duration
    }


def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    """
    
    tool_functions = {
        "continue_train_gru_ppo_agent": continue_training_gru_ppo_agent,
        "train_new_gru_ppo_agent": train_new_gru_ppo_agent,
        "evaluate_trained_gru_ppo_agent": evaluate_trained_gru_ppo_agent
    }
    
    if tool_name not in tool_functions:
        error_msg = f"错误: 未知的PPO工具 '{tool_name}'"
        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        result = tool_functions[tool_name](*args)
        print(f"PPO工具执行成功: {tool_name}")
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"执行PPO工具时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}


def main():
    """
    主函数，用于直接运行此脚本
    """
    import sys
    if len(sys.argv) < 2:
        print("默认运行继续训练...")
        continue_training_gru_ppo_agent()
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练GRU门搜索智能体: python ppo_training.py train_new_gru_ppo_agent [model_path]")
        print("2. 继续训练GRU门搜索智能体: python ppo_training.py continue_train_gru_ppo_agent [model_path]")
        print("3. 评估已训练GRU模型: python ppo_training.py evaluate_trained_gru_ppo_agent [model_path]")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(str(response))


if __name__ == "__main__":
    main()