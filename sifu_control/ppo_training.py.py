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
import json
from pathlib import Path


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['config']

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


def train_gate_search_ppo_agent_medium():
    """
    中等规模门搜索PPO智能体训练 - 使用全局配置
    """
    config = CONFIG
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练中等规模门搜索PPO智能体，目标: {config['TARGET_DESCRIPTION']}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(config['MEDIUM_MODEL_PATH'])
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 导入并初始化环境
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4  # 4种移动动作: forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # 2种转向动作: turn_left, turn_right
    
    # 使用改进的超参数
    from sifu_control.ppo_agents import PPOAgent
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    start_episode = 0
    # 自动查找最新的检查点文件
    latest_checkpoint = find_latest_checkpoint(config['MEDIUM_MODEL_PATH'])
    
    if latest_checkpoint:
        # 从检查点加载
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        logger.info(f"从检查点继续训练: {latest_checkpoint}, 从第 {start_episode} 轮开始")
    elif os.path.exists(config['MEDIUM_MODEL_PATH']):
        # 检查点不存在，但主模型文件存在，从主模型加载
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(config['MEDIUM_MODEL_PATH'], map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(config['MEDIUM_MODEL_PATH'], map_location=device))
            logger.info(f"从预训练模型加载: {config['MEDIUM_MODEL_PATH']}")
        except Exception as e:
            logger.warning(f"加载主模型失败: {e}")
    
    from sifu_control.ppo_agents import Memory
    memory = Memory()
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = Memory()
    
    # 如果从检查点开始，调整总训练轮数
    total_training_episodes = start_episode + config['EPISODES']
    
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
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, MStep: {move_forward_step:.3f}, TAngle: {turn_angle:.3f}")
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
        
        # 每UPDATE_INTERVAL个episode更新一次
        if (episode + 1) % config['UPDATE_INTERVAL'] == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-config['UPDATE_INTERVAL']+1)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_path = config['MEDIUM_MODEL_PATH'].replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
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
    
    torch.save(ppo_agent.policy.state_dict(), config['MEDIUM_MODEL_PATH'])
    logger.info(f"PPO模型已保存为 {config['MEDIUM_MODEL_PATH']}")
    
    return ppo_agent


def continue_training_ppo_agent(model_path=None):
    """
    基于现有模型继续训练（改进版）- 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MEDIUM_MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"基于现有模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 导入并初始化环境
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4  # 4种移动动作: forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # 2种转向动作: turn_left, turn_right
    
    # 创建改进的PPO智能体
    from sifu_control.ppo_agents import PPOAgent
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
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
    
    from sifu_control.ppo_agents import Memory
    memory = Memory()
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = Memory()
    
    total_training_episodes = start_episode + config['EPISODES']
    
    logger.info(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
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
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, MStep: {move_forward_step:.3f}, TAngle: {turn_angle:.3f}")
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
        
        # 每UPDATE_INTERVAL个episode更新一次
        if (episode + 1) % config['UPDATE_INTERVAL'] == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-config['UPDATE_INTERVAL']+1)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % config['CHECKPOINT_INTERVAL'] == 0:
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
    
    return {"status": "success", "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮", "final_episode": total_training_episodes}


def evaluate_trained_ppo_agent_medium(model_path=None):
    """
    评估中等规模已训练好的PPO模型性能（改进版）- 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MEDIUM_MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估中等规模PPO模型: {model_path}")
    
    # 定义状态和动作维度
    move_action_dim = 4
    turn_action_dim = 2
    
    # 创建改进的PPO智能体
    from sifu_control.ppo_agents import PPOAgent
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
    from sifu_control.ppo_agents import TargetSearchEnvironment
    
    for episode in range(config['EVAL_EPISODES']):
        # 初始化环境
        env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
        state = env.reset()
        from sifu_control.ppo_agents import Memory
        memory = Memory()  # 创建空的记忆对象，仅用于act函数
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        while not done and step_count < CONFIG['EVAL_MAX_STEPS']:  # 统一使用EVAL_MAX_STEPS
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据，因为我们只是在评估
            memory.clear_memory()
            
            # 打印动作
            move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["turn_left", "turn_right"]
            
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                       f"Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
            
            state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {config['TARGET_DESCRIPTION']}")
                
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
        'total_episodes': config['EVAL_EPISODES'],
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / config['EVAL_EPISODES'] if config['EVAL_EPISODES'] > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'avg_final_area': avg_final_area,
        'details': evaluation_results
    }
    
    logger.info(f"PPO评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{config['EVAL_EPISODES']})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}, 平均最终面积: {avg_final_area:.2f}")
    
    return stats


def load_and_test_ppo_agent_medium(model_path=None):
    """
    加载并测试中等规模已训练的PPO模型（改进版）- 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MEDIUM_MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"加载中等规模PPO模型并测试: {model_path}")
    
    # 定义动作维度
    move_action_dim = 4
    turn_action_dim = 2
    
    # 创建改进的PPO智能体
    from sifu_control.ppo_agents import PPOAgent
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
    from sifu_control.ppo_agents import TargetSearchEnvironment
    
    # 初始化环境
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    state = env.reset()
    from sifu_control.ppo_agents import Memory
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < CONFIG['EVAL_MAX_STEPS']:  # 统一使用EVAL_MAX_STEPS
        move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        logger.info(f"Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                   f"Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
        
        state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {config['TARGET_DESCRIPTION']}")
        
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
            return {"status": "success", "result": f"成功找到{config['TARGET_DESCRIPTION']}！", "steps": step_count, "total_reward": total_reward}
        elif done:
            logger.info("未能找到目标")
            return {"status": "partial_success", "result": f"未能找到{config['TARGET_DESCRIPTION']}", "steps": step_count, "total_reward": total_reward}
    
    # 执行重置到原点操作
    env.reset_to_origin()
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    logger.info(result["result"])
    return result


def find_gate_with_ppo_medium():
    """
    使用中等规模PPO算法寻找门（训练+测试流程，改进版）- 使用全局配置
    """
    config = CONFIG
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用中等规模PPO寻找目标: {config['TARGET_DESCRIPTION']}")
    
    # 确保模型保存目录存在
    model_path = config['MEDIUM_MODEL_PATH']
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练PPO智能体
    trained_agent = train_gate_search_ppo_agent_medium()
    
    # 测试训练好的智能体
    test_result = load_and_test_ppo_agent_medium()
    
    return test_result


# GRU相关函数
def train_gate_search_gru_ppo_agent():
    """
    使用GRU的门搜索PPO智能体训练 - 使用全局配置
    """
    config = CONFIG
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练GRU门搜索PPO智能体，目标: {config['TARGET_DESCRIPTION']}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(config['MODEL_PATH'])
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 导入并初始化环境
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4  # 4种移动动作
    turn_action_dim = 2  # 2种转向动作
    
    # 使用GRU PPO智能体
    from sifu_control.ppo_agents import GRUPPOAgent
    ppo_agent = GRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    start_episode = 0
    latest_checkpoint = find_latest_checkpoint(config['MODEL_PATH'])
    
    if latest_checkpoint:
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        logger.info(f"从检查点继续训练: {latest_checkpoint}, 从第 {start_episode} 轮开始")
    elif os.path.exists(config['MODEL_PATH']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(config['MODEL_PATH'], map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(config['MODEL_PATH'], map_location=device))
            logger.info(f"从预训练模型加载: {config['MODEL_PATH']}")
        except Exception as e:
            logger.warning(f"加载主模型失败: {e}")
    
    from sifu_control.ppo_agents import GRUMemory
    memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
    
    total_training_episodes = start_episode + config['EPISODES']
    
    for episode in range(start_episode, total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        
        # 重置智能体的历史状态
        ppo_agent.state_history.clear()
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 环境交互
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            # 更新memory中的奖励和终止状态
            if memory.rewards:  # 如果不是第一次迭代
                memory.rewards[-1] = reward
                memory.is_terminals[-1] = done
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
                turn_action_names = ["turn_left", "turn_right"]
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, "
                      f"MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, "
                      f"MStep: {move_forward_step:.3f}, TAngle: {turn_angle:.3f}")
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
        
        # 每UPDATE_INTERVAL个episode更新一次
        if (episode + 1) % config['UPDATE_INTERVAL'] == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-config['UPDATE_INTERVAL']+1)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_path = config['MODEL_PATH'].replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
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
    
    logger.info("GRU PPO训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), config['MODEL_PATH'])
    logger.info(f"GRU PPO模型已保存为 {config['MODEL_PATH']}")
    
    return ppo_agent


def continue_training_gru_ppo_agent(model_path=None):
    """
    基于现有GRU模型继续训练 - 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"基于现有GRU模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4
    turn_action_dim = 2
    
    from sifu_control.ppo_agents import GRUPPOAgent
    ppo_agent = GRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint:
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"从主模型文件加载: {model_path}")
            start_episode = 0
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return {"status": "error", "message": f"加载模型失败: {e}"}
    
    from sifu_control.ppo_agents import GRUMemory
    memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
    
    total_training_episodes = start_episode + config['EPISODES']
    
    logger.info(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    for episode in range(start_episode, total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        
        ppo_agent.state_history.clear()
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            if memory.rewards:
                memory.rewards[-1] = reward
                memory.is_terminals[-1] = done
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
                turn_action_names = ["turn_left", "turn_right"]
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, "
                      f"MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, "
                      f"MStep: {move_forward_step:.3f}, TAngle: {turn_angle:.3f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                env.reset_to_origin()
                break

        batch_memory.states.extend(memory.states)
        batch_memory.move_actions.extend(memory.move_actions)
        batch_memory.turn_actions.extend(memory.turn_actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        if hasattr(memory, 'action_params'):
            batch_memory.action_params.extend(memory.action_params)
        
        memory.clear_memory()
        
        if (episode + 1) % config['UPDATE_INTERVAL'] == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-config['UPDATE_INTERVAL']+1)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        if (episode + 1) % config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            ppo_agent.save_checkpoint(checkpoint_path, episode, ppo_agent.optimizer.state_dict())
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.2f}, Avg Reward: {avg_total_reward:.2f}")
    
    if len(batch_memory.rewards) > 0:
        print("更新最终策略...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    logger.info("GRU继续训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"更新后的GRU PPO模型已保存为 {model_path}")
    
    return {"status": "success", "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮", "final_episode": total_training_episodes}


def evaluate_trained_gru_ppo_agent(model_path=None):
    """
    评估已训练的GRU PPO模型性能 - 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估GRU PPO模型: {model_path}")
    
    move_action_dim = 4
    turn_action_dim = 2
    
    from sifu_control.ppo_agents import GRUPPOAgent
    ppo_agent = GRUPPOAgent(
        (3, 480, 640), move_action_dim, turn_action_dim
    )
    
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()
        logger.info(f"GRU PPO模型已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件 {model_path} 不存在，使用随机模型")
    
    evaluation_results = []
    
    from sifu_control.ppo_agents import TargetSearchEnvironment
    
    for episode in range(config['EVAL_EPISODES']):
        env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
        state = env.reset()
        from sifu_control.ppo_agents import GRUMemory
        memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        ppo_agent.state_history.clear()
        
        while not done and step_count < CONFIG['EVAL_MAX_STEPS']:  # 统一使用EVAL_MAX_STEPS
            move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据
            memory.clear_memory()
            
            move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["turn_left", "turn_right"]
            
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - Move: {move_action_names[move_action]}, "
                       f"Turn: {turn_action_names[turn_action]}, Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
            
            state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {config['TARGET_DESCRIPTION']}")
                
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
        
        env.reset_to_origin()
        
        logger.info(f"Episode {episode+1} 完成 - Steps: {step_count}, Total Reward: {total_reward}, Success: {success}, Final Area: {current_area}")
    
    successful_episodes = sum(1 for r in evaluation_results if r['success'])
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
    avg_final_area = np.mean([r['final_area'] for r in evaluation_results])
    
    stats = {
        'total_episodes': config['EVAL_EPISODES'],
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / config['EVAL_EPISODES'] if config['EVAL_EPISODES'] > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'avg_final_area': avg_final_area,
        'details': evaluation_results
    }
    
    logger.info(f"GRU PPO评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{config['EVAL_EPISODES']})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}, 平均最终面积: {avg_final_area:.2f}")
    
    return stats


def load_and_test_gru_ppo_agent(model_path=None):
    """
    加载并测试已训练的GRU PPO模型 - 使用全局配置
    """
    config = CONFIG
    
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"加载GRU PPO模型并测试: {model_path}")
    
    move_action_dim = 4
    turn_action_dim = 2
    
    from sifu_control.ppo_agents import GRUPPOAgent
    ppo_agent = GRUPPOAgent(
        (3, 480, 640), move_action_dim, turn_action_dim
    )
    
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()
        logger.info(f"GRU PPO模型已从 {model_path} 加载")
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    from sifu_control.ppo_agents import TargetSearchEnvironment
    
    env = TargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    state = env.reset()
    from sifu_control.ppo_agents import GRUMemory
    memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
    
    total_reward = 0
    step_count = 0
    done = False
    
    ppo_agent.state_history.clear()
    
    logger.info("开始测试GRU PPO智能体...")
    
    while not done and step_count < CONFIG['EVAL_MAX_STEPS']:  # 统一使用EVAL_MAX_STEPS
        move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
        
        memory.clear_memory()
        
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        logger.info(f"Step {step_count}: Taking action - Move: {move_action_names[move_action]}, "
                   f"Turn: {turn_action_names[turn_action]}, Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
        
        state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {config['TARGET_DESCRIPTION']}")
        
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
            return {"status": "success", "result": f"成功找到{config['TARGET_DESCRIPTION']}！", "steps": step_count, "total_reward": total_reward}
        elif done:
            logger.info("未能找到目标")
            return {"status": "partial_success", "result": f"未能找到{config['TARGET_DESCRIPTION']}", "steps": step_count, "total_reward": total_reward}
    
    env.reset_to_origin()
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    logger.info(result["result"])
    return result


def find_gate_with_gru_ppo():
    """
    使用GRU PPO算法寻找门（训练+测试流程）- 使用全局配置
    """
    config = CONFIG
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用GRU PPO寻找目标: {config['TARGET_DESCRIPTION']}")
    
    model_path = config['MODEL_PATH']
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    trained_agent = train_gate_search_gru_ppo_agent()
    
    test_result = load_and_test_gru_ppo_agent()
    
    return test_result


def train_simple_ppo_agent(episodes=100, model_path="simple_ppo_model.pth"):
    """
    简单PPO训练函数
    """
    logger = logging.getLogger(__name__)
    
    # 初始化环境
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    
    # 初始化智能体
    from sifu_control.ppo_agents import SimplePPOAgent
    agent = SimplePPOAgent()
    
    # 尝试加载已有模型
    agent.load_model(model_path)
    
    from sifu_control.ppo_agents import SimpleMemory
    memory = SimpleMemory()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        while not done:
            move_action, turn_action, move_step, turn_angle = agent.act(state, memory)
            
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_step, turn_angle)
            
            # 存储奖励和终端状态
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Steps: {step_count}")
                break
        
        # 更新策略
        agent.update(memory)
        memory.clear_memory()
        
        # 保存模型
        if (episode + 1) % 10 == 0:
            agent.save_model(model_path)
    
    logger.info("简单PPO训练完成！")
    return agent


def test_simple_ppo_agent(model_path="simple_ppo_model.pth"):
    """
    测试简单PPO智能体
    """
    logger = logging.getLogger(__name__)
    
    # 初始化环境
    from sifu_control.ppo_agents import TargetSearchEnvironment
    env = TargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    
    # 初始化智能体
    from sifu_control.ppo_agents import SimplePPOAgent
    agent = SimplePPOAgent()
    agent.load_model(model_path)
    
    # 测试一轮
    state = env.reset()
    total_reward = 0
    step_count = 0
    done = False
    
    from sifu_control.ppo_agents import SimpleMemory
    memory = SimpleMemory()
    
    while not done and step_count < 100:  # 限制测试步数
        move_action, turn_action, move_step, turn_angle = agent.act(state, memory)
        
        # 清除临时内存
        memory.clear_memory()
        
        state, reward, done, detection_results = env.step(move_action, turn_action, move_step, turn_angle)
        
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"检测到目标: {[d['label'] for d in detection_results]}")
        
        if done:
            if detection_results:
                logger.info("成功找到目标！")
            else:
                logger.info("未找到目标")
            break
    
    env.reset_to_origin()
    
    result = {
        "steps": step_count,
        "total_reward": total_reward,
        "success": bool(detection_results)
    }
    
    logger.info(f"测试完成: {result}")
    return result


def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    """
    logger = logging.getLogger(__name__)
    
    # 简单PPO功能
    if tool_name == "simple_train":
        return train_simple_ppo_agent(*args)
    elif tool_name == "simple_test":
        return test_simple_ppo_agent(*args)
    
    # 动态导入高级功能以避免循环导入
    if tool_name == "find_gate_with_ppo":
        from ppo_agents import find_gate_with_ppo_medium
        func = find_gate_with_ppo_medium
    elif tool_name == "find_gate_with_gru_ppo":
        from ppo_agents import find_gate_with_gru_ppo
        func = find_gate_with_gru_ppo
    elif tool_name == "train_gate_search_ppo_agent":
        from ppo_agents import train_gate_search_ppo_agent_medium
        func = train_gate_search_ppo_agent_medium
    elif tool_name == "train_gate_search_gru_ppo_agent":
        from ppo_agents import train_gate_search_gru_ppo_agent
        func = train_gate_search_gru_ppo_agent
    elif tool_name == "continue_train_ppo_agent":
        from ppo_agents import continue_training_ppo_agent
        func = continue_training_ppo_agent
    elif tool_name == "continue_train_gru_ppo_agent":
        from ppo_agents import continue_training_gru_ppo_agent
        func = continue_training_gru_ppo_agent
    elif tool_name == "evaluate_trained_ppo_agent":
        from ppo_agents import evaluate_trained_ppo_agent_medium
        func = evaluate_trained_ppo_agent_medium
    elif tool_name == "evaluate_trained_gru_ppo_agent":
        from ppo_agents import evaluate_trained_gru_ppo_agent
        func = evaluate_trained_gru_ppo_agent
    elif tool_name == "load_and_test_ppo_agent":
        from ppo_agents import load_and_test_ppo_agent_medium
        func = load_and_test_ppo_agent_medium
    elif tool_name == "load_and_test_gru_ppo_agent":
        from ppo_agents import load_and_test_gru_ppo_agent
        func = load_and_test_gru_ppo_agent
    else:
        logger.error(f"错误: 未知的PPO工具 '{tool_name}'")
        print(f"错误: 未知的PPO工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的PPO工具 '{tool_name}'"}

    try:
        result = func(*args)
        logger.info(f"PPO工具执行成功: {tool_name}")
        return {"status": "success", "result": str(result)}
    except Exception as e:
        logger.error(f"执行PPO工具时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}


def setup_logging():
    """设置日志配置"""
    import time
    import logging
    import os
    from pathlib import Path
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"ppo_networks_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    import sys
    if len(sys.argv) < 2:
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练简单PPO智能体: simple_train")
        print("2. 测试简单PPO智能体: simple_test")
        print("3. 训练门搜索智能体: train_gate_search_ppo_agent")
        print("4. 训练GRU门搜索智能体: train_gate_search_gru_ppo_agent")
        print("5. 寻找门（包含训练）: find_gate_with_ppo")
        print("6. 使用GRU寻找门（包含训练）: find_gate_with_gru_ppo")
        print("7. 评估已训练模型: evaluate_trained_ppo_agent")
        print("8. 评估已训练GRU模型: evaluate_trained_gru_ppo_agent")
        print("9. 加载并测试模型: load_and_test_ppo_agent")
        print("10. 加载并测试GRU模型: load_and_test_gru_ppo_agent")
        
        # 运行快速训练
        print("\n=== 开始简单PPO训练 ===")
        try:
            agent = train_simple_ppo_agent(episodes=50)
            print("\n简单PPO训练完成！")
        except Exception as e:
            logger.error(f"简单PPO训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)

if __name__ == "__main__":
    main()