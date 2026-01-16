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


def continue_training_gru_ppo_agent(model_path=None):
    """
    基于现有GRU模型继续训练 - 使用全局配置，带收敛监控
    """
    config = CONFIG
    convergence_info = {}
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = logging.getLogger(__name__)
    logger.info(f"基于现有GRU模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    from ppo_agents import EnhancedTargetSearchEnvironment
    env = EnhancedTargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4
    turn_action_dim = 2
    
    from ppo_agents import EnhancedGRUPPOAgent
    ppo_agent = EnhancedGRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    # 首先查找检查点文件
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint:
        # 如果找到检查点，优先加载检查点
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        logger.info(f"从最新检查点加载: {latest_checkpoint}")
    elif os.path.exists(model_path):
        # 没有检查点但主模型存在，加载主模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"从主模型文件加载: {model_path}")
            start_episode = 0
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return {"status": "error", "message": f"加载模型失败: {e}"}
    else:
        # 模型文件和检查点都不存在，创建新模型
        logger.info(f"模型文件和检查点都不存在: {model_path}，创建新模型开始训练")
        start_episode = 0
        
        # 确保模型目录存在
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    from ppo_agents import GRUMemory
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
        success_flag = False  # 任务成功标志
        
        ppo_agent.state_history.clear()
        
        # 修复：将while循环正确缩进到for循环内部
        while not done:
            move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(state, memory, return_debug_info=True)
                        
            # 打印神经网络输出和动作参数在一起
            print(f"\n--- 动作详情 ---")
            print(f"GRU前5个值: {[f'{val:.2f}' for val in debug_info['gru_last_output'][0][:5]]}")
            
            # Move动作分支输出 - 4个离散动作的概率
            move_probs = debug_info['move_logits'][0]
            move_softmax = np.exp(move_probs) / np.sum(np.exp(move_probs))
            print(f"MoveProbabilities: {[f'{p:.2f}' for p in move_softmax]}")
            
            # Turn动作分支输出 - 2个离散动作的概率
            turn_probs = debug_info['turn_logits'][0]
            turn_softmax = np.exp(turn_probs) / np.sum(np.exp(turn_probs))
            print(f"TurnProbabilities: {[f'{p:.2f}' for p in turn_softmax]}")
            
            # 价值函数输出
            print(f"价值函数输出: {debug_info['value'][0][0]:.2f}")
            
            # 动作和参数信息
            move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["turn_left", "turn_right"]
            print(f"执行动作: Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, Move Step: {move_forward_step:.2f}, Turn Angle: {turn_angle:.2f}")
            print(f"------------------\n")
            
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            if len(memory.rewards) > 0:
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
                
                # 检查是否成功找到目标
                climb_detected = any(
                    detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                    for detection in detection_results
                )
                success_flag = climb_detected
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}, Success: {success_flag}")
                print(f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}, "
                    f"Success: {success_flag}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, "
                    f"MStep: {move_forward_step:.2f}, TAngle: {turn_angle:.2f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                # 使用收敛监控
                convergence_info = ppo_agent.check_convergence_status(total_reward, step_count, success_flag)
                
                # 每10轮打印一次收敛报告
                if episode % 10 == 0:
                    print(f"收敛监控 - 平均奖励: {convergence_info['avg_recent_reward']:.2f}, "
                        f"成功率: {convergence_info['recent_success_rate']:.2f}, "
                        f"学习率: {convergence_info['current_learning_rate']:.6f}, "
                        f"剩余耐心: {convergence_info['current_patience']}")
                
                # 检查早停条件
                if convergence_info['should_stop_early']:
                    logger.info(f"早停触发: 连续{ppo_agent.patience}轮无改善")
                    print(f"早停触发: 连续{ppo_agent.patience}轮无改善，训练结束")
                    break
                
                # 重置环境并获取新的初始状态
                env.reset_to_origin()
                state = env.reset()
                break

        # 每个episode结束后都将其记忆加入批量记忆
        batch_memory.states.extend(memory.states)
        batch_memory.move_actions.extend(memory.move_actions)
        batch_memory.turn_actions.extend(memory.turn_actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        if hasattr(memory, 'action_params'):
            batch_memory.action_params.extend(memory.action_params)
        
        memory.clear_memory()
        
        # 每个episode结束后都尝试更新策略
        if len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episode {episode+1}...")
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
        
        # 检查早停
        if convergence_info.get('should_stop_early', False):
            break
    
    if len(batch_memory.rewards) > 0:
        print("更新最终策略...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    # 打印最终收敛报告
    final_report = ppo_agent.get_convergence_report()
    print("\n=== 训练收敛报告 ===")
    for key, value in final_report.items():
        print(f"{key}: {value}")
    
    logger.info("GRU继续训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"更新后的GRU PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮", 
        "final_episode": total_training_episodes,
        "convergence_report": final_report
    }

def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    """
    logger = logging.getLogger(__name__)
    
    # 动态导入高级功能以避免循环导入
   

    if tool_name == "continue_train_gru_ppo_agent":
        
        func = continue_training_gru_ppo_agent

  
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
        continue_training_gru_ppo_agent()
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练GRU门搜索智能体: train_gate_search_gru_ppo_agent")
        print("2. 使用GRU寻找门（包含训练）: find_gate_with_gru_ppo")
        print("3. 评估已训练GRU模型: evaluate_trained_gru_ppo_agent")
        print("4. 加载并测试GRU模型: load_and_test_gru_ppo_agent")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)

if __name__ == "__main__":
    main()