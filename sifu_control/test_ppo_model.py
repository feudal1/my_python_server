import os
import sys
import time
import numpy as np
import cv2
import torch
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 模拟ImprovedMovementController类，避免依赖外部模块
class MockImprovedMovementController:
    def move_forward(self, duration=0.3):
        pass
    def turn_left(self, angle=30):
        pass
    def turn_right(self, angle=30):
        pass

# 替换control_api_tool模块
sys.modules['control_api_tool'] = type('obj', (object,), {'ImprovedMovementController': MockImprovedMovementController})()

from sifu_control.ppo_training import TargetSearchEnvironment, PPOAgent, CONFIG, Memory

def test_ppo_model(model_path, test_episodes=5, show_visualization=False):
    """
    测试训练好的PPO模型
    
    参数:
    - model_path: 模型文件路径
    - test_episodes: 测试轮数
    - show_visualization: 是否显示可视化
    
    返回:
    - 测试结果字典
    """
    print(f"\n=== 开始测试PPO模型 ===")
    print(f"测试模型: {model_path}")
    print(f"测试轮数: {test_episodes}")
    
    # 创建测试环境
    env = TargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    
    # 初始化智能体
    turn_action_dim = 2  # turn_left, turn_right
    ppo_agent = PPOAgent(
        (3, CONFIG.get('IMAGE_HEIGHT', 480), CONFIG.get('IMAGE_WIDTH', 640)),
        turn_action_dim
    )
    
    # 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        ppo_agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        ppo_agent.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        print(f"成功加载模型: {model_path}")
        
        # 显示训练历史信息
        if 'loss_history' in checkpoint and 'reward_history' in checkpoint:
            print(f"模型训练历史: {len(checkpoint['reward_history'])} 轮")
            if len(checkpoint['reward_history']) > 0:
                avg_reward = np.mean(checkpoint['reward_history'][-20:])
                print(f"最近20轮平均奖励: {avg_reward:.2f}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 测试结果记录
    test_results = {
        'episodes': test_episodes,
        'success_count': 0,
        'total_rewards': [],
        'steps_per_episode': [],
        'final_areas': [],
        'detection_success_rate': 0
    }
    
    # 创建测试结果保存目录
    test_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    # 开始测试
    for episode in range(test_episodes):
        print(f"\n=== 测试轮次 {episode+1}/{test_episodes} ===")
        
        # 重置环境
        state = env.reset()
        total_reward = 0
        step_count = 0
        success_flag = False
        climb_detected = False
        
        # 检查初始状态
        initial_detections = env.detect_target(state)
        climb_detected = env._check_climb_conditions(initial_detections)
        
        # 测试循环
        while step_count < env.max_steps and not climb_detected:
            # 生成CNN解释化掩码
            cnn_heatmap = ppo_agent.generate_cnn_heatmap(state)
            
            # 保存测试过程中的掩码
            if step_count % 5 == 0 or step_count == env.max_steps - 1:
                mask_filename = os.path.join(test_results_dir, f"test_episode_{episode+1}_step_{step_count+1}.png")
                cv2.imwrite(mask_filename, cnn_heatmap)
                print(f"保存测试掩码: {mask_filename}")
            
            # 选择动作
            test_memory = Memory()
            move_action, turn_action, move_duration, turn_angle = ppo_agent.select_action(
                state, test_memory, return_debug_info=False
            )
            
            # 执行动作
            next_state, reward, done, detections = env.step(
                move_action, turn_action, move_duration, turn_angle
            )
            
            # 检查是否检测到climb
            climb_detected = env._check_climb_conditions(detections)
            
            # 更新状态
            total_reward += reward
            state = next_state
            step_count += 1
            
            # 打印步骤信息
            print(f"步骤 {step_count}: 奖励 = {reward:.2f}, 累计奖励 = {total_reward:.2f}")
            
            if done:
                break
        
        # 记录测试结果
        success_flag = climb_detected
        test_results['total_rewards'].append(total_reward)
        test_results['steps_per_episode'].append(step_count)
        test_results['final_areas'].append(env.last_area)
        
        if success_flag:
            test_results['success_count'] += 1
            print(f"✅ 测试轮次 {episode+1} 成功！")
        else:
            print(f"❌ 测试轮次 {episode+1} 失败")
        
        print(f"轮次 {episode+1} 结果: 奖励 = {total_reward:.2f}, 步数 = {step_count}, 最终面积 = {env.last_area:.2f}")
    
    # 计算测试指标
    test_results['success_rate'] = test_results['success_count'] / test_episodes
    test_results['avg_reward'] = np.mean(test_results['total_rewards']) if test_results['total_rewards'] else 0
    test_results['avg_steps'] = np.mean(test_results['steps_per_episode']) if test_results['steps_per_episode'] else 0
    test_results['avg_final_area'] = np.mean(test_results['final_areas']) if test_results['final_areas'] else 0
    
    # 生成测试报告
    print(f"\n=== 测试报告 ===")
    print(f"测试轮数: {test_episodes}")
    print(f"成功轮数: {test_results['success_count']}")
    print(f"成功率: {test_results['success_rate']:.2f}")
    print(f"平均奖励: {test_results['avg_reward']:.2f}")
    print(f"平均步数: {test_results['avg_steps']:.2f}")
    print(f"平均最终面积: {test_results['avg_final_area']:.2f}")
    
    # 保存测试结果
    test_report_path = os.path.join(test_results_dir, f"test_report_{int(time.time())}.txt")
    with open(test_report_path, 'w', encoding='utf-8') as f:
        f.write("=== PPO模型测试报告 ===\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试模型: {model_path}\n")
        f.write(f"测试轮数: {test_episodes}\n")
        f.write(f"成功轮数: {test_results['success_count']}\n")
        f.write(f"成功率: {test_results['success_rate']:.2f}\n")
        f.write(f"平均奖励: {test_results['avg_reward']:.2f}\n")
        f.write(f"平均步数: {test_results['avg_steps']:.2f}\n")
        f.write(f"平均最终面积: {test_results['avg_final_area']:.2f}\n\n")
        f.write("详细结果:\n")
        for i, (reward, steps, area) in enumerate(zip(
            test_results['total_rewards'], 
            test_results['steps_per_episode'], 
            test_results['final_areas']
        )):
            success = "成功" if i < test_results['success_count'] else "失败"
            f.write(f"轮次 {i+1}: 奖励 = {reward:.2f}, 步数 = {steps}, 面积 = {area:.2f}, 结果 = {success}\n")
    
    print(f"测试报告已保存: {test_report_path}")
    
    return test_results

def find_latest_model(model_dir='sifu_model'):
    """
    查找最新的模型文件
    """
    model_dir = os.path.join(Path(__file__).parent.parent, model_dir)
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        return None
    
    # 查找所有检查点文件
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('ppo_model_checkpoint_ep_') and file.endswith('.pth'):
            try:
                ep_num = int(file.split('_')[-1].replace('.pth', ''))
                checkpoint_files.append((file, ep_num))
            except:
                pass
    
    if not checkpoint_files:
        print("错误: 未找到模型文件")
        return None
    
    # 按轮次编号排序，返回最新的
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    latest_file = checkpoint_files[0][0]
    latest_path = os.path.join(model_dir, latest_file)
    
    print(f"找到最新模型: {latest_path}")
    return latest_path

def main():
    """
    主函数
    """
    print("=== PPO模型测试工具 ===")
    
    # 查找最新模型
    latest_model = find_latest_model()
    if not latest_model:
        print("请指定模型文件路径")
        return
    
    # 测试参数
    test_episodes = 5
    show_visualization = False
    
    # 运行测试
    test_results = test_ppo_model(
        latest_model,
        test_episodes=test_episodes,
        show_visualization=show_visualization
    )
    
    if test_results:
        print("\n=== 测试完成 ===")
    else:
        print("\n=== 测试失败 ===")

if __name__ == "__main__":
    main()
