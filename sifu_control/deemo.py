"""
门搜索DQN调用演示
直接调用现有的gate_find_dqn.py中的功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    主函数 - 直接调用现有的DQN功能
    """
    print("正在导入现有的门搜索DQN功能...")
    
    try:
        # 导入现有的功能
        from sifu_control.gate_find_dqn import execute_tool
        
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: train_gate_search_agent")
        print("2. 寻找门（包含训练）: find_gate_with_dqn")
        print("3. 评估已训练模型: evaluate_trained_agent")
        print("4. 加载并测试模型: load_and_test_agent")
        
        # 首先执行完整的训练过程，将模型保存到model文件夹
        print("\n=== 开始训练门搜索智能体 (100轮)，模型将保存到model文件夹) ===")
        result = execute_tool("train_gate_search_agent", "100", "./model/gate_search_dqn_model.pth", "gate")
        print(f"\n训练结果: {result}")
        
        print("\n=== 使用训练好的模型进行测试 ===")
        test_result = execute_tool("load_and_test_agent", "./model/gate_search_dqn_model.pth", "gate")
        print(f"\n测试结果: {test_result}")
        
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保sifu_control目录下有gate_find_dqn.py文件")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()