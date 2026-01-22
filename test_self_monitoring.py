"""
测试自我监控功能
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from llm_server.llm_class import LLMService, VLMService

# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("测试自我监控功能")
    print("=" * 50)

    # 初始化服务
    print("\n1. 初始化LLM服务...")
    try:
        llm_service = LLMService()
        print("   ✓ LLM服务初始化成功")
    except Exception as e:
        print(f"   ✗ LLM服务初始化失败: {e}")
        sys.exit(1)

    print("\n2. 初始化VLM服务...")
    try:
        vlm_service = VLMService()
        print("   ✓ VLM服务初始化成功")
    except Exception as e:
        print(f"   ✗ VLM服务初始化失败: {e}")
        sys.exit(1)

    print("\n3. 导入自我监控模块...")
    try:
        from mcp_cline.self_monitoring import SelfMonitoringThread
        print("   ✓ 自我监控模块导入成功")
    except Exception as e:
        print(f"   ✗ 自我监控模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n4. 创建自我监控线程...")
    try:
        monitoring_thread = SelfMonitoringThread(
            vlm_service=vlm_service,
            llm_service=llm_service,
            callback_analysis=lambda x: print(f"[VLM分析] {x}"),
            callback_commentary=lambda x: print(f"[吐槽] {x}")
        )
        print("   ✓ 自我监控线程创建成功")
    except Exception as e:
        print(f"   ✗ 自我监控线程创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 50)
    print("所有模块初始化完成！")
    print("=" * 50)

    print("\n说明:")
    print("  - 自我监控线程已创建但未启动")
    print("  - 使用 monitoring_thread.start_monitoring() 启动")
    print("  - 每10秒截5张图，间隔2秒")
    print("  - 每3个VLM分析后生成一次吐槽")
    print("\n命令:")
    print("  - monitoring_thread.start_monitoring()  # 启动")
    print("  - monitoring_thread.stop_monitoring()   # 停止")
    print("  - monitoring_thread.pause_monitoring()  # 暂停")
    print("  - monitoring_thread.resume_monitoring() # 恢复")
    print("\n" + "=" * 50)

    # 交互式测试
    print("\n是否要启动自我监控进行测试? (y/n): ", end="")
    choice = input().strip().lower()

    if choice == 'y':
        print("\n启动自我监控...")
        monitoring_thread.start_monitoring()

        print("按 Ctrl+C 停止监控")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n停止监控...")
            monitoring_thread.stop_monitoring()
            print("已停止")
    else:
        print("\n测试完成")
