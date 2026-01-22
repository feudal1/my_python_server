"""
测试向量记忆系统
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_cline.vector_memory import VectorMemory

def main():
    print("=" * 60)
    print("测试向量记忆系统")
    print("=" * 60)

    # 创建记忆系统
    print("\n1. 创建向量记忆系统...")
    memory = VectorMemory()
    print("   ✓ 记忆系统创建成功")

    # 测试保存记忆
    print("\n2. 测试保存记忆...")
    memory.save_memory(
        vlm_analysis="一只猫在沙发上睡觉",
        llm_commentary="这只猫看起来很嚣张",
        metadata={"category": "cat", "time": "morning"}
    )
    print("   ✓ 保存记忆1")

    memory.save_memory(
        vlm_analysis="猫从沙发上跳到地板",
        llm_commentary="哟，嚣张猫下地视察民情了？",
        metadata={"category": "cat", "time": "morning"}
    )
    print("   ✓ 保存记忆2")

    memory.save_memory(
        vlm_analysis="用户正在写代码",
        llm_commentary="代码写得不错，继续加油",
        metadata={"category": "work", "time": "afternoon"}
    )
    print("   ✓ 保存记忆3")

    # 测试检索记忆
    print("\n3. 测试检索记忆 (查询: '猫在地上')...")
    results = memory.retrieve_memory("猫在地上", top_k=2)
    print(f"   检索到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['document'][:50]}... (相似度: {1 - result['distance']:.2f})")

    # 测试检索工作相关记忆
    print("\n4. 测试检索记忆 (查询: '写代码')...")
    results = memory.retrieve_memory("写代码", top_k=2)
    print(f"   检索到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['document'][:50]}... (相似度: {1 - result['distance']:.2f})")

    # 测试格式化上下文
    print("\n5. 测试格式化上下文...")
    results = memory.retrieve_memory("猫", top_k=3)
    context = memory.format_memories_for_context(results, max_count=3)
    print("   格式化后的上下文:")
    print("   " + "\n   ".join(context.split("\n")))

    # 测试获取最近记忆
    print("\n6. 测试获取最近记忆...")
    recent_memories = memory.get_recent_memories(limit=3)
    print(f"   最近 {len(recent_memories)} 条记忆:")
    for i, mem in enumerate(recent_memories, 1):
        print(f"   {i}. {mem['document'][:50]}...")

    # 测试统计信息
    print("\n7. 记忆系统统计:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    # 提示
    print("\n提示:")
    print("  - 向量数据库已保存在: ./vector_memory_db")
    print("  - 运行主程序时,记忆系统会自动加载")
    print("  - 每次VLM分析和LLM吐槽都会保存到记忆库")
    print("  - 每次吐槽前会自动检索相关记忆")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
