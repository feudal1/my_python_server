#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试记忆模块 - 验证存储和检索用户偏好相关的记忆
"""
import time
from mcp_cline.vector_memory import VectorMemory


def test_memory_module():
    """测试记忆模块的功能"""
    print("=" * 60)
    print("测试记忆模块 - 草莓偏好测试")
    print("=" * 60)

    # 创建记忆系统实例
    memory = VectorMemory()

    # 1. 测试保存关于草莓偏好的记忆
    print("\n1. 保存记忆: '我喜欢草莓'")
    memory_id = memory.save_memory(
        vlm_analysis="用户表示喜欢草莓",
        llm_commentary="用户喜欢草莓，这是一个水果偏好",
        metadata={
            "category": "preference",
            "topic": "food",
            "user_inputs": ["我喜欢草莓"]
        }
    )
    print(f"   记忆保存成功，ID: {memory_id}")

    # 等待一小段时间，确保时间戳不同
    time.sleep(0.5)

    # 2. 测试检索记忆 - 查询"我喜欢什么"
    print("\n2. 检索记忆: '我喜欢什么'")
    results = memory.retrieve_memory("我喜欢什么", top_k=3)
    print(f"   检索到 {len(results)} 条相关记忆:")
    
    for i, result in enumerate(results, 1):
        document = result['document']
        distance = result['distance']
        similarity = 1 - distance
        metadata = result.get('metadata', {})
        timestamp = metadata.get('datetime', '未知时间')
        memory_type = metadata.get('type', 'unknown')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      类型: {memory_type}")
        print(f"      内容: {document}")
        print(f"      相似度: {similarity:.2f}")
        print(f"      元数据: {metadata}")
        print()

    # 3. 测试语义检索 - 专门检索主记忆
    print("\n3. 测试语义检索 (只检索主记忆): '我喜欢什么'")
    results_main = memory.retrieve_memory("我喜欢什么", top_k=3, memory_type="monitoring")
    print(f"   检索到 {len(results_main)} 条相关主记忆:")
    
    for i, result in enumerate(results_main, 1):
        document = result['document']
        distance = result['distance']
        similarity = 1 - distance
        metadata = result.get('metadata', {})
        timestamp = metadata.get('datetime', '未知时间')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      内容: {document}")
        print(f"      相似度: {similarity:.2f}")
        print(f"      元数据: {metadata}")
        print()

    # 3. 测试获取最近记忆
    print("\n3. 获取最近记忆:")
    recent_memories = memory.get_recent_memories(limit=5)
    print(f"   最近 {len(recent_memories)} 条记忆:")
    
    for i, mem in enumerate(recent_memories, 1):
        if isinstance(mem, dict) and 'main' in mem:
            # 完整版格式
            main_doc = mem['main']['document']
            timestamp = mem['main']['metadata'].get('datetime', '未知时间')
            user_inputs = mem.get('user_inputs', [])
            user_input_texts = [ui['document'] for ui in user_inputs]
            
            print(f"   {i}. 时间: {timestamp}")
            print(f"      主记忆: {main_doc}")
            if user_input_texts:
                print(f"      用户输入: {user_input_texts}")
        else:
            # 简化版格式
            print(f"   {i}. {mem}")
        print()

    # 4. 测试格式化上下文
    print("\n4. 测试格式化上下文:")
    context_results = memory.retrieve_memory("草莓", top_k=2)
    context = memory.format_memories_for_context(context_results)
    print("   格式化后的上下文:")
    print(context)

    # 5. 显示记忆系统统计信息
    print("\n5. 记忆系统统计:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_module()
