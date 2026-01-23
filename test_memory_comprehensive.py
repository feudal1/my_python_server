#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试记忆模块 - 全面验证存储和检索用户偏好相关的记忆
"""
import time
from mcp_cline.vector_memory import VectorMemory


def test_memory_comprehensive():
    """综合测试记忆模块的功能"""
    print("=" * 70)
    print("综合测试记忆模块 - 用户偏好测试")
    print("=" * 70)

    # 创建记忆系统实例
    memory = VectorMemory()

    # 1. 测试保存多条记忆
    print("\n1. 保存用户偏好记忆...")
    
    # 保存草莓偏好
    memory_id_1 = memory.save_memory(
        vlm_analysis="用户表示喜欢草莓",
        llm_commentary="用户喜欢草莓，这是一个水果偏好",
        metadata={
            "category": "preference",
            "topic": "food",
            "user_inputs": ["我喜欢草莓"]
        }
    )
    print(f"   草莓偏好保存成功，ID: {memory_id_1}")

    time.sleep(0.3)

    # 保存巧克力偏好
    memory_id_2 = memory.save_memory(
        vlm_analysis="用户表示喜欢巧克力",
        llm_commentary="用户喜欢巧克力，这是一个零食偏好",
        metadata={
            "category": "preference",
            "topic": "food",
            "user_inputs": ["我喜欢巧克力"]
        }
    )
    print(f"   巧克力偏好保存成功，ID: {memory_id_2}")

    time.sleep(0.3)

    # 保存编程偏好
    memory_id_3 = memory.save_memory(
        vlm_analysis="用户表示喜欢编程",
        llm_commentary="用户喜欢编程，这是一个兴趣爱好",
        metadata={
            "category": "preference",
            "topic": "hobby",
            "user_inputs": ["我喜欢编程"]
        }
    )
    print(f"   编程偏好保存成功，ID: {memory_id_3}")

    # 2. 测试检索记忆 - 查询"我喜欢什么"
    print("\n2. 语义检索测试: '我喜欢什么'")
    results = memory.retrieve_memory("我喜欢什么", top_k=5)
    print(f"   检索到 {len(results)} 条相关记忆:")
    
    for i, result in enumerate(results, 1):
        document = result['document']
        distance = result['distance']
        similarity = 1 - distance
        metadata = result.get('metadata', {})
        timestamp = metadata.get('datetime', '未知时间')
        memory_type = metadata.get('type', 'unknown')
        category = metadata.get('category', 'unknown')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      类型: {memory_type} | 分类: {category}")
        print(f"      内容: {document}")
        print(f"      相似度: {similarity:.2f}")
        print()

    # 3. 测试针对性检索 - 只检索食物偏好
    print("\n3. 针对性检索测试: '我喜欢的食物'")
    results_food = memory.retrieve_memory("我喜欢的食物", top_k=3)
    print(f"   检索到 {len(results_food)} 条相关食物偏好记忆:")
    
    for i, result in enumerate(results_food, 1):
        document = result['document']
        similarity = 1 - result['distance']
        metadata = result.get('metadata', {})
        timestamp = metadata.get('datetime', '未知时间')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      内容: {document}")
        print(f"      相似度: {similarity:.2f}")
        print()

    # 4. 测试获取最近记忆（完整格式）
    print("\n4. 获取最近记忆（完整格式）:")
    recent_memories = memory.get_recent_memories(limit=10)
    print(f"   最近 {len(recent_memories)} 条记忆:")
    
    for i, mem in enumerate(recent_memories, 1):
        if isinstance(mem, dict) and 'main' in mem:
            # 完整版格式
            main_doc = mem['main']['document']
            timestamp = mem['main']['metadata'].get('datetime', '未知时间')
            memory_type = mem['main']['metadata'].get('type', 'unknown')
            category = mem['main']['metadata'].get('category', 'unknown')
            topic = mem['main']['metadata'].get('topic', 'unknown')
            user_inputs = mem.get('user_inputs', [])
            user_input_texts = [ui['document'] for ui in user_inputs]
            
            print(f"   {i}. 时间: {timestamp}")
            print(f"      类型: {memory_type} | 分类: {category} | 主题: {topic}")
            print(f"      主记忆: {main_doc}")
            if user_input_texts:
                print(f"      用户输入: {user_input_texts}")
            print()

    # 5. 测试记忆系统统计信息
    print("\n5. 记忆系统统计:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 6. 测试记忆持久性 - 重新创建实例并检索
    print("\n6. 测试记忆持久性...")
    print("   重新创建记忆系统实例...")
    new_memory = VectorMemory()
    
    # 重新检索
    persistent_results = new_memory.retrieve_memory("我喜欢草莓", top_k=3)
    print(f"   重新检索到 {len(persistent_results)} 条关于草莓的记忆:")
    
    for i, result in enumerate(persistent_results, 1):
        document = result['document']
        similarity = 1 - result['distance']
        timestamp = result['metadata'].get('datetime', '未知时间')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      内容: {document}")
        print(f"      相似度: {similarity:.2f}")
        print()

    print("\n" + "=" * 70)
    print("综合测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_memory_comprehensive()
