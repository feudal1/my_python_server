#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复效果 - 验证用户询问"我喜欢什么"时系统能够正确回应
"""
import time
from mcp_cline.vector_memory import VectorMemory


def test_fix_verification():
    """测试修复效果"""
    print("=" * 70)
    print("测试修复效果 - 用户偏好询问验证")
    print("=" * 70)

    # 创建记忆系统实例
    memory = VectorMemory()

    # 1. 确保系统中有用户偏好记忆
    print("\n1. 验证用户偏好记忆...")
    
    # 先检索是否已有草莓偏好记忆
    existing_strawberry = memory.retrieve_memory("我喜欢草莓", top_k=2)
    if existing_strawberry:
        print(f"   已存在 {len(existing_strawberry)} 条草莓偏好记忆")
    else:
        # 保存草莓偏好记忆
        memory_id = memory.save_memory(
            vlm_analysis="用户表示喜欢草莓",
            llm_commentary="用户喜欢草莓，这是一个水果偏好",
            metadata={
                "category": "preference",
                "topic": "food",
                "user_inputs": ["我喜欢草莓"]
            }
        )
        print(f"   已保存草莓偏好记忆，ID: {memory_id}")

    # 保存巧克力偏好记忆
    chocolate_id = memory.save_memory(
        vlm_analysis="用户表示喜欢巧克力",
        llm_commentary="用户喜欢巧克力，这是一个零食偏好",
        metadata={
            "category": "preference",
            "topic": "food",
            "user_inputs": ["我喜欢巧克力"]
        }
    )
    print(f"   已保存巧克力偏好记忆，ID: {chocolate_id}")

    # 保存编程偏好记忆
    coding_id = memory.save_memory(
        vlm_analysis="用户表示喜欢编程",
        llm_commentary="用户喜欢编程，这是一个兴趣爱好",
        metadata={
            "category": "preference",
            "topic": "hobby",
            "user_inputs": ["我喜欢编程"]
        }
    )
    print(f"   已保存编程偏好记忆，ID: {coding_id}")

    time.sleep(1)

    # 2. 测试修复后的检索逻辑
    print("\n2. 测试修复后的检索逻辑...")
    
    # 模拟用户输入"我喜欢什么"
    user_query = "我喜欢什么"
    print(f"   用户查询: '{user_query}'")
    
    # 单独使用用户输入检索偏好记忆
    preference_results = memory.retrieve_memory(user_query, top_k=3, memory_type="monitoring")
    print(f"   单独检索偏好记忆: {len(preference_results)} 条")
    
    for i, result in enumerate(preference_results, 1):
        document = result['document']
        similarity = 1 - result['distance']
        metadata = result['metadata']
        timestamp = metadata.get('datetime', '未知时间')
        category = metadata.get('category', 'unknown')
        topic = metadata.get('topic', 'unknown')
        
        print(f"   {i}. 时间: {timestamp}")
        print(f"      内容: {document}")
        print(f"      分类: {category} | 主题: {topic}")
        print(f"      相似度: {similarity:.2f}")
        print()

    # 3. 模拟游戏场景VLM分析
    print("\n3. 模拟游戏场景分析...")
    game_analysis = "陈千语：快动手修复起重机！老人：漫不经心，路自己走，废话两句而已。"
    
    # 检索游戏相关记忆
    game_results = memory.retrieve_memory(game_analysis, top_k=2, memory_type="monitoring")
    print(f"   游戏记忆检索: {len(game_results)} 条")
    
    for i, result in enumerate(game_results, 1):
        document = result['document'][:50]
        similarity = 1 - result['distance']
        print(f"   {i}. 内容: {document}...")
        print(f"      相似度: {similarity:.2f}")
        print()

    # 4. 验证记忆合并逻辑
    print("\n4. 验证记忆合并效果...")
    
    # 模拟合并结果（偏好记忆优先）
    all_results = preference_results + game_results
    seen_docs = set()
    unique_results = []
    for result in all_results:
        doc = result.get('document', '')
        if doc not in seen_docs:
            seen_docs.add(doc)
            unique_results.append(result)
            if len(unique_results) >= 3:
                break
    
    print(f"   合并后记忆: {len(unique_results)} 条")
    print("   记忆顺序（偏好优先）:")
    for i, result in enumerate(unique_results, 1):
        document = result.get('document', '')
        print(f"   {i}. {document}")

    # 5. 测试持久性
    print("\n5. 测试记忆持久性...")
    new_memory = VectorMemory()
    persistent_results = new_memory.retrieve_memory("我喜欢什么", top_k=3)
    print(f"   重新检索到 {len(persistent_results)} 条偏好记忆")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("修复验证结果: 系统现在应该能够正确检索用户偏好记忆")
    print("=" * 70)


if __name__ == "__main__":
    test_fix_verification()
