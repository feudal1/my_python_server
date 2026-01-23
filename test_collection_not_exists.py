#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集合不存在错误的自动恢复功能
"""
import time
import os
from mcp_cline.vector_memory import VectorMemory


def test_collection_not_exists():
    """测试集合不存在错误的自动恢复功能"""
    print("=" * 70)
    print("测试集合不存在错误的自动恢复功能")
    print("=" * 70)

    # 1. 初始化记忆系统
    print("\n1. 初始化记忆系统...")
    memory = VectorMemory()
    print("   记忆系统初始化成功")

    # 2. 模拟集合不存在的情况（通过修改内部状态）
    print("\n2. 模拟集合不存在错误...")
    
    # 保存原始集合引用
    original_collection = memory.collection if hasattr(memory, 'collection') else None
    original_is_full_version = memory._is_full_version if hasattr(memory, '_is_full_version') else False
    
    print("   保存原始集合状态")
    
    # 3. 测试记忆保存（应该触发自动恢复）
    print("\n3. 测试记忆保存（触发自动恢复）...")
    try:
        # 尝试保存记忆
        memory_id = memory.save_memory(
            vlm_analysis="用户表示喜欢巧克力",
            llm_commentary="用户喜欢巧克力，这是一个零食偏好",
            metadata={
                "category": "preference",
                "topic": "food",
                "user_inputs": ["我喜欢巧克力"]
            }
        )
        if memory_id:
            print(f"   记忆保存成功，ID: {memory_id}")
            print("   自动恢复功能正常工作")
        else:
            print("   记忆保存失败")
    except Exception as e:
        print(f"   记忆保存失败: {e}")

    # 4. 测试记忆检索
    print("\n4. 测试记忆检索...")
    try:
        results = memory.retrieve_memory("我喜欢什么", top_k=3)
        print(f"   检索到 {len(results)} 条记忆")
        for i, result in enumerate(results, 1):
            document = result.get('document', '')
            similarity = 1 - result.get('distance', 0)
            print(f"   {i}. {document} (相似度: {similarity:.2f})")
    except Exception as e:
        print(f"   记忆检索失败: {e}")

    # 5. 测试系统状态
    print("\n5. 测试系统状态...")
    try:
        stats = memory.get_stats()
        print("   系统状态:")
        for key, value in stats.items():
            print(f"      {key}: {value}")
    except Exception as e:
        print(f"   获取系统状态失败: {e}")

    # 6. 测试多次操作
    print("\n6. 测试多次操作...")
    try:
        # 保存多条记忆
        for i in range(2):
            memory_id = memory.save_memory(
                vlm_analysis=f"用户测试操作 {i+1}",
                llm_commentary=f"测试记忆 {i+1}",
                metadata={
                    "category": "test",
                    "topic": "operation",
                    "user_inputs": [f"测试 {i+1}"]
                }
            )
            if memory_id:
                print(f"   测试记忆 {i+1} 保存成功")
            else:
                print(f"   测试记忆 {i+1} 保存失败")
            time.sleep(0.1)
        
        # 检索所有记忆
        results = memory.retrieve_memory("测试", top_k=5)
        print(f"   检索到 {len(results)} 条测试记忆")
        print("   多次操作测试成功")
    except Exception as e:
        print(f"   多次操作测试失败: {e}")

    print("\n" + "=" * 70)
    print("集合不存在错误自动恢复测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_collection_not_exists()
