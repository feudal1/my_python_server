#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集合恢复功能 - 验证系统能够在集合不存在的情况下自动恢复
"""
import time
import shutil
import os
from mcp_cline.vector_memory import VectorMemory


def test_collection_recovery():
    """测试集合恢复功能"""
    print("=" * 70)
    print("测试集合恢复功能 - 自动重新初始化验证")
    print("=" * 70)

    # 1. 模拟集合损坏场景
    print("\n1. 模拟集合损坏场景...")
    
    # 获取向量数据库路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vector_db_path = os.path.join(base_dir, "vector_memory_db")
    
    print(f"   向量数据库路径: {vector_db_path}")
    
    # 检查数据库目录是否存在
    if os.path.exists(vector_db_path):
        print("   发现现有数据库目录，将模拟损坏场景")
        
        # 创建临时备份
        backup_path = f"{vector_db_path}_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(vector_db_path, backup_path)
        print(f"   已创建数据库备份: {backup_path}")
        
        # 删除数据库目录（模拟损坏）
        shutil.rmtree(vector_db_path)
        print("   已删除数据库目录，模拟集合不存在场景")
    else:
        print("   未发现数据库目录，直接测试新环境")

    # 2. 测试系统初始化
    print("\n2. 测试系统初始化...")
    try:
        memory = VectorMemory()
        print("   记忆系统初始化成功")
        
        # 检查集合状态
        print("   系统状态:")
        stats = memory.get_stats()
        for key, value in stats.items():
            print(f"      {key}: {value}")
    except Exception as e:
        print(f"   系统初始化失败: {e}")
        return

    # 3. 测试记忆保存
    print("\n3. 测试记忆保存...")
    try:
        memory_id = memory.save_memory(
            vlm_analysis="用户表示喜欢草莓",
            llm_commentary="用户喜欢草莓，这是一个水果偏好",
            metadata={
                "category": "preference",
                "topic": "food",
                "user_inputs": ["我喜欢草莓"]
            }
        )
        if memory_id:
            print(f"   记忆保存成功，ID: {memory_id}")
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

    # 5. 测试系统恢复
    print("\n5. 测试系统恢复...")
    try:
        # 重新创建实例
        new_memory = VectorMemory()
        print("   重新创建记忆系统实例成功")
        
        # 再次检索
        results = new_memory.retrieve_memory("我喜欢草莓", top_k=3)
        print(f"   重新检索到 {len(results)} 条草莓相关记忆")
        
        if results:
            print("   系统恢复验证成功")
        else:
            print("   系统恢复验证失败")
    except Exception as e:
        print(f"   系统恢复测试失败: {e}")

    # 6. 清理测试环境
    print("\n6. 清理测试环境...")
    backup_path = f"{vector_db_path}_backup"
    if os.path.exists(backup_path):
        # 恢复备份
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
        shutil.copytree(backup_path, vector_db_path)
        shutil.rmtree(backup_path)
        print("   已恢复数据库备份")
    else:
        print("   无备份可恢复")

    print("\n" + "=" * 70)
    print("集合恢复功能测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_collection_recovery()
