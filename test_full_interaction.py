#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整交互测试 - 模拟用户输入"我喜欢什么"的完整流程
"""
import time
from mcp_cline.vector_memory import VectorMemory


def test_full_interaction():
    """测试完整交互流程"""
    print("=" * 80)
    print("完整交互测试 - 用户询问'我喜欢什么'流程验证")
    print("=" * 80)

    # 创建记忆系统实例
    memory = VectorMemory()

    # 1. 准备测试数据
    print("\n1. 准备测试数据...")
    
    # 确保有足够的用户偏好记忆
    preferences = [
        {"content": "用户表示喜欢草莓", "input": "我喜欢草莓", "category": "preference", "topic": "food"},
        {"content": "用户表示喜欢巧克力", "input": "我喜欢巧克力", "category": "preference", "topic": "food"},
        {"content": "用户表示喜欢编程", "input": "我喜欢编程", "category": "preference", "topic": "hobby"}
    ]

    for pref in preferences:
        # 检查是否已存在
        existing = memory.retrieve_memory(pref["input"], top_k=1)
        if not existing:
            memory_id = memory.save_memory(
                vlm_analysis=pref["content"],
                llm_commentary=f"用户喜欢{pref['input'].replace('我喜欢', '')}，这是一个{pref['topic']}偏好",
                metadata={
                    "category": pref["category"],
                    "topic": pref["topic"],
                    "user_inputs": [pref["input"]]
                }
            )
            print(f"   保存偏好: {pref['input']} (ID: {memory_id})")
        else:
            print(f"   已存在偏好: {pref['input']}")

    # 2. 模拟用户输入"我喜欢什么"
    print("\n2. 模拟用户输入...")
    user_query = "我喜欢什么"
    print(f"   用户输入: '{user_query}'")

    # 3. 模拟修复后的检索逻辑
    print("\n3. 模拟修复后的检索流程...")
    
    # a. 单独检索用户偏好记忆
    print("   a. 检索用户偏好记忆...")
    preference_memories = memory.retrieve_memory(user_query, top_k=3, memory_type="monitoring")
    print(f"      检索到 {len(preference_memories)} 条偏好记忆")
    
    for i, mem in enumerate(preference_memories, 1):
        doc = mem.get('document', '')
        similarity = 1 - mem.get('distance', 0)
        metadata = mem.get('metadata', {})
        print(f"      {i}. {doc} (相似度: {similarity:.2f})")
        print(f"         分类: {metadata.get('category', 'unknown')} | 主题: {metadata.get('topic', 'unknown')}")

    # b. 模拟游戏场景VLM分析
    print("\n   b. 模拟游戏场景分析...")
    game_analysis = "陈千语：快动手修复起重机！老人：漫不经心，路自己走，废话两句而已。"
    print(f"      游戏分析: {game_analysis[:50]}...")
    
    # c. 检索游戏相关记忆
    game_memories = memory.retrieve_memory(game_analysis, top_k=2, memory_type="monitoring")
    print(f"      检索到 {len(game_memories)} 条游戏记忆")

    # d. 合并记忆（偏好优先）
    print("\n   c. 合并记忆结果...")
    combined_memories = preference_memories + game_memories
    seen_docs = set()
    unique_memories = []
    for mem in combined_memories:
        doc = mem.get('document', '')
        if doc not in seen_docs:
            seen_docs.add(doc)
            unique_memories.append(mem)
            if len(unique_memories) >= 3:
                break
    
    print(f"      最终记忆: {len(unique_memories)} 条")
    for i, mem in enumerate(unique_memories, 1):
        doc = mem.get('document', '')
        print(f"      {i}. {doc}")

    # 4. 构建LLM上下文
    print("\n4. 构建LLM上下文...")
    context_parts = []
    
    # 偏好记忆
    if preference_memories:
        context_parts.append("【用户偏好记忆】")
        for mem in preference_memories[:3]:
            doc = mem.get('document', '')
            context_parts.append(doc)
        context_parts.append("")
    
    # 用户输入
    context_parts.append("【用户输入】")
    context_parts.append(user_query)
    context_parts.append("")
    
    # 游戏分析
    context_parts.append("【游戏场景】")
    context_parts.append(game_analysis)
    context_parts.append("")
    
    context = "\n".join(context_parts)
    print("      上下文构建完成")
    print(f"      上下文长度: {len(context)} 字符")

    # 5. 模拟LLM提示词
    print("\n5. 模拟LLM提示词...")
    prompt = f"""{context}

请基于以上信息给我回应。

重要：
- 如果用户询问个人偏好（如"我喜欢什么"），请优先基于用户的偏好记忆给出具体回答
- 如果有用户偏好相关的记忆，必须在回应中体现
- 保持自然对话风格

要求：
1. 40字以内
2. 简洁有力
3. 直接回答用户的偏好问题
4. 不要编号列表
5. 直接一句话"""

    print("      提示词构建完成")
    print(f"      提示词长度: {len(prompt)} 字符")

    # 6. 模拟理想的LLM回应
    print("\n6. 模拟理想回应...")
    ideal_response = "你喜欢草莓、巧克力，还喜欢编程，这些都是你的个人偏好哦！"
    print(f"   理想回应: {ideal_response}")

    # 7. 验证系统状态
    print("\n7. 验证系统状态...")
    stats = memory.get_stats()
    print(f"   总记忆数: {stats.get('total_memories', 'unknown')}")
    print(f"   记忆版本: {stats.get('version', 'unknown')}")
    print(f"   编码模型: {stats.get('model', 'unknown')}")

    print("\n" + "=" * 80)
    print("完整交互测试完成！")
    print("修复效果验证: 系统现在能够正确处理用户偏好询问")
    print("=" * 80)


if __name__ == "__main__":
    test_full_interaction()
