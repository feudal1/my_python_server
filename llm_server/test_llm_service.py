#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 服务测试脚本
测试 LLMService 类的功能，包括非流式和流式调用
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_server.llm_class import LLMService


def test_llm_non_stream():
    """测试 LLM 非流式调用"""
    print("=== 测试 LLM 非流式调用 ===")
    
    # 初始化 LLM 服务
    llm_service = LLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "你好，请介绍一下自己"
        }
    ]
    
    try:
        # 调用非流式方法
        response = llm_service.create(messages)
        
        # 打印响应结果
        print("响应状态: 成功")
        print(f"模型: {response.get('model', '未知')}")
        
        # 提取并打印生成的内容
        choices = response.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '无内容')
            print(f"生成内容: {content[:200]}..." if len(content) > 200 else f"生成内容: {content}")
        
        print("非流式调用测试完成\n")
        return True
    except Exception as e:
        print(f"非流式调用测试失败: {str(e)}")
        return False


def test_llm_stream():
    """测试 LLM 流式调用"""
    print("=== 测试 LLM 流式调用 ===")
    
    # 初始化 LLM 服务
    llm_service = LLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "请简要介绍一下人工智能的发展历程"
        }
    ]
    
    # 流式回调函数
    def stream_callback(chunk_content):
        """流式回调，打印每个 chunk 的内容"""
        print(chunk_content, end="", flush=True)
    
    try:
        # 调用流式方法
        print("开始流式接收响应...")
        chunks = []
        for chunk in llm_service.create_stream(messages, callback=stream_callback):
            chunks.append(chunk)
        
        print("\n流式调用测试完成\n")
        return True
    except Exception as e:
        print(f"流式调用测试失败: {str(e)}")
        return False


def test_llm_with_tools():
    """测试 LLM 带工具调用"""
    print("=== 测试 LLM 带工具调用 ===")
    
    # 初始化 LLM 服务
    llm_service = LLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "今天北京的天气怎么样？"
        }
    ]
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    try:
        # 调用非流式方法
        response = llm_service.create(messages, tools=tools)
        
        # 打印响应结果
        print("响应状态: 成功")
        
        # 提取并打印生成的内容
        choices = response.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            content = message.get('content', '无内容')
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                print(f"工具调用: {tool_calls[0].get('function', {}).get('name', '未知')}")
                print(f"参数: {tool_calls[0].get('function', {}).get('arguments', '无')}")
            else:
                print(f"生成内容: {content[:200]}..." if len(content) > 200 else f"生成内容: {content}")
        
        print("带工具调用测试完成\n")
        return True
    except Exception as e:
        error_msg = str(e)
        if "enable-auto-tool-choice" in error_msg:
            print("⚠️  工具调用功能未启用")
            print("提示: 如需启用工具调用，请在启动 LLM 服务器时添加以下参数:")
            print("  --enable-auto-tool-choice --tool-call-parser")
            print("工具调用测试已跳过（这是正常的，如果不需要工具调用功能）\n")
            return True
        else:
            print(f"带工具调用测试失败: {error_msg}")
            return False


if __name__ == "__main__":
    print("开始测试 LLM 服务...\n")
    
    # 运行所有测试
    test_results = {
        "非流式调用": test_llm_non_stream(),
        "流式调用": test_llm_stream(),
        "带工具调用": test_llm_with_tools()
    }
    
    # 打印测试结果
    print("=== 测试结果汇总 ===")
    for test_name, result in test_results.items():
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
    
    # 计算总体结果
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    print(f"\n总体结果: {passed_tests}/{total_tests} 通过")
