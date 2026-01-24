#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI生成响应
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_server.llm_class import LLMService


def test_ai_response():
    """测试AI生成响应"""
    print("=== 测试AI生成响应 ===")
    
    try:
        # 初始化 LLM 服务
        llm_service = LLMService()
        
        # 测试消息
        messages = [
            {
                "role": "user",
                "content": "你好，请介绍一下你自己"
            }
        ]
        
        # 调用 LLM 服务
        print("发送请求到 AI 服务...")
        response = llm_service.create(messages)
        
        # 打印响应
        print("\nAI 响应:")
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(content)
        else:
            print("响应格式不正确:", response)
        
        print("\nAI 测试成功！")
        return True
        
    except Exception as e:
        print(f"\nAI 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    test_ai_response()
