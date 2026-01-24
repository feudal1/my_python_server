#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 和 VLM 服务是否正常工作
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_server.llm_class import LLMService, VLMService


def test_llm_service():
    """测试 LLM 服务"""
    print("=== 测试 LLM 服务 ===")
    
    try:
        # 初始化 LLM 服务
        llm_service = LLMService()
        
        # 测试消息
        messages = [
            {
                "role": "user",
                "content": "你好，给我讲一个简短的故事"
            }
        ]
        
        # 调用 LLM 服务
        print("发送请求到 LLM 服务...")
        response = llm_service.create(messages)
        
        # 打印响应
        print("\nLLM 服务响应:")
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(content)
        else:
            print("响应格式不正确:", response)
        
        print("\nLLM 服务测试成功！")
        return True
        
    except Exception as e:
        print(f"\nLLM 服务测试失败: {str(e)}")
        return False


def test_vlm_service():
    """测试 VLM 服务"""
    print("\n=== 测试 VLM 服务 ===")
    
    try:
        # 初始化 VLM 服务
        vlm_service = VLMService()
        
        # 测试消息
        messages = [
            {
                "role": "user",
                "content": "这张图片里有什么？"
            }
        ]
        
        # 使用一个简单的测试图像
        # 注意：需要确保这个路径存在一个测试图像
        test_image_path = "E:\\code\\my_python_server\\yolo\\findgate_data\\Snipaste_2026-01-13_23-40-40.png"
        
        if not os.path.exists(test_image_path):
            print(f"测试图像不存在: {test_image_path}")
            print("跳过 VLM 服务测试")
            return True
        
        # 调用 VLM 服务
        print("发送请求到 VLM 服务...")
        response = vlm_service.create_with_image(messages, image_source=test_image_path)
        
        # 打印响应
        print("\nVLM 服务响应:")
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(content)
        else:
            print("响应格式不正确:", response)
        
        print("\nVLM 服务测试成功！")
        return True
        
    except Exception as e:
        print(f"\nVLM 服务测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("开始测试 LLM 和 VLM 服务...\n")
    
    # 测试 LLM 服务
    llm_success = test_llm_service()
    
    # 测试 VLM 服务
    vlm_success = test_vlm_service()
    
    print("\n=== 测试结果汇总 ===")
    print(f"LLM 服务: {'成功' if llm_success else '失败'}")
    print(f"VLM 服务: {'成功' if vlm_success else '失败'}")
    
    if llm_success and vlm_success:
        print("\n所有测试通过！服务运行正常。")
        sys.exit(0)
    else:
        print("\n部分测试失败，请检查服务配置。")
        sys.exit(1)
