#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM 和 VLM 服务的连接状态
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_server.llm_class import LLMService, VLMService


def test_llm_connection():
    """测试 LLM 服务连接"""
    print("=== 测试 LLM 服务连接 ===")
    
    try:
        # 初始化 LLM 服务
        llm_service = LLMService()
        
        # 测试消息
        messages = [
            {
                "role": "user",
                "content": "你好"
            }
        ]
        
        # 调用 LLM 服务
        print("发送请求到 LLM 服务...")
        response = llm_service.create(messages)
        
        # 检查响应
        if 'choices' in response and len(response['choices']) > 0:
            print("✓ LLM 服务连接成功！")
            return True
        else:
            print("✗ LLM 服务响应格式不正确")
            return False
            
    except Exception as e:
        print(f"✗ LLM 服务连接失败: {str(e)}")
        return False


def test_vlm_connection():
    """测试 VLM 服务连接"""
    print("\n=== 测试 VLM 服务连接 ===")
    
    try:
        # 初始化 VLM 服务
        vlm_service = VLMService()
        
        # 测试消息（仅文本，不包含图像）
        messages = [
            {
                "role": "user",
                "content": "你好"
            }
        ]
        
        # 调用 VLM 服务
        print("发送请求到 VLM 服务...")
        response = vlm_service.create_with_image(messages)
        
        # 检查响应
        if 'choices' in response and len(response['choices']) > 0:
            print("✓ VLM 服务连接成功！")
            return True
        else:
            print("✗ VLM 服务响应格式不正确")
            return False
            
    except Exception as e:
        print(f"✗ VLM 服务连接失败: {str(e)}")
        print("注意：VLM 服务可能需要正确的多模态模型配置")
        return False


if __name__ == "__main__":
    print("开始测试 LLM 和 VLM 服务连接...\n")
    
    # 测试 LLM 服务
    llm_success = test_llm_connection()
    
    # 测试 VLM 服务
    vlm_success = test_vlm_connection()
    
    print("\n=== 测试结果汇总 ===")
    print(f"LLM 服务: {'成功' if llm_success else '失败'}")
    print(f"VLM 服务: {'成功' if vlm_success else '失败'}")
    
    if llm_success:
        print("\n🎉 LLM 服务正常运行！")
    
    if vlm_success:
        print("🎉 VLM 服务正常运行！")
    else:
        print("\n⚠️ VLM 服务需要正确配置多模态模型")
        print("请检查 WSL 中的 VLM 服务是否使用了正确的 MiniCPM-V 模型")
    
    if llm_success or vlm_success:
        print("\n✅ 至少有一个服务正常运行，可以开始使用！")
        sys.exit(0)
    else:
        print("\n❌ 所有服务测试失败，请检查配置")
        sys.exit(1)
