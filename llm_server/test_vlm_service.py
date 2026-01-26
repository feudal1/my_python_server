#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 服务测试脚本
测试 VLMService 类的功能，包括单图像、多图像和多轮对话
"""

import sys
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_server.llm_class import VLMService


def create_test_image(image_path):
    """
    创建一个测试图像
    
    Args:
        image_path: 图像保存路径
    """
    # 创建一个 400x300 的图像
    img = Image.new('RGB', (400, 300), color='white')
    d = ImageDraw.Draw(img)
    
    # 绘制一些基本图形
    d.rectangle([50, 50, 150, 150], fill='blue')
    d.circle([250, 100], 50, fill='red')
    d.line([50, 200, 350, 200], fill='green', width=5)
    
    # 添加文本
    try:
        # 尝试使用默认字体
        font = ImageFont.load_default()
        d.text((50, 220), "Test Image", fill='black', font=font)
    except Exception:
        # 如果没有字体，跳过文本
        pass
    
    img.save(image_path)
    print(f"测试图像已创建: {image_path}")


def test_vlm_single_image():
    """测试 VLM 单图像识别"""
    print("=== 测试 VLM 单图像识别 ===")
    
    # 初始化 VLM 服务
    vlm_service = VLMService()
    
    # 创建测试图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image_path = tmp.name
    
    try:
        create_test_image(test_image_path)
        
        # 准备测试消息
        messages = [
            {
                "role": "user",
                "content": "描述一下这张图片的内容"
            }
        ]
        
        # 调用 VLM 服务
        try:
            response = vlm_service.create_with_image(messages, image_source=test_image_path)
            
            # 打印响应结果
            print("响应状态: 成功")
            
            # 提取并打印生成的内容
            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '无内容')
                print(f"生成内容: {content[:300]}..." if len(content) > 300 else f"生成内容: {content}")
            
            print("单图像识别测试完成\n")
            return True
        except Exception as e:
            print(f"单图像识别测试失败: {str(e)}")
            return False
    finally:
        # 清理测试图像
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


def test_vlm_multiple_images():
    """测试 VLM 多图像识别"""
    print("=== 测试 VLM 多图像识别 ===")
    
    # 初始化 VLM 服务
    vlm_service = VLMService()
    
    # 创建测试图像
    test_image_paths = []
    try:
        # 创建两张测试图像
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                test_image_path = tmp.name
                test_image_paths.append(test_image_path)
            
            # 创建不同内容的图像
            img = Image.new('RGB', (400, 300), color='white')
            d = ImageDraw.Draw(img)
            
            if i == 0:
                # 第一张图像：蓝色方块
                d.rectangle([50, 50, 350, 250], fill='blue')
                d.text((150, 130), "Image 1", fill='white')
            else:
                # 第二张图像：红色圆形
                d.ellipse([100, 50, 300, 250], fill='red')
                d.text((150, 130), "Image 2", fill='white')
            
            img.save(test_image_path)
            print(f"测试图像 {i+1} 已创建: {test_image_path}")
        
        # 准备测试消息
        messages = [
            {
                "role": "user",
                "content": "比较这两张图片的不同之处"
            }
        ]
        
        # 调用 VLM 服务
        try:
            response = vlm_service.create_with_multiple_images(messages, image_sources=test_image_paths)
            
            # 打印响应结果
            print("响应状态: 成功")
            
            # 提取并打印生成的内容
            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '无内容')
                print(f"生成内容: {content[:300]}..." if len(content) > 300 else f"生成内容: {content}")
            
            print("多图像识别测试完成\n")
            return True
        except Exception as e:
            print(f"多图像识别测试失败: {str(e)}")
            return False
    finally:
        # 清理测试图像
        for img_path in test_image_paths:
            if os.path.exists(img_path):
                os.unlink(img_path)


def test_vlm_multimodal_conversation():
    """测试 VLM 多轮对话"""
    print("=== 测试 VLM 多轮对话 ===")
    
    # 初始化 VLM 服务
    vlm_service = VLMService()
    
    # 创建测试图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image_path = tmp.name
    
    try:
        create_test_image(test_image_path)
        
        # 准备历史对话
        conversation_history = []
        
        # 第一轮对话：描述图像
        print("第一轮对话：描述图像")
        current_query = "描述一下这张图片的内容"
        
        try:
            # 调用 VLM 服务
            response = vlm_service.create_multimodal_conversation(
                conversation_history, 
                current_query, 
                image_source=test_image_path
            )
            
            # 提取响应内容
            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '无内容')
                print(f"AI 回复: {content[:200]}..." if len(content) > 200 else f"AI 回复: {content}")
                
                # 添加到历史对话
                conversation_history.append({'role': 'user', 'content': current_query})
                conversation_history.append({'role': 'assistant', 'content': content})
            
            # 第二轮对话：基于前一轮的问题
            print("\n第二轮对话：基于前一轮的问题")
            current_query = "图片中有哪些颜色？"
            
            response = vlm_service.create_multimodal_conversation(
                conversation_history, 
                current_query
            )
            
            # 提取响应内容
            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '无内容')
                print(f"AI 回复: {content[:200]}..." if len(content) > 200 else f"AI 回复: {content}")
            
            print("多轮对话测试完成\n")
            return True
        except Exception as e:
            print(f"多轮对话测试失败: {str(e)}")
            return False
    finally:
        # 清理测试图像
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


def test_vlm_direct_image_content():
    """测试 VLM 直接使用图像内容格式"""
    print("=== 测试 VLM 直接使用图像内容格式 ===")
    
    # 初始化 VLM 服务
    vlm_service = VLMService()
    
    # 创建测试图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image_path = tmp.name
    
    try:
        create_test_image(test_image_path)
        
        # 读取图像并转换为base64
        import base64
        from io import BytesIO
        
        with Image.open(test_image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        # 构建data URI
        image_data = f"data:image/png;base64,{img_str}"
        
        # 准备测试消息（直接包含图像内容）
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "描述一下这张图片的内容"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ]
            }
        ]
        
        # 调用 VLM 服务
        try:
            response = vlm_service.create_with_image(messages)
            
            # 打印响应结果
            print("响应状态: 成功")
            
            # 提取并打印生成的内容
            choices = response.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                content = message.get('content', '无内容')
                print(f"生成内容: {content[:300]}..." if len(content) > 300 else f"生成内容: {content}")
            
            print("直接图像内容格式测试完成\n")
            return True
        except Exception as e:
            print(f"直接图像内容格式测试失败: {str(e)}")
            return False
    finally:
        # 清理测试图像
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


def test_vlm_with_tools():
    """测试 VLM 带工具调用"""
    print("=== 测试 VLM 带工具调用 ===")
    
    # 初始化 VLM 服务
    vlm_service = VLMService()
    
    # 创建测试图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image_path = tmp.name
    
    try:
        create_test_image(test_image_path)
        
        # 准备测试消息
        messages = [
            {
                "role": "user",
                "content": "请分析这张图片，并告诉我图片中有哪些颜色"
            }
        ]
        
        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_colors",
                    "description": "分析图片中的颜色信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "colors": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "图片中识别到的颜色列表"
                            }
                        },
                        "required": ["colors"]
                    }
                }
            }
        ]
        
        # 调用 VLM 服务
        try:
            response = vlm_service.create_with_image(messages, image_source=test_image_path, tools=tools)
            
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
                print("提示: 如需启用工具调用，请在启动 VLM 服务器时添加以下参数:")
                print("  --enable-auto-tool-choice --tool-call-parser")
                print("工具调用测试已跳过（这是正常的，如果不需要工具调用功能）\n")
                return True
            else:
                print(f"带工具调用测试失败: {error_msg}")
                return False
    finally:
        # 清理测试图像
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)


if __name__ == "__main__":
    print("开始测试 VLM 服务...\n")
    
    # 运行所有测试
    test_results = {
        "单图像识别": test_vlm_single_image(),
        "多图像识别": test_vlm_multiple_images(),
        "多轮对话": test_vlm_multimodal_conversation(),
        "直接图像内容格式": test_vlm_direct_image_content(),
        "带工具调用": test_vlm_with_tools()
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
