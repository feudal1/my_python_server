import os
import json
from urllib.parse import urlparse
import base64
from io import BytesIO
import tempfile
import urllib3
from datetime import datetime
from PIL import Image
import requests

# 禁用 SSL 警告（仅开发环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

debug = False


class LLMService:
    def __init__(self):
        # 使用 WSL 本地服务
        self.api_url = 'http://localhost:8000/v1'
        self.model_name = '/root/models/Qwen_Qwen2.5-0.5B-Instruct'
        self.api_key = 'sk-xxx'  # 本地服务不需要真实 API 密钥

        # 配置 HTTP 会话，添加重试和 SSL 优化
        self._configure_http_session()

        print(f"LLM服务初始化完成，模型: {self.model_name}")

    def _configure_http_session(self):
        """配置 HTTP 会话，优化连接稳定性"""
        print("[LLM] 使用 requests 直接调用 API")

    def create(self, messages, tools=None):
        """非流式调用：返回完整响应"""
        print("开始调用LLM服务（非流式）")

        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, port, path = parsed.hostname, parsed.port, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")
        if not port:
            port = 80  # 默认端口

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.9
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # ===== 立即保存发送给LLM的messages到文件 =====
        os.makedirs('output', exist_ok=True)
        request_file_path = os.path.join('output', 'llm_request.json')
        with open(request_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'request': {
                    'model': self.model_name,
                    'messages': messages,
                    'temperature': 0.9
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"LLM请求已保存到: {request_file_path}")
        # ================================================

        print("正在调用LLM服务...")
        response = requests.post(
            f"http://{host}:{port}{path}",
            json=request_body,
            headers=headers,
            verify=False,
            timeout=30  # 设置30秒超时
        )

        print(f"LLM服务响应状态码: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"LLM服务器错误: {response.status_code} - {response.text}")

        data = response.json()

        # ===== 保存LLM响应到文件 =====
        output_file_path = os.path.join('output', 'llm_response.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'response': data
            }, f, indent=2, ensure_ascii=False)
        print(f"LLM响应已保存到: {output_file_path}")
        # ================================================

        print("LLM服务调用完成")
        return data

    def create_stream(self, messages, tools=None, callback=None):
        """
        流式调用：生成器，每次 yield 一个解析后的 chunk 字典
        兼容旧版 callback（可选）
        """
        print("开始调用LLM服务（流式）")

        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, port, path = parsed.hostname, parsed.port, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")
        if not port:
            port = 80  # 默认端口

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.9,
            "stream": True
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }

        # ===== 立即保存发送给LLM的流式messages到文件 =====
        os.makedirs('output', exist_ok=True)
        request_file_path = os.path.join('output', 'llm_stream_request.json')
        with open(request_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'request': {
                    'model': self.model_name,
                    'messages': messages,
                    'temperature': 0.9,
                    'stream': True
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"LLM流式请求已保存到: {request_file_path}")
        # ================================================

        # ===== 保存流式响应 =====
        # 流式响应收集完成后保存
        collected_chunks = []
        # ================================================

        response = requests.post(
            f"http://{host}:{port}{path}",
            json=request_body,
            headers=headers,
            stream=True,
            verify=False
        )

        print(f"LLM服务流式响应状态码: {response.status_code}")

        if response.status_code != 200:
            raise Exception(f"LLM服务器错误: {response.status_code} - {response.text}")

        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)

                    if "error" in chunk:
                        raise Exception(f"LLM服务器错误: {chunk['error']}")

                    content = ""
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")

                    # 收集所有chunks用于保存
                    collected_chunks.append(chunk)

                    if callback and isinstance(content, str) and content:
                        callback(content)

                    yield chunk

                except json.JSONDecodeError:
                    continue

        # 保存完整的流式响应
        if collected_chunks:
            os.makedirs('output', exist_ok=True)
            output_file_path = os.path.join('output', 'llm_stream_response.json')
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'response': {
                        'chunks': collected_chunks
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"LLM流式响应已保存到: {output_file_path}")

        print("LLM服务流式调用完成")


class VLMService:
    def __init__(self):
        # 使用 WSL 本地服务
        self.api_url = 'http://localhost:8001/v1'
        self.model_name = '/root/models/OpenBMB_MiniCPM-V-2_6-int4'
        self.api_key = 'sk-xxx'  # 本地服务不需要真实 API 密钥

        print(f"VLM服务初始化完成，模型: {self.model_name}")

    def create_with_image(self, messages, image_source=None, tools=None):
        """
        使用OpenAI兼容API发送带有图像的消息到VLM服务
        
        Args:
            messages: 消息列表，支持多轮对话
            image_source: 图像源（路径或URL），仅在第一轮对话中使用
            tools: 可用工具列表
            
        Returns:
            API响应结果
        """
        # 准备消息内容
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                openai_messages.append({
                    'role': role,
                    'content': content
                })
            elif isinstance(content, list):
                # 复合消息（可能包含图像）
                openai_messages.append({
                    'role': role,
                    'content': content
                })

        # 如果提供了图像源且是第一轮对话（没有图像消息），将其添加到第一个用户消息
        if image_source and not any(isinstance(msg['content'], list) and any('image_url' in item for item in msg['content']) for msg in openai_messages):
            if openai_messages and openai_messages[0]['role'] == 'user':
                # 读取图像并转换为base64
                with Image.open(image_source) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                # 构建data URI
                image_data = f"data:image/png;base64,{img_str}"
                
                # 更新消息内容
                if isinstance(openai_messages[0]['content'], str):
                    # 将纯文本转换为复合内容
                    openai_messages[0]['content'] = [
                        {'type': 'text', 'text': openai_messages[0]['content']},
                        {'type': 'image_url', 'image_url': {'url': image_data}}
                    ]
                else:
                    # 添加图像到现有复合内容
                    openai_messages[0]['content'].append({
                        'type': 'image_url', 'image_url': {'url': image_data}
                    })

        # 构建请求体
        request_body = {
            "model": self.model_name,
            "messages": openai_messages,
            "tools": tools,
            "temperature": 0.9
        }

        # 构建请求URL
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, port, path = parsed.hostname, parsed.port, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")
        if not port:
            port = 80  # 默认端口

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 发送请求
        try:
            response = requests.post(
                f"http://{host}:{port}{path}",
                json=request_body,
                headers=headers,
                verify=False,
                timeout=60  # 设置60秒超时
            )
            
            # 检查响应是否成功
            if response.status_code == 200:
                result = response.json()
                print(f"[VLM调试] 成功收到响应")
                return result
            else:
                raise Exception(f"VLM服务器错误: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[VLM错误] 调用失败: {str(e)}")
            raise

    def create_with_multiple_images(self, messages, image_sources=None):
        """
        使用OpenAI兼容API发送多张图像到VLM服务
        
        Args:
            messages: 消息列表
            image_sources: 图像源列表（路径或URL）
            
        Returns:
            API响应结果
        """
        if not image_sources:
            return self.create_with_image(messages)
        
        # 准备消息内容
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                openai_messages.append({
                    'role': role,
                    'content': content
                })
            elif isinstance(content, list):
                openai_messages.append({
                    'role': role,
                    'content': content
                })

        # 如果有第一个用户消息，添加所有图像
        if openai_messages and openai_messages[0]['role'] == 'user':
            # 确保content是列表格式
            if isinstance(openai_messages[0]['content'], str):
                # 将纯文本转换为复合内容
                base_content = [{'type': 'text', 'text': openai_messages[0]['content']}]
            else:
                # 保留现有复合内容
                base_content = openai_messages[0]['content']
            
            # 添加所有图像
            for image_source in image_sources:
                # 读取图像并转换为base64
                with Image.open(image_source) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                # 构建data URI
                image_data = f"data:image/png;base64,{img_str}"
                
                # 添加图像到内容
                base_content.append({
                    'type': 'image_url', 'image_url': {'url': image_data}
                })
            
            # 更新消息内容
            openai_messages[0]['content'] = base_content

        # 调用create_with_image方法
        return self.create_with_image(openai_messages)

    def create_multimodal_conversation(self, conversation_history, current_query, image_source=None):
        """
        支持多轮对话的多模态对话
        
        Args:
            conversation_history: 历史对话列表
            current_query: 当前查询
            image_source: 图像源（仅在第一轮对话中使用）
            
        Returns:
            API响应结果
        """
        # 构建完整的消息列表
        messages = []
        
        # 添加历史对话
        for conv in conversation_history:
            messages.append({
                'role': conv['role'],
                'content': conv['content']
            })
        
        # 添加当前查询
        messages.append({
            'role': 'user',
            'content': current_query
        })
        
        # 调用create_with_image方法
        return self.create_with_image(messages, image_source=image_source)
