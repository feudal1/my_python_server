import os
from dotenv import load_dotenv
import json
import dashscope
from dashscope import MultiModalConversation
from urllib.parse import urlparse
import base64
from io import BytesIO
import tempfile
import urllib3
from datetime import datetime
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 禁用 SSL 警告（仅开发环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

debug = False


class LLMService:
    def __init__(self):
        dotenv_path = r'E:\code\my_python_server\my_python_server_private\.env'
        load_dotenv(dotenv_path)

        self.api_url = os.getenv('LLM_OPENAI_API_URL')
        self.model_name = os.getenv('LLM_MODEL_NAME')
        self.api_key = os.getenv('LLM_OPENAI_API_KEY')

        if not self.api_url:
            raise ValueError("环境变量 'LLM_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'LLM_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'LLM_OPENAI_API_KEY' 未设置或为空")

        # 配置 DashScope 的 HTTP 会话，添加重试和 SSL 优化
        self._configure_dashscope_session()

        print(f"LLM服务初始化完成，模型: {self.model_name}")

    def _configure_dashscope_session(self):
        """配置 DashScope SDK 的 HTTP 会话，优化连接稳定性"""
        try:
            # 创建带重试机制的 session
            session = requests.Session()

            # 配置重试策略
            retry_strategy = Retry(
                total=3,  # 总重试次数
                backoff_factor=1,  # 重试间隔因子（指数退避）
                status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
                allowed_methods=["POST", "GET"]  # 允许重试的 HTTP 方法
            )

            # 创建适配器并挂载到 session
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,  # 连接池大小
                pool_maxsize=10
            )

            # 将适配器挂载到所有协议
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # 配置 DashScope 使用这个 session
            # DashScope SDK 会使用全局的 HTTP 配置
            import dashscope
            dashscope.http_connection.HTTPConnection.api_key = self.api_key

            print("[LLM] HTTP会话已配置，启用重试机制")

        except Exception as e:
            print(f"[LLM警告] HTTP会话配置失败，使用默认配置: {e}")

    def create(self, messages, tools=None):
        """非流式调用：返回完整响应"""
        print("开始调用LLM服务（非流式）")

        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")

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
        import requests
        response = requests.post(
            f"https://{host}{path}",
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
        host, path = parsed.hostname, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")

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
        import requests
        collected_chunks = []
        # ================================================

        response = requests.post(
            f"https://{host}{path}",
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
        dotenv_path = r'E:\code\my_python_server\my_python_server_private\.env'
        load_dotenv(dotenv_path)

        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')

        if not self.model_name:
            raise ValueError("环境变量 'VLM_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'VLM_OPENAI_API_KEY' 未设置或为空")

        # 设置DashScope API密钥
        dashscope.api_key = self.api_key

        print(f"VLM服务初始化完成，模型: {self.model_name}")

    def create_with_image(self, messages, image_source=None, tools=None):
        """
        使用DashScope SDK发送带有图像的消息到VLM服务
        
        Args:
            messages: 消息列表，支持多轮对话
            image_source: 图像源（路径或URL），仅在第一轮对话中使用
            tools: 可用工具列表
            
        Returns:
            API响应结果
        """
        # 准备消息内容
        dashscope_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                dashscope_messages.append({
                    'role': role,
                    'content': [{'text': content}]
                })
            elif isinstance(content, list):
                # 复合消息（可能包含图像）
                msg_content = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            msg_content.append({'text': item["text"]})
                        elif "image" in item:
                            # DashScope格式的图像
                            if item["image"].startswith("file://"):
                                # 本地文件
                                image_path = item["image"][7:]  # 移除 "file://" 前缀
                                msg_content.append({'image': f"file://{image_path}"})
                            elif item["image"].startswith("data:"):
                                # Data URI，需要保存到临时文件
                                header, encoded = item["image"].split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                extension = {
                                    "image/jpeg": ".jpg",
                                    "image/png": ".png",
                                    "image/gif": ".gif",
                                    "image/webp": ".webp"
                                }.get(mime_type, ".jpg")
                                
                                # 解码并保存到临时文件
                                decoded = base64.b64decode(encoded)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                                    tmp_file.write(decoded)
                                    temp_path = tmp_file.name
                                
                                msg_content.append({'image': f"file://{temp_path}"})
                            else:
                                # 假设是URL
                                msg_content.append({'image': item["image"]})
                        elif "type" in item and item["type"] == "text":
                            msg_content.append({'text': item["text"]})
                        elif "type" in item and item["type"] == "image_url":
                            # 处理标准格式的图像URL
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:"):
                                # Data URI，需要保存到临时文件
                                header, encoded = image_url.split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                extension = {
                                    "image/jpeg": ".jpg",
                                    "image/png": ".png",
                                    "image/gif": ".gif",
                                    "image/webp": ".webp"
                                }.get(mime_type, ".jpg")
                                
                                # 解码并保存到临时文件
                                decoded = base64.b64decode(encoded)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                                    tmp_file.write(decoded)
                                    temp_path = tmp_file.name
                                
                                msg_content.append({'image': f"file://{temp_path}"})
                            else:
                                # URL或其他格式
                                msg_content.append({'image': image_url})
                
                dashscope_messages.append({
                    'role': role,
                    'content': msg_content
                })

        # 如果提供了图像源且是第一轮对话（没有图像消息），将其添加到第一个用户消息
        if image_source and not any('image' in item for msg in dashscope_messages for item in msg['content']):
            if dashscope_messages and dashscope_messages[0]['role'] == 'user':
                # 添加图像到现有内容
                dashscope_messages[0]['content'].append({'image': f"file://{os.path.abspath(image_source)}"})

        # 调用DashScope多模态对话服务
        

        try:
            response = MultiModalConversation.call(
                model=self.model_name,
                messages=dashscope_messages
            )
            
            # 检查响应是否成功
            if response.status_code == 200:
                result = {
                    'choices': [{
                        'message': {
                            'content': response.output.choices[0].message.content[0]["text"]
                        },
                        'finish_reason': 'stop'
                    }],
                    'model': self.model_name,
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    }
                }
                
                print(f"[VLM调试] 成功收到响应")
                return result
            else:
                raise Exception(f"VLM服务器错误: {response.code} - {response.message}")
        except Exception as e:
            print(f"[VLM错误] 调用失败: {str(e)}")
            raise

    def create_with_multiple_images(self, messages, image_sources=None):
        """
        使用DashScope SDK发送多张图像到VLM服务
        
        Args:
            messages: 消息列表
            image_sources: 图像源列表（路径或URL）
            
        Returns:
            API响应结果
        """
        if not image_sources:
            return self.create_with_image(messages)
        
        # 准备消息内容
        dashscope_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                dashscope_messages.append({
                    'role': role,
                    'content': [{'text': content}]
                })
            elif isinstance(content, list):
                dashscope_messages.append({
                    'role': role,
                    'content': content
                })

        # 如果有第一个用户消息，添加所有图像
        if dashscope_messages and dashscope_messages[0]['role'] == 'user':
            # 确保content是列表格式
            if not isinstance(dashscope_messages[0]['content'], list):
                dashscope_messages[0]['content'] = []

            # 添加所有图像
            for image_source in image_sources:
                dashscope_messages[0]['content'].append({'image': f"file://{os.path.abspath(image_source)}"})

        # 添加重试机制处理SSL错误
        max_retries = 3
        retry_delay = 2  # 秒
        import time

        for attempt in range(max_retries):
            try:
                # 调用DashScope多模态对话服务
                response = MultiModalConversation.call(
                    model=self.model_name,
                    messages=dashscope_messages
                )

                # 检查响应是否成功
                if response.status_code == 200:
                    result = {
                        'choices': [{
                            'message': {
                                'content': response.output.choices[0].message.content[0]["text"]
                            },
                            'finish_reason': 'stop'
                        }],
                        'model': self.model_name,
                        'usage': {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0
                        }
                    }

                    print(f"[VLM调试] 成功收到多图响应")
                    return result
                else:
                    raise Exception(f"VLM服务器错误: {response.code} - {response.message}")

            except Exception as e:
                error_str = str(e)
                # 检查是否是SSL相关错误
                if 'SSLError' in error_str or 'EOF occurred in violation of protocol' in error_str:
                    if attempt < max_retries - 1:
                        print(f"[VLM警告] SSL错误，第{attempt + 1}次重试... ({error_str})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[VLM错误] 多图调用失败（已重试{max_retries}次）: {error_str}")
                        raise
                else:
                    # 非SSL错误，直接抛出
                    print(f"[VLM错误] 多图调用失败: {error_str}")
                    raise

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