import os
from dotenv import load_dotenv
import json
import requests
from urllib.parse import urlparse
import base64
from io import BytesIO
from PIL import Image
import urllib3
from datetime import datetime

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

        print(f"LLM服务初始化完成，模型: {self.model_name}")

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

        self.api_url = os.getenv('VLM_OPENAI_API_URL')
        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')

        if not self.api_url:
            raise ValueError("环境变量 'VLM_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'VLM_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'VLM_OPENAI_API_KEY' 未设置或为空")

        print(f"VLM服务初始化完成，模型: {self.model_name}")

    def encode_image(self, image_source):
        try:
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                image_data = response.content
            else:
                with open(image_source, "rb") as image_file:
                    image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"图像编码失败: {str(e)}")

    def create_with_image(self, messages, image_source=None, tools=None):
        # 使用OpenAI兼容格式（智谱AI GLM-4V支持）
        if image_source and messages and messages[0]["role"] == "user":
            base64_image = self.encode_image(image_source)
            print(f"[VLM调试] Base64编码长度: {len(base64_image)}")

            current_content = messages[0]["content"]

            # OpenAI兼容格式：使用 image_url
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }

            if isinstance(current_content, str):
                text_content = {"type": "text", "text": current_content}
                messages[0]["content"] = [text_content, image_content]
            elif isinstance(current_content, list):
                messages[0]["content"].append(image_content)

        # 智能处理URL：如果api_url已包含完整路径，直接使用；否则添加默认路径
        if '/chat/completions' in self.api_url or '/v1/chat/completions' in self.api_url:
            full_url = self.api_url
        else:
            full_url = f"{self.api_url}/chat/completions"

        parsed = urlparse(full_url)
        host = parsed.netloc or parsed.hostname
        path = parsed.path if parsed.path else '/v1/chat/completions'

        if not host:
            print(f"解析URL失败: {full_url}")
            raise ValueError("API URL 无效，无法解析主机名")

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # ===== 立即保存发送给VLM的messages到文件 =====
        os.makedirs('output', exist_ok=True)
        request_file_path = os.path.join('output', 'vlm_request.json')

        # 创建截断版本的请求体用于保存
        truncated_body = request_body.copy()
        if (isinstance(truncated_body.get("messages"), list) and
            len(truncated_body["messages"]) > 0 and
            isinstance(truncated_body["messages"][0].get("content"), list)):
            for item in truncated_body["messages"][0]["content"]:
                if item.get("type") == "image_url":
                    item["image_url"]["url"] = "[BASE64_IMAGE_DATA_TRUNCATED]"

        with open(request_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_source': image_source if image_source else None,
                'request': truncated_body
            }, f, indent=2, ensure_ascii=False)
        print(f"VLM请求已保存到: {request_file_path}")
        # ================================================

        # ===== 调试日志：打印发送给 VLM 的请求内容 =====
        debug_headers = headers.copy()
        if "Authorization" in debug_headers:
            debug_headers["Authorization"] = "Bearer [REDACTED]"

        print(f"\n【VLM 请求调试信息】")
        print(f"原始API URL: {self.api_url}")
        print(f"完整请求URL: https://{host}{path}")
        print("Headers:")
        print(json.dumps(debug_headers, indent=2))
        print("Request Body (Base64 图像已截断):")
        print(json.dumps(truncated_body, indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")
        # ================================================

        print("正在调用VLM服务...")
        response = requests.post(
            f"https://{host}{path}",
            json=request_body,
            headers=headers,
            verify=False,
            timeout=30
        )

        print(f"VLM服务响应状态码: {response.status_code}")
        if response.status_code != 200:
            error_msg = response.text
            print(f"VLM服务器错误响应: {error_msg}")
            raise Exception(f"VLM服务器错误: {response.status_code} - {error_msg}")

        data = response.json()

        os.makedirs('output', exist_ok=True)
        output_file_path = os.path.join('output', 'vlm_response.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'response': data
            }, f, indent=4, ensure_ascii=False)
        print(f"VLM响应已保存到: {output_file_path}")

        return data
        print(f"VLM响应已保存到: {output_file_path}")

        return data