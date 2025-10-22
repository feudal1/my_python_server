# llm_server.py
import os
from dotenv import load_dotenv
import json
import http.client
from urllib.parse import urlparse
from flask import Flask, request, jsonify
import logging
import argparse
import sys
import base64
from io import BytesIO
from PIL import Image
import requests

# 解析命令行参数 (只保留一个argparse实例)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
parser.add_argument('--log-file', help='日志文件路径')
args, unknown = parser.parse_known_args()

# 配置日志
if args.log_file:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
dotenv_path = r'E:\code\apikey\.env'
load_dotenv(dotenv_path)
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

class LLMService:
    def __init__(self):
        # 从环境变量中获取 DeepSeek 参数
        self.api_url = os.getenv('deepseek_OPENAI_API_URL')
        self.model_name = os.getenv('deepseek_MODEL_NAME')
        self.api_key = os.getenv('deepseek_OPENAI_API_KEY')
            # 检查必需的环境变量是否存在
        if not self.api_url:
            raise ValueError("环境变量 'deepseek_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'deepseek_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'deepseek_OPENAI_API_KEY' 未设置或为空")
        logger.info(f"LLM服务初始化完成，模型: {self.model_name}")

    def create(self, messages, tools=None):
        logger.info("开始调用LLM服务")
        logger.debug(f"消息内容: {messages}")
        
        # 解析 URL（去掉协议部分）
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        if not host:
            logger.error("API URL 无效，无法解析主机名")
            raise ValueError("API URL 无效，无法解析主机名")

        # 创建 HTTP 连接
        conn = http.client.HTTPSConnection(host)

        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.9  # 添加温度参数
        }

        # 发送 POST 请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(request_body),
            headers=headers
        )

        # 获取响应
        response = conn.getresponse()
        logger.info(f"LLM服务响应状态码: {response.status}")
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            logger.error(f"LLM服务器错误: {response.status} - {error_msg}")
            raise Exception(f"LLM服务器错误: {response.status} - {error_msg}")

        # 读取响应内容
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)

        # 确保output目录存在
        os.makedirs('output', exist_ok=True)
        
        # 将响应保存到文件 (修复路径分隔符问题)
        output_file_path = os.path.join('output', 'formatted_data.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 关闭连接
        conn.close()
        
        logger.info("LLM服务调用完成")
        return data

class VLMService:
    def __init__(self):
        # 从环境变量中获取 VLM 参数
        self.api_url = os.getenv('VLM_OPENAI_API_URL')
        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')
        
        # 检查必需的环境变量是否存在
        if not self.api_url:
            raise ValueError("环境变量 'VLM_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'VLM_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'VLM_OPENAI_API_KEY' 未设置或为空")
        logger.info(f"VLM服务初始化完成，模型: {self.model_name}")

    def encode_image(self, image_source):
        """
        编码图像为base64字符串
        支持URL和本地文件路径
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # 从URL获取图像
                response = requests.get(image_source)
                image_data = response.content
            else:
                # 从本地文件路径获取图像
                with open(image_source, "rb") as image_file:
                    image_data = image_file.read()
            
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
            raise Exception(f"图像编码失败: {str(e)}")

    def create_with_image(self, messages, image_source=None, tools=None):
        logger.info("开始调用VLM服务")
        
        # 如果提供了图像源，则处理图像
        if image_source:
            # 编码图像
            base64_image = self.encode_image(image_source)
            
            # 在第一条消息中添加图像
            if messages and len(messages) > 0:
                # 构建包含图像的消息内容
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                
                # 如果第一条消息的内容是字符串，转换为列表格式
                if isinstance(messages[0]["content"], str):
                    messages[0]["content"] = [
                        {
                            "type": "text",
                            "text": messages[0]["content"]
                        },
                        image_content
                    ]
                # 如果已经是列表格式，追加图像内容
                elif isinstance(messages[0]["content"], list):
                    messages[0]["content"].append(image_content)
        
        logger.debug(f"消息内容: {messages}")
        
        # 解析 URL（去掉协议部分）
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        if not host:
            logger.error("API URL 无效，无法解析主机名")
            raise ValueError("API URL 无效，无法解析主机名")

        # 创建 HTTP 连接
        conn = http.client.HTTPSConnection(host)

        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }

        # 发送 POST 请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(request_body),
            headers=headers
        )

        # 获取响应
        response = conn.getresponse()
        logger.info(f"VLM服务响应状态码: {response.status}")
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            logger.error(f"VLM服务器错误: {response.status} - {error_msg}")
            raise Exception(f"VLM服务器错误: {response.status} - {error_msg}")

        # 读取响应内容
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)

        # 确保output目录存在
        os.makedirs('output', exist_ok=True)
        
        # 将响应保存到文件
        output_file_path = os.path.join('output', 'vlm_formatted_data.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 关闭连接
        conn.close()
        
        logger.info("VLM服务调用完成")
        return data

# 创建服务实例
llm_service = LLMService()
vlm_service = VLMService()

@app.route('/chat', methods=['GET', 'POST'])
def chat_endpoint():
    try:
        logger.info("收到聊天请求")
        
        if request.method == 'GET':
            # 从查询参数中获取 messages 和 tools
            messages_json = request.args.get('messages', '[]')
            tools_json = request.args.get('tools', None)
            
            # 解析 JSON 字符串
            messages = json.loads(messages_json) if messages_json else []
            tools = json.loads(tools_json) if tools_json else None
            
        elif request.method == 'POST':
            # 从请求体中获取数据
            data = request.get_json()
            if not data:
                logger.error("POST请求缺少JSON数据")
                return jsonify({"error": "请求体必须包含JSON数据"}), 400
                
            messages = data.get('messages', [])
            tools = data.get('tools', None)
        
        logger.debug(f"请求参数 - 消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        
        # 调用 LLMService 的 create 方法
        result = llm_service.create(messages, tools)
        logger.info("聊天请求处理完成")
        return jsonify(result)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        return jsonify({"error": f"JSON解析失败: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 在现有导入和代码之后添加以下内容
@app.route('/vlm/chat', methods=['GET'])
def vlm_chat_get_endpoint():
    try:
        logger.info("收到VLM聊天GET请求")
        
        # 从查询参数中获取数据
        messages_json = request.args.get('messages', '[]')
        image_source = request.args.get('image_source', None)
        tools_json = request.args.get('tools', None)
        
        # 解析 JSON 字符串
        try:
            messages = json.loads(messages_json) if messages_json else []
            tools = json.loads(tools_json) if tools_json else None
        except json.JSONDecodeError as e:
            logger.error(f"查询参数JSON解析失败: {str(e)}")
            return jsonify({"error": f"查询参数JSON解析失败: {str(e)}"}), 400
        
        logger.debug(f"VLM请求参数 - 消息数量: {len(messages)}, 包含图像: {image_source is not None}, 工具数量: {len(tools) if tools else 0}")
        
        # 调用 VLMService 的 create_with_image 方法
        result = vlm_service.create_with_image(messages, image_source, tools)
        logger.info("VLM聊天GET请求处理完成")
        return jsonify(result)
    except Exception as e:
        logger.error(f"VLM聊天GET请求处理失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    logger.info("健康检查请求")
    return jsonify({
        "status": "healthy",
        "message": "LLM/VLM服务运行正常",
        "models": {
            "llm": llm_service.model_name,
            "vlm": vlm_service.model_name
        }
    })

@app.route('/routes')
def show_routes():
    """
    显示所有可用路由
    """
    routes = []
    for rule in app.url_map.iter_rules():
        # 获取端点函数的docstring
        endpoint_func = app.view_functions.get(rule.endpoint)
        docstring = endpoint_func.__doc__.strip() if endpoint_func and endpoint_func.__doc__ else "无描述"
        
        routes.append({
            'endpoint': rule.endpoint,
            'methods': sorted(list(rule.methods - {'HEAD'})),  # 排除HEAD方法
            'rule': str(rule),
            'description': docstring
        })
    
    # 按路由规则排序
    routes.sort(key=lambda x: x['rule'])
    
    return jsonify({'routes': routes})

# 使用指定的端口运行 Flask 应用 (移除重复的argparse)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)