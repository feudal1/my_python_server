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

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--log-file', help='日志文件路径')
args = parser.parse_args()

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
load_dotenv()

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

# 创建 LLMService 实例
llm_service = LLMService()

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

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    logger.info("健康检查请求")
    return jsonify({
        "status": "healthy",
        "message": "LLM服务运行正常",
        "model": llm_service.model_name
    })

@app.route('/', methods=['GET'])
def index():
    """
    根路径，返回API说明
    """
    logger.info("根路径访问")
    return jsonify({
        "message": "LLM API服务",
        "endpoints": {
            "POST /chat": "与LLM进行对话",
            "GET /health": "健康检查"
        },
        "port": 5003
    })

if __name__ == "__main__":
    # 确保output目录存在
    os.makedirs('output', exist_ok=True)
    logger.info("启动LLM API服务...")
    logger.info("访问端口: 5003")
    app.run(host='0.0.0.0', port=5003, debug=False,use_reloader=False)