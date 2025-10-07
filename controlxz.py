# controlxz.py
import logging
from datetime import datetime
from pynput.keyboard import Key, Controller
import threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import json
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

# 全局变量用于存储按键状态
key_state = {
    'X': False,
    'Z': False,
    'last_updated': None
}

# 存储日志记录的列表
log_records = []

# 创建键盘控制器
keyboard = Controller()

# Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

from flask_cors import CORS
CORS(app)

# 初始化SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# 自定义日志处理器，用于通过WebSocket发送日志
class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        }
        log_records.append(log_entry)
        # 保持日志记录在合理范围内
        if len(log_records) > 1000:
            log_records.pop(0)
        # 通过WebSocket广播日志
        socketio.emit('log_entry', log_entry)

# 添加自定义日志处理器
ws_handler = WebSocketLogHandler()
ws_handler.setLevel(logging.INFO)
logger.addHandler(ws_handler)

def press_key(key):
    """模拟按键按下"""
    logger.info(f"模拟按键按下: {key}")
    
    try:
        # 使用独立线程避免阻塞
        def _press():
            if key == 'X':
                keyboard.press('x')
                keyboard.release('x')
            elif key == 'Z':
                keyboard.press('z')
                keyboard.release('z')
        
        thread = threading.Thread(target=_press)
        thread.daemon = True
        thread.start()
        
    except Exception as e:
        logger.error(f"按键模拟失败: {e}")

def control_keys(action, key):
    """控制X和Z按键状态"""
    try:
        if action not in ['press', 'release'] or key not in ['X', 'Z']:
            logger.warning(f"无效的操作或按键: action={action}, key={key}")
            return {'error': '无效的操作或按键'}, 400
        
        # 实际按键控制（只在按下时触发）
        if action == 'press':
            press_key(key)
        
        # 更新按键状态
        global key_state
        key_state[key] = (action == 'press')
        key_state['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"按键 {key} 状态更新为: {action}")
        
        # 通过WebSocket广播状态更新
        socketio.emit('key_state_update', key_state)
        
        return {
            'status': 'success',
            'message': f'按键 {key} 已{action}',
            'key_state': key_state
        }, 200
        
    except Exception as e:
        logger.error(f"控制按键时出错: {e}")
        return {'error': str(e)}, 500

def get_key_state():
    """获取当前按键状态"""
    logger.info("获取按键状态")
    return key_state

def get_recent_logs(limit=50):
    """获取最近的日志记录"""
    logger.info(f"获取最近 {limit} 条日志记录")
    return log_records[-limit:] if len(log_records) > limit else log_records

def control_health_check():
    """按键控制健康检查"""
    logger.info("执行健康检查")
    return {
        'status': 'healthy', 
        'message': '按键控制模块正在运行',
        'timestamp': datetime.now().isoformat()
    }

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    logger.info('客户端已连接')
    # 发送当前状态给新连接的客户端
    emit('key_state_update', key_state)
    # 发送最近的日志记录
    emit('recent_logs', get_recent_logs())

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('客户端已断开连接')

@socketio.on('control_key')
def handle_control_key(data):
    """处理按键控制请求"""
    action = data.get('action')
    key = data.get('key')
    
    logger.info(f"收到按键控制请求: action={action}, key={key}")
    
    # 调用已有的控制函数
    result, status_code = control_keys(action, key)
    if status_code == 200:
        emit('control_result', result)
    else:
        emit('control_error', result)

@socketio.on('get_state')
def handle_get_state():
    """获取按键状态"""
    logger.info("WebSocket请求获取按键状态")
    state = get_key_state()
    emit('key_state_update', state)

@socketio.on('get_logs')
def handle_get_logs(data):
    """获取日志记录"""
    limit = data.get('limit', 50)
    logs = get_recent_logs(limit)
    emit('recent_logs', logs)

# HTTP 接口 (保持兼容性)
@app.route('/control', methods=['POST'])
def handle_control():
    """处理按键控制请求"""
    data = request.get_json()
    action = data.get('action')
    key = data.get('key')
    
    logger.info(f"HTTP请求按键控制: action={action}, key={key}")
    
    # 调用已有的控制函数
    result, status_code = control_keys(action, key)
    return jsonify(result), status_code

@app.route('/state', methods=['GET'])
def handle_get_state_http():
    """获取按键状态"""
    logger.info("HTTP请求获取按键状态")
    state = get_key_state()
    return jsonify(state)

@app.route('/logs', methods=['GET'])
def handle_get_logs_http():
    """获取日志记录的HTTP接口"""
    limit = request.args.get('limit', default=50, type=int)
    logs = get_recent_logs(limit)
    return jsonify(logs)

@app.route('/health', methods=['GET'])
def handle_health_check():
    """健康检查"""
    logger.info("HTTP请求健康检查")
    health_info = control_health_check()
    return jsonify(health_info)

import subprocess
import re
from flask_socketio import SocketIO, emit

def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        # 方法1: 使用ipconfig命令(Windows)
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, encoding='gbk')
        # 匹配IPv4地址(排除127.0.0.1)
        ip_pattern = r'IPv4.*?(\d+\.\d+\.\d+\.\d+)'
        matches = re.findall(ip_pattern, result.stdout)
        # 返回第一个非回环地址
        for ip in matches:
            if not ip.startswith('127.'):
                return ip
        return '127.0.0.1'
    except Exception as e:
        logger.warning(f"获取IP地址失败: {e}")
        return '0.0.0.0'

# 在主程序中使用
if __name__ == '__main__':
    local_ip = get_local_ip()
    logger.info(f"启动控制服务...")
    logger.info(f"本机IP地址: {local_ip}")
    logger.info("请确保以管理员权限运行此程序")
    socketio.run(app, host=local_ip, port=5001, debug=True, use_reloader=False)