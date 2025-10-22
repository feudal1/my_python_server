# mousecontrol.py
from flask import Flask, jsonify, request
import pyautogui
import logging
import base64
import io
from PIL import Image
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

# 创建Flask应用
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
# 禁用pyautogui的安全限制（在生产环境中请谨慎使用）
pyautogui.FAILSAFE = False

@app.route('/click', methods=['POST'])
def click_mouse():
    """
    鼠标点击接口
    可以接收JSON数据指定x,y坐标和点击类型
    """
    try:
        data = request.get_json()
        
        # 获取点击坐标，默认为当前鼠标位置
        x = data.get('x')
        y = data.get('y')
        
        # 获取点击类型，默认为left
        click_type = data.get('type', 'left')
        
        # 获取点击次数，默认为1
        clicks = data.get('clicks', 1)
        
        # 获取间隔时间，默认为0.0
        interval = data.get('interval', 0.0)
        
        # 执行点击操作
        if x is not None and y is not None:
            pyautogui.click(x, y, clicks=clicks, interval=interval, button=click_type)
            logger.info(f"点击位置: ({x}, {y}), 类型: {click_type}, 次数: {clicks}")
        else:
            pyautogui.click(clicks=clicks, interval=interval, button=click_type)
            logger.info(f"点击当前位置, 类型: {click_type}, 次数: {clicks}")
            
        return jsonify({
            "status": "success",
            "message": "鼠标点击成功",
            "coordinates": {"x": x, "y": y} if x is not None and y is not None else "current position"
        })
        
    except Exception as e:
        logger.error(f"点击操作失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"鼠标点击失败: {str(e)}"
        }), 500

@app.route('/position', methods=['GET'])
def get_position():
    """
    获取当前鼠标位置
    """
    try:
        x, y = pyautogui.position()
        logger.info(f"获取鼠标位置: ({x}, {y})")
        return jsonify({
            "status": "success",
            "position": {"x": x, "y": y}
        })
    except Exception as e:
        logger.error(f"获取鼠标位置失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"获取鼠标位置失败: {str(e)}"
        }), 500

@app.route('/move', methods=['POST'])
def move_mouse():
    """
    移动鼠标到指定位置
    """
    try:
        data = request.get_json()
        x = data.get('x')
        y = data.get('y')
        duration = data.get('duration', 0.0)  # 移动持续时间
        
        if x is None or y is None:
            logger.warning("移动鼠标请求缺少坐标参数")
            return jsonify({
                "status": "error",
                "message": "请提供x和y坐标"
            }), 400
            
        pyautogui.moveTo(x, y, duration=duration)
        logger.info(f"移动鼠标到: ({x}, {y})")
        
        return jsonify({
            "status": "success",
            "message": "鼠标移动成功",
            "position": {"x": x, "y": y}
        })
        
    except Exception as e:
        logger.error(f"移动鼠标失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"鼠标移动失败: {str(e)}"
        }), 500

@app.route('/screenshot', methods=['GET'])
def take_screenshot():
    """
    截取屏幕截图并返回base64编码的图像（调整为384x640分辨率）
    """
    try:
        logger.info("开始屏幕截图")
        # 截取整个屏幕
        screenshot = pyautogui.screenshot()
        
        # 调整图像大小为384x640
        resized_screenshot = screenshot.resize((640,384), resample=Image.Resampling.LANCZOS)
        
        # 将图像转换为base64编码
        img_buffer = io.BytesIO()
        resized_screenshot.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        logger.info("屏幕截图完成（已调整为384x640分辨率）")
        
        return jsonify({
            "status": "success",
            "message": "屏幕截图成功",
            "image": img_base64
        })
        
    except Exception as e:
        logger.error(f"截图失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"截图失败: {str(e)}"
        }), 500

@app.route('/routes')
def show_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        # 获取端点函数的docstring
        endpoint_func = app.view_functions.get(rule.endpoint)
        docstring = endpoint_func.__doc__.strip() if endpoint_func and endpoint_func.__doc__ else "无描述"
        
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule),
            'description': docstring
        })
    return {'routes': routes}

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
args = parser.parse_args()

# 使用指定的端口运行 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)