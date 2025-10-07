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

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    logger.info("健康检查请求")
    return jsonify({
        "status": "healthy",
        "message": "鼠标控制服务运行正常"
    })

@app.route('/', methods=['GET'])
def index():
    """
    根路径，返回API说明
    """
    logger.info("根路径访问")
    return jsonify({
        "message": "鼠标控制API服务",
        "endpoints": {
            "POST /click": "执行鼠标点击操作",
            "GET /position": "获取当前鼠标位置",
            "POST /move": "移动鼠标到指定位置",
            "GET /screenshot": "截取屏幕截图",
            "GET /health": "健康检查"
        },
        "port": 5002
    })

if __name__ == '__main__':
    logger.info("启动鼠标控制API服务...")
    logger.info("访问端口: 5002")
    logger.info("请确保以管理员权限运行此程序")
    app.run(host='0.0.0.0', port=5002, debug=False,use_reloader=False)