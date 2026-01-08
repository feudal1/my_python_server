import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import subprocess
import time

app = Flask(__name__)
CORS(app)  # 允许跨域请求，方便前端访问

# 用于跟踪按键状态
key_states = {
    'x': False,
    'z': False
}

def call_mousecontrol_function(function_name, *args):
    """调用mousecontrol.py中的函数"""
    try:
        cmd = ["python", os.path.join(os.path.dirname(__file__), "mousecontrol.py"), function_name] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            return {"status": "success", "message": result.stdout.strip()}
        else:
            return {"status": "error", "message": result.stderr.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XZ键控制面板</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            font-family: Arial, sans-serif;
            overflow: hidden;
            background-color: #f0f0f0;
        }
        
        .half {
            width: 50%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            transition: background-color 0.1s;
            user-select: none;
        }
        
        .x-half {
            background-color: #ffcccc;
            border-right: 2px solid #ccc;
        }
        
        .z-half {
            background-color: #ccccff;
        }
        
        .half.active {
            opacity: 0.7;
        }
        
        .key-label {
            font-size: 8em;
            font-weight: bold;
            color: #333;
        }
        
        .instructions {
            margin-top: 20px;
            font-size: 1.2em;
            color: #666;
            text-align: center;
        }
        
        .status {
            margin-top: 20px;
            font-size: 1.2em;
            color: #333;
            text-align: center;
        }
        
        .pressed {
            background-color: #ff9999 !important;
        }
        
        .z-pressed {
            background-color: #9999ff !important;
        }
    </style>
</head>
<body>
    <div class="half x-half" id="xHalf">
        <div class="key-label">X</div>
        <div class="instructions">点击按住或松开X键</div>
        <div class="status" id="xStatus">状态: 未按下</div>
    </div>
    
    <div class="half z-half" id="zHalf">
        <div class="key-label">Z</div>
        <div class="instructions">点击按住或松开Z键</div>
        <div class="status" id="zStatus">状态: 未按下</div>
    </div>

    <script>
        // 按键状态
        let keyStates = {
            x: false,
            z: false
        };
        
        // 获取DOM元素
        const xHalf = document.getElementById('xHalf');
        const zHalf = document.getElementById('zHalf');
        const xStatus = document.getElementById('xStatus');
        const zStatus = document.getElementById('zStatus');
        
        // 发送按键状态到服务器
        async function sendKeyAction(action, key) {
            try {
                let endpoint;
                if (action === 'press_key') {
                    endpoint = '/press_key';
                } else {
                    endpoint = '/key_' + action;
                }
                
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({key: key})
                });
                const result = await response.json();
                console.log(`Key ${key} ${action}:`, result);
                return result;
            } catch (error) {
                console.error(`Error sending ${action} for key ${key}:`, error);
            }
        }
        
        // 更新按键状态显示
        function updateStatus() {
            xStatus.textContent = `状态: ${keyStates.x ? '按下' : '未按下'}`;
            zStatus.textContent = `状态: ${keyStates.z ? '按下' : '未按下'}`;
            
            // 更新视觉效果
            if (keyStates.x) {
                xHalf.classList.add('pressed');
            } else {
                xHalf.classList.remove('pressed');
            }
            
            if (keyStates.z) {
                zHalf.classList.add('z-pressed');
            } else {
                zHalf.classList.remove('z-pressed');
            }
        }
        
        // 为X区域添加事件监听器
        xHalf.addEventListener('mousedown', function(e) {
            e.preventDefault();
            // 按下X键（长按）
            sendKeyAction('down', 'x');
            keyStates.x = true;
            updateStatus();
        });
        
        xHalf.addEventListener('mouseup', function() {
            // 释放X键
            sendKeyAction('up', 'x');
            keyStates.x = false;
            updateStatus();
        });
        
        xHalf.addEventListener('mouseleave', function() {
            // 鼠标离开区域，也应释放按键
            if (keyStates.x) {
                sendKeyAction('up', 'x');
                keyStates.x = false;
                updateStatus();
            }
        });
        
        // 为Z区域添加事件监听器
        zHalf.addEventListener('mousedown', function(e) {
            e.preventDefault();
            // 按下Z键（长按）
            sendKeyAction('down', 'z');
            keyStates.z = true;
            updateStatus();
        });
        
        zHalf.addEventListener('mouseup', function() {
            // 释放Z键
            sendKeyAction('up', 'z');
            keyStates.z = false;
            updateStatus();
        });
        
        zHalf.addEventListener('mouseleave', function() {
            // 鼠标离开区域，也应释放按键
            if (keyStates.z) {
                sendKeyAction('up', 'z');
                keyStates.z = false;
                updateStatus();
            }
        });
        
        // 触摸设备支持
        xHalf.addEventListener('touchstart', function(e) {
            e.preventDefault();
            // 按下X键（长按）
            sendKeyAction('down', 'x');
            keyStates.x = true;
            updateStatus();
        });
        
        xHalf.addEventListener('touchend', function(e) {
            e.preventDefault();
            // 释放X键
            sendKeyAction('up', 'x');
            keyStates.x = false;
            updateStatus();
        });
        
        xHalf.addEventListener('touchcancel', function() {
            // 触摸取消，释放按键
            if (keyStates.x) {
                sendKeyAction('up', 'x');
                keyStates.x = false;
                updateStatus();
            }
        });
        
        zHalf.addEventListener('touchstart', function(e) {
            e.preventDefault();
            // 按下Z键（长按）
            sendKeyAction('down', 'z');
            keyStates.z = true;
            updateStatus();
        });
        
        zHalf.addEventListener('touchend', function(e) {
            e.preventDefault();
            // 释放Z键
            sendKeyAction('up', 'z');
            keyStates.z = false;
            updateStatus();
        });
        
        zHalf.addEventListener('touchcancel', function() {
            // 触摸取消，释放按键
            if (keyStates.z) {
                sendKeyAction('up', 'z');
                keyStates.z = false;
                updateStatus();
            }
        });
        
        // 页面加载时获取初始状态
        window.onload = function() {
            updateStatus();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """主页，提供HTML界面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/press_key', methods=['POST'])
def press_key():
    """按下按键（短按）"""
    data = request.json
    key = data.get('key', '').lower()
    
    if key in ['x', 'z']:
        result = call_mousecontrol_function('press_key', key)
        return jsonify(result)
    else:
        return jsonify({"status": "error", "message": "只支持X和Z键"})

@app.route('/key_down', methods=['POST'])
def key_down():
    """按下并持续按住按键"""
    data = request.json
    key = data.get('key', '').lower()
    
    if key in ['x', 'z']:
        result = call_mousecontrol_function('key_down', key)
        key_states[key] = True
        return jsonify(result)
    else:
        return jsonify({"status": "error", "message": "只支持X和Z键"})

@app.route('/key_up', methods=['POST'])
def key_up():
    """释放按键"""
    data = request.json
    key = data.get('key', '').lower()
    
    if key in ['x', 'z']:
        result = call_mousecontrol_function('key_up', key)
        key_states[key] = False
        return jsonify(result)
    else:
        return jsonify({"status": "error", "message": "只支持X和Z键"})

@app.route('/key_state', methods=['GET'])
def get_key_state():
    """获取当前按键状态"""
    return jsonify(key_states)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)