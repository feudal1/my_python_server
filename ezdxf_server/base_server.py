# base_server.py
import ezdxf
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
from flask import Flask, request, jsonify, send_file, render_template_string
import os
import subprocess
import tempfile
import time
import json
from datetime import datetime
from functools import wraps

SHOW_TIMING = True
app = Flask(__name__)

def convert_dwg_to_dxf(dwg_file_path):
    """
    使用ODA File Converter将DWG文件转换为DXF文件
    
    Args:
        dwg_file_path (str): DWG文件路径
        
    Returns:
        dict: 转换结果，包含状态和DXF文件路径或错误信息
    """
    start_time = time.time()
    try:
        # 检查DWG文件是否存在
        if not os.path.exists(dwg_file_path):
            return {
                "status": "error",
                "message": f"DWG文件不存在: {dwg_file_path}"
            }

        # 获取文件目录和文件名
        dwg_dir = os.path.dirname(dwg_file_path)
        dwg_filename = os.path.basename(dwg_file_path)
        dxf_filename = dwg_filename.replace('.dwg', '.dxf')
        dxf_file_path = dwg_file_path.replace('.dwg', '.dxf')
        if dxf_file_path == dwg_file_path:
            dxf_file_path = f"{dwg_file_path}.dxf"

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(dxf_file_path)
        os.makedirs(output_dir, exist_ok=True)

        # 参数定义（按官方规范）
        TEIGHA_PATH = "ODAFileConverter"
        INPUT_FOLDER = dwg_dir
        OUTPUT_FOLDER = output_dir
        OUTVER = "ACAD2018"           # 输入版本（如原文件是 ACAD2018）
        OUTFORMAT = "DXF"             # 输出格式：只能是 DXF / DWG / DXB
        RECURSIVE = "0"
        AUDIT = "1"
        INPUTFILTER = "*.DWG"         # 只处理DWG文件

        # 构建命令（注意顺序！）
        cmd = [
            TEIGHA_PATH,
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            OUTVER,           # 输入版本
            OUTFORMAT,        # 输出格式 ← 必须是 DXF/DWG/DXB
            RECURSIVE,
            AUDIT,
            INPUTFILTER
        ]

        print(f"执行命令: {' '.join(cmd)}")

        # 执行转换命令
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # 打印调试信息
        print("ODA 输出:", result.stdout)
        print("ODA 错误:", result.stderr)

        if result.returncode == 0:
            # 检查DXF文件是否生成
            if os.path.exists(dxf_file_path):
                # 验证DXF文件是否有效
                if is_valid_dxf(dxf_file_path):
                    response = {
                        "status": "success",
                        "message": f"DWG文件已成功转换为DXF: {dxf_file_path}",
                        "dxf_path": dxf_file_path
                    }
                    if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                    return response
                else:
                    response = {
                        "status": "error",
                        "message": f"生成的DXF文件结构不完整或损坏: {dxf_file_path}"
                    }
                    if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                    return response
            else:
                response = {
                    "status": "error",
                    "message": f"转换完成但未生成DXF文件。请检查输出目录: {output_dir}"
                }
                if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                return response
        else:
            response = {
                "status": "error",
                "message": f"ODA转换失败: {result.stderr}"
            }
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return response

    except subprocess.TimeoutExpired:
        response = {
            "status": "error",
            "message": "转换超时"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response
    except FileNotFoundError:
        response = {
            "status": "error",
            "message": "未找到ODAFileConverter命令，请确保ODA File Converter已正确安装并加入PATH"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response
    except Exception as e:
        response = {
            "status": "error",
            "message": f"转换过程中发生错误: {str(e)}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

def is_valid_dxf(dxf_path):
    """检查DXF文件是否有效"""
    try:
        doc = ezdxf.readfile(dxf_path)
        return True
    except Exception as e:
        print(f"DXF文件校验失败: {e}")
        return False







@app.route('/objects/texts', methods=['GET'])
def get_dxf_texts():
    """
    获取DXF/DWG文件中所有文本对象的内容
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        JSON格式的文本内容列表
    """
    start_time = time.time()
    file_path = request.args.get('dxf_path')
    if not file_path:
        response = {
            "status": "error", 
            "message": "缺少dxf_path参数"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(file_path):
        response = {
            "status": "error", 
            "message": f"文件不存在: {file_path}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # 如果是DWG文件，先转换为DXF
        if file_ext == '.dwg':
            conversion_result = convert_dwg_to_dxf(file_path)
            if conversion_result["status"] != "success":
                if SHOW_TIMING:
                    conversion_result["processing_time"] = time.time() - start_time
                return jsonify(conversion_result), 500
            dxf_file_path = conversion_result["dxf_path"]
        elif file_ext == '.dxf':
            dxf_file_path = file_path
        else:
            response = {
                "status": "error",
                "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
            }
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return jsonify(response), 400

        # 读取DXF文件并收集文本内容
        doc_read_start = time.time()
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()
        doc_read_time = time.time() - doc_read_start

        # 收集所有文本内容
        text_collection_start = time.time()
        texts = []
        for entity in msp:
            if entity.dxftype() == 'TEXT':
                texts.append({
                    'content': entity.dxf.text,
                    'position': {
                        'x': entity.dxf.insert.x,
                        'y': entity.dxf.insert.y
                    }
                })
            elif entity.dxftype() == 'MTEXT':
                texts.append({
                    'content': entity.text,
                    'position': {
                        'x': entity.dxf.insert.x,
                        'y': entity.dxf.insert.y
                    }
                })
        text_collection_time = time.time() - text_collection_start

        # 将所有文本内容合并为一个字符串
        text_merge_start = time.time()
        all_text_content = '\n'.join([text['content'] for text in texts])
        text_merge_time = time.time() - text_merge_start

        response = {
            'status': 'success',
            'texts': texts,
            'all_text_content': all_text_content,
            'text_count': len(texts),
            'file_path': dxf_file_path
        }
        
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
            response["timing_details"] = {
                "doc_read_time": doc_read_time,
                "text_collection_time": text_collection_time,
                "text_merge_time": text_merge_time
            }
            
        return jsonify(response)

    except Exception as e:
        response = {
            'status': 'error',
            'message': f"处理文件时出错: {str(e)}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500
# HTML模板
@app.route('/list_dxf_files', methods=['GET'])
def list_dxf_files():
    """
    获取指定文件夹内所有DXF文件的名称
    
    Query Parameters:
        folder_path: 文件夹路径
        
    Returns:
        JSON格式的DXF文件列表
    """
    start_time = time.time()
    
    # 获取请求参数
    folder_path = request.args.get('folder_path')
    
    # 检查是否提供了文件夹路径
    if not folder_path:
        response = {
            "status": "error",
            "message": "缺少folder_path参数"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400
    
    # 检查路径是否存在且为文件夹
    if not os.path.exists(folder_path):
        response = {
            "status": "error",
            "message": f"路径不存在: {folder_path}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404
        
    if not os.path.isdir(folder_path):
        response = {
            "status": "error",
            "message": f"指定路径不是文件夹: {folder_path}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400
    
    try:
        # 获取文件夹内所有文件
        all_files = os.listdir(folder_path)
        
        # 筛选出DXF文件（不区分大小写）
        dxf_files = [f for f in all_files if f.lower().endswith('.dxf')]
        
        response = {
            "status": "success",
            "folder_path": folder_path,
            "dxf_files": dxf_files,
            "count": len(dxf_files)
        }
        
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
            
        return jsonify(response)
        
    except PermissionError:
        response = {
            "status": "error",
            "message": f"没有权限访问文件夹: {folder_path}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 403
        
    except Exception as e:
        response = {
            "status": "error",
            "message": f"读取文件夹时出错: {str(e)}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .message { padding: 20px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 20px; }
        .error { background-color: #ffe6e6; border-left: 5px solid #ff0000; }
        .success { background-color: #e6ffe6; border-left: 5px solid #00cc00; }
        .info { background-color: #e6f3ff; border-left: 5px solid #0066cc; }
        .image-container { text-align: center; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .note { padding: 15px; background-color: #fff9e6; border-left: 5px solid #ffcc00; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        {% if message %}
        <div class="message {% if title == '处理失败' or title == '错误' or title == '转换失败' %}error{% elif title == '文件处理结果' or title == '高亮对象处理结果' %}success{% else %}info{% endif %}">
            <p>{{ message|safe }}</p>
        </div>
        {% endif %}
        
        {% if image_url %}
        <div class="image-container">
            <h2>处理结果图像</h2>
            <img src="{{ image_url }}" alt="文件处理结果">
        </div>
        {% elif title != '错误' and title != '转换失败' %}
        <p>无法显示图像。</p>
        {% endif %}
        
        <div class="note">
            <h3>支持的文件格式：</h3>
            <ul>
                <li>DXF文件（直接处理）</li>
                <li>DWG文件（自动转换为DXF后处理）</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "DXF/DWG基础处理服务已启动",
        "endpoints": {
            "process_dxf": "GET /process_dxf?dxf_path=<path_to_dxf_or_dwg_file> (处理DXF/DWG文件并返回JSON结果)",
            "objects_signatures": "GET /objects/signatures?dxf_path=<path_to_dxf_or_dwg_file> (获取所有对象签名)",
            "image": "GET /image?path=<path_to_image> (获取图像文件)"
        },
        "notes": "支持DXF和DWG文件格式。DWG文件会自动转换为DXF格式后再处理。"
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5303, debug=True)