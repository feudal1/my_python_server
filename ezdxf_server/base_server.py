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
import math
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dxf_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

SHOW_TIMING = True
app = Flask(__name__)


# ==================== 配置 matplotlib 支持中文 ====================
matplotlib.rcParams['font.family'] = 'SimHei'  # Windows 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方框


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
        if not os.path.exists(dwg_file_path):
            return {
                "status": "error",
                "message": f"DWG文件不存在: {dwg_file_path}"
            }

        dwg_dir = os.path.dirname(dwg_file_path)
        dwg_filename = os.path.basename(dwg_file_path)
        dxf_filename = dwg_filename.replace('.dwg', '.dxf')
        dxf_file_path = dwg_file_path.replace('.dwg', '.dxf')
        if dxf_file_path == dwg_file_path:
            dxf_file_path = f"{dwg_file_path}.dxf"

        output_dir = os.path.dirname(dxf_file_path)
        os.makedirs(output_dir, exist_ok=True)

        TEIGHA_PATH = "ODAFileConverter"
        INPUT_FOLDER = dwg_dir
        OUTPUT_FOLDER = output_dir
        OUTVER = "ACAD2018"
        OUTFORMAT = "DXF"
        RECURSIVE = "0"
        AUDIT = "1"
        INPUTFILTER = "*.DWG"

        cmd = [
            TEIGHA_PATH,
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            OUTVER,
            OUTFORMAT,
            RECURSIVE,
            AUDIT,
            INPUTFILTER
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            if os.path.exists(dxf_file_path) and is_valid_dxf(dxf_file_path):
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
                    "message": f"生成的DXF文件无效或未创建: {dxf_file_path}"
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
            "message": "未找到ODAFileConverter命令，请确保已安装并加入PATH"
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
        logger.info(f"DXF文件校验失败: {e}")
        return False


@app.route('/objects/texts', methods=['GET'])
def get_dxf_texts():
    """
    获取DXF/DWG文件中所有文本对象的内容
    """
    start_time = time.time()
    file_path = request.args.get('dxf_path')
    if not file_path:
        response = {"status": "error", "message": "缺少dxf_path参数"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(file_path):
        response = {"status": "error", "message": f"文件不存在: {file_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
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
            response = {"status": "error", "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"}
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return jsonify(response), 400

        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        texts = []
        for entity in msp:
            if entity.dxftype() == 'TEXT':
                texts.append({
                    'content': entity.dxf.text,
                    'position': {'x': entity.dxf.insert.x, 'y': entity.dxf.insert.y}
                })
            elif entity.dxftype() == 'MTEXT':
                texts.append({
                    'content': entity.text,
                    'position': {'x': entity.dxf.insert.x, 'y': entity.dxf.insert.y}
                })

        all_text_content = '\n'.join([text['content'] for text in texts])

        response = {
            'status': 'success',
            'texts': texts,
            'all_text_content': all_text_content,
            'text_count': len(texts),
            'file_path': dxf_file_path
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response)

    except Exception as e:
        response = {"status": "error", "message": f"处理文件时出错: {str(e)}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500


@app.route('/list_dxf_files', methods=['GET'])
def list_dxf_files():
    """
    获取指定文件夹内所有DXF文件的名称
    """
    start_time = time.time()
    folder_path = request.args.get('folder_path')
    if not folder_path:
        response = {"status": "error", "message": "缺少folder_path参数"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(folder_path):
        response = {"status": "error", "message": f"路径不存在: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    if not os.path.isdir(folder_path):
        response = {"status": "error", "message": f"指定路径不是文件夹: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    try:
        all_files = os.listdir(folder_path)
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
        response = {"status": "error", "message": f"没有权限访问文件夹: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 403
    except Exception as e:
        response = {"status": "error", "message": f"读取文件夹时出错: {str(e)}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500


@app.route('/render_dxf_image', methods=['GET'])
def render_dxf_image():
    """
    渲染DXF/DWG文件为图像
    """
    start_time = time.time()
    file_path = request.args.get('dxf_path')
    if not file_path:
        response = {"status": "error", "message": "缺少dxf_path参数"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(file_path):
        response = {"status": "error", "message": f"文件不存在: {file_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
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
            response = {"status": "error", "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"}
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return jsonify(response), 400

        result = process_dxf_file_simple(dxf_file_path)
        if result["status"] == "success":
            if request.args.get('return_image') == 'true':
                return send_file(result["output_path"], mimetype='image/png')
            else:
                if SHOW_TIMING:
                    result["processing_time"] = time.time() - start_time
                return jsonify(result)
        else:
            if SHOW_TIMING:
                result["processing_time"] = time.time() - start_time
            return jsonify(result), 500
    except Exception as e:
        response = {"status": "error", "message": f"处理文件时出错: {str(e)}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500

def process_dxf_file_simple(dxf_file_path):
    """
    处理DXF文件并生成图像（支持中文、自动字体大小）
    """
    start_time = time.time()
    dxf_color_map = {
        0: 'black', 1: 'red', 2: 'yellow', 3: 'green', 4: 'cyan',
        5: 'blue', 6: 'magenta', 7: 'white', 8: '#a5a5a5', 9: '#c0c0c0',
        10: 'red', 11: '#ffaaaa', 12: '#bd0000', 13: '#bd7373', 14: '#800000',
        15: '#ff0000', 16: '#ffff00', 17: '#ffff73', 18: '#bda000', 19: '#bdae73',
        20: '#808000', 21: '#ffff00', 22: '#00ff00', 23: '#aaffaa', 24: '#00bd00',
        25: '#73bd73', 26: '#008000', 27: '#00ff00', 28: '#00ffff', 29: '#aaffff',
        30: '#00bfbf', 31: '#73bfbf', 32: '#008080', 33: '#00ffff', 34: '#0000ff',
        35: '#aaaaff', 36: '#0000bd', 37: '#7373bf', 38: '#000080', 39: '#0000ff',
        40: '#ff00ff', 41: '#ffaaff', 42: '#bd00bd', 43: '#bd73bd', 44: '#800080',
        45: '#ff00ff', 'default': 'black'
    }

    try:
        if not os.path.exists(dxf_file_path):
            return {"status": "error", "message": f"文件不存在: {dxf_file_path}"}

        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        fig, ax = plt.subplots(figsize=(12, 12), dpi=200)  # 提高分辨率
        all_x_coords = []
        all_y_coords = []

        entity_counts = {}
        processed_entities = 0

        for entity in msp:
            entity_type = entity.dxftype()
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            try:
                color = dxf_color_map.get(entity.dxf.color, dxf_color_map['default'])
            except:
                color = dxf_color_map['default']

            linewidth = 0.5

            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                ax.plot([start.x, end.x], [start.y, end.y], color=color, linewidth=linewidth)
                all_x_coords.extend([start.x, end.x])
                all_y_coords.extend([start.y, end.y])
                processed_entities += 1

            elif entity.dxftype() == 'SPLINE':
                try:
                    points = [(p.x, p.y) for p in entity.fit_points] if hasattr(entity, 'fit_points') and entity.fit_points else \
                             [(p[0], p[1]) for p in entity.control_points]
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)
                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
                    processed_entities += 1
                except Exception as e:
                    logger.info(f'处理SPLINE实体时出错: {e}')

            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                if start_angle > end_angle:
                    end_angle += 360
                arc_patch = Arc((center.x, center.y), 2*radius, 2*radius, angle=0,
                                theta1=start_angle, theta2=end_angle,
                                color=color, linewidth=linewidth)
                ax.add_patch(arc_patch)
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                processed_entities += 1

            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                circle_patch = plt.Circle((center.x, center.y), radius, fill=False, color=color, linewidth=linewidth)
                ax.add_patch(circle_patch)
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                processed_entities += 1

            # 修改 TEXT 实体处理部分
            elif entity.dxftype() == 'TEXT':
                insert = entity.dxf.insert
                text = entity.dxf.text
                height = entity.dxf.height if hasattr(entity.dxf, 'height') else 0.5
                # 使用原始高度，但根据图像整体尺寸适度调整
                font_size = height * 0.3  # 保持原始比例，调整倍数使显示效果更好
                ax.text(insert.x, insert.y, text, color=color, fontsize=font_size)
                all_x_coords.append(insert.x)
                all_y_coords.append(insert.y)
                processed_entities += 1

            # 修改 MTEXT 实体处理部分
            elif entity.dxftype() == 'MTEXT':
                insert = entity.dxf.insert
                text = entity.text
                height = entity.dxf.char_height if hasattr(entity.dxf, 'char_height') else 0.5
                # 使用原始高度，但根据图像整体尺寸适度调整
                font_size = height * 0.3  # 保持原始比例，调整倍数使显示效果更好
                ax.text(insert.x, insert.y, text, color=color, fontsize=font_size)
                all_x_coords.append(insert.x)
                all_y_coords.append(insert.y)
                processed_entities += 1

        # 设置坐标范围
        if all_x_coords and all_y_coords:
            margin = 10
            ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
            ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        # 保存图像
        output_dir = "dxf_output/pictures"
        os.makedirs(output_dir, exist_ok=True)
        dxf_basename = os.path.basename(dxf_file_path)
        output_filename = f"{os.path.splitext(dxf_basename)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=200, format='png')
        plt.close(fig)
        plt.close('all')

        response = {
            "status": "success",
            "message": f"DXF文件已处理并保存为 {output_path}",
            "output_path": output_path,
            "entity_stats": {
                "total_processed": processed_entities,
                "type_breakdown": entity_counts
            }
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

    except Exception as e:
        response = {"status": "error", "message": str(e)}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response
@app.route('/objects/break_all_blocks', methods=['GET'])
def break_all_blocks():
    """
    分解DXF/DWG文件中的所有块引用为基本图形元素
    """
    start_time = time.time()
    file_path = request.args.get('dxf_path')
    if not file_path:
        response = {"status": "error", "message": "缺少dxf_path参数"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(file_path):
        response = {"status": "error", "message": f"文件不存在: {file_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
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
            response = {"status": "error", "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"}
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return jsonify(response), 400

        # 读取DXF文件
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        # 统计分解前的实体数量
        original_entity_count = len(msp)

        # 记录分解的块引用数量
        blocks_broken = 0
        exploded_entities = 0
        
        # 收集所有需要分解的INSERT实体
        inserts = [entity for entity in msp if entity.dxftype() == 'INSERT']
        
        # 分解所有块引用
        for insert in inserts:
            try:
                # 使用ezdxf内置的explode方法分解块
                exploded = insert.explode()
                blocks_broken += 1
                exploded_entities += len(exploded)
            except Exception as e:
                logger.info(f"分解块 '{insert.dxf.name}' 时出错: {e}")

        # 保存修改后的DXF文件
        output_dir = "dxf_output/broken"
        os.makedirs(output_dir, exist_ok=True)
        dxf_basename = os.path.basename(dxf_file_path)
        output_filename = f"{os.path.splitext(dxf_basename)[0]}_broken.dxf"
        output_path = os.path.join(output_dir, output_filename)
        
        doc.saveas(output_path)

        response = {
            "status": "success",
            "message": f"成功分解 {blocks_broken} 个块引用",
            "original_entity_count": original_entity_count,
            "exploded_entities_count": exploded_entities,
            "output_path": output_path,
            "blocks_broken": blocks_broken
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response)

    except Exception as e:
        response = {"status": "error", "message": f"处理文件时出错: {str(e)}"}
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
def explode_all_blocks(msp):
    """
    递归分解模型空间中的所有块引用和多段线
    
    Args:
        msp: 模型空间对象
        
    Returns:
        tuple: (分解的块数量, 分解出的实体数量)
    """
    blocks_broken = 0
    exploded_entities = 0
    
    # 多次遍历直到没有更多的INSERT实体和可分解的多段线
    while True:
        # 收集所有需要分解的实体（INSERT和多段线）
        inserts = [entity for entity in msp if entity.dxftype() == 'INSERT']
        polylines = [entity for entity in msp if entity.dxftype() in ['POLYLINE', 'LWPOLYLINE']]
        
        # 如果没有需要分解的实体，则退出循环
        if not inserts and not polylines:
            break
            
        # 分解所有块引用
        for insert in inserts:
            try:
                exploded = insert.explode()
                blocks_broken += 1
                exploded_entities += len(exploded)
                logger.info(f"分解块 '{insert.dxf.name}'，获得 {len(exploded)} 个实体")
            except Exception as e:
                logger.info(f"分解块 '{insert.dxf.name}' 时出错: {e}")
                
        # 分解所有多段线
        for polyline in polylines:
            try:
                exploded = polyline.explode()
                blocks_broken += 1
                exploded_entities += len(exploded)
                logger.info(f"分解多段线，获得 {len(exploded)} 个实体")
            except Exception as e:
                logger.info(f"分解多段线时出错: {e}")
                
    return blocks_broken, exploded_entities
@app.route('/merge_dxf_files', methods=['GET'])
def merge_dxf_files():
    """
    合并文件夹内所有DXF文件到一个DXF文件中，按文件名顺序排列，并在合并前分解所有块
    
    参数:
    - folder_path: 包含DXF文件的文件夹路径
    - output_name (可选): 输出文件名，默认为 merged_dxf.dxf
    
    返回:
    - status: 状态(success/error)
    - message: 处理结果消息
    - output_path: 合并后的DXF文件路径
    - file_count: 合并的文件数量
    """
    start_time = time.time()
    folder_path = request.args.get('folder_path')
    output_name = request.args.get('output_name', 'merged_dxf.dxf')
    
    if not folder_path:
        response = {"status": "error", "message": "缺少folder_path参数"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    if not os.path.exists(folder_path):
        response = {"status": "error", "message": f"路径不存在: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 404

    if not os.path.isdir(folder_path):
        response = {"status": "error", "message": f"指定路径不是文件夹: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 400

    try:
        # 获取所有DXF文件并排序
        all_files = os.listdir(folder_path)
        dxf_files = sorted([f for f in all_files if f.lower().endswith('.dxf')])
        
        if not dxf_files:
            response = {"status": "error", "message": "文件夹中没有找到DXF文件"}
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return jsonify(response), 404
            
        # 创建新的DXF文档
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        # 初始化偏移量
        current_offset_y = 0
        max_height = 0
        merged_count = 0
        
        # 在 merge_dxf_files 中
        for filename in dxf_files:
            try:
                file_path = os.path.join(folder_path, filename)
                source_doc = ezdxf.readfile(file_path)
                source_msp = source_doc.modelspace()

                # 分解块
                blocks_broken, exploded_entities = explode_all_blocks(source_msp)
                if blocks_broken > 0:
                    source_msp = source_doc.modelspace()  # 重新获取分解后的实体

                # 计算边界框
                extents = calculate_dxf_extents(source_msp)
                if not extents:
                    continue
                min_x, min_y, max_x, max_y = extents

                # 设置偏移（左对齐，竖直堆叠）
                if merged_count == 0:
                    offset_x = -min_x
                    offset_y = -min_y
                else:
                    offset_x = -min_x
                    offset_y = current_offset_y - min_y

                # 复制实体
                copy_entities_with_offset(source_msp, msp, offset_x, offset_y)

                # 更新下一个文件的起始位置
                current_offset_y += max_y - min_y + 10  # 加间距

                merged_count += 1
            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {e}")
                continue
        
        # 保存合并后的DXF文件
        output_dir = "dxf_output/merged"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        doc.saveas(output_path)
        
        response = {
            "status": "success",
            "message": f"成功合并 {merged_count} 个DXF文件（已分解块）",
            "output_path": output_path,
            "file_count": merged_count,
            "merged_files": dxf_files
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response)
        
    except PermissionError:
        response = {"status": "error", "message": f"没有权限访问文件夹: {folder_path}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 403
    except Exception as e:
        response = {"status": "error", "message": f"合并文件时出错: {str(e)}"}
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return jsonify(response), 500

def calculate_dxf_extents(msp):
    """
    计算模型空间中所有实体的边界框
    
    Args:
        msp: 模型空间对象
        
    Returns:
        tuple: (min_x, min_y, max_x, max_y) 或 None(如果没有实体)
    """
    all_x_coords = []
    all_y_coords = []
    
    for entity in msp:
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                all_x_coords.extend([start.x, end.x])
                all_y_coords.extend([start.y, end.y])
                
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                
            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                insert = entity.dxf.insert
                all_x_coords.append(insert.x)
                all_y_coords.append(insert.y)
                
            elif entity.dxftype() == 'SPLINE':
                if hasattr(entity, 'fit_points') and entity.fit_points:
                    points = [(p.x, p.y) for p in entity.fit_points]
                elif hasattr(entity, 'control_points'):
                    points = [(p[0], p[1]) for p in entity.control_points]
                else:
                    continue
                    
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)
                
        except Exception as e:
            logger.info(f"计算实体边界时出错: {e}")
            continue
    
    if not all_x_coords or not all_y_coords:
        return None
        
    return (min(all_x_coords), min(all_y_coords), max(all_x_coords), max(all_y_coords))


def copy_entities_with_offset(source_msp, target_msp, offset_x, offset_y):
    """
    将源模型空间中的实体复制到目标模型空间，并应用偏移
    
    Args:
        source_msp: 源模型空间
        target_msp: 目标模型空间
        offset_x: X轴偏移量
        offset_y: Y轴偏移量
    """
    for entity in source_msp:
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                target_msp.add_line(
                    (start.x + offset_x, start.y + offset_y),
                    (end.x + offset_x, end.y + offset_y),
                    dxfattribs={'color': entity.dxf.color}
                )
                
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                target_msp.add_circle(
                    (center.x + offset_x, center.y + offset_y),
                    entity.dxf.radius,
                    dxfattribs={'color': entity.dxf.color}
                )
                
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                target_msp.add_arc(
                    (center.x + offset_x, center.y + offset_y),
                    entity.dxf.radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle,
                    dxfattribs={'color': entity.dxf.color}
                )
                
            elif entity.dxftype() == 'TEXT':
                insert = entity.dxf.insert
                target_msp.add_text(
                    entity.dxf.text,
                    dxfattribs={
                        'insert': (insert.x + offset_x, insert.y + offset_y),
                        'height': entity.dxf.height,
                        'color': entity.dxf.color
                    }
                )
                
            elif entity.dxftype() == 'MTEXT':
                insert = entity.dxf.insert
                target_msp.add_mtext(
                    entity.text,
                    dxfattribs={
                        'insert': (insert.x + offset_x, insert.y + offset_y),
                        'char_height':  entity.dxf.char_height*0.7 if hasattr(entity.dxf, 'char_height') else 0.5,
                        'color': entity.dxf.color
                    }
                )
                
        except Exception as e:
            logger.info(f"复制实体时出错: {e}")
            continue

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