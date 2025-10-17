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

app = Flask(__name__)

import subprocess
import os

def convert_dwg_to_dxf(dwg_file_path):
    """
    使用ODA File Converter将DWG文件转换为DXF文件
    
    Args:
        dwg_file_path (str): DWG文件路径
        
    Returns:
        dict: 转换结果，包含状态和DXF文件路径或错误信息
    """
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
                    return {
                        "status": "success",
                        "message": f"DWG文件已成功转换为DXF: {dxf_file_path}",
                        "dxf_path": dxf_file_path
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"生成的DXF文件结构不完整或损坏: {dxf_file_path}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"转换完成但未生成DXF文件。请检查输出目录: {output_dir}"
                }
        else:
            return {
                "status": "error",
                "message": f"ODA转换失败: {result.stderr}"
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "转换超时"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "未找到ODAFileConverter命令，请确保ODA File Converter已正确安装并加入PATH"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"转换过程中发生错误: {str(e)}"
        }

def is_valid_dxf(dxf_path):
    """检查DXF文件是否有效"""
    try:
        doc = ezdxf.readfile(dxf_path)
        return True
    except Exception as e:
        print(f"DXF文件校验失败: {e}")
        return False

# 在 readdxf.py 文件中添加新的API端点，放在其他@app.route装饰器附近
@app.route('/objects/all', methods=['GET'])
def get_dxf_objects():
    """
    获取DXF/DWG文件中所有对象的类型
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        JSON格式的对象类型列表和统计信息
    """
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({
            "status": "error", 
            "message": "缺少dxf_path参数"
        }), 400

    if not os.path.exists(file_path):
        return jsonify({
            "status": "error", 
            "message": f"文件不存在: {file_path}"
        }), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # 如果是DWG文件，先转换为DXF
        if file_ext == '.dwg':
            conversion_result = convert_dwg_to_dxf(file_path)
            if conversion_result["status"] != "success":
                return jsonify(conversion_result), 500
            dxf_file_path = conversion_result["dxf_path"]
        elif file_ext == '.dxf':
            dxf_file_path = file_path
        else:
            return jsonify({
                "status": "error",
                "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
            }), 400

        # 读取DXF文件并收集对象类型
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        # 收集所有对象类型
        object_types = []
        for entity in msp:
            object_types.append(entity.dxftype())

        # 统计各类对象数量
        type_count = {}
        for obj_type in object_types:
            type_count[obj_type] = type_count.get(obj_type, 0) + 1

        return jsonify({
            'status': 'success',
            'object_types': object_types,
            'object_count': len(object_types),
            'type_statistics': type_count,
            'file_path': dxf_file_path
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"处理文件时出错: {str(e)}"
        }), 500

@app.route('/objects/texts', methods=['GET'])
def get_dxf_texts():
    """
    获取DXF/DWG文件中所有文本对象的内容
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        JSON格式的文本内容列表
    """
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({
            "status": "error", 
            "message": "缺少dxf_path参数"
        }), 400

    if not os.path.exists(file_path):
        return jsonify({
            "status": "error", 
            "message": f"文件不存在: {file_path}"
        }), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # 如果是DWG文件，先转换为DXF
        if file_ext == '.dwg':
            conversion_result = convert_dwg_to_dxf(file_path)
            if conversion_result["status"] != "success":
                return jsonify(conversion_result), 500
            dxf_file_path = conversion_result["dxf_path"]
        elif file_ext == '.dxf':
            dxf_file_path = file_path
        else:
            return jsonify({
                "status": "error",
                "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
            }), 400

        # 读取DXF文件并收集文本内容
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        # 收集所有文本内容
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

        # 将所有文本内容合并为一个字符串
        all_text_content = '\n'.join([text['content'] for text in texts])

        return jsonify({
            'status': 'success',
            'texts': texts,
            'all_text_content': all_text_content,
            'text_count': len(texts),
            'file_path': dxf_file_path
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"处理文件时出错: {str(e)}"
        }), 500

def generate_entity_signature(entity):
    """
    为实体生成特征签名，用于跨文件匹配
    
    Args:
        entity: DXF实体对象
        
    Returns:
        dict: 实体的特征签名
    """
    signature = {
        "type": entity.dxftype(),
        "handle": entity.dxf.handle if hasattr(entity.dxf, 'handle') else None
    }
    
    if entity.dxftype() == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        # 为了便于匹配，对起点和终点进行排序
        points = sorted([(round(start.x, 6), round(start.y, 6)), 
                         (round(end.x, 6), round(end.y, 6))])
        signature.update({
            "start": points[0],
            "end": points[1],
            "length": round(((end.x - start.x)**2 + (end.y - start.y)**2)**0.5, 6)
        })
    elif entity.dxftype() == 'CIRCLE':
        center = entity.dxf.center
        signature.update({
            "center": (round(center.x, 6), round(center.y, 6)),
            "radius": round(entity.dxf.radius, 6)
        })
    elif entity.dxftype() == 'ARC':
        center = entity.dxf.center
        signature.update({
            "center": (round(center.x, 6), round(center.y, 6)),
            "radius": round(entity.dxf.radius, 6),
            "start_angle": round(entity.dxf.start_angle, 6),
            "end_angle": round(entity.dxf.end_angle, 6)
        })
    elif entity.dxftype() == 'SPLINE':
        signature.update({
            "degree": entity.dxf.degree if hasattr(entity.dxf, 'degree') else None,
            "fit_points_count": len(entity.fit_points) if hasattr(entity, 'fit_points') else 0,
            "control_points_count": len(entity.control_points) if hasattr(entity, 'control_points') else 0
        })
        
    return signature
@app.route('/objects/signatures', methods=['GET'])
def get_dxf_signatures():
    """
    获取DXF/DWG文件中所有对象的签名信息
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        JSON格式的对象签名列表
    """
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({
            "status": "error", 
            "message": "缺少dxf_path参数"
        }), 400

    if not os.path.exists(file_path):
        return jsonify({
            "status": "error", 
            "message": f"文件不存在: {file_path}"
        }), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # 如果是DWG文件，先转换为DXF
        if file_ext == '.dwg':
            conversion_result = convert_dwg_to_dxf(file_path)
            if conversion_result["status"] != "success":
                return jsonify(conversion_result), 500
            dxf_file_path = conversion_result["dxf_path"]
        elif file_ext == '.dxf':
            dxf_file_path = file_path
        else:
            return jsonify({
                "status": "error",
                "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
            }), 400

        # 读取DXF文件并收集对象签名
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()

        # 收集所有对象签名
        signatures = []
        for entity in msp:
            signature = generate_entity_signature(entity)
            signatures.append(signature)

        return jsonify({
            'status': 'success',
            'signatures': signatures,
            'object_count': len(signatures),
            'file_path': dxf_file_path
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"处理文件时出错: {str(e)}"
        }), 500
def process_dxf_file(dxf_file_path, highlight_random=False):
    """
    处理DXF文件并生成图像
    
    Args:
        dxf_file_path (str): DXF文件路径
        highlight_random (bool): 是否随机高亮一个对象
        
    Returns:
        dict: 处理结果
    """
    # DXF颜色索引到matplotlib颜色的映射
    dxf_color_map = {
        0: 'black',      # 黑色
        1: 'red',        # 红色
        2: 'yellow',     # 黄色
        3: 'green',      # 绿色
        4: 'cyan',       # 青色
        5: 'blue',       # 蓝色
        6: 'magenta',    # 洋红色
        7: 'white',      # 白色
        8: '#a5a5a5',    # 灰色
        9: '#c0c0c0',    # 浅灰
        10: 'red',       # 红色
        11: '#ffaaaa',   # 粉红色
        12: '#bd0000',   # 深红色
        13: '#bd7373',   # 玫瑰色
        14: '#800000',   # 棕红色
        15: '#ff0000',   # 鲜红色
        16: '#ffff00',   # 黄色
        17: '#ffff73',   # 金黄色
        18: '#bda000',   # 深黄色
        19: '#bdae73',   # 橄榄绿
        20: '#808000',   # 深橄榄绿
        21: '#ffff00',   # 鲜黄色
        22: '#00ff00',   # 绿色
        23: '#aaffaa',   # 浅绿色
        24: '#00bd00',   # 深绿色
        25: '#73bd73',   # 海绿色
        26: '#008000',   # 深绿
        27: '#00ff00',   # 鲜绿色
        28: '#00ffff',   # 青色
        29: '#aaffff',   # 浅青色
        30: '#00bfbf',   # 深青色
        31: '#73bfbf',   # 青绿
        32: '#008080',   # 深青绿
        33: '#00ffff',   # 鲜青色
        34: '#0000ff',   # 蓝色
        35: '#aaaaff',   # 浅蓝色
        36: '#0000bd',   # 深蓝色
        37: '#7373bf',   # 紫蓝色
        38: '#000080',   # 深紫蓝
        39: '#0000ff',   # 鲜蓝色
        40: '#ff00ff',   # 洋红色
        41: '#ffaaff',   # 浅洋红
        42: '#bd00bd',   # 深洋红
        43: '#bd73bd',   # 紫色
        44: '#800080',   # 深紫
        45: '#ff00ff',   # 鲜洋红
        # 默认颜色
        'default': 'black'
    }

    try:
        # 检查文件是否存在
        if not os.path.exists(dxf_file_path):
            return {
                "status": "error",
                "message": f"文件不存在: {dxf_file_path}"
            }

        # 加载DXF文件
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()  # 获取模型空间

        # 创建图形对象
        fig, ax = plt.subplots()
        
        # 获取所有可显示的实体列表（只有LINE, SPLINE, ARC, CIRCLE可显示）
        displayable_entities = []
        for entity in msp:
            if entity.dxftype() in ['LINE', 'SPLINE', 'ARC', 'CIRCLE']:
                displayable_entities.append(entity)
        
        # 如果需要高亮且存在可显示实体，则随机选择一个
        highlighted_entity = None
        if highlight_random and displayable_entities:
            highlighted_entity = np.random.choice(displayable_entities)

        lines = []
        line_colors = []
        highlighted_elements = []
        
        # 存储所有坐标点用于设置视图范围
        all_x_coords = []
        all_y_coords = []

        highlighted_entity_signature = None

        for entity in msp:
            # 判断当前实体是否为高亮实体
            is_highlighted = (entity is highlighted_entity)
            linewidth = 1.5 if is_highlighted else 0.5

            # 获取实体颜色
            try:
                # 尝试获取实体的颜色
                if hasattr(entity.dxf, 'color'):
                    dxf_color = entity.dxf.color
                    # 使用DXF颜色索引映射到实际颜色
                    if dxf_color is not None and dxf_color in dxf_color_map:
                        actual_color = dxf_color_map[dxf_color]
                    else:
                        actual_color = dxf_color_map['default']  # 默认黑色
                else:
                    actual_color = dxf_color_map['default']
            except:
                actual_color = dxf_color_map['default']  # 出错时使用默认黑色
            
            # 如果是高亮状态，使用红色
            color = 'red' if is_highlighted else actual_color

            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                if is_highlighted:
                    # 单独绘制高亮线段
                    ax.plot([start.x, end.x], [start.y, end.y], color=color, linewidth=linewidth)
                    # 生成高亮实体签名
                    highlighted_entity_signature = generate_entity_signature(entity)
                else:
                    # 添加到普通线段集合
                    lines.append([(start.x, start.y), (end.x, end.y)])
                    line_colors.append(color)
                
                # 记录坐标用于计算范围
                all_x_coords.extend([start.x, end.x])
                all_y_coords.extend([start.y, end.y])
                    
            elif entity.dxftype() == 'SPLINE':
                try:
                    # 获取拟合点或控制点
                    if hasattr(entity, 'fit_points') and entity.fit_points:
                        points = [(p.x, p.y) for p in entity.fit_points]
                    else:
                        points = [(p[0], p[1]) for p in entity.control_points]
                        
                    if is_highlighted:
                        # 单独绘制高亮样条曲线
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)
                        # 生成高亮实体签名
                        highlighted_entity_signature = generate_entity_signature(entity)
                    else:
                        # 对于普通样条曲线，只绘制起点和终点之间的连线
                        if len(points) >= 2:
                            lines.append([points[0], points[-1]])
                            line_colors.append(color)
                            
                    # 记录坐标用于计算范围
                    all_x_coords.extend([p[0] for p in points])
                    all_y_coords.extend([p[1] for p in points])
                except Exception as e:
                    print(f'处理SPLINE实体时出错: {e}')
                    
            elif entity.dxftype() == 'ARC':
                arc = entity
                center = arc.dxf.center
                radius = arc.dxf.radius
                start_angle = arc.dxf.start_angle
                end_angle = arc.dxf.end_angle
                
                # 处理圆弧角度
                if start_angle > end_angle:
                    end_angle += 360
                    
                if is_highlighted:
                    # 单独绘制高亮圆弧
                    arc_patch = Arc((center.x, center.y), 2*radius, 2*radius, angle=0,
                                    theta1=start_angle, theta2=end_angle,
                                    color=color, linewidth=linewidth)
                    ax.add_patch(arc_patch)
                    # 生成高亮实体签名
                    highlighted_entity_signature = generate_entity_signature(entity)
                else:
                    # 普通圆弧用原始颜色绘制
                    arc_patch = Arc((center.x, center.y), 2*radius, 2*radius, angle=0,
                                    theta1=start_angle, theta2=end_angle,
                                    color=color, linewidth=0.5)
                    ax.add_patch(arc_patch)
                    
                # 记录坐标用于计算范围（近似）
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                    
            elif entity.dxftype() == 'CIRCLE':
                circle = entity
                center = circle.dxf.center
                radius = circle.dxf.radius
                
                if is_highlighted:
                    # 单独绘制高亮圆形
                    circle_patch = plt.Circle((center.x, center.y), radius, fill=False, 
                                            color=color, linewidth=linewidth)
                    ax.add_patch(circle_patch)
                    # 生成高亮实体签名
                    highlighted_entity_signature = generate_entity_signature(entity)
                else:
                    # 普通圆形用原始颜色绘制
                    circle_patch = plt.Circle((center.x, center.y), radius, fill=False, 
                                            color=color, linewidth=0.5)
                    ax.add_patch(circle_patch)
                    
                # 记录坐标用于计算范围
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
            else:
                # 其他实体类型虽然处理了但没有可视化
                print('未处理的实体类型:', entity.dxftype())

        # 绘制普通线条（使用原始颜色）
        if lines:
            line_segments = LineCollection(lines, linewidths=0.5, colors=line_colors if line_colors else 'black')
            ax.add_collection(line_segments)

        # 设置坐标范围
        if all_x_coords and all_y_coords:
            margin = 5
            ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
            ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
        else:
            # 如果没有任何实体，设置默认范围
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        plt.gca().set_aspect('equal', adjustable='box')
        output_path = f"{dxf_file_path}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)  # 添加bbox_inches='tight'确保完整保存
        plt.close(fig)

        result = {
            "status": "success",
            "message": f"DXF文件已处理并保存为 {output_path}",
            "output_path": output_path
        }
        
        # 如果有高亮实体，返回其类型和特征信息及签名
        if highlighted_entity:
            entity_info = {
                "type": highlighted_entity.dxftype(),
                "signature": highlighted_entity_signature
            }
            
            # 添加不同类型实体的特征信息
            if highlighted_entity.dxftype() == 'LINE':
                start = highlighted_entity.dxf.start
                end = highlighted_entity.dxf.end
                entity_info["start"] = {"x": start.x, "y": start.y}
                entity_info["end"] = {"x": end.x, "y": end.y}
                entity_info["length"] = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5
                
            elif highlighted_entity.dxftype() == 'CIRCLE':
                center = highlighted_entity.dxf.center
                entity_info["center"] = {"x": center.x, "y": center.y}
                entity_info["radius"] = highlighted_entity.dxf.radius
                
            elif highlighted_entity.dxftype() == 'ARC':
                center = highlighted_entity.dxf.center
                entity_info["center"] = {"x": center.x, "y": center.y}
                entity_info["radius"] = highlighted_entity.dxf.radius
                entity_info["start_angle"] = highlighted_entity.dxf.start_angle
                entity_info["end_angle"] = highlighted_entity.dxf.end_angle
                
            elif highlighted_entity.dxftype() == 'SPLINE':
                if hasattr(highlighted_entity, 'fit_points') and highlighted_entity.fit_points:
                    entity_info["fit_points_count"] = len(highlighted_entity.fit_points)
                if hasattr(highlighted_entity, 'control_points'):
                    entity_info["control_points_count"] = len(highlighted_entity.control_points)
                    
            result["highlighted_entity"] = entity_info

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
@app.route('/process_dxf', methods=['GET'])
def handle_process_dxf():
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({"status": "error", "message": "缺少dxf_path参数"}), 400

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"文件不存在: {file_path}"}), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return jsonify(conversion_result), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return jsonify({
            "status": "error",
            "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
        }), 400

    result = process_dxf_file(dxf_file_path)
    status_code = 200 if result["status"] == "success" else 500
    return jsonify(result), status_code


@app.route('/view_dxf', methods=['GET'])
def view_dxf():
    file_path = request.args.get('dxf_path')
    if not file_path:
        return render_template_string(HTML_TEMPLATE, title="错误", message="缺少dxf_path参数", image_url=None), 400

    if not os.path.exists(file_path):
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"文件不存在: {file_path}", image_url=None), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return render_template_string(HTML_TEMPLATE, title="转换失败", message=conversion_result["message"], image_url=None), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。", image_url=None), 400

    result = process_dxf_file(dxf_file_path)
    if result["status"] == "success":
        image_path = result["output_path"]
        image_url = f"/image?path={image_path}"
        return render_template_string(HTML_TEMPLATE, title="文件处理结果", message=result["message"], image_url=image_url)
    else:
        return render_template_string(HTML_TEMPLATE, title="处理失败", message=result["message"], image_url=None), 500


@app.route('/image', methods=['GET'])
def get_image():
    image_path = request.args.get('path')
    if not image_path or not os.path.exists(image_path):
        return "图像不存在", 404
    return send_file(image_path, mimetype='image/png')


# 新增：随机高亮一个对象并返回JSON结果
@app.route('/highlight_random', methods=['GET'])
def highlight_random_object():
    """
    随机高亮DXF/DWG文件中的一个对象并生成图像
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        JSON格式的处理结果
    """
    file_path = request.args.get('dxf_path')
    if not file_path:
        return jsonify({"status": "error", "message": "缺少dxf_path参数"}), 400

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"文件不存在: {file_path}"}), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return jsonify(conversion_result), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return jsonify({
            "status": "error",
            "message": f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。"
        }), 400

    result = process_dxf_file(dxf_file_path, highlight_random=True)
    status_code = 200 if result["status"] == "success" else 500
    return jsonify(result), status_code


# 新增：在网页中查看随机高亮结果
@app.route('/view_highlighted', methods=['GET'])
def view_highlighted_dxf():
    """
    在网页中查看随机高亮对象的DXF/DWG文件
    
    Query Parameters:
        dxf_path: DXF或DWG文件路径
        
    Returns:
        HTML页面显示处理结果
    """
    file_path = request.args.get('dxf_path')
    if not file_path:
        return render_template_string(HTML_TEMPLATE, title="错误", message="缺少dxf_path参数", image_url=None), 400

    if not os.path.exists(file_path):
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"文件不存在: {file_path}", image_url=None), 404

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.dwg':
        conversion_result = convert_dwg_to_dxf(file_path)
        if conversion_result["status"] != "success":
            return render_template_string(HTML_TEMPLATE, title="转换失败", message=conversion_result["message"], image_url=None), 500
        dxf_file_path = conversion_result["dxf_path"]
    elif file_ext == '.dxf':
        dxf_file_path = file_path
    else:
        return render_template_string(HTML_TEMPLATE, title="错误", message=f"不支持的文件格式: {file_ext}。仅支持DXF和DWG文件。", image_url=None), 400

    result = process_dxf_file(dxf_file_path, highlight_random=True)
    if result["status"] == "success":
        image_path = result["output_path"]
        image_url = f"/image?path={image_path}"
        highlighted_msg = f"随机高亮的对象类型: {result['highlighted_entity']}" if result['highlighted_entity'] else "未找到可高亮的对象"
        message = f"{result['message']}<br>{highlighted_msg}"
        return render_template_string(HTML_TEMPLATE, title="高亮对象处理结果", message=message, image_url=image_url)
    else:
        return render_template_string(HTML_TEMPLATE, title="处理失败", message=result["message"], image_url=None), 500

@app.route('/dxf_files', methods=['GET'])
def get_dxf_files():
    """
    获取指定文件夹下的所有DXF文件名称
    
    Query Parameters:
        folder_path: 文件夹路径（可选，默认为当前工作目录）
        
    Returns:
        JSON格式的DXF文件列表
    """
    folder_path = request.args.get('folder_path', os.getcwd())
    
    if not os.path.exists(folder_path):
        return jsonify({
            "status": "error",
            "message": f"文件夹不存在: {folder_path}"
        }), 404
        
    if not os.path.isdir(folder_path):
        return jsonify({
            "status": "error",
            "message": f"路径不是文件夹: {folder_path}"
        }), 400
    
    try:
        # 获取文件夹下所有文件
        all_files = os.listdir(folder_path)
        
        # 筛选出DXF文件（包括大写和小写扩展名）
        dxf_files = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f)) 
                     and os.path.splitext(f)[1].lower() == '.dxf']
        
        return jsonify({
            "status": "success",
            "folder_path": folder_path,
            "dxf_files": dxf_files,
            "count": len(dxf_files)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"读取文件夹时出错: {str(e)}"
        }), 500

# HTML模板
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
        "message": "DXF/DWG处理服务已启动",
        "endpoints": {
            "process_dxf": "GET /process_dxf?dxf_path=<path_to_dxf_or_dwg_file> (返回JSON结果)",
            "view_dxf": "GET /view_dxf?dxf_path=<path_to_dxf_or_dwg_file> (在网页中查看图像)",
            "highlight_random": "GET /highlight_random?dxf_path=<path_to_dxf_or_dwg_file> (随机高亮一个对象并返回JSON结果)",
            "view_highlighted": "GET /view_highlighted?dxf_path=<path_to_dxf_or_dwg_file> (在网页中查看随机高亮结果)",
            "image": "GET /image?path=<path_to_image> (获取图像文件)",
            "objects_all": "GET /objects/all?dxf_path=<path_to_dxf_or_dwg_file> (获取所有对象类型)",
            "objects_signatures": "GET /objects/signatures?dxf_path=<path_to_dxf_or_dwg_file> (获取所有对象签名)",
            "dxf_files": "GET /dxf_files?folder_path=<path_to_folder> (获取文件夹下所有DXF文件)"
        },
        "notes": "支持DXF和DWG文件格式。DWG文件会自动转换为DXF格式后再处理。",
        "example": "GET /view_dxf?dxf_path=C:\\test\\example.dwg"
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5301, debug=True)