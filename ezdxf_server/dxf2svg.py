# dxf_to_svg.py
import ezdxf
import svgwrite
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import math
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_dxf_files():
    """
    使用弹窗选择一个或多个DXF文件
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，支持多选
    file_paths = filedialog.askopenfilenames(
        title="选择要转换的DXF文件",
        filetypes=[
            ("DXF files", "*.dxf"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return list(file_paths)  # 转换为列表

def dxf_to_svg(dxf_path, svg_path):
    """
    将DXF文件转换为SVG文件
    
    Args:
        dxf_path: DXF文件路径
        svg_path: 输出SVG文件路径
    """
    # 读取DXF文件
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # 计算边界框
    extents = calculate_dxf_extents(msp)
    if not extents:
        raise ValueError("DXF文件中没有有效的实体")
    
    min_x, min_y, max_x, max_y = extents
    width = max_x - min_x
    height = max_y - min_y
    
    # 创建SVG绘图对象
    # 注意：SVG的坐标系与DXF不同，需要进行转换
    dwg = svgwrite.Drawing(svg_path, size=(f"{width}px", f"{height}px"), 
                          viewBox=(f"{min_x} {min_y} {width} {height}"))
    
    # 设置SVG属性
    dwg.attribs['xmlns:xlink'] = "http://www.w3.org/1999/xlink"
    
    # 处理实体
    entity_count = 0
    entity_stats = {}
    
    for entity in msp:
        try:
            entity_type = entity.dxftype()
            entity_stats[entity_type] = entity_stats.get(entity_type, 0) + 1
            
            if entity_type == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                line = dwg.line(start=(start[0], start[1]), 
                               end=(end[0], end[1]),
                               stroke='black')
                # 设置线宽（如果存在）
                if hasattr(entity.dxf, 'lineweight') and entity.dxf.lineweight != ezdxf.const.LINEWEIGHT_DEFAULT:
                    line.attribs['stroke-width'] = str(entity.dxf.lineweight / 100)  # 转换为像素
                dwg.add(line)
                entity_count += 1
                
            elif entity_type == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                circle = dwg.circle(center=(center[0], center[1]), 
                                   r=radius,
                                   stroke='black', fill='none')
                dwg.add(circle)
                entity_count += 1
                
            elif entity_type == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = math.radians(entity.dxf.start_angle)
                end_angle = math.radians(entity.dxf.end_angle)
                
                # 计算起点和终点
                start_point = (center[0] + radius * math.cos(start_angle),
                              center[1] + radius * math.sin(start_angle))
                end_point = (center[0] + radius * math.cos(end_angle),
                            center[1] + radius * math.sin(end_angle))
                
                # 判断是大弧还是小弧
                angle_diff = (entity.dxf.end_angle - entity.dxf.start_angle) % 360
                large_arc_flag = 1 if angle_diff > 180 else 0
                
                # SVG路径数据
                path_data = f"M {start_point[0]},{start_point[1]} " \
                           f"A {radius},{radius} 0 {large_arc_flag},1 {end_point[0]},{end_point[1]}"
                
                path = dwg.path(d=path_data, fill='none', stroke='black')
                dwg.add(path)
                entity_count += 1
                
            elif entity_type == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in entity.vertices()]
                if len(points) > 1:
                    if entity.closed:
                        # 闭合多段线
                        polyline = dwg.polyline(points, stroke='black', fill='none')
                        polyline.attribs['fill'] = 'none'
                    else:
                        # 开放多段线
                        polyline = dwg.polyline(points, stroke='black', fill='none')
                    dwg.add(polyline)
                    entity_count += 1
                    
            elif entity_type == 'POLYLINE':
                # 处理复杂多段线
                points = []
                for vertex in entity.vertices:
                    if vertex.dxftype() == 'VERTEX':
                        points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                
                if len(points) > 1:
                    if entity.is_closed:
                        polyline = dwg.polyline(points, stroke='black', fill='none')
                        polyline.attribs['fill'] = 'none'
                    else:
                        polyline = dwg.polyline(points, stroke='black', fill='none')
                    dwg.add(polyline)
                    entity_count += 1
                    
            elif entity_type == 'ELLIPSE':
                # 椭圆处理 - 使用SVG路径近似
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                
                # 简化处理：使用椭圆元素
                ellipse = dwg.ellipse(center=(center[0], center[1]),
                                     r=(major_axis.magnitude, major_axis.magnitude * ratio),
                                     stroke='black', fill='none')
                dwg.add(ellipse)
                entity_count += 1
                
            elif entity_type == 'SPLINE':
                # 样条曲线处理 - 简化为折线
                try:
                    # 将样条曲线近似为多段线
                    points = entity.approximate(20)  # 获取近似点
                    if points:
                        points = [(p[0], p[1]) for p in points]
                        if len(points) > 1:
                            polyline = dwg.polyline(points, stroke='black', fill='none')
                            dwg.add(polyline)
                            entity_count += 1
                except Exception as e:
                    logging.warning(f"处理SPLINE实体时出错: {e}")
                    
            elif entity_type == 'TEXT':
                insert = entity.dxf.insert
                text_content = entity.dxf.text
                if isinstance(text_content, bytes):
                    text_content = text_content.decode('utf-8')
                
                # 添加文本
                txt = dwg.text(text_content, insert=(insert[0], insert[1]), 
                              fill='black')
                # 设置字体大小
                if hasattr(entity.dxf, 'height'):
                    txt.attribs['font-size'] = str(entity.dxf.height)
                dwg.add(txt)
                entity_count += 1
                
            elif entity_type == 'MTEXT':
                insert = entity.dxf.insert
                text_content = entity.text
                if isinstance(text_content, bytes):
                    text_content = text_content.decode('utf-8')
                
                # 添加多行文本
                txt = dwg.text(text_content, insert=(insert[0], insert[1]), 
                              fill='black')
                # 设置字体大小
                if hasattr(entity.dxf, 'char_height'):
                    txt.attribs['font-size'] = str(entity.dxf.char_height)
                dwg.add(txt)
                entity_count += 1
                
            else:
                logging.warning(f"未支持的实体类型: {entity_type}")
                
        except Exception as e:
            logging.warning(f"转换实体 {entity.dxftype()} 时出错: {e}")
            continue
    
    logging.info(f"实体统计信息: {entity_stats}")
    
    # 保存SVG文件
    dwg.save()
    return entity_count

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
                all_x_coords.extend([start[0], end[0]])
                all_y_coords.extend([start[1], end[1]])
                
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center[0] - radius, center[0] + radius])
                all_y_coords.extend([center[1] - radius, center[1] + radius])
                
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center[0] - radius, center[0] + radius])
                all_y_coords.extend([center[1] - radius, center[1] + radius])
                
            elif entity.dxftype() == 'ELLIPSE':
                center = entity.dxf.center
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                major_radius = major_axis.magnitude
                minor_radius = major_radius * ratio
                all_x_coords.extend([center[0] - major_radius, center[0] + major_radius])
                all_y_coords.extend([center[1] - minor_radius, center[1] + minor_radius])
                
            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                insert = entity.dxf.insert
                all_x_coords.append(insert[0])
                all_y_coords.append(insert[1])
                
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.vertices())
                for point in points:
                    if len(point) >= 2:
                        all_x_coords.append(point[0])
                        all_y_coords.append(point[1])
                        
            elif entity.dxftype() == 'POLYLINE':
                for vertex in entity.vertices:
                    if vertex.dxftype() == 'VERTEX':
                        location = vertex.dxf.location
                        all_x_coords.append(location.x)
                        all_y_coords.append(location.y)
                        
            elif entity.dxftype() == 'SPLINE':
                try:
                    # 近似样条曲线的边界
                    points = entity.approximate(20)
                    for point in points:
                        all_x_coords.append(point[0])
                        all_y_coords.append(point[1])
                except Exception as e:
                    logging.warning(f"计算SPLINE边界时出错: {e}")
                        
        except Exception as e:
            logging.warning(f"计算实体边界时出错: {e}")
            continue
    
    if not all_x_coords or not all_y_coords:
        return None
        
    return (min(all_x_coords), min(all_y_coords), max(all_x_coords), max(all_y_coords))

def main():
    """
    主函数：图形界面转换DXF文件为SVG（支持多文件）
    """
    try:
        # 选择DXF文件（支持多选）
        print("请选择要转换的DXF文件...")
        dxf_files = select_dxf_files()
        
        if not dxf_files:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("提示", "未选择任何文件")
            root.destroy()
            return
        
        print(f"已选择 {len(dxf_files)} 个文件:")
        for i, file in enumerate(dxf_files, 1):
            print(f"{i}. {file}")
        
        total_converted = 0
        conversion_results = []
        
        # 批量转换文件
        for dxf_file in dxf_files:
            try:
                print(f"\n正在转换: {dxf_file}")
                
                # 确定输出SVG文件路径
                base_name = os.path.splitext(dxf_file)[0]
                svg_file = base_name + ".svg"
                
                # 执行转换操作
                entity_count = dxf_to_svg(dxf_file, svg_file)
                total_converted += 1
                
                # 记录结果
                result_message = f"成功转换 {entity_count} 个实体"
                conversion_results.append({
                    'dxf': dxf_file,
                    'svg': svg_file,
                    'entities': entity_count,
                    'status': 'success'
                })
                
                print(f"  -> {result_message}")
                
            except Exception as e:
                error_msg = f"转换文件 {dxf_file} 时出错: {str(e)}"
                print(error_msg)
                logging.error(error_msg)
                
                conversion_results.append({
                    'dxf': dxf_file,
                    'error': str(e),
                    'status': 'error'
                })
        
        # 显示总体结果
        success_count = sum(1 for r in conversion_results if r['status'] == 'success')
        error_count = len(conversion_results) - success_count
        
        summary_message = f"批量转换完成!\n\n" \
                         f"总计文件数: {len(dxf_files)}\n" \
                         f"成功转换: {success_count}\n" \
                         f"转换失败: {error_count}"
        
        print(f"\n{summary_message}")
        
        # 显示详细结果对话框
        detail_message = summary_message + "\n\n详细信息:\n"
        for result in conversion_results:
            if result['status'] == 'success':
                detail_message += f"✓ {os.path.basename(result['dxf'])} -> {result['entities']} 个实体\n"
            else:
                detail_message += f"✗ {os.path.basename(result['dxf'])} -> 错误: {result['error']}\n"
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("批量转换完成", detail_message)
        root.destroy()
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        
        # 显示错误对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", error_msg)
        root.destroy()

if __name__ == "__main__":
    main()