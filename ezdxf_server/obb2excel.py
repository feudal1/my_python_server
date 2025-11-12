import os
import tkinter as tk
from tkinter import filedialog
import ezdxf
import pandas as pd
import math
import re
import subprocess
import platform
import sys

def get_points_from_dxf(doc):
    """
    从DXF文档中提取线段端点、圆弧采样点、多段线顶点和圆的采样点
    """
    points = []
    modelspace = doc.modelspace()
    
    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            # 正确访问Vec3对象的x,y属性
            start = (entity.dxf.start.x, entity.dxf.start.y)
            end = (entity.dxf.end.x, entity.dxf.end.y)
            points.append(start)
            points.append(end)
        elif entity.dxftype() == 'ARC':
            # 将圆弧拆分成20个点
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            
            # 确保角度范围正确
            if end_angle < start_angle:
                end_angle += 360
                
            # 在圆弧上均匀采样20个点
            for i in range(20):
                angle = start_angle + (end_angle - start_angle) * i / 19
                x = center[0] + radius * math.cos(math.radians(angle))
                y = center[1] + radius * math.sin(math.radians(angle))
                points.append((x, y))
        elif entity.dxftype() == 'CIRCLE':
            # 将圆拆分成40个点（更精细）
            center = (entity.dxf.center.x, entity.dxf.center.y)
            radius = entity.dxf.radius
            
            for i in range(40):
                angle = 360 * i / 40
                x = center[0] + radius * math.cos(math.radians(angle))
                y = center[1] + radius * math.sin(math.radians(angle))
                points.append((x, y))
        elif entity.dxftype() == 'LWPOLYLINE':
            # 轻量多段线，顶点存储在 `entity.vertices` 中
            with entity.points() as vertices:
                for vertex in vertices:
                    points.append((vertex[0], vertex[1]))
        elif entity.dxftype() == 'POLYLINE':
            # 传统多段线，需要遍历子实体
            for vertex in entity.vertices:
                point = vertex.dxf.location
                points.append((point.x, point.y))

    # 去除重复点
    points = list(set(points))
    return points

def rotate_point(point, angle, center=(0, 0)):
    """
    绕指定中心点旋转点
    """
    x, y = point
    cx, cy = center
    
    x -= cx
    y -= cy
    
    rad = math.radians(angle)
    cos_rad, sin_rad = math.cos(rad), math.sin(rad)
    new_x = x * cos_rad - y * sin_rad
    new_y = x * sin_rad + y * cos_rad
    
    new_x += cx
    new_y += cy
    
    return (new_x, new_y)

def get_bounding_box_area(points, angle):
    """
    计算给定角度下的包围盒面积和边界信息
    """
    rotated_points = [rotate_point(p, -angle) for p in points]
    
    if not rotated_points:
        return float('inf'), (0, 0, 0, 0)
    
    x_coords = [p[0] for p in rotated_points]
    y_coords = [p[1] for p in rotated_points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return area, (min_x, max_x, min_y, max_y)

def ternary_search_min_area(points, left, right, eps=1e-6):
    """
    使用三分法搜索最小面积角度
    """
    while right - left > eps:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        area1, _ = get_bounding_box_area(points, mid1)
        area2, _ = get_bounding_box_area(points, mid2)
        
        if area1 < area2:
            right = mid2
        else:
            left = mid1
    
    optimal_angle = (left + right) / 2
    min_area, bounds = get_bounding_box_area(points, optimal_angle)
    return optimal_angle, min_area, bounds

def get_oriented_bounding_box_approx(points):
    """
    使用三分搜索获取最小面积包围矩形
    """
    if len(points) < 2:
        return None
    
    # 首先找到一个较好的初始角度范围
    angles_to_check = []
    
    # 使用较小步长进行初步搜索
    for i in range(0, 180, 2):
        angles_to_check.append(i)
    
    # 找到距离最远的点对
    max_dist = 0
    farthest_pair = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            if dist > max_dist:
                max_dist = dist
                farthest_pair = (points[i], points[j])
    
    if farthest_pair:
        p1, p2 = farthest_pair
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        angles_to_check.extend([angle, angle + 90])
    
    angles_to_check = list(set([a % 180 for a in angles_to_check]))
    
    # 找到初步的最小面积角度
    min_area = float('inf')
    best_angle = 0
    best_bounds = None
    
    for angle in angles_to_check:
        area, bounds = get_bounding_box_area(points, angle)
        if area < min_area:
            min_area = area
            best_angle = angle
            best_bounds = bounds
    
    # 在最优角度附近使用三分法进行精细搜索
    search_range = 5  # 搜索范围±5度
    left_angle = (best_angle - search_range) % 180
    right_angle = (best_angle + search_range) % 180
    
    # 处理跨越0度的情况
    if left_angle > right_angle:
        # 在[0, right_angle]和[left_angle, 180]两个区间分别搜索
        try:
            optimal_angle1, min_area1, bounds1 = ternary_search_min_area(points, 0, right_angle)
        except:
            optimal_angle1, min_area1, bounds1 = best_angle, min_area, best_bounds
            
        try:
            optimal_angle2, min_area2, bounds2 = ternary_search_min_area(points, left_angle, 180)
        except:
            optimal_angle2, min_area2, bounds2 = best_angle, min_area, best_bounds
        
        if min_area1 < min_area2:
            optimal_angle = optimal_angle1
            min_area = min_area1
            best_bounds = bounds1
        else:
            optimal_angle = optimal_angle2
            min_area = min_area2
            best_bounds = bounds2
    else:
        try:
            optimal_angle, min_area, best_bounds = ternary_search_min_area(points, left_angle, right_angle)
        except:
            optimal_angle, min_area, best_bounds = best_angle, min_area, best_bounds
    
    # 构造最终的包围盒
    if best_bounds is None:
        return None
        
    min_x, max_x, min_y, max_y = best_bounds
    width = max_x - min_x
    height = max_y - min_y
    
    return {
        'width': width,
        'height': height,
        'area': min_area,
        'angle': optimal_angle
    }

def extract_quantity_from_filename(filename):
    """
    从文件名中提取数量信息
    例如: 三宝7-4PL20x570x4810=12个攻牙.dxf -> 12
    """
    # 使用正则表达式匹配文件名中的数字（在等号后）
    match = re.search(r'=([0-9]+)', filename)
    if match:
        return int(match.group(1))
    return 1  # 默认数量为1

def process_dxf_files(file_paths):
    """
    处理多个DXF文件并获取OBB信息
    """
    results = []
    
    for file_path in file_paths:
        try:
            # 获取文件名
            file_name = os.path.basename(file_path)
            
            # 从文件名提取数量信息
            quantity = extract_quantity_from_filename(file_name)
            
            # 读取DXF文件
            doc = ezdxf.readfile(file_path)
            
            # 提取点
            points = get_points_from_dxf(doc)
            
            if len(points) < 2:
                results.append({
                    '文件名': file_name,
                    'OBB长度': 0,
                    'OBB宽度': 0,
                    '数量': quantity
                })
                continue
            
            # 计算OBB
            obb = get_oriented_bounding_box_approx(points)
            
            if obb:
                # 确保长度>=宽度
                length = max(obb['width'], obb['height'])
                width = min(obb['width'], obb['height'])
                
                results.append({
                    '文件名': file_name,
                    'OBB长度': round(length, 3),
                    'OBB宽度': round(width, 3),
                    '数量': quantity
                })
            else:
                results.append({
                    '文件名': file_name,
                    'OBB长度': 0,
                    'OBB宽度': 0,
                    '数量': quantity
                })
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            results.append({
                '文件名': os.path.basename(file_path),
                'OBB长度': '错误',
                'OBB宽度': '错误',
                '数量': '错误'
            })
    
    return results

def select_dxf_files():
    """
    弹出文件选择对话框选择DXF文件
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_paths = filedialog.askopenfilenames(
        title="选择DXF文件",
        filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
    )
    
    return list(file_paths)

def export_to_csv(data, output_path):
    """
    将数据导出到CSV文件
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

def open_file(filepath):
    """
    跨平台打开文件
    """
    system = platform.system()
    try:
        if system == "Windows":
            # 使用 subprocess.Popen 并分离进程，避免随主程序退出而关闭
            subprocess.Popen(['start', filepath], shell=True)
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", filepath])
        else:  # Linux
            subprocess.Popen(["xdg-open", filepath])
        return True
    except Exception as e:
        print(f"无法自动打开文件: {e}")
        return False

def main():
    """
    主函数
    """
    print("请选择DXF文件...")
    
    # 选择DXF文件
    dxf_files = select_dxf_files()
    
    if not dxf_files:
        print("未选择任何文件")
        return
    
    print(f"已选择 {len(dxf_files)} 个文件")
    
    # 处理文件并获取OBB信息
    print("正在处理文件...")
    results = process_dxf_files(dxf_files)
    
    # 导出到CSV - 保存在第一个DXF文件的目录下
    if dxf_files:
        output_dir = os.path.dirname(dxf_files[0])
        output_file = os.path.join(output_dir, "obb_results.csv")
    else:
        output_file = "obb_results.csv"
    
    export_to_csv(results, output_file)
    
    print(f"结果已导出到: {output_file}")
    
    # 显示结果
    print("\n处理结果:")
    for result in results:
        print(f"文件: {result['文件名']}, "
              f"OBB长度: {result['OBB长度']}, "
              f"OBB宽度: {result['OBB宽度']}, "
              f"数量: {result['数量']}")
    
    # 自动打开生成的CSV文件
    print(f"\n正在打开文件: {output_file}")
    if open_file(output_file):
        print("文件已成功打开，请查看。")
    else:
        print("文件打开失败，请手动打开。")
    
    # 等待用户输入后再退出，防止窗口闪退
    input("\n按回车键退出程序...")

if __name__ == "__main__":
    main()