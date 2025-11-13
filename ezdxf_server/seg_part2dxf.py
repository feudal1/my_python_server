# cluster_obb_export.py
import ezdxf
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import tkinter as tk
from tkinter import filedialog

def get_points_from_lines(lines):
    """从线段中提取所有点"""
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
    return list(set(points))

def get_aabb_bounding_box(points):
    """获取轴对齐的最小包围矩形"""
    if len(points) < 1:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    
    return {
        'type': 'AABB',
        'corners': corners,
        'width': width,
        'height': height,
        'area': width * height,
        'angle': 0,
        'center': ((min_x + max_x) / 2, (min_y + max_y) / 2)
    }

def rotate_point(point, angle, center=(0, 0)):
    """绕指定中心点旋转点"""
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
    """计算给定角度下的包围盒面积和边界信息"""
    rotated_points = [rotate_point(p, -angle) for p in points]
    
    x_coords = [p[0] for p in rotated_points]
    y_coords = [p[1] for p in rotated_points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return area, (min_x, max_x, min_y, max_y)

def ternary_search_min_area(points, left, right, eps=1e-6):
    """使用三分法搜索最小面积角度"""
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
    """使用三分搜索获取最小面积包围矩形"""
    if len(points) < 2:
        return None
    
    angles_to_check = []
    
    for i in range(0, 180, 2):
        angles_to_check.append(i)
    
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
    
    min_area = float('inf')
    best_angle = 0
    best_bounds = None
    
    for angle in angles_to_check:
        area, bounds = get_bounding_box_area(points, angle)
        if area < min_area:
            min_area = area
            best_angle = angle
            best_bounds = bounds
    
    search_range = 5
    left_angle = (best_angle - search_range) % 180
    right_angle = (best_angle + search_range) % 180
    
    if left_angle > right_angle:
        optimal_angle1, min_area1, bounds1 = ternary_search_min_area(points, 0, right_angle)
        optimal_angle2, min_area2, bounds2 = ternary_search_min_area(points, left_angle, 180)
        
        if min_area1 < min_area2:
            optimal_angle = optimal_angle1
            min_area = min_area1
            best_bounds = bounds1
        else:
            optimal_angle = optimal_angle2
            min_area = min_area2
            best_bounds = bounds2
    else:
        optimal_angle, min_area, best_bounds = ternary_search_min_area(points, left_angle, right_angle)
    
    min_x, max_x, min_y, max_y = best_bounds
    width = max_x - min_x
    height = max_y - min_y
    
    rotated_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    
    corners = [rotate_point(p, optimal_angle) for p in rotated_corners]
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_original = rotate_point((center_x, center_y), optimal_angle)
    
    return {
        'type': 'OBB',
        'corners': corners,
        'width': width,
        'height': height,
        'area': min_area,
        'angle': optimal_angle,
        'center': center_original
    }

def create_dxf_with_lines(lines, filename):
    """创建包含指定线段的DXF文件"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    for line in lines:
        msp.add_line(line[0], line[1])
    
    doc.saveas(filename)

def cluster_lines(lines, distance_threshold=5):
    """聚类线段到不同的聚落"""
    class Cluster:
        def __init__(self, lines=[], min_x=0, max_x=0, min_y=0, max_y=0):
            self.lines = lines
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y
    
    clusters = []
    remaining_lines = list(lines)
    
    while remaining_lines:
        seed = remaining_lines.pop(0)
        current_cluster = [seed]
        
        # 初始化聚落的边界
        min_x = min(seed[0][0], seed[1][0])
        max_x = max(seed[0][0], seed[1][0])
        min_y = min(seed[0][1], seed[1][1])
        max_y = max(seed[0][1], seed[1][1])
        
        while True:
            # 扩展边界
            expanded_min_x = min_x - distance_threshold
            expanded_max_x = max_x + distance_threshold
            expanded_min_y = min_y - distance_threshold
            expanded_max_y = max_y + distance_threshold
            
            # 寻找在扩展边界内的线段
            to_add = []
            for line in list(remaining_lines):
                in_cluster = False
                # 检查线段的两个端点是否在扩展后的边界内
                for point in line:
                    if (expanded_min_x <= point[0] <= expanded_max_x and
                        expanded_min_y <= point[1] <= expanded_max_y):
                        in_cluster = True
                        break
                if in_cluster:
                    to_add.append(line)
            
            # 如果没有找到，结束循环
            if not to_add:
                break
            
            # 将找到的线段加入当前聚落，并更新边界
            for line in to_add:
                current_cluster.append(line)
                remaining_lines.remove(line)
                # 更新当前聚落的边界
                line_min_x = min(p[0] for p in line)
                line_max_x = max(p[0] for p in line)
                line_min_y = min(p[1] for p in line)
                line_max_y = max(p[1] for p in line)
                min_x = min(min_x, line_min_x)
                max_x = max(max_x, line_max_x)
                min_y = min(min_y, line_min_y)
                max_y = max(max_y, line_max_y)
        
        # 检查是否有包含关系
        to_remove = []
        for cluster in clusters:
            if (min_x <= cluster.min_x and min_y <= cluster.min_y and 
                max_x >= cluster.max_x and max_y >= cluster.max_y):
                current_cluster.extend(cluster.lines)
                to_remove.append(cluster)
            elif (min_x >= cluster.min_x and min_y >= cluster.min_y and 
                  max_x <= cluster.max_x and max_y <= cluster.max_y):
                cluster.lines.extend(current_cluster)
                current_cluster = []
                break
                
        for cluster in to_remove:
            clusters.remove(cluster)
        
        # 将当前聚落加入结果列表
        if current_cluster:
            current_cluster_c = Cluster(
                lines=current_cluster,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y
            )
            clusters.append(current_cluster_c)
    
    return clusters

def main():
    # 创建一个隐藏的根窗口用于文件对话框
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 弹窗选择DXF文件
    dxf_file_path = filedialog.askopenfilename(
        title="选择DXF文件",
        filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
    )

    # 检查用户是否选择了文件
    if not dxf_file_path:
        print("未选择文件，程序退出")
        return

    # 获取文件名（不含扩展名）
    base_filename = os.path.splitext(os.path.basename(dxf_file_path))[0]
    directory = os.path.dirname(dxf_file_path)
    
    print(f"基础文件名: {base_filename}")
    print(f"目录: {directory}")

    # 加载DXF文件
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()  # 获取模型空间

    # 准备一个列表用于存储所有的线段
    lines = []
            
    # 遍历模型空间中的所有实体
    for entity in msp:
        if entity.dxftype() == 'LINE':  # 如果是直线
            # 添加起点和终点到lines列表
            lines.append([(entity.dxf.start.x, entity.dxf.start.y), 
                          (entity.dxf.end.x, entity.dxf.end.y)])
        elif entity.dxftype() == 'LWPOLYLINE':  # 处理轻量多段线
            points = entity.get_points()  # 获取多段线的所有顶点
            # 将连续的点连接成线段
            for i in range(len(points) - 1):
                lines.append([(points[i][0], points[i][1]), 
                              (points[i+1][0], points[i+1][1])])
            # 如果多段线闭合，则连接最后一个点与第一个点
            if entity.is_closed:
                lines.append([(points[-1][0], points[-1][1]), 
                              (points[0][0], points[0][1])])
        elif entity.dxftype() == 'ARC':  # 如果是圆弧
            arc = entity
            center = np.array([arc.dxf.center.x, arc.dxf.center.y])
            radius = arc.dxf.radius
            start_angle = arc.dxf.start_angle
            end_angle = arc.dxf.end_angle
            if start_angle > end_angle:
                end_angle += 360  # 确保角度范围正确
            angle_step = (end_angle - start_angle) / 15  # 分割成小段
            angles = np.arange(start_angle, end_angle, angle_step)
            arc_points = [center + radius * np.array([np.cos(np.deg2rad(angle)), 
                          np.sin(np.deg2rad(angle))]) for angle in angles]
            # 将圆弧分割为多个线段
            for i in range(len(arc_points) - 1):
                lines.append([tuple(arc_points[i]), tuple(arc_points[i+1])])

    print(f"总共读取到 {len(lines)} 条线段")

    # 聚类线段
    clusters = cluster_lines(lines)
    print(f"找到 {len(clusters)} 个聚落")

    # 为每个聚落计算OBB并导出DXF
    for i, cluster in enumerate(clusters):
        # 提取点
        points = get_points_from_lines(cluster.lines)
        
        # 计算OBB
        obb = get_oriented_bounding_box_approx(points)
        
        if obb:
            # 格式化尺寸（保留一位小数）
            width = round(obb['width'], 1)
            height = round(obb['height'], 1)
            
            # 创建新的文件名
            new_filename = f"{base_filename}{width}x{height}.dxf"
            full_path = os.path.join(directory, new_filename)
            
            # 导出DXF文件
            create_dxf_with_lines(cluster.lines, full_path)
            
            print(f"聚落 {i+1}: 导出文件 '{new_filename}' (尺寸: {width} x {height})")
        else:
            print(f"聚落 {i+1}: 无法计算OBB")

if __name__ == "__main__":
    main()