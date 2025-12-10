import math
import numpy as np
from pyautocad import Autocad, APoint
import tkinter as tk
from tkinter import filedialog

def get_selection_or_model_space(acad, doc):
    """获取用户选择的对象"""
    print("请选择对象")
    
    try:
        import time
        unique_name = f"Temp_Selection_Set_{int(time.time() * 1000) % 10000}"
        
        selection_set = doc.SelectionSets.Add(unique_name)
        selection_set.SelectOnScreen()
        
        if selection_set.Count > 0:
            print(f"检测到 {selection_set.Count} 个选中对象")
            selection = []
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    selection.append(entity)
                except Exception as e:
                    print(f"无法访问选中对象 {i}: {e}")
            selection_set.Delete()
            return selection
        else:
            selection_set.Delete()
            return []
    except Exception as e:
        print(f"无法获取选择集: {e}")
        return None

def get_lines_from_entities(entities):
    """从AutoCAD实体中提取线段"""
    lines = []
    
    for i, entity in enumerate(entities):
        try:
            if entity.ObjectName == "AcDbLine":
                start = entity.StartPoint[:2]
                end = entity.EndPoint[:2]
                lines.append((tuple(start), tuple(end)))
                print(f"找到线段 {len(lines)}: 起点({start[0]:.2f}, {start[1]:.2f}), 终点({end[0]:.2f}, {end[1]:.2f})")
            # 可以根据需要添加对其他实体类型的处理...
            elif entity.ObjectName == "AcDbCircle":
                # 处理圆对象，将其拆分成40个点
                center = entity.Center[:2]
                radius = entity.Radius
                
                # 在圆上均匀采样40个点并连接成线段
                circle_points = []
                for k in range(80):
                    angle = 2 * math.pi * k / 80
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    circle_points.append((x, y))
                
                # 将相邻点连接成线段
                for k in range(len(circle_points)):
                    start_point = circle_points[k]
                    end_point = circle_points[(k + 1) % len(circle_points)]  # 最后一个点连接到第一个点
                    lines.append((start_point, end_point))
                    
                print(f"找到圆 {i+1}: 采样40个点，生成40条线段")
            elif entity.ObjectName == "AcDbArc":
                # 处理圆弧对象，将其拆分成20个点
                center = entity.Center[:2]
                radius = entity.Radius
                start_angle = entity.StartAngle
                end_angle = entity.EndAngle
                
                # 确保角度范围正确
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                    
                # 在圆弧上均匀采样20个点并连接成线段
                arc_points = []
                for k in range(40):
                    angle = start_angle + (end_angle - start_angle) * k / 39
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    arc_points.append((x, y))
                
                # 将相邻点连接成线段
                for k in range(len(arc_points) - 1):
                    start_point = arc_points[k]
                    end_point = arc_points[k + 1]
                    lines.append((start_point, end_point))
                    
                print(f"找到圆弧 {i+1}: 采样20个点，生成19条线段")
        except Exception as e:
            print(f"处理对象 {i+1} 时出错: {e}")
            continue
    
    print(f"共提取到 {len(lines)} 条线段")
    return lines
                    


class LineCluster:
    def __init__(self, lines=[], min_x=0, max_x=0, min_y=0, max_y=0):
        self.lines = lines
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

def cluster_lines(lines, distance_threshold=50.0):
    """对线段进行聚类，基于您之前成功的算法"""
    if not lines:
        return []
    
    clusters = []
    remaining_lines = list(lines)  # 创建副本用于处理

    while remaining_lines:
        # 取出第一条线段作为种子
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
            for line in list(remaining_lines):  # 遍历副本避免问题
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
        
        # 检查是否有其他聚落被当前聚落完全包含，如果有则合并
        clusters_to_remove = []
        for cluster in clusters:
            if (min_x <= cluster.min_x and min_y <= cluster.min_y and
                max_x >= cluster.max_x and max_y >= cluster.max_y):
                current_cluster.extend(cluster.lines)
                clusters_to_remove.append(cluster)
        
        # 移除被包含的聚落
        for cluster in clusters_to_remove:
            clusters.remove(cluster)
        
        # 将当前聚落加入结果列表
        current_cluster_obj = LineCluster(
            lines=current_cluster,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y
        )
        clusters.append(current_cluster_obj)

    return clusters

def get_points_from_cluster(cluster):
    """从线段聚类中提取所有点"""
    points = []
    for line in cluster.lines:
        points.append(line[0])  # 起点
        points.append(line[1])  # 终点
    return list(set(points))  # 去重

def get_aabb_bounding_box(points):
    """获取轴对齐的最小包围矩形"""
    if len(points) < 1:
        return None
    
    # 处理单点情况
    if len(points) == 1:
        x, y = points[0]
        size = 1.0  # 创建一个很小的正方形包围框
        corners = [
            (x - size/2, y - size/2),
            (x + size/2, y - size/2),
            (x + size/2, y + size/2),
            (x - size/2, y + size/2)
        ]
        return {
            'type': 'AABB',
            'corners': corners,
            'width': size,
            'height': size,
            'area': size * size,
            'angle': 0,
            'center': (x, y)
        }
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # 计算四个角点
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
    # 处理单点情况
    if len(points) < 1:
        return None
    
    if len(points) == 1:
        x, y = points[0]
        size = 1.0  # 创建一个很小的正方形包围框
        corners = [
            (x - size/2, y - size/2),
            (x + size/2, y - size/2),
            (x + size/2, y + size/2),
            (x - size/2, y + size/2)
        ]
        return {
            'type': 'OBB',
            'corners': corners,
            'width': size,
            'height': size,
            'area': size * size,
            'angle': 0,
            'center': (x, y)
        }
    
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
    
    # 构造最终的包围盒
    min_x, max_x, min_y, max_y = best_bounds
    width = max_x - min_x
    height = max_y - min_y
    
    # 计算旋转后的四个角点
    rotated_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    
    # 将角点旋转回原始坐标系
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

def ensure_layer_exists(acad, layer_name):
    """确保图层存在，如果不存在则创建它"""
    try:
        # 尝试获取现有图层
        layers = acad.doc.Layers
        layer = layers.Item(layer_name)
        return layer
    except:
        # 图层不存在，创建新图层
        try:
            layers = acad.doc.Layers
            new_layer = layers.Add(layer_name)
            return new_layer
        except Exception as e:
            print(f"创建图层 '{layer_name}' 时出错: {e}")
            return None

def draw_bounding_box_with_text(acad, box, color_index, cluster_id=None, width=None, height=None):
    """在AutoCAD中绘制包围框并在旁边添加长宽数字文本"""
    if not box or 'corners' not in box:
        return None
    
    corners = box['corners']
    model = acad.model
    
    # 确保"obb_clusters"图层存在
    obb_layer = ensure_layer_exists(acad, "obb_clusters")
    
    # 绘制四条边，形成闭合矩形
    lines = []
    for i in range(4):
        p1 = APoint(corners[i][0], corners[i][1], 0)
        p2 = APoint(corners[(i+1)%4][0], corners[(i+1)%4][1], 0)
        line = model.AddLine(p1, p2)
        line.Color = color_index
        # 如果图层创建成功，将线条也放到obb_clusters图层
        if obb_layer:
            try:
                line.Layer = "obb_clusters"
            except:
                pass  # 如果设置图层失败，保持默认图层
        lines.append(line)
    
    # 添加聚类编号
    if cluster_id is not None:
        # 在包围框中心放置编号
        center_x = sum(corner[0] for corner in corners) / 4
        center_y = sum(corner[1] for corner in corners) / 4
        text_position = APoint(center_x, center_y, 0)
        text_content = str(cluster_id)
        text = model.AddText(text_content, text_position, 5)  # 文本高度设为5
        text.Color = color_index
        if obb_layer:
            try:
                text.Layer = "obb_clusters"
            except Exception as e:
                print(f"设置文字图层时出错: {e}")
        lines.append(text)
    
    # 如果提供了宽度和高度，则添加文本标注
    if width is not None and height is not None:
        # 找到包围框的右上角点
        top_right = max(corners, key=lambda p: p[0] + p[1])
        
        # 在右上角附近放置文本
        text_position = APoint(top_right[0] + 5, top_right[1] + 5, 0)
        text_content = f"{width:.0f}*{height:.0f}"
        text = model.AddText(text_content, text_position, 200)  # 文本高度设为5
        text.Color = color_index
        
        # 将文字设置到"obb_clusters"图层
        if obb_layer:
            try:
                text.Layer = "obb_clusters"
            except Exception as e:
                print(f"设置文字图层时出错: {e}")
        
        lines.append(text)
    
    return lines

def process_clusters(acad, clusters):
    """处理聚类并绘制OBB框"""
    print(f"\n处理 {len(clusters)} 个聚类:")
    
    # 预定义颜色列表 (AutoCAD颜色索引)
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    for i, cluster in enumerate(clusters):
        # 从线段聚类中提取点
        points = get_points_from_cluster(cluster)
        print(f"\n聚类 {i} 包含 {len(cluster.lines)} 条线段 ({len(points)} 个点)")
        
        # 计算AABB
        aabb = get_aabb_bounding_box(points)
        if aabb:
            print(f"  AABB尺寸: {aabb['width']:.3f} * {aabb['height']:.3f}")
            print(f"  AABB面积: {aabb['area']:.3f}")
        
        # 计算OBB
        obb = get_oriented_bounding_box_approx(points)
        if obb:
            print(f"  OBB尺寸: {obb['width']:.3f} x {obb['height']:.3f}")
            print(f"  OBB面积: {obb['area']:.3f}")
            if 'angle' in obb:
                print(f"  旋转角度: {obb['angle']:.2f}度")
            
            if aabb and aabb['area'] > 0:
                saving = (1 - obb['area'] / aabb['area']) * 100
                print(f"  相比AABB节省: {saving:.2f}%")
            
            # 绘制OBB框并标注编号和尺寸
            color_index = colors[i % len(colors)]
            draw_bounding_box_with_text(acad, obb, color_index, i, obb['width'], obb['height'])
            print(f"  已绘制聚类 {i} 的OBB框 (颜色索引: {color_index})")

def main():
    """主函数"""
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    
    try:
        entities = get_selection_or_model_space(acad, doc)
        
        if entities is None:
            print("获取对象过程中发生错误，程序退出")
            return
        
        if not entities:
            print("没有找到任何对象，程序退出")
            return
        
        print(f"处理 {len(entities)} 个对象")
        
        # 提取线段数据
        lines = get_lines_from_entities(entities)
        
        if len(lines) < 1:
            print("未找到有效线段")
            return
        
        # 对线段进行聚类
        clusters = cluster_lines(lines, distance_threshold=200.0)  # 根据您的数据调整阈值
        print(f"\n总共识别到 {len(clusters)} 个聚类")
        
        # 处理每个聚类并绘制OBB框
        process_clusters(acad, clusters)
        
        print("\n处理完成！所有聚类的OBB框已绘制在'obb_clusters'图层中")
        print("每个框中心显示聚类编号，右上角显示尺寸信息")
        
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()