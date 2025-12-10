import math
import numpy as np
from pyautocad import Autocad, APoint
import pyperclip

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

def get_points_from_entities(entities):
    """从AutoCAD实体中提取线段的端点"""
    points = []
    
    for i, entity in enumerate(entities):
        try:
            if entity.ObjectName == "AcDbLine":
                start = entity.StartPoint[:2]
                end = entity.EndPoint[:2]
                points.append(tuple(start))
                points.append(tuple(end))
                print(f"找到线段 {len(points)//2}: 起点({start[0]:.2f}, {start[1]:.2f}), 终点({end[0]:.2f}, {end[1]:.2f})")
            elif entity.ObjectName == "AcDbPolyline":
                # 处理多段线（包括带圆弧的多段线）
                vertex_count = entity.Coordinates.count
                coords = entity.Coordinates
                
                # 获取多段线的顶点
                polyline_points = []
                for j in range(0, vertex_count, 2):
                    if j + 1 < vertex_count:
                        x, y = coords[j], coords[j+1]
                        polyline_points.append((x, y))
                
                # 如果是封闭多段线，移除重复点
                if entity.Closed and len(polyline_points) > 1:
                    polyline_points.pop()  # 移除最后一个与第一个重合的点
                
                points.extend(polyline_points)
                print(f"找到多段线 {i+1}: 共{len(polyline_points)}个顶点")
                
                # 对于带圆弧的多段线，我们还可以采样圆弧部分以获得更精确的边界
                for j in range(entity.Coordinates.count // 2 - 1):
                    if hasattr(entity, 'GetBulge') and entity.GetBulge(j) != 0:
                        # 如果存在凸度（bulge），表示这是一段圆弧
                        bulge = entity.GetBulge(j)
                        if j*2+3 < entity.Coordinates.count:
                            start_point = (coords[j*2], coords[j*2+1])
                            end_point = (coords[j*2+2], coords[j*2+3])
                            # 可以在这里添加对圆弧的采样点，以提高精度
                            # 为简化起见，暂时只取端点，实际应用中可增加采样点
                            
            elif entity.ObjectName == "AcDbArc":
                # 处理圆弧对象，将其拆分成20个点
                center = entity.Center[:2]
                radius = entity.Radius
                start_angle = entity.StartAngle
                end_angle = entity.EndAngle
                
                # 确保角度范围正确
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                    
                # 在圆弧上均匀采样20个点
                for k in range(40):
                    angle = start_angle + (end_angle - start_angle) * k / 39
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((x, y))
                    
                print(f"找到圆弧 {i+1}: 采样20个点")
                
            elif entity.ObjectName == "AcDbCircle":
                # 处理圆对象，将其拆分成40个点
                center = entity.Center[:2]
                radius = entity.Radius
                
                # 在圆上均匀采样40个点
                for k in range(80):
                    angle = 2 * math.pi * k / 80
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((x, y))
                    
                print(f"找到圆 {i+1}: 采样40个点")

                    
            else:
                print(f"跳过非线段对象 {i+1}: {entity.ObjectName}")
        except Exception as e:
            print(f"处理对象 {i+1} 时出错: {e}")
            continue
    
    points = list(set(points))
    print(f"共提取到 {len(points)} 个不重复的点")
    return points


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
    """确保图层存在，如果不存在则创建它
    :param acad: Autocad实例
    :param layer_name: 图层名称
    :return: 图层对象
    """
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

def draw_bounding_box_with_text(acad, box, color_index, width=None, height=None):
    """在AutoCAD中绘制包围框并在旁边添加长宽数字文本
    
    :param acad: Autocad实例
    :param box: 包围框字典，包含corners列表
    :param color_index: AutoCAD颜色索引 (1=红色, 3=绿色, 5=蓝色等)
    :param width: 包围框宽度（可选）
    :param height: 包围框高度（可选）
    """
    if not box or 'corners' not in box:
        return None
    
    corners = box['corners']
    model = acad.model
    
    # 确保"obb"图层存在
    obb_layer = ensure_layer_exists(acad, "obb")
    
    # 绘制四条边，形成闭合矩形
    lines = []
    for i in range(4):
        p1 = APoint(corners[i][0], corners[i][1], 0)
        p2 = APoint(corners[(i+1)%4][0], corners[(i+1)%4][1], 0)
        line = model.AddLine(p1, p2)
        line.Color = color_index
        # 如果图层创建成功，将线条也放到obb图层
        if obb_layer:
            try:
                line.Layer = "obb"
            except:
                pass  # 如果设置图层失败，保持默认图层
        lines.append(line)
    
    # 如果提供了宽度和高度，则添加文本标注
    if width is not None and height is not None:
        # 计算包围框的中心点
        center_x = sum(corner[0] for corner in corners) / 4
        center_y = sum(corner[1] for corner in corners) / 4
        
        # 找到包围框的右上角点
        top_right = max(corners, key=lambda p: p[0] + p[1])
        
        # 在右上角附近放置文本
        text_position = APoint(top_right[0] + 5, top_right[1] + 5, 0)
        text_content = f"长度{width:.0f}*宽度{height:.0f}"
        text = model.AddText(text_content, text_position, 50)  # 文本高度设为5
        text.Color = color_index
        
        # 将文字设置到"obb"图层
        if obb_layer:
            try:
                text.Layer = "obb"
            except Exception as e:
                print(f"设置文字图层时出错: {e}")
        
        lines.append(text)
    
    return lines

def analyze_and_draw_bounding_boxes(acad, entities, draw_boxes=True):
    """分析实体并绘制包围框
    :param acad: Autocad实例
    :param entities: 实体列表
    :param draw_boxes: 是否绘制边界框，默认为True
    """
    print(f"正在分析 {len(entities)} 个实体")
    
    points = get_points_from_entities(entities)
    
    if len(points) < 1:
        print("未找到有效点")
        return None
    
    # 计算AABB
    aabb = get_aabb_bounding_box(points)
    if aabb:
        print("\n轴对齐包围盒 (AABB):")
        print(f"  尺寸: {aabb['width']:.3f} * {aabb['height']:.3f}")
        print(f"  面积: {aabb['area']:.3f}")
        
        # 绘制AABB (红色)
        if draw_boxes:
           # draw_bounding_box_with_text(acad, aabb, 1)
            print("  已绘制AABB (红色)")
    
    # 计算OBB
    obb = get_oriented_bounding_box_approx(points)
    if obb:
        print("\n近似最小包围盒 (OBB):")
        print(f"  尺寸: {obb['width']:.3f} x {obb['height']:.3f}")
        print(f"  面积: {obb['area']:.3f}")
        print(f"  旋转角度: {obb['angle']:.2f}度")
        print(f"  中心点: ({obb['center'][0]:.3f}, {obb['center'][1]:.3f})")
        
        # 将OBB尺寸复制到剪贴板（去掉空格）
        obb_dimensions = f"{obb['width']:.0f}*{obb['height']:.0f}"
        pyperclip.copy(obb_dimensions)
        print(f"  OBB尺寸已复制到剪贴板: {obb_dimensions}")
        
        if aabb:
            saving = (1 - obb['area'] / aabb['area']) * 100
            print(f"  相比AABB节省: {saving:.2f}%")
        
        # 绘制OBB (绿色) 并添加文本标注
        if draw_boxes:
            draw_bounding_box_with_text(acad, obb, 3, obb['width'], obb['height'])
            print("  已绘制OBB (绿色) 和尺寸文本")
    
    return {
        'points': points,
        'aabb': aabb,
        'obb': obb
    }

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
        
        # 控制是否绘制边界框的变量
        DRAW_BOUNDING_BOXES = True
        
        result = analyze_and_draw_bounding_boxes(acad, entities, DRAW_BOUNDING_BOXES)
        
        if result:
            if DRAW_BOUNDING_BOXES:
                print("\n分析完成! AABB为红色，OBB为绿色")
            else:
                print("\n分析完成! 边界框未绘制")
        else:
            print("分析失败")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()