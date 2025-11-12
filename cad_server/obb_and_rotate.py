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
                vertex_count = entity.Coordinates.count
                coords = entity.Coordinates
                
                polyline_points = []
                for j in range(0, vertex_count, 2):
                    if j + 1 < vertex_count:
                        x, y = coords[j], coords[j+1]
                        polyline_points.append((x, y))
                
                if entity.Closed and len(polyline_points) > 1:
                    polyline_points.pop()
                
                points.extend(polyline_points)
                print(f"找到多段线 {i+1}: 共{len(polyline_points)}个顶点")
                
            elif entity.ObjectName == "AcDbArc":
                center = entity.Center[:2]
                radius = entity.Radius
                start_angle = entity.StartAngle
                end_angle = entity.EndAngle
                
                if end_angle < start_angle:
                    end_angle += 2 * math.pi
                    
                for k in range(20):
                    angle = start_angle + (end_angle - start_angle) * k / 19
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((x, y))
                    
                print(f"找到圆弧 {i+1}: 采样20个点")
                
            elif entity.ObjectName == "AcDbCircle":
                center = entity.Center[:2]
                radius = entity.Radius
                
                for k in range(40):
                    angle = 2 * math.pi * k / 40
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append((x, y))
                    
                print(f"找到圆 {i+1}: 采样40个点")
            elif entity.ObjectName == "AcDbPolyline" or entity.ObjectName == "AcDb2dPolyline":
                try:
                    # 获取坐标数据
                    coords = entity.Coordinates
                    
                    # 处理不同的坐标数据类型
                    if hasattr(coords, 'count'):
                        # 如果 count 是方法，则调用它
                        if callable(coords.count):
                            vertex_count = coords.count()
                        else:
                            vertex_count = coords.count
                    else:
                        # 如果没有 count 属性，则使用 len()
                        vertex_count = len(coords)
                    
                    polyline_points = []
                    for j in range(0, vertex_count, 2):
                        if j + 1 < vertex_count:
                            x, y = coords[j], coords[j+1]
                            polyline_points.append((x, y))
                    
                    # 检查是否闭合
                    is_closed = False
                    try:
                        is_closed = entity.Closed
                    except:
                        # 如果无法直接获取 Closed 属性，尝试其他方式
                        try:
                            is_closed = entity.GetClosed() if hasattr(entity, 'GetClosed') else False
                        except:
                            is_closed = False
                    
                    if is_closed and len(polyline_points) > 1:
                        # 对于闭合多段线，检查首尾点是否相同
                        if polyline_points and polyline_points[0] == polyline_points[-1]:
                            polyline_points.pop()
                    
                    points.extend(polyline_points)
                    obj_type = "二维多段线" if entity.ObjectName == "AcDb2dPolyline" else "多段线"
                    print(f"找到{obj_type} {i+1}: 共{len(polyline_points)}个顶点")
                    
                except Exception as e:
                    print(f"处理多段线 {i+1} 时出错: {e}")
                except Exception as e:
                    print(f"处理多段线 {i+1} 时出错: {e}")                
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

def rotate_entities(acad, entities, angle, center):
    """旋转所有实体
    
    :param acad: Autocad实例
    :param entities: 实体列表
    :param angle: 旋转角度（度）
    :param center: 旋转中心点 (x, y)
    """
    print(f"\n开始旋转 {len(entities)} 个实体，角度: {angle:.2f}度")
    
    # 转换为弧度（AutoCAD使用弧度）
    angle_rad = math.radians(angle)
    base_point = APoint(center[0], center[1], 0)
    
    rotated_count = 0
    for i, entity in enumerate(entities):
        try:
            # 使用AutoCAD的Rotate方法
            entity.Rotate(base_point, angle_rad)
            rotated_count += 1
        except Exception as e:
            print(f"旋转对象 {i+1} 时出错: {e}")
            continue
    
    print(f"成功旋转 {rotated_count} 个对象")
    return rotated_count

def draw_bounding_box(acad, box, color_index):
    """在AutoCAD中绘制包围框"""
    if not box or 'corners' not in box:
        return None
    
    corners = box['corners']
    model = acad.model
    
    lines = []
    for i in range(4):
        p1 = APoint(corners[i][0], corners[i][1], 0)
        p2 = APoint(corners[(i+1)%4][0], corners[(i+1)%4][1], 0)
        line = model.AddLine(p1, p2)
        line.Color = color_index
        lines.append(line)
    
    return lines

def draw_dimensions(acad, box, color_index=4, hide_border=False):
    """
    在包围框上绘制长度和宽度的尺寸标注
    :param acad: Autocad实例
    :param box: 包围框信息字典
    :param color_index: 尺寸线颜色索引（默认蓝色）
    :param hide_border: 是否隐藏边界框
    """
    if not box or 'corners' not in box:
        return None
    
    corners = box['corners']
    model = acad.model
    
    # 获取包围框的四个角点
    p1 = APoint(corners[0][0], corners[0][1], 0)  # 左下角
    p2 = APoint(corners[1][0], corners[1][1], 0)  # 右下角
    p3 = APoint(corners[2][0], corners[2][1], 0)  # 右上角
    p4 = APoint(corners[3][0], corners[3][1], 0)  # 左上角
    
    # 计算尺寸线的偏移位置（距离包围框一定距离）
    offset_distance = max(box['width'], box['height']) * 0.1  # 偏移10%的长度
    
    try:
        # 绘制底部宽度尺寸线（在包围框下方）
        dim_start_x = APoint(corners[0][0], corners[0][1] - offset_distance, 0)
        dim_end_x = APoint(corners[1][0], corners[1][1] - offset_distance, 0)
        dim_x = model.AddDimAligned(p1, p2, dim_start_x)
        dim_x.Color = color_index
        
        # 绘制右侧高度尺寸线（在包围框右侧）
        dim_start_y = APoint(corners[1][0] + offset_distance, corners[1][1], 0)
        dim_end_y = APoint(corners[2][0] + offset_distance, corners[2][1], 0)
        dim_y = model.AddDimAligned(p2, p3, dim_start_y)
        dim_y.Color = color_index
        
        print(f"已绘制尺寸标注 (蓝色): 宽度={box['width']:.3f}, 高度={box['height']:.3f}")
        
        # 如果需要隐藏边界框，则删除边界线
        if hide_border and 'border_lines' in box:
            for line in box['border_lines']:
                try:
                    line.Delete()
                except:
                    pass
            print("边界框已隐藏")
        
        return [dim_x, dim_y]
    except Exception as e:
        print(f"绘制尺寸标注时出错: {e}")
        return None

def draw_blue_dimensions_only(acad, box):
    """
    仅绘制蓝色尺寸标注，不绘制边界框
    :param acad: Autocad实例
    :param box: 包围框信息字典
    """
    if not box or 'corners' not in box:
        return None
    
    corners = box['corners']
    model = acad.model
    
    # 获取包围框的四个角点
    p1 = APoint(corners[0][0], corners[0][1], 0)  # 左下角
    p2 = APoint(corners[1][0], corners[1][1], 0)  # 右下角
    p3 = APoint(corners[2][0], corners[2][1], 0)  # 右上角
    p4 = APoint(corners[3][0], corners[3][1], 0)  # 左上角
    
    # 计算尺寸线的偏移位置（距离包围框一定距离）
    offset_distance = max(box['width'], box['height']) * 0.1  # 偏移10%的长度
    
    try:
        # 绘制底部宽度尺寸线（在包围框下方）
        dim_start_x = APoint(corners[0][0], corners[0][1] - offset_distance, 0)
        dim_end_x = APoint(corners[1][0], corners[1][1] - offset_distance, 0)
        dim_x = model.AddDimAligned(p1, p2, dim_start_x)
        dim_x.Color = 7  # 蓝色
        
        # 绘制右侧高度尺寸线（在包围框右侧）
        dim_start_y = APoint(corners[1][0] + offset_distance, corners[1][1], 0)
        dim_end_y = APoint(corners[2][0] + offset_distance, corners[2][1], 0)
        dim_y = model.AddDimAligned(p2, p3, dim_start_y)
        dim_y.Color = 7  # 蓝色
        
        print(f"已绘制蓝色尺寸标注: 宽度={box['width']:.3f}, 高度={box['height']:.3f}")
        
        return [dim_x, dim_y]
    except Exception as e:
        print(f"绘制蓝色尺寸标注时出错: {e}")
        return None

def analyze_rotate_and_draw(acad, entities, rotate_to_align=True, draw_boxes=False, blue_dimensions_only=True):
    """分析、旋转实体并绘制包围框
    
    :param acad: Autocad实例
    :param entities: 实体列表
    :param rotate_to_align: 是否旋转实体使其轴对齐
    :param draw_boxes: 是否绘制边界框
    :param blue_dimensions_only: 是否仅绘制蓝色尺寸标注
    """
    print(f"正在分析 {len(entities)} 个实体")
    
    # 提取点
    points = get_points_from_entities(entities)
    
    if len(points) < 1:
        print("未找到有效点")
        return None
    
    # 计算原始AABB
    aabb_original = get_aabb_bounding_box(points)
    if aabb_original:
        print("\n原始轴对齐包围盒 (AABB):")
        print(f"  尺寸: {aabb_original['width']:.3f} x {aabb_original['height']:.3f}")
        print(f"  面积: {aabb_original['area']:.3f}")
        
        # 不绘制任何标注或边界框，除了旋转后的蓝色标注
    
    # 计算OBB
    obb = get_oriented_bounding_box_approx(points)
    if obb:
        print("\n最小包围盒 (OBB):")
        print(f"  尺寸: {obb['width']:.3f} x {obb['height']:.3f}")
        print(f"  面积: {obb['area']:.3f}")
        print(f"  旋转角度: {obb['angle']:.2f}度")
        print(f"  中心点: ({obb['center'][0]:.3f}, {obb['center'][1]:.3f})")
        
        # 将OBB尺寸复制到剪贴板
        obb_dimensions = f"{obb['width']:.0f}x{obb['height']:.0f}"
        pyperclip.copy(obb_dimensions)
        print(f"  OBB尺寸已复制到剪贴板: {obb_dimensions}")
        
        if aabb_original:
            saving = (1 - obb['area'] / aabb_original['area']) * 100
            print(f"  相比AABB节省: {saving:.2f}%")
        
        # 不绘制任何标注或边界框，除了旋转后的蓝色标注
        
        # 如果需要旋转实体使其轴对齐
        if rotate_to_align and len(entities) > 0:
            # 旋转角度为-OBB的角度，使OBB与坐标轴对齐
            rotation_angle = -obb['angle']
            
            # 旋转所有实体
            rotate_entities(acad, entities, rotation_angle, obb['center'])
            
            # 重新计算旋转后的点
            rotated_points = [rotate_point(p, rotation_angle, obb['center']) for p in points]
            
            # 计算旋转后的AABB（应该与原来的OBB尺寸相同）
            aabb_rotated = get_aabb_bounding_box(rotated_points)
            if aabb_rotated:
                print("\n旋转后轴对齐包围盒:")
                print(f"  尺寸: {aabb_rotated['width']:.3f} x {aabb_rotated['height']:.3f}")
                print(f"  面积: {aabb_rotated['area']:.3f}")
                
                # 仅绘制旋转后的AABB蓝色尺寸标注
                if blue_dimensions_only and not draw_boxes:
                    draw_blue_dimensions_only(acad, aabb_rotated)
                    print("  已绘制旋转后AABB蓝色尺寸标注")
                
                # 验证旋转后的AABB与原OBB是否一致
                width_diff = abs(aabb_rotated['width'] - obb['width'])
                height_diff = abs(aabb_rotated['height'] - obb['height'])
                area_diff = abs(aabb_rotated['area'] - obb['area'])
                
                print(f"\n验证结果:")
                print(f"  宽度差异: {width_diff:.6f}")
                print(f"  高度差异: {height_diff:.6f}")
                print(f"  面积差异: {area_diff:.6f}")
                
                if width_diff < 1e-3 and height_diff < 1e-3:
                    print("  ✓ 旋转后AABB与原OBB基本一致")
                else:
                    print("  ✗ 旋转后AABB与原OBB存在较大差异")
    
    return {
        'points': points,
        'aabb_original': aabb_original,
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
        
        # 控制参数
        ROTATE_TO_ALIGN = True         # 是否旋转实体使其轴对齐
        DRAW_BOUNDING_BOXES = False    # 是否绘制边界框
        BLUE_DIMENSIONS_ONLY = True    # 是否仅绘制蓝色尺寸标注
        
        result = analyze_rotate_and_draw(acad, entities, ROTATE_TO_ALIGN, DRAW_BOUNDING_BOXES, BLUE_DIMENSIONS_ONLY)
        
        if result:
            if BLUE_DIMENSIONS_ONLY and not DRAW_BOUNDING_BOXES:
                print("\n分析完成! 仅绘制蓝色尺寸标注，未绘制其他任何内容")
            elif DRAW_BOUNDING_BOXES:
                if ROTATE_TO_ALIGN:
                    print("\n分析完成!")
                    print("  红色框: 原始AABB")
                    print("  绿色框: 原始OBB")
                    print("  蓝色框: 旋转后AABB（应与绿色OBB一致）")
                    print("  青色标注: 旋转后AABB的长宽尺寸")
                else:
                    print("\n分析完成!")
                    print("  红色框: 原始AABB")
                    print("  绿色框: 原始OBB")
            else:
                print("\n分析完成! 仅绘制蓝色尺寸标注")
        else:
            print("分析失败")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()