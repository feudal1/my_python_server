import math
from collections import defaultdict
from pyautocad import Autocad, APoint

def connect_to_autocad():
    """连接到AutoCAD"""
    try:
        acad = Autocad(create_if_not_exists=True)
        return acad
    except Exception as e:
        print(f"连接失败: {e}")
        return None

def distance(point1, point2):
    """计算两点间距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def points_equal(p1, p2, tolerance=1e-6):
    """判断两点是否相等"""
    return distance(p1, p2) < tolerance

def find_total_objects(acad):
    """查找类圆形对象组合"""
    try:
        doc = acad.doc
        print(f"当前文档: {doc.FullName}")
        
        # 分类对象
        arcs = []
        lines = []
        
        for obj in doc.ModelSpace:
            if not hasattr(obj, 'ObjectName'):
                continue
                
            try:
                if obj.ObjectName == 'AcDbArc':
                    # 验证圆弧对象的基本属性
                    _ = obj.Center
                    _ = obj.Radius
                    _ = obj.StartPoint
                    _ = obj.EndPoint
                    _ = obj.StartAngle
                    _ = obj.EndAngle
                    arcs.append(obj)
                elif obj.ObjectName == 'AcDbLine':
                    # 验证线对象的基本属性
                    _ = obj.StartPoint
                    _ = obj.EndPoint
                    lines.append(obj)
            except Exception:
                # 忽略不能访问必要属性的对象
                continue
        
        print(f"找到圆弧对象数: {len(arcs)}")
        print(f"找到直线对象数: {len(lines)}")
        
        return arcs, lines
        
    except Exception as e:
        print(f"获取对象时出错: {e}")
        return [],[]

def group_points_by_circle(acad, arcs, lines):
    """将属于同一圆的点分组"""
    circles = {}  # 用(中心点, 半径)作为键存储圆的信息
    circle_points = defaultdict(list)  # 存储每个圆上的点
    objects_to_delete = set()  # 记录需要删除的对象
    
    # 处理圆弧对象的端点
    for arc in arcs:
        try:
            center = (arc.Center[0], arc.Center[1])
            radius = round(arc.Radius, 6)  # 四舍五入避免浮点误差
            circle_key = (center, radius)
            
            # 添加圆弧的起始点和终止点
            start_point = (arc.StartPoint[0], arc.StartPoint[1])
            end_point = (arc.EndPoint[0], arc.EndPoint[1])
            
            circle_points[circle_key].append(('arc_start', start_point, arc))
            circle_points[circle_key].append(('arc_end', end_point, arc))
            
            circles[circle_key] = {
                'center': center,
                'radius': radius
            }
            
            # 记录需要删除的圆弧对象
            objects_to_delete.add(arc)
        except Exception as e:
            print(f"处理圆弧点时出错: {e}")
            continue
    
    # 处理直线对象的端点
    for line in lines:
        try:
            start_point = (line.StartPoint[0], line.StartPoint[1])
            end_point = (line.EndPoint[0], line.EndPoint[1])
            
            # 尝试匹配到已有的圆
            matched = False
            for circle_key, circle_info in circles.items():
                center, radius = circle_key
                if (point_on_circle(center, radius, start_point) and 
                    point_on_circle(center, radius, end_point)):
                    circle_points[circle_key].append(('line_start', start_point, line))
                    circle_points[circle_key].append(('line_end', end_point, line))
                    matched = True
                    
                    # 记录需要删除的直线对象
                    objects_to_delete.add(line)
                    break
            
            # 如果没有匹配到现有圆，检查是否构成新圆（三点定圆）
            if not matched:
                # 这里简化处理，实际上需要更复杂的算法来确定新圆
                pass
        except Exception as e:
            print(f"处理直线点时出错: {e}")
            continue
    
    # 合并近似相同的圆
    merged_circle_points = defaultdict(list)
    processed_keys = set()
    
    circle_keys = list(circle_points.keys())
    
    for i, circle_key1 in enumerate(circle_keys):
        if circle_key1 in processed_keys:
            continue
            
        # 将当前圆作为基准
        merged_circle_points[circle_key1].extend(circle_points[circle_key1])
        processed_keys.add(circle_key1)
        
        # 检查后续的圆是否与当前圆近似相同
        for circle_key2 in circle_keys[i+1:]:
            if circle_key2 in processed_keys:
                continue
                
            if circles_equal(circle_key1, circle_key2):
                # 合并点列表
                merged_circle_points[circle_key1].extend(circle_points[circle_key2])
                processed_keys.add(circle_key2)
    
    # 去除每个圆内重复的点
    for circle_key in merged_circle_points:
        unique_points = []
        seen_points = set()
        
        for point_type, point_coords, obj in merged_circle_points[circle_key]:
            # 使用四舍五入的坐标作为唯一标识
            point_key = (round(point_coords[0], 3), round(point_coords[1], 3))
            if point_key not in seen_points:
                seen_points.add(point_key)
                unique_points.append((point_type, point_coords, obj))
        
        merged_circle_points[circle_key] = unique_points
    
    return merged_circle_points, objects_to_delete

def point_on_circle(center, radius, point, tolerance=1):
    """判断点是否在圆上"""
    try:
        # 计算点到圆心的距离
        dist = distance(center, point)
        # 检查距离是否等于半径
        return abs(dist - radius) < tolerance
    except:
        return False

def circles_equal(circle1, circle2, center_tolerance=1e-3, radius_tolerance=1e-3):
    """判断两个圆是否相等"""
    center1, radius1 = circle1
    center2, radius2 = circle2
    
    # 检查圆心距离
    center_distance = distance(center1, center2)
    # 检查半径差
    radius_diff = abs(radius1 - radius2)
    
    return center_distance < center_tolerance and radius_diff < radius_tolerance

def print_points_on_same_circle(circle_points):
    """打印位于同一圆上的所有点"""
    print("\n=== 圆上点分组信息 ===")
    
    for i, (circle_key, points) in enumerate(circle_points.items()):
        center, radius = circle_key
        print(f"\n圆 {i+1}:")
        print(f"  中心点: ({center[0]:.3f}, {center[1]:.3f})")
        print(f"  半径: {radius:.3f}")
        print(f"  点数量: {len(points)}")
        print("  点详情:")
        
        for j, (point_type, point_coords, obj) in enumerate(points):
            print(f"    {j+1}. 类型: {point_type}, 坐标: ({point_coords[0]:.3f}, {point_coords[1]:.3f})")

def create_arcs_from_points(acad, circle_points):
    """根据同一圆上的点创建圆弧"""
    created_arcs = []
    
    for circle_key, points in circle_points.items():
        center, radius = circle_key
        
        # 如果圆上有足够多的点（比如超过一定阈值），考虑创建圆弧
        if len(points) >= 3:  # 至少有3个点在圆上才考虑处理
            try:
                # 计算所有点的角度
                angles = []
                for point_type, point_coords, obj in points:
                    angle = math.atan2(point_coords[1] - center[1], point_coords[0] - center[0])
                    angle = angle_normalize(angle)
                    angles.append((angle, point_coords, obj))
                
                # 对角度进行排序
                angles.sort(key=lambda x: x[0])
                
                # 确定覆盖范围
                if len(angles) >= 2:
                    # 找到角度间隔最大的位置，这通常是圆弧的断点
                    max_gap = 0
                    max_gap_index = 0
                    
                    for i in range(len(angles)):
                        next_i = (i + 1) % len(angles)
                        gap = angles[next_i][0] - angles[i][0]
                        if gap < 0:
                            gap += 2 * math.pi
                        if gap > max_gap:
                            max_gap = gap
                            max_gap_index = next_i
                    
                    # 从最大间隔的下一个点开始作为起点
                    start_angle = angles[max_gap_index][0]
                    end_angle = angles[(max_gap_index - 1) % len(angles)][0]
                    
                    # 确保角度在正确范围内
                    if end_angle <= start_angle:
                        end_angle += 2 * math.pi
                    
                    # 创建圆弧
                    new_arc = acad.doc.ModelSpace.AddArc(
                        APoint(center[0], center[1]),
                        radius,
                        start_angle,
                        end_angle
                    )
                    created_arcs.append(new_arc)
                    print(f"在中心点({center[0]:.3f}, {center[1]:.3f})处创建了圆弧，半径为{radius:.3f}，角度从{math.degrees(start_angle):.1f}°到{math.degrees(end_angle):.1f}°")
                        
            except Exception as e:
                print(f"创建圆弧时出错: {e}")
                continue
    
    return created_arcs

def delete_original_objects(acad, objects_to_delete):
    """删除原始对象"""
    deleted_count = 0
    for obj in objects_to_delete:
        try:
            obj.Delete()
            deleted_count += 1
        except Exception as e:
            print(f"删除对象时出错: {e}")
            continue
    print(f"成功删除 {deleted_count} 个原始对象")
    return deleted_count

def angle_normalize(angle):
    """将角度标准化到[0, 2π]范围内"""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle

# 在文件末尾添加以下函数
def main():
    """主函数入口"""
    # 连接AutoCAD
    print("正在连接AutoCAD...")
    acad = connect_to_autocad()
    if not acad:
        print("无法连接到AutoCAD，退出程序")
        exit()
    
    print("AutoCAD连接成功")
    
    # 获取当前文档和模型空间
    try:
        doc = acad.doc
        print(f"活动文档: {doc.FullName}")
        model = doc.ModelSpace
        print("成功获取模型空间")
    except Exception as e:
        print(f"获取文档或模型空间时出错: {e}")
        exit()
    
    # 查找类圆形对象
    print("\n开始查找类圆形对象...")
    arcs, lines = find_total_objects(acad)
    
    # 打印位于同一圆弧的所有点
    print("\n分析并打印位于同一圆上的点...")
    circle_points, objects_to_delete = group_points_by_circle(acad, arcs, lines)
    print_points_on_same_circle(circle_points)
    
    # 根据点的数量决定是否创建圆弧
    print("\n检查是否可以创建圆弧...")
    full_arcs = create_arcs_from_points(acad, circle_points)
    print(f"创建了 {len(full_arcs)} 个圆弧")
    
    # 删除原始对象
    print("\n开始删除原始对象...")
    delete_original_objects(acad, objects_to_delete)
    
    print("\n处理完成！")

# 修改 if __name__ == "__main__": 部分为:
if __name__ == "__main__":
    main()