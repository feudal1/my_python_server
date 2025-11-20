# arc_dimension_tool.py
from pyautocad import Autocad, APoint
import math

import time

def get_selected_arcs(acad, doc):
    """
    获取用户选择的圆弧对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 选中的圆弧对象列表
    """
    try:
        # 提示用户选择对象
        print("请选择圆弧对象...")
        
        # 使用时间戳创建唯一的选择集名称
        selection_set_name = f"ArcSelection_{int(time.time())}"
        selection_set = doc.SelectionSets.Add(selection_set_name)
        selection_set.SelectOnScreen()
        
        # 筛选出圆弧对象
        arcs = []
        if selection_set.Count > 0:
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    if entity.ObjectName == "AcDbArc":
                        arcs.append(entity)
                except Exception as e:
                    print(f"检查对象时出错: {e}")
                    continue
            
            print(f"共选择 {selection_set.Count} 个对象，其中 {len(arcs)} 个为圆弧")
        else:
            print("未选择任何对象")
        
        # 清理选择集
        selection_set.Delete()
        
        return arcs
        
    except Exception as e:
        print(f"选择对象时出错: {e}")
        return []

def calculate_arc_properties(arc):
    """
    计算圆弧的各项属性
    
    Args:
        arc: AutoCAD Arc对象
    
    Returns:
        dict: 包含各项属性的字典
    """
    try:
        # 获取基本属性
        center = arc.Center
        radius = arc.Radius
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算弧长
        if end_angle >= start_angle:
            angle_diff = end_angle - start_angle
        else:
            angle_diff = (2 * math.pi - start_angle) + end_angle
            
        arc_length = radius * angle_diff
        
        # 计算起点和终点坐标
        start_x = center[0] + radius * math.cos(start_angle)
        start_y = center[1] + radius * math.sin(start_angle)
        end_x = center[0] + radius * math.cos(end_angle)
        end_y = center[1] + radius * math.sin(end_angle)
        
        # 计算弦长
        chord_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # 计算矢高（拱高）
        # 弦的中点
        mid_chord_x = (start_x + end_x) / 2
        mid_chord_y = (start_y + end_y) / 2
        
        # 圆心到弦中点的距离
        dist_center_to_chord_mid = math.sqrt((mid_chord_x - center[0])**2 + (mid_chord_y - center[1])**2)
        
        # 矢高 = 半径 - 圆心到弦的距离
        sagitta = radius - dist_center_to_chord_mid
        
        # 计算弧中点角度
        mid_angle = (start_angle + end_angle) / 2
        if end_angle < start_angle:
            mid_angle = (start_angle + end_angle + 2*math.pi) / 2
            if mid_angle > 2*math.pi:
                mid_angle -= 2*math.pi
                
        # 弧中点坐标
        arc_mid_x = center[0] + radius * math.cos(mid_angle)
        arc_mid_y = center[1] + radius * math.sin(mid_angle)
        
        return {
            'arc_length': arc_length,
            'chord_length': chord_length,
            'sagitta': sagitta,
            'start_point': (start_x, start_y),
            'end_point': (end_x, end_y),
            'mid_arc_point': (arc_mid_x, arc_mid_y),
            'center': center,
            'radius': radius,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'mid_angle': mid_angle
        }
    except Exception as e:
        print(f"计算圆弧属性时出错: {e}")
        return None

def ensure_layer_exists(doc, layer_name):
    """
    确保指定图层存在，如果不存在则创建
    
    Args:
        doc: AutoCAD文档对象
        layer_name: 图层名称
    
    Returns:
        layer: 图层对象
    """
    try:
        # 检查图层是否已存在
        layers = doc.Layers
        for i in range(layers.Count):
            if layers.Item(i).Name == layer_name:
                return layers.Item(i)
        
        # 如果不存在则创建新图层
        new_layer = layers.Add(layer_name)
        return new_layer
    except Exception as e:
        print(f"创建或获取图层 {layer_name} 时出错: {e}")
        return None

def add_chord_dimension(doc, arc_props, index, offset_distance=200, text_height=50):
    """
    添加弦长标注
    
    Args:
        doc: AutoCAD文档对象
        arc_props: 圆弧属性字典
        index: 圆弧索引号
        offset_distance: 标注偏移距离，默认为200
        text_height: 文字高度，默认为50
    
    Returns:
        标注对象或None
    """
    try:
        # 确保图层存在
        ensure_layer_exists(doc, "弦长标注")
        
        # 在弦下方标注弦长
        # 计算弦中点
        chord_mid_x = (arc_props['start_point'][0] + arc_props['end_point'][0]) / 2
        chord_mid_y = (arc_props['start_point'][1] + arc_props['end_point'][1]) / 2
        
        # 偏移一定距离来放置标注文本
        # 计算垂直于弦的单位向量
        chord_dx = arc_props['end_point'][0] - arc_props['start_point'][0]
        chord_dy = arc_props['end_point'][1] - arc_props['start_point'][1]
        chord_length = math.sqrt(chord_dx**2 + chord_dy**2)
        
        if chord_length > 0:
            # 单位向量
            unit_x = chord_dx / chord_length
            unit_y = chord_dy / chord_length
            
            # 垂直向量（逆时针旋转90度）
            perp_x = -unit_y
            perp_y = unit_x
            
            # 标注点（弦中点向下偏移指定单位）
            dim_x = chord_mid_x + perp_x * offset_distance
            dim_y = chord_mid_y + perp_y * offset_distance
            
            chord_dim_point = APoint(dim_x, dim_y)
            
            # 添加弦长标注
            chord_start = APoint(arc_props['start_point'][0], arc_props['start_point'][1])
            chord_end = APoint(arc_props['end_point'][0], arc_props['end_point'][1])
            
            chord_length_dim = doc.ModelSpace.AddDimAligned(chord_start, chord_end, chord_dim_point)
            chord_length_value = round(arc_props['chord_length'])
            chord_length_dim.TextOverride = f"弦长={chord_length_value}"
            
            # 设置文字大小
            chord_length_dim.TextHeight = text_height
            
            # 控制文字位置
            chord_length_dim.TextMovement = 2  # 固定文字位置
            chord_length_dim.TextPosition = chord_dim_point
            # 分配到弦长标注图层
            chord_length_dim.Layer = "弦长标注"
            
            print(f"成功为圆弧 {index} 添加弦长标注: 弦长={chord_length_value}")
            return chord_length_dim
            
    except Exception as e:
        print(f"为圆弧 {index} 添加弦长标注时出错: {e}")
    
    return None

def add_sagitta_dimension(doc, arc_props, index, offset_distance=50, text_height=50):
    """
    添加矢高标注
    
    Args:
        doc: AutoCAD文档对象
        arc_props: 圆弧属性字典
        index: 圆弧索引号
        offset_distance: 标注偏移距离，默认为50
        text_height: 文字高度，默认为50
    
    Returns:
        标注对象或None
    """
    try:
        # 确保图层存在
        ensure_layer_exists(doc, "矢高标注")
        
        # 矢高标注放在弦的中垂线上
        chord_mid_x = (arc_props['start_point'][0] + arc_props['end_point'][0]) / 2
        chord_mid_y = (arc_props['start_point'][1] + arc_props['end_point'][1]) / 2
        
        # 圆心到弦中点的向量
        to_center_x = arc_props['center'][0] - chord_mid_x
        to_center_y = arc_props['center'][1] - chord_mid_y
        to_center_length = math.sqrt(to_center_x**2 + to_center_y**2)
        
        if to_center_length > 0:
            # 单位向量
            unit_x = to_center_x / to_center_length
            unit_y = to_center_y / to_center_length
            
            # 矢高标注点（固定偏移指定距离）
            dim_x = chord_mid_x + unit_x * offset_distance
            dim_y = chord_mid_y + unit_y * offset_distance
            
            sagitta_dim_point = APoint(dim_x, dim_y)
            
            # 矢高起点（弦中点）
            sagitta_start = APoint(chord_mid_x, chord_mid_y)
            
            # 矢高终点（弧中点）
            sagitta_end = APoint(arc_props['mid_arc_point'][0], arc_props['mid_arc_point'][1])
            
            # 添加矢高标注
            sagitta_dim = doc.ModelSpace.AddDimAligned(sagitta_start, sagitta_end, sagitta_dim_point)
            sagitta_value = round(arc_props['sagitta'])
            sagitta_dim.TextOverride = f"矢高={sagitta_value}"
            
            # 设置文字大小
            sagitta_dim.TextHeight = text_height
            
            # 设置图层为"矢高标注"
            sagitta_dim.Layer = "矢高标注"
            
            # 控制文字位置
            sagitta_dim.TextMovement = 2  # 固定文字位置
            sagitta_dim.TextPosition = sagitta_dim_point
            
            print(f"成功为圆弧 {index} 添加矢高标注: 矢高={sagitta_value}")
            return sagitta_dim
            
    except Exception as e:
        print(f"为圆弧 {index} 添加矢高标注时出错: {e}")
    
    return None

def add_radius_dimension(doc, arc_props, index, offset_distance=900, text_height=50):
    """
    添加半径标注
    
    Args:
        doc: AutoCAD文档对象
        arc_props: 圆弧属性字典
        index: 圆弧索引号
        offset_distance: 标注偏移距离，默认为900
        text_height: 文字高度，默认为50
    
    Returns:
        标注对象或None
    """
    try:
        # 确保图层存在
        ensure_layer_exists(doc, "半径标注")
        
        # 半径标注需要圆心点、弧上一点和引线长度  
        center_point = APoint(arc_props['center'][0], arc_props['center'][1])  
        
        # 在弧上选择一个点作为半径标注的引线点（使用中点）  
        mid_arc_point = APoint(arc_props['mid_arc_point'][0], arc_props['mid_arc_point'][1])  
        
        # 添加半径标注
        radius_dim = doc.ModelSpace.AddDimRadial(center_point, mid_arc_point, offset_distance)  
        radius_value = round(arc_props['radius'])  
        radius_dim.TextOverride = f"R{radius_value}"  
        
        # 设置文字大小
        radius_dim.TextHeight = text_height
        
   
        
        # 分配到半径标注图层
        radius_dim.Layer = "半径标注"
        
        print(f"成功为圆弧 {index} 添加半径标注: R={radius_value}")
        return radius_dim
        
    except Exception as e:
        print(f"为圆弧 {index} 添加半径标注时出错: {e}")
    
    return None

def main():
    """
    主函数 - 为选中的圆弧添加矢高、弦长和半径标注
    """
    # 连接到正在运行的 AutoCAD
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    unit=1
    # 定义标注偏移参数，在此处统一修改
    CHORD_OFFSET = 300*unit    # 弦长标注偏移距离
    SAGITTA_OFFSET = 950   *unit# 矢高标注偏移距离
    RADIUS_OFFSET = -900    *unit# 半径标注偏移距离
    
    TEXT_HEIGHT = 150      *unit# 文字高度统一设置
    
    try:
        # 获取用户选择的圆弧
        selected_arcs = get_selected_arcs(acad, doc)
        
        if not selected_arcs:
            print("没有选择有效的圆弧对象")
            return
        
        # 统计成功添加的标注数量
        success_count = 0
        
        # 为每个选中的圆弧添加标注
        for i, arc in enumerate(selected_arcs, 1):
            print(f"\n处理第 {i} 个圆弧:")
            
            # 计算圆弧属性
            props = calculate_arc_properties(arc)
            if not props:
                print(f"无法计算第 {i} 个圆弧的属性")
                continue
            
            # 添加弦长标注
            chord_dim = add_chord_dimension(doc, props, i, CHORD_OFFSET, TEXT_HEIGHT)
            
            # 添加矢高标注
            sagitta_dim = add_sagitta_dimension(doc, props, i, SAGITTA_OFFSET, TEXT_HEIGHT)
            
            # 添加半径标注
            radius_dim = add_radius_dimension(doc, props, i, RADIUS_OFFSET, TEXT_HEIGHT)
            
            # 统计成功添加的标注
            dims_added = sum([1 for dim in [chord_dim, sagitta_dim, radius_dim] if dim is not None])
            success_count += dims_added
            
            print(f"第 {i} 个圆弧添加了 {dims_added} 个标注")
        
        print(f"\n总共为 {len(selected_arcs)} 个圆弧添加了 {success_count} 个标注")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()