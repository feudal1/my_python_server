#dr_arc_length.py
from pyautocad import Autocad, APoint
import math
import time

def get_selection_or_model_space(acad, doc):
    """
    获取用户选择的对象，如果没有选择则遍历模型空间
    """
    # 检查是否已有选择集
    try:
        # 遍历现有的选择集
        for i in range(doc.SelectionSets.Count):
            selection_set = doc.SelectionSets.Item(i)
            if selection_set.Count > 0:
                print(f"使用现有选择集: {selection_set.Count} 个对象")
                # 收集选中的对象
                selection = []
                for j in range(selection_set.Count):
                    try:
                        entity = selection_set.Item(j)
                        selection.append(entity)
                    except Exception as e:
                        print(f"无法访问选中对象 {j}: {e}")
                return selection
    except Exception as e:
        print(f"检查现有选择集时出错: {e}")
    
    # 如果没有现成的选择集，则提示用户选择
    print("请选择对象")
    
    try:
        # 先尝试删除可能已存在的临时选择集
        try:
            existing_selection_set = doc.SelectionSets.Item("Temp_Selection_Set")
            existing_selection_set.Delete()
        except:
            # 如果不存在则忽略错误
            pass
        
        # 使用唯一名称避免冲突
        unique_name = f"Temp_Selection_Set_{int(time.time() * 1000) % 10000}"
        
        selection_set = doc.SelectionSets.Add(unique_name)
        selection_set.SelectOnScreen()
        
        if selection_set.Count > 0:
            print(f"检测到 {selection_set.Count} 个选中对象")
            # 收集选中的对象
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
    except Exception as e:
        print(f"无法获取选择集: {e}")
    
    # 如果没有选择对象，遍历模型空间
    print("未检测到选择集，遍历模型空间...")
    try:
        ms = doc.ModelSpace
        selection = []
        for i in range(ms.Count):
            try:
                entity = ms.Item(i)
                selection.append(entity)
            except Exception as e:
                print(f"无法访问模型空间对象 {i}: {e}")
        return selection
    except Exception as e:
        print(f"无法访问模型空间: {e}")
        return []

def process_arcs(entities):
    """
    从实体列表中筛选出圆弧对象并计算相关信息，排除"外弧"图层上的圆弧
    """
    arcs = []
    
    for i, entity in enumerate(entities):
        try:
            # 判断是否为圆弧且不在"外弧"图层上
            if entity.ObjectName == "AcDbArc":
                # 检查图层名称，排除"外弧"图层
                layer_name = entity.Layer
                if layer_name != "外弧":
                    arcs.append(entity)
                    print(f"找到圆弧对象 {len(arcs)} (图层: {layer_name})")
                else:
                    print(f"跳过'外弧'图层上的圆弧对象 {i}")
            else:
                print(f"对象 {i} 类型: {entity.ObjectName}")
        except Exception as e:
            print(f"检查对象 {i} 时出错: {e}")
            continue
    
    return arcs

def calculate_arc_properties(arc):
    """
    计算圆弧的各种属性：弧长、弦长、矢高
    
    Args:
        arc: AutoCAD圆弧对象
    
    Returns:
        dict: 包含弧长、弦长、矢高等信息的字典
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
        
        # 计算标注位置点
        # 弧中点角度
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

def add_arc_dimensions(doc, arcs):
    """
    为选中的圆弧添加弧长、弦长和矢高标注
    
    Args:
        doc: AutoCAD文档对象
        arcs: 圆弧对象列表
    
    Returns:
        list: 成功添加的标注对象列表
    """
    added_dims = []
    
    # 确保"矢高"图层存在
    sagitta_layer = ensure_layer_exists(doc, "矢高")
    
    for i, arc in enumerate(arcs):
        try:
            # 计算圆弧属性
            props = calculate_arc_properties(arc)
            if not props:
                continue
                
            # 1. 添加弧长标注（使用专门的弧长标注）
            try:
                # 弧长标注需要圆心点、起点、终点和标注位置点
                center_point = APoint(props['center'][0], props['center'][1])
                start_point = APoint(props['start_point'][0], props['start_point'][1])
                end_point = APoint(props['end_point'][0], props['end_point'][1])
                
                # 标注点放在弧中点外侧更远的位置
                # 计算弧中点并向外延伸更多距离作为标注位置
                offset_distance = max(props['radius'] * 0.3, 10)  # 至少10个单位距离
                mid_offset_x = props['center'][0] + (props['radius'] + offset_distance) * math.cos(props['mid_angle'])
                mid_offset_y = props['center'][1] + (props['radius'] + offset_distance) * math.sin(props['mid_angle'])
                dim_point = APoint(mid_offset_x, mid_offset_y)
                
                # 添加真正的弧长标注
                arc_length_dim = doc.ModelSpace.AddDimArc(center_point, start_point, end_point, dim_point)
                arc_length_value = round(props['arc_length'])
                arc_length_dim.TextOverride = f"L={arc_length_value}"
                added_dims.append(arc_length_dim)
                print(f"成功为圆弧 {i+1} 添加弧长标注: L={arc_length_value}")
            except Exception as e:
                print(f"为圆弧 {i+1} 添加弧长标注时出错: {e}")
            
            # 2. 添加弦长标注
            try:
                # 在弦下方标注弦长
                # 计算弦中点
                chord_mid_x = (props['start_point'][0] + props['end_point'][0]) / 2
                chord_mid_y = (props['start_point'][1] + props['end_point'][1]) / 2
                
                # 偏移一定距离来放置标注文本
                # 计算垂直于弦的单位向量
                chord_dx = props['end_point'][0] - props['start_point'][0]
                chord_dy = props['end_point'][1] - props['start_point'][1]
                chord_length = math.sqrt(chord_dx**2 + chord_dy**2)
                
                if chord_length > 0:
                    # 单位向量
                    unit_x = chord_dx / chord_length
                    unit_y = chord_dy / chord_length
                    
                    # 垂直向量（逆时针旋转90度）
                    perp_x = -unit_y
                    perp_y = unit_x
                    
                    # 标注点（弦中点向下偏移更远）
                    offset_distance = max(props['radius'] * 0.3, 15)  # 至少10个单位距离
                    dim_x = chord_mid_x + perp_x * offset_distance
                    dim_y = chord_mid_y - perp_y * offset_distance*3/2
                    
                    chord_dim_point = APoint(dim_x,dim_y)
                    
                    # 添加弦长标注
                    chord_start = APoint(props['start_point'][0], props['start_point'][1])
                    chord_end = APoint(props['end_point'][0], props['end_point'][1])
                    
                    chord_length_dim = doc.ModelSpace.AddDimAligned(chord_start, chord_end, chord_dim_point)
                    chord_length_value = round(props['chord_length'])
                    chord_length_dim.TextOverride = f"弦长={chord_length_value}"
                                        # 控制文字位置
                    chord_length_dim.TextMovement = 2  # 固定文字位置
                    chord_length_dim.TextPosition =chord_dim_point
                    added_dims.append(chord_length_dim)
                    print(f"成功为圆弧 {i+1} 添加弦长标注: 弦长={chord_length_value}")
            except Exception as e:
                print(f"为圆弧 {i+1} 添加弦长标注时出错: {e}")
            
            # 3. 添加矢高标注
            try:
                # 矢高标注放在弦的中垂线上
                chord_mid_x = (props['start_point'][0] + props['end_point'][0]) / 2
                chord_mid_y = (props['start_point'][1] + props['end_point'][1]) / 2
                
                # 圆心到弦中点的向量
                to_center_x = props['center'][0] - chord_mid_x
                to_center_y = props['center'][1] - chord_mid_y
                to_center_length = math.sqrt(to_center_x**2 + to_center_y**2)
                
                if to_center_length > 0:
                    # 单位向量
                    unit_x = to_center_x / to_center_length
                    unit_y = to_center_y / to_center_length
                    
                    # 矢高标注点（放在更精确的中间位置）
                    offset_distance = max(props['radius'] * 0.3, 15)
                    dim_x = chord_mid_x + unit_x * offset_distance
                    dim_y = chord_mid_y + unit_y * offset_distance
                    
                    sagitta_dim_point = APoint(dim_x,props['end_point'][1]-  unit_y * offset_distance/2)
                    
                    # 矢高起点（弦中点）
                    sagitta_start = APoint(chord_mid_x, chord_mid_y)
                    
                    # 矢高终点（弧中点）
                    sagitta_end = APoint(props['mid_arc_point'][0], props['mid_arc_point'][1])
                    
                    # 添加矢高标注
                    sagitta_dim = doc.ModelSpace.AddDimAligned(sagitta_start, sagitta_end, sagitta_dim_point)
                    sagitta_value = round(props['sagitta'])
                    sagitta_dim.TextOverride = f"矢高={sagitta_value}"
                    
                    # 设置图层为"矢高"
                    sagitta_dim.Layer = "矢高"
                    
                    # 控制文字位置
                    sagitta_dim.TextMovement = 2  # 固定文字位置
                    sagitta_dim.TextPosition = sagitta_dim_point
                    
                    added_dims.append(sagitta_dim)
                    print(f"成功为圆弧 {i+1} 添加矢高标注: 矢高={sagitta_value}")
            except Exception as e:
                print(f"为圆弧 {i+1} 添加矢高标注时出错: {e}")
                
        except Exception as e:
            print(f"为圆弧 {i+1} 添加标注时出错: {e}")
            continue
    
    return added_dims

def main():
    """
    主函数 - 为选中圆弧添加弧长、弦长和矢高标注
    """
    # 连接到正在运行的 AutoCAD
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    
    try:
        # 获取要处理的对象
        entities = get_selection_or_model_space(acad, doc)
        
        if not entities:
            print("没有找到任何对象")
            return
        
        print(f"处理 {len(entities)} 个对象")
        
        # 筛选出圆弧对象
        arcs = process_arcs(entities)
        
        if not arcs:
            print("没有找到圆弧对象")
            return
        
        print(f"找到 {len(arcs)} 个圆弧对象")
        
        # 为圆弧添加各种标注
        dims = add_arc_dimensions(doc, arcs)
        
        if dims:
            print(f"成功为 {len(dims)} 个标注添加完成")
        else:
            print("未能添加任何标注")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()