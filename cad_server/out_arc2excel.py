# out_arc2excel.py
from pyautocad import Autocad, APoint
import math
import openpyxl
from openpyxl import Workbook
import os
import uuid

# 定义全局图层名称常量
OUTER_ARC_LAYER = "外弧"
INNER_ARC_LAYER = "内弧"
OUTER_CIRCLE_LAYER = "外圆"
INNER_CIRCLE_LAYER = "内圆"
NUMBERING_LAYER = "编号"

def get_all_objects_from_selection(acad, doc):
    """
    一次性获取所有选中的圆弧和圆对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        tuple: (圆弧对象列表, 圆对象列表)
    """
    arcs = []
    circles = []
    
    try:
        # 生成唯一的选择集名称以避免冲突
        selection_set_name = "SelectAllObjects_" + str(uuid.uuid4())[:8]
        
        # 提示用户选择对象
        selection_set = doc.SelectionSets.Add(selection_set_name)
        
        print("请在AutoCAD中选择所有需要处理的圆弧和圆对象...")
        selection_set.SelectOnScreen()
        
        # 遍历选中的对象，筛选出圆弧和圆
        for i in range(selection_set.Count):
            try:
                entity = selection_set.Item(i)
                if entity.ObjectName == "AcDbArc":
                    arcs.append(entity)
                elif entity.ObjectName == "AcDbCircle":
                    circles.append(entity)
            except Exception as e:
                print(f"检查选中对象 {i} 时出错: {e}")
                continue
        
        # 清理选择集
        selection_set.Delete()
        
        print(f"总共选中了 {len(arcs)} 个圆弧对象和 {len(circles)} 个圆对象")
        
    except Exception as e:
        print(f"获取选中对象时出错: {e}")
        
    return arcs, circles

def separate_objects_by_layer(arcs, circles):
    """
    根据图层名称将对象分为外对象和内对象
    
    Args:
        arcs: 圆弧对象列表
        circles: 圆对象列表
    
    Returns:
        tuple: (外对象列表, 内对象列表)
    """
    outer_objects = []
    inner_objects = []
    
    # 处理圆弧
    for arc in arcs:
        layer_name = arc.Layer
        if layer_name == OUTER_ARC_LAYER:
            outer_objects.append(('arc', arc))
        elif layer_name == INNER_ARC_LAYER:
            inner_objects.append(('arc', arc))
    
    # 处理圆
    for circle in circles:
        layer_name = circle.Layer
        if layer_name == OUTER_CIRCLE_LAYER:
            outer_objects.append(('circle', circle))
        elif layer_name == INNER_CIRCLE_LAYER:
            inner_objects.append(('circle', circle))
    
    print(f"分离结果: 外对象 {len(outer_objects)} 个, 内对象 {len(inner_objects)} 个")
    return outer_objects, inner_objects

def get_object_start_point(obj_type, obj):
    """
    获取对象的起始点坐标（圆弧有起始点，圆使用最上方点作为起始点）
    
    Args:
        obj_type: 对象类型 ('arc' 或 'circle')
        obj: 对象实体
    
    Returns:
        tuple: (x, y) 坐标
    """
    if obj_type == 'arc':
        start_point = obj.StartPoint
        return (start_point[0], start_point[1])
    elif obj_type == 'circle':
        # 对于圆，我们使用最上方的点作为"起始点"
        center = obj.Center
        radius = obj.Radius
        return (center[0], center[1] + radius)

def sort_objects_by_start_point(objects, y_tolerance=1000):
    """
    按起点X坐标对对象进行排序（仅按X轴排序）
    
    Args:
        objects: 对象列表 [('type', object), ...]
        y_tolerance: 保留参数，但不再使用
    
    Returns:
        list: 按X坐标升序排序后的对象列表
    """
    if not objects:
        return []
    
    # 仅按X坐标升序排序所有对象
    objects_sorted_by_x = sorted(objects, key=lambda o: get_object_start_point(o[0], o[1])[0])
    
    print(f"排序前对象数量: {len(objects)}")
    print(f"按X坐标排序后对象数量: {len(objects_sorted_by_x)}")
    
    return objects_sorted_by_x

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
        # 尝试获取现有图层
        layer = doc.Layers.Item(layer_name)
        print(f"图层 '{layer_name}' 已存在")
        return layer
    except:
        # 如果图层不存在，则创建新图层
        try:
            layer = doc.Layers.Add(layer_name)
            print(f"已创建图层 '{layer_name}'")
            return layer
        except Exception as e:
            print(f"创建图层 '{layer_name}' 失败: {e}")
            return None

def add_number_to_object(acad, doc, obj_type, obj, number):  
    """
    在对象附近添加编号文本
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
        obj_type: 对象类型 ('arc' 或 'circle')
        obj: 对象实体
        number: 编号
    """
    try:
        if obj_type == 'arc':
            # 获取圆弧相关信息
            start_point = obj.StartPoint
            end_point = obj.EndPoint
            center = obj.Center   
            start_angle = obj.StartAngle
            end_angle = obj.EndAngle
            radius = obj.Radius  
            
            # 计算中点角度
            mid_angle = (start_angle + end_angle) / 2
            if start_angle > end_angle and not (start_angle >= 0 and end_angle <= 2 * math.pi):
                mid_angle = (start_angle + end_angle + 2 * math.pi) / 2
            
            # 计算中点坐标
            mid_x = center[0] + radius * math.cos(mid_angle)
            mid_y = center[1] + radius * math.sin(mid_angle)
                
            # 将文本位置稍微偏移至对象内侧
            direction_x = mid_x - center[0]
            direction_y = mid_y - center[1]
            
            # 归一化方向向量
            length = math.sqrt(direction_x**2 + direction_y**2)
            if length > 0:
                direction_x /= length
                direction_y /= length
                
                # 将文本位置稍微偏移至对象内侧
                text_x = mid_x - direction_x * radius * 0.1
                text_y = mid_y - direction_y * radius * 0.1
            else:
                # 回退方案
                text_x = mid_x
                text_y = mid_y
                
            # 使用 APoint 创建坐标点
            text_point = APoint(text_x, text_y)  
              
            # 添加文本，使用 MText 替代 Text
            text_width = radius * 0.5  # 设置文本宽度为半径的一半
            text_obj = doc.ModelSpace.AddMText(text_point, text_width, str(number))
            text_obj.Height = radius * 0.08  # 字体大小为半径的0.08倍
            text_obj.Layer = NUMBERING_LAYER
              
            print(f"已为圆弧添加序号: {number}")
            
        elif obj_type == 'circle':
            # 获取圆相关信息
            center = obj.Center
            radius = obj.Radius
            
            # 在圆的正上方稍偏内侧位置放置文本
            text_x = center[0]
            text_y = center[1] + radius * 0.9  # 稍微偏内一点的位置
            
            # 使用 APoint 创建坐标点
            text_point = APoint(text_x, text_y)
            
            # 添加文本，使用 MText 替代 Text
            text_width = radius * 0.5  # 设置文本宽度为半径的一半
            text_obj = doc.ModelSpace.AddMText(text_point, text_width, str(number))
            text_obj.Height = 500 # 字体大小为半径的0.08倍
            text_obj.Layer = NUMBERING_LAYER
            
            print(f"已为圆添加序号: {number}")
            
    except Exception as e:  
        print(f"为对象添加序号 {number} 时出错: {e}")

def calculate_object_length(obj_type, obj):
    """
    计算对象的长度并向上取整（圆弧计算弧长，圆计算周长）
    
    Args:
        obj_type: 对象类型 ('arc' 或 'circle')
        obj: AutoCAD对象实体
    
    Returns:
        tuple: (向上取整后的长度, 向上取整后的半径)
    """
    try:
        if obj_type == 'arc':
            radius = obj.Radius
            start_angle = obj.StartAngle
            end_angle = obj.EndAngle
            
            # 计算圆弧角度差
            if end_angle < start_angle:
                angle_diff = (2 * math.pi - start_angle) + end_angle
            else:
                angle_diff = end_angle - start_angle
                
            # 计算弧长 = 半径 × 弧度
            arc_length = radius * angle_diff
            
            # 向上取整
            arc_length_ceil = math.ceil(arc_length)
            radius_ceil = math.ceil(radius)
            
            return arc_length_ceil, radius_ceil
            
        elif obj_type == 'circle':
            radius = obj.Radius
            
            # 计算周长 = 2 × π × 半径
            circumference = 2 * math.pi * radius
            
            # 向上取整
            circumference_ceil = math.ceil(circumference)
            radius_ceil = math.ceil(radius)
            
            return circumference_ceil, radius_ceil
            
    except Exception as e:
        print(f"计算对象长度时出错: {e}")
        return 0, 0

def export_paired_object_data_to_excel(paired_objects, dxf_directory):
    """
    将配对对象数据导出到Excel文件（每对只记录一次）
    
    Args:
        paired_objects: 配对对象列表 [(outer_obj, inner_obj), ...]
        dxf_directory: DXF文件所在目录
    """
    try:
        # 构建Excel文件路径
        filename = os.path.join(dxf_directory, "弧数据.xlsx")
        
        # 确保文件路径存在
        if not os.path.exists(dxf_directory):
            os.makedirs(dxf_directory)
            print(f"创建目录: {dxf_directory}")
        
        # 计算配对对象的数据
        paired_object_data = []
        for index, (outer_obj_info, inner_obj_info) in enumerate(paired_objects, start=1):
            # 计算外对象长度
            if outer_obj_info:
                outer_type, outer_obj = outer_obj_info
                outer_length, _ = calculate_object_length(outer_type, outer_obj)
            else:
                outer_length = 0
                
            # 计算内对象半径
            if inner_obj_info:
                inner_type, inner_obj = inner_obj_info
                _, inner_radius = calculate_object_length(inner_type, inner_obj)
            else:
                inner_radius = 0
                
            paired_object_data.append((outer_length, inner_radius))
            print(f"配对 {index}: 外对象长度={outer_length}, 内对象半径={inner_radius}")
        
        # 创建工作簿和工作表
        wb = Workbook()
        
        # 配对数据工作表
        ws_paired = wb.active
        ws_paired.title = "配对对象数据"
        
        # 写入表头
        ws_paired['A1'] = "序号"
        ws_paired['B1'] = "外对象长度(取整)"
        ws_paired['C1'] = "内对象半径"
        
        # 写入配对对象数据
        for idx, (outer_length, inner_radius) in enumerate(paired_object_data, start=1):
            ws_paired[f'A{idx+1}'] = idx
            ws_paired[f'B{idx+1}'] = outer_length
            ws_paired[f'C{idx+1}'] = inner_radius
            
            outer_type = paired_objects[idx-1][0][0] if paired_objects[idx-1][0] else None
            inner_type = paired_objects[idx-1][1][0] if paired_objects[idx-1][1] else None
            
            outer_name = "圆" if outer_type == "circle" else "弧" if outer_type == "arc" else "无"
            inner_name = "圆" if inner_type == "circle" else "弧" if inner_type == "arc" else "无"
            
            print(f"配对 {idx}: 外{outer_name}长度={outer_length}, 内{inner_name}半径={inner_radius}")
        
        # 保存文件
        wb.save(filename)
        print(f"成功导出配对对象数据到 {filename}")
        
    except Exception as e:
        print(f"导出到Excel时出错: {e}")

def pair_objects(outer_objects, inner_objects):
    """
    将外对象和内对象按顺序配对
    
    Args:
        outer_objects: 外对象列表 [('type', object), ...]
        inner_objects: 内对象列表 [('type', object), ...]
    
    Returns:
        list: 配对对象列表 [((outer_type, outer_obj), (inner_type, inner_obj)), ...]
    """
    # 排序对象
    sorted_outer_objects = sort_objects_by_start_point(outer_objects) if outer_objects else []
    sorted_inner_objects = sort_objects_by_start_point(inner_objects) if inner_objects else []
    
    # 确定最大长度以完成所有配对
    max_count = max(len(sorted_outer_objects), len(sorted_inner_objects))
    
    # 创建配对列表
    paired_objects = []
    for i in range(max_count):
        outer_obj = sorted_outer_objects[i] if i < len(sorted_outer_objects) else None
        inner_obj = sorted_inner_objects[i] if i < len(sorted_inner_objects) else None
        paired_objects.append((outer_obj, inner_obj))
    
    print(f"共创建 {len(paired_objects)} 对对象")
    return paired_objects

def main():
    """
    主函数 - 一次性选择所有弧和圆，然后分离内外对象进行处理
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
        # 获取DXF文件所在目录
        dxf_path = doc.FullName  # 获取完整文件路径
        dxf_directory = os.path.dirname(dxf_path)  # 提取目录部分
        print(f"DXF文件目录: {dxf_directory}")
        
        # 确保"编号"图层存在（只需执行一次）
        ensure_layer_exists(doc, NUMBERING_LAYER)
        
        # 一次性选择所有圆弧和圆对象
        all_arcs, all_circles = get_all_objects_from_selection(acad, doc)
        
        if not all_arcs and not all_circles:
            print("没有选中任何圆弧或圆对象")
            return
        
        # 根据图层分离内外对象
        outer_objects, inner_objects = separate_objects_by_layer(all_arcs, all_circles)
        
        if not outer_objects and not inner_objects:
            print("选中的对象中不包含'外弧'/'外圆'或'内弧'/'内圆'图层的对象")
            return
            
        # 将内外对象配对
        paired_objects = pair_objects(outer_objects, inner_objects)
        
        # 为每对对象添加相同编号
        for index, (outer_obj_info, inner_obj_info) in enumerate(paired_objects, start=1):
            # 为外对象添加编号（如果存在）
            if outer_obj_info:
                obj_type, obj = outer_obj_info
                add_number_to_object(acad, doc, obj_type, obj, index)
            
     
        # 导出配对数据到Excel
        if paired_objects:
            export_paired_object_data_to_excel(paired_objects, dxf_directory)
        
        # 重新生成视图以显示新添加的编号
        doc.Regen()
        print("已完成所有操作")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()