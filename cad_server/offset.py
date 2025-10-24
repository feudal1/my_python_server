from pyautocad import Autocad, APoint
import math


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
        # 尝试获取当前选择集
        selection_set = doc.SelectionSets.Add("Temp_Selection_Set")
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
    从实体列表中筛选出圆弧对象
    """
    arcs = []
    
    for i, entity in enumerate(entities):
        try:
            # 判断是否为圆弧
            if entity.ObjectName == "AcDbArc":
                arcs.append(entity)
                print(f"找到圆弧对象 {len(arcs)}")
            else:
                print(f"对象 {i} 类型: {entity.ObjectName}")
        except Exception as e:
            print(f"检查对象 {i} 时出错: {e}")
            continue
    
    return arcs


def offset_arcs_outward(doc, arcs, offset_distance=70, layer_name="外弧"):
    """
    对选中的圆弧进行向外偏移，并将结果放在指定图层上
    
    Args:
        doc: AutoCAD文档对象
        arcs: 圆弧对象列表
        offset_distance: 偏移距离，默认为70
        layer_name: 目标图层名称，默认为"外弧"
    
    Returns:
        list: 成功偏移的圆弧对象列表
    """
    # 确保目标图层存在
    try:
        layers = doc.Layers
        try:
            # 尝试获取现有图层
            target_layer = layers.Item(layer_name)
        except:
            # 如果图层不存在则创建新图层
            target_layer = layers.Add(layer_name)
            print(f"创建新图层: {layer_name}")
    except Exception as e:
        print(f"处理图层时出错: {e}")
        target_layer = None
    
    offset_arcs_list = []
    
    for i, arc in enumerate(arcs):
        try:
            # 获取圆弧属性
            center = arc.Center
            radius = arc.Radius
            start_angle = arc.StartAngle
            end_angle = arc.EndAngle
            
            # 计算偏移后的半径
            # 向外偏移就是增加半径值
            new_radius = radius + offset_distance
            
            # 创建新的圆弧对象
            new_center = APoint(center[0], center[1])
            
            # 使用AddArc方法创建新的圆弧
            new_arc = doc.ModelSpace.AddArc(new_center, new_radius, start_angle, end_angle)
            
            # 将新圆弧分配到指定图层
            if target_layer:
                new_arc.Layer = layer_name
            
            offset_arcs_list.append(new_arc)
            print(f"成功为圆弧 {i+1} 创建向外偏移 {offset_distance} 的新圆弧，已放置到图层 '{layer_name}'")
            
        except Exception as e:
            print(f"为圆弧 {i+1} 创建偏移时出错: {e}")
            continue
    
    return offset_arcs_list


def main():
    """
    主函数 - 对选中圆弧进行向外偏移70，并放置到"外弧"图层
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
        
        # 对圆弧进行向外偏移70，并放置到"外弧"图层
        offset_arcs = offset_arcs_outward(doc, arcs, 70, "外弧")
        
        if offset_arcs:
            print(f"成功为 {len(offset_arcs)} 个圆弧创建了向外偏移70的新圆弧，并放置到'外弧'图层")
        else:
            print("未能为任何圆弧创建偏移")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")


if __name__ == "__main__":
    main()