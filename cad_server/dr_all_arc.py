from pyautocad import Autocad, APoint
import math


# 修改 get_selection_or_model_space 函数部分代码
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
        import time
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


def add_radius_dimension_to_arcs(doc, arcs):
    """
    为选中的圆弧添加半径标注
    
    Args:
        doc: AutoCAD文档对象
        arcs: 圆弧对象列表
    
    Returns:
        list: 成功添加的标注对象列表
    """
    added_dims = []
    
    for i, arc in enumerate(arcs):
        try:
            # 获取圆弧的中心点
            center = arc.Center
            
            # 获取圆弧半径
            radius = arc.Radius
            
            # 获取圆弧起始角度和终止角度
            start_angle = arc.StartAngle
            end_angle = arc.EndAngle
            
            # 计算标注点位置（从圆心到圆弧边缘的一个点）
            # 使用起始角度来确定半径线的方向
            dim_point_x = center[0] + radius * math.cos(start_angle)
            dim_point_y = center[1] + radius * math.sin(start_angle)
            mid_point_y = center[1] + radius * math.sin(start_angle)/2
            hight=  mid_point_y -dim_point_y 
            # 圆心点
            center_point = APoint(center[0], center[1])
            # 圆弧边缘点（沿半径方向）
            arc_point = APoint(dim_point_x, dim_point_y)
            
            # 添加半径标注
            dim = doc.ModelSpace.AddDimRadial(center_point,  arc_point, 0)
            
            # 设置标注文字为半径格式
            radius_value = round(radius)
            dim.TextOverride = f"R{radius_value}"
                                # 控制文字位置
            dim.TextMovement = 2  # 固定文字位置
            # 计算起点坐标
            start_y = center[1] + radius * math.sin(start_angle)+ hight
            # 设置文字位置点：x为圆弧中心x，y为起点y
            text_position = APoint( dim_point_x, start_y)
            dim.TextPosition = text_position
            added_dims.append(dim)
            print(f"成功为圆弧 {i+1} 添加半径标注: R{radius_value}")
            
        except Exception as e:
            print(f"为圆弧 {i+1} 添加半径标注时出错: {e}")
            continue
    
    return added_dims


def main():
    """
    主函数 - 为选中圆弧添加半径标注
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
        
        # 为圆弧添加半径标注
        dims = add_radius_dimension_to_arcs(doc, arcs)
        
        if dims:
            print(f"成功为 {len(dims)} 个圆弧添加了半径标注")
        else:
            print("未能为任何圆弧添加半径标注")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")


if __name__ == "__main__":
    main()