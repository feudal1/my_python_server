from pyautocad import Autocad, APoint


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

def process_entities(entities):
    """
    处理实体列表，提取直线并计算边界
    """
    lines = []
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    has_valid_bounds = False
    
    for i, entity in enumerate(entities):
        try:
            # 判断是否为直线
            if entity.ObjectName == "AcDbLine":
                lines.append(entity)
                
                # 获取实体的边界框
                try:
                    bounds = entity.Bounds
                    if bounds and len(bounds) >= 6:
                        # bounds格式: (min_x, min_y, min_z, max_x, max_y, max_z)
                        min_x = min(min_x, bounds[0])
                        min_y = min(min_y, bounds[1])
                        min_z = min(min_z, bounds[2])
                        max_x = max(max_x, bounds[3])
                        max_y = max(max_y, bounds[4])
                        max_z = max(max_z, bounds[5])
                        has_valid_bounds = True
                    else:
                        print(f"对象 {i} 没有有效的边界信息")
                except Exception as bounds_error:
                    print(f"无法获取对象 {i} 的边界: {bounds_error}")
            
            # 处理多段线
            elif entity.ObjectName == "AcDbPolyline":
                print(f"对象 {i} 是多段线，正在提取线段...")
                # 获取多段线的顶点数
                try:
                    coordinates = entity.Coordinates
                    # 处理坐标数组，每两个点构成一条线段
                    segment_count = 0
                    for j in range(0, len(coordinates) - 2, 2):
                        # 对于每一对点，我们创建一个虚拟的直线表示
                        x1, y1 = coordinates[j], coordinates[j+1]
                        x2, y2 = coordinates[j+2], coordinates[j+3]
                        
                        # 更新边界
                        min_x = min(min_x, x1, x2)
                        min_y = min(min_y, y1, y2)
                        max_x = max(max_x, x1, x2)
                        max_y = max(max_y, y1, y2)
                        has_valid_bounds = True
                        segment_count += 1
                        
                    print(f"从多段线中提取了 {segment_count} 条线段")
                except Exception as poly_error:
                    print(f"处理多段线 {i} 时出错: {poly_error}")
            
            else:
                print(f"对象 {i} 类型: {entity.ObjectName}")
        except Exception as e:
            print(f"处理对象 {i} 时出错: {e}")
            continue
    
    return lines, (min_x, min_y, min_z, max_x, max_y, max_z), has_valid_bounds

def calculate_dimensions(bounds):
    """
    根据边界计算长宽
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    if all(isinstance(x, (int, float)) and abs(x) != float('inf') for x in [min_x, min_y, max_x, max_y]):
        length = round(max_x - min_x, 1)
        width = round(max_y - min_y, 1)
        return length, width
    return None, None

def add_dal_aligned_dimension(doc, bounds):
    """
    添加DAL对齐标注
    """
    try:
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        length = round(max_x - min_x, 1)
        width = round(max_y - min_y, 1)
        
        # 计算偏移量为尺寸的十分之一
        offset = max(length, width) * 0.1
        
        # 定义标注的两个点（水平方向）
        pt1 = APoint(min_x, min_y)
        pt2 = APoint(max_x, min_y)
        # 标注线的位置（偏移）
        line_point = APoint((min_x + max_x) / 2, min_y - offset)
        
        # 添加水平对齐标注
        dim1 = doc.ModelSpace.AddDimAligned(pt1, pt2, line_point)
        dim1.TextOverride = f"{length}"
        
        # 定义标注的两个点（垂直方向）
        pt3 = APoint(min_x, min_y)
        pt4 = APoint(min_x, max_y)
        # 标注线的位置（偏移）
        line_point2 = APoint(min_x - offset, (min_y + max_y) / 2)
        
        # 添加垂直对齐标注
        dim2 = doc.ModelSpace.AddDimAligned(pt3, pt4, line_point2)
        dim2.TextOverride = f"{width}"
        
        return dim1, dim2
    except Exception as e:
        print(f"添加对齐标注失败: {e}")
        return None, None

def main():
    """
    主函数
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
        
        # 处理实体
        lines, bounds, has_valid_bounds = process_entities(entities)
        
        # 计算长宽
        if has_valid_bounds:
            length, width = calculate_dimensions(bounds)
            if length is not None and width is not None:
                print(f"长：{length}, 宽：{width}")
                # 添加DAL对齐标注
                dim1, dim2 = add_dal_aligned_dimension(doc, bounds)
                if dim1 and dim2:
                    print("成功添加DAL对齐标注")
                else:
                    print("添加DAL对齐标注失败")
        
        print(f"找到的直线数量: {len(lines)}")
        

    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()