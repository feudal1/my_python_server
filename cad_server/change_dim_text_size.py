# modify_dimension_text_size.py    
from pyautocad import Autocad, APoint    
  
def format_decimal(value):  
    """  
    智能格式化数值:整数保留0位,非整数保留1位小数  
      
    :param value: 要格式化的数值  
    :return: 格式化后的数值(整数或保留1位小数的浮点数)  
    """  
    if value == int(value):  
        return int(value)  
    else:  
        return int(value) 
        return round(value, 0)  
  
def get_selected_dimensions(acad):    
    """     
    获取用户选择的标注对象    
    """    
    try:    
        # 使用 PyAutoCAD 的 get_selection 方法    
        selection = acad.get_selection("请选择标注对象")    
            
        dimensions = []     
        for i in range(selection.Count):    
            try:    
                obj = selection.Item(i)    
                # 检查是否为标注对象    
                if "Dimension" in obj.ObjectName:    
                    dimensions.append(obj)    
            except Exception as e:    
                print(f"无法访问选中对象 {i}: {e}")    
            
        return dimensions    
    except Exception as e:    
        print(f"获取选择集时出错: {e}")    
        return []    
    
def get_current_text_height(dimensions):    
    """    
    获取当前选中标注的文字高度(取第一个作为参考)    
    """    
    if dimensions:    
        try:    
            current_height = dimensions[0].TextHeight    
            return format_decimal(current_height)  # 使用智能格式化  
        except Exception as e:    
            print(f"获取当前文字高度时出错: {e}")    
    return 2.5    
    
def get_text_height_from_cad(acad, current_height):    
    """    
    通过CAD命令行获取用户输入的文字高度    
    """    
    try:    
        # 显示提示信息    
        acad.prompt(f"当前文字高度: {current_height}\n")    
            
        # GetReal 只接受一个参数(提示字符串)    
        prompt = f"请输入新的标注文字高度 <{current_height}>: "    
        result = acad.doc.Utility.GetReal(prompt)    
        return result    
    except Exception as e:    
        print(f"从CAD获取输入时出错: {e}")    
        return None    
    
def modify_dimension_properties(acad, dimensions, text_height=None, color=None):    
    """    
    修改标注的属性(文字高度、颜色、图层颜色和文字内容的小数位数)    
        
    :param acad: Autocad 实例    
    :param dimensions: 标注对象列表    
    :param text_height: 新的文字高度(可选)    
    :param color: 新的颜色索引(可选,3=绿色)    
    """    
    modified_count = 0    
    modified_layers = set()  # 记录已修改的图层,避免重复修改    
        
    for i, dim in enumerate(dimensions):    
        try:    
            if text_height is not None:    
                # 格式化文字高度  
                formatted_height = format_decimal(text_height)  
                dim.TextHeight = formatted_height    
                print(f"已修改标注 {i+1} 的文字高度为 {formatted_height}")    
                  
                # 修改标注文字内容的小数位数  
                try:  
                    # 获取标注的测量值  
                    measurement = dim.Measurement  
                    # 格式化测量值  
                    formatted_text = str(format_decimal(measurement))  
                    # 设置文字覆盖  
                    dim.TextOverride = formatted_text  
                    print(f"已修改标注 {i+1} 的文字内容为 {formatted_text}")  
                except Exception as text_error:  
                    print(f"修改标注 {i+1} 文字内容时出错: {text_error}")  
                
            if color is not None:    
                # 修改标注对象的颜色    
                dim.Color = color    
                print(f"已修改标注 {i+1} 的颜色为 {color}")    
                    
                # 获取标注所在的图层并修改图层颜色    
                layer_name = dim.Layer    
                if layer_name not in modified_layers:    
                    try:    
                        layer = acad.doc.Layers.Item(layer_name)    
                        layer.Color = color    
                        modified_layers.add(layer_name)    
                        print(f"已修改图层 '{layer_name}' 的颜色为 {color}")    
                    except Exception as layer_error:    
                        print(f"修改图层 '{layer_name}' 颜色时出错: {layer_error}")    
                
            modified_count += 1    
        except Exception as e:    
            print(f"修改标注 {i+1} 时出错: {e}")    
        
    return modified_count    
    
def main():    
    """    
    主函数    
    """    
    try:    
        # 连接到正在运行的 AutoCAD    
        acad = Autocad(create_if_not_exists=True)    
        print(f"成功连接到 AutoCAD 文档: {acad.doc.Name}")    
        acad.doc.SetVariable("DIMTXT", 150) 
        acad.doc.SetVariable("DIMDEC", 0)
    except Exception as e:    
        print(f"无法连接到 AutoCAD: {e}")    
        return    
        
    try:    
        # 获取选中的标注对象    
        dimensions = get_selected_dimensions(acad)    
            
        if not dimensions:    
            print("没有找到任何标注对象")    
            return    
            
        print(f"找到 {len(dimensions)} 个标注对象")    
            
        # 获取当前文字高度    
        current_height = get_current_text_height(dimensions)    
        print(f"当前文字高度: {current_height}")    
            
        # 通过CAD命令行获取输入    
        text_height = get_text_height_from_cad(acad, current_height)    
            
        if text_height is None:    
            print("用户取消了操作或输入无效")    
            return    
            
        # 使用智能格式化(整数保留0位,非整数保留1位小数)  
        text_height = format_decimal(text_height)  
        print(f"设置新的文字高度为: {text_height}")    
            
        # 修改标注的文字高度、颜色、图层颜色和文字内容(绿色=3)    
        modified_count = modify_dimension_properties(acad, dimensions, text_height=text_height, color=7)    
            
        print(f"成功修改了 {modified_count} 个标注的文字高度、颜色和文字内容")    
            
        # 刷新视图    
        try:    
            acad.doc.Regen(1)  # acAllViewports = 1    
        except Exception as regen_error:    
            print(f"视图刷新失败: {regen_error}")    
                
    except Exception as e:    
        print(f"处理对象时出错: {e}")    
    
if __name__ == "__main__":    
    main()