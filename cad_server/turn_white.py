# change_dimensions_to_white.py
from pyautocad import Autocad, APoint

def get_selected_dimensions(acad):
    """
    获取用户选择的标注对象
    """
    try:
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

def change_dimensions_to_white(acad, dimensions):
    """
    将选中的标注对象全部改为白色显示
    
    :param acad: Autocad实例
    :param dimensions: 标注对象列表
    :return: 成功修改的数量
    """
    modified_count = 0
    modified_layers = set()  # 记录已修改的图层，避免重复修改
    
    for i, dim in enumerate(dimensions):
        try:
            # 修改标注对象的颜色为白色（颜色索引7表示白色）
            dim.Color = 7
            print(f"已修改标注 {i+1} 的颜色为白色")
                
            # 获取标注所在的图层并修改图层颜色为白色
            layer_name = dim.Layer
            if layer_name not in modified_layers:
                try:
                    layer = acad.doc.Layers.Item(layer_name)
                    layer.Color = 7
                    modified_layers.add(layer_name)
                    print(f"已修改图层 '{layer_name}' 的颜色为白色")
                except Exception as layer_error:
                    print(f"修改图层 '{layer_name}' 颜色时出错: {layer_error}")
            
            # 同时修改标注文字的颜色为白色
            try:
                dim.TextColor = 7
                print(f"已修改标注 {i+1} 的文字颜色为白色")
            except AttributeError:
                # 某些版本可能不支持TextColor属性
                pass
                        # 添加尺寸线颜色修改为白色  
            try:  
                dim.DimensionLineColor = 7  
                print(f"已修改标注 {i+1} 的尺寸线颜色为白色")  
            except AttributeError:  
                # 某些版本可能不支持DimensionLineColor属性  
                pass  
                        # 添加尺寸界线颜色修改为白色  
            try:  
                dim.ExtensionLineColor = 7  
                print(f"已修改标注 {i+1} 的尺寸界线颜色为白色")  
            except AttributeError:  
                # 某些版本可能不支持ExtensionLineColor属性  
                pass 
            modified_count += 1
        except Exception as e:
            print(f"修改标注 {i+1} 时出错: {e}")
    
    return modified_count

def main():
    """
    主函数 - 将选中的所有标注元素变为白色
    """
    try:
        # 连接到正在运行的 AutoCAD
        acad = Autocad(create_if_not_exists=True)
        print(f"成功连接到 AutoCAD 文档: {acad.doc.Name}")
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
        
        # 将所有选中的标注改为白色
        modified_count = change_dimensions_to_white(acad, dimensions)
        
        print(f"成功将 {modified_count} 个标注对象改为白色显示")
        
        # 刷新视图
        try:
            acad.doc.Regen(1)  # acAllViewports = 1
            print("视图刷新完成")
        except Exception as regen_error:
            print(f"视图刷新失败: {regen_error}")
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()