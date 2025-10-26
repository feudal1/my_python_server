import win32com.client
import pandas as pd
from collections import defaultdict
import os

def export_part_list_to_excel():
    """
    通过SolidWorks COM接口获取当前文档零件信息并导出到Excel
    """
    try:
        # 连接SolidWorks应用程序
        sw_app = win32com.client.Dispatch("SldWorks.Application")
        if not sw_app:
            raise Exception("无法连接到SolidWorks应用程序")
        
        # 获取当前活动文档
        sw_model = sw_app.ActiveDoc
        if not sw_model:
            raise Exception("没有打开的文档")
        
        # 检查文档类型
        doc_type = sw_model.GetType()
        if doc_type != 2:  # 2表示装配体文档
            raise Exception("当前文档不是装配体文档")
        
        # 获取装配体中的所有零部件
        part_count_dict = get_part_counts(sw_model)
        
        # 导出到Excel
        export_to_excel(part_count_dict)
        
        print("零件清单已成功导出到Excel文件")
        
    except Exception as e:
        print(f"导出失败: {str(e)}")

def get_part_counts(sw_model):
    """
    获取装配体中各零件的数量统计
    
    Args:
        sw_model: SolidWorks文档对象
        
    Returns:
        dict: 零件名称和数量的字典
    """
    part_count_dict = defaultdict(int)
    
    # 获取装配体所有组件
    assembly_doc = sw_model.GetAssemblyDoc()
    components = assembly_doc.GetComponents(False)  # False表示不包括轻化组件
    
    if components:
        for component in components:
            # 获取组件名称
            component_name = component.Name2
            # 获取组件路径
            component_path = component.GetPathName
            
            # 从路径中提取文件名（不包含扩展名）
            if component_path:
                file_name = os.path.splitext(os.path.basename(component_path))[0]
            else:
                file_name = component_name
            
            # 统计数量
            part_count_dict[file_name] += 1
    
    return dict(part_count_dict)

def export_to_excel(part_count_dict, filename="零件清单.xlsx"):
    """
    将零件统计信息导出到Excel文件
    
    Args:
        part_count_dict (dict): 零件名称和数量字典
        filename (str): 输出文件名
    """
    # 准备数据
    data = []
    for part_name, count in part_count_dict.items():
        data.append({
            '零件名称': part_name,
            '数量': count
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 导出到Excel
    df.to_excel(filename, index=False, sheet_name='零件清单')
    
    print(f"数据已导出到: {os.path.abspath(filename)}")

# 使用示例
if __name__ == "__main__":
    export_part_list_to_excel()