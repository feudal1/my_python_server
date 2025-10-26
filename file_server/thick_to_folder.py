import os
import re
import shutil

def organize_dxf_files_by_thickness(folder_path):
    """
    根据DXF文件中的厚度信息对文件进行分类整理
    
    Args:
        folder_path (str): 包含DXF文件的文件夹路径
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.dxf'):
            file_path = os.path.join(folder_path, filename)
            
            thickness = None
            
            # 首先尝试从文件内容中提取厚度
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试其他编码
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            
            # 从文件内容中提取厚度信息
            thickness_match = re.search(r'(\d+(?:\.\d+)?)mm', content, re.IGNORECASE)
            if thickness_match:
                thickness = thickness_match.group(1)
            
            # 如果文件内容中没有找到厚度信息，则尝试从文件名中提取
            if not thickness:
                name_thickness_match = re.search(r'(\d+(?:\.\d+)?)\s*mm', filename, re.IGNORECASE)
                if name_thickness_match:
                    thickness = name_thickness_match.group(1)
            
            if thickness:
                # 创建对应厚度的子文件夹
                thickness_folder = os.path.join(folder_path, f"{thickness}mm")
                if not os.path.exists(thickness_folder):
                    os.makedirs(thickness_folder)
                
                # 复制文件到对应的子文件夹（而不是移动）
                destination_path = os.path.join(thickness_folder, filename)
                shutil.copy2(file_path, destination_path)
                print(f"已复制 {filename} 到 {thickness}mm 文件夹")
            else:
                print(f"未找到厚度信息: {filename}")

# 使用示例
if __name__ == "__main__":
    # 获取用户输入的文件夹路径
    dxf_folder = input("请输入包含DXF文件的文件夹路径（直接回车默认使用 'dxf'）: ").strip()
    if not dxf_folder:
        dxf_folder = "dxf"  # 默认路径
    
    organize_dxf_files_by_thickness(dxf_folder)