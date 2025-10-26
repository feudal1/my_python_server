import pandas as pd
import os
import re
from pathlib import Path

def merge_excel_data(source_file, target_file, output_file):
    """
    将源Excel文件中的数量信息按零件号匹配到目标Excel文件
    
    Args:
        source_file: 包含完整信息的源Excel文件路径
        target_file: 需要补充信息的目标Excel文件路径
        output_file: 输出文件路径
    """
    # 标准化路径
    source_file = str(Path(source_file).resolve())
    target_file = str(Path(target_file).resolve())
    output_file = str(Path(output_file).resolve())
    
    # 验证源文件和目标文件是否存在
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"目标文件不存在: {target_file}")
    
    # 读取两个Excel文件，源文件使用灵活读取，目标文件使用第二行作为标题
    source_df = read_excel_with_flexible_header(source_file)
    # 直接使用第二行(索引为1)作为目标文件的标题行
    target_df = pd.read_excel(target_file, header=1)
    
    # 显示两个文件的列名
    print("源文件列名:", source_df.columns.tolist())
    print("目标文件列名:", target_df.columns.tolist())
    
    # 自动检测源文件中的相关列
    # 定义可能的列名映射
    column_mapping = {
        '零件号': ['零件号', '图号', '部件号', 'Part Number', 'PartNo', '图号/零件号'],
        '数量': ['数量', 'Qty', 'Quantity', '件数'],

    }
    
    # 查找实际存在的列名
    actual_columns = {}
    for standard_name, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in source_df.columns:
                actual_columns[standard_name] = name
                found = True
                break
        if not found:
            print(f"警告: 未找到 '{standard_name}' 对应的列")
    
    print("映射后的列名:", actual_columns)
    
    # 如果自动检测失败，尝试手动映射（基于列位置）
    if not actual_columns:
        print("尝试基于列位置进行手动映射...")
        actual_columns = manual_column_mapping(source_df)
        print("手动映射后的列名:", actual_columns)
    
    # 检查必需的零件号列是否存在
    if '零件号' not in actual_columns:
        raise KeyError("源文件中未找到零件号相关列")
    
    # 提取需要的列（使用实际列名）
    source_columns_to_extract = [actual_columns['零件号']]
    if '数量' in actual_columns:
        source_columns_to_extract.append(actual_columns['数量'])

    
    source_needed = source_df[source_columns_to_extract]
    
    # 重命名列以匹配标准名称
    rename_dict = {actual_columns[key]: key for key in actual_columns}
    source_needed = source_needed.rename(columns=rename_dict)
    
    # 添加标准化零件号列用于匹配
    source_needed['标准化零件号'] = source_needed['零件号'].apply(normalize_part_number)
    target_df['标准化零件号'] = target_df['零件号'].apply(normalize_part_number)
    
    # 使用标准化零件号合并数据
    result_df = pd.merge(target_df, source_needed, on='标准化零件号', how='left', suffixes=('', '_source'))
    
    # 删除临时的标准化零件号列
    result_df = result_df.drop(['标准化零件号'], axis=1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    result_df.to_excel(output_file, index=False)
    print(f"数据合并完成，已保存到 {output_file}")

def normalize_part_number(part_number):
    """
    标准化零件号，提取基本的数字和连字符部分用于匹配
    例如: 
    "21085-3000-1001-0100-01-" -> "21085-3000-1001-0100-01"
    "21085-3000-1001-0100-01.stp" -> "21085-3000-1001-0100-01"
    """
    if pd.isna(part_number):
        return part_number
    
    part_str = str(part_number)
    
    # 提取以数字开头，由数字和连字符组成的部分，并确保不以连字符结尾
    match = re.search(r'^([\d\-]*\d)', part_str)
    if match:
        return match.group(1)
    
    # 如果没有匹配到，则返回原始值
    return part_str

def read_excel_with_flexible_header(file_path):
    """
    灵活读取Excel文件，尝试不同的header行
    """
    # 首先尝试默认读取
    df = pd.read_excel(file_path)
    
    # 检查是否有很多"Unnamed"列
    unnamed_count = sum(1 for col in df.columns if 'Unnamed' in str(col))
    
    # 如果Unnamed列太多，尝试其他header行
    if unnamed_count > len(df.columns) / 2:
        print(f"检测到文件 {file_path} 可能有标题行问题，尝试不同header行...")
        for header_row in range(0, min(5, len(df))):  # 尝试前5行作为header
            try:
                temp_df = pd.read_excel(file_path, header=header_row)
                unnamed_count_temp = sum(1 for col in temp_df.columns if 'Unnamed' in str(col))
                if unnamed_count_temp < unnamed_count:
                    print(f"使用第 {header_row} 行作为标题行")
                    return temp_df
            except Exception as e:
                continue
    return df

def manual_column_mapping(df):
    """
    当自动检测失败时，基于列位置进行手动映射
    """
    columns = df.columns.tolist()
    mapping = {}
    
    # 如果列数足够，基于位置映射
    if len(columns) >= 2:
        mapping['零件号'] = columns[0]  # 假设第1列是零件号
        mapping['数量'] = columns[1]   # 假设第2列是数量

    
    return mapping

# 加载环境变量
dotenv_path = r'E:\code\apikey\.env'  
from dotenv import load_dotenv
load_dotenv(dotenv_path)  

# 获取用户输入并处理路径
source_file = os.getenv('info_copy_source_file') 
target_file = os.getenv('info_copy_target_file') 
output_file = os.getenv('info_copy_output_file') 
print(f"- info_copy_source_file{source_file}")
print(f"- info_copy_target_file{target_file}")
print(f"- info_copy_output_file{output_file}")
# 检查环境变量是否正确加载
if not source_file or not target_file or not output_file:
    print("错误: 请检查.env文件中的配置，确保以下环境变量已设置:")
    print("- info_copy_source_file")
    print("- info_copy_target_file")
    print("- info_copy_output_file")
    input("按回车键退出...")
else:
    # 标准化路径
    source_file = source_file
    target_file = target_file
    output_file = output_file

    # 调用函数
    try:
        merge_excel_data(source_file, target_file, output_file)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")