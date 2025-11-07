import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def merge_excel_files_no_header():
    """
    弹出文件选择对话框，让用户选择多个Excel文件，
    直接合并所有数据，不依赖列名
    """
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title="请选择要合并的Excel文件",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    
    # 如果没有选择文件，则退出
    if not file_paths:
        messagebox.showinfo("提示", "未选择任何文件")
        return
    
    # 检查是否选择了多个文件
    if len(file_paths) < 2:
        messagebox.showwarning("警告", "至少需要选择两个Excel文件进行合并")
        return
    
    try:
        # 存储所有数据框的列表
        dataframes = []
        file_names = []
        
        # 读取每个Excel文件，不使用任何行作为列名
        for file_path in file_paths:
            # 读取Excel文件，将所有行都作为数据处理
            df = pd.read_excel(file_path, header=None)
            
            # 添加一列来源文件名，便于追踪数据来源
            # 创建一个与数据行数相同的文件名列
            df[len(df.columns)] = [os.path.basename(file_path)] * len(df)
            
            dataframes.append(df)
            file_names.append(os.path.basename(file_path))
            
            print(f"已读取文件: {file_path}, 共 {len(df)} 行数据")
            print(f"列数: {len(df.columns)}")
        
        # 合并所有数据框，outer join确保所有列都被保留
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # 获取第一个文件所在的目录作为输出目录
        output_dir = os.path.dirname(file_paths[0])
        
        # 生成输出文件路径
        output_file = os.path.join(output_dir, "合并结果.xlsx")
        
        # 如果文件已存在，添加序号避免覆盖
        counter = 1
        original_output_file = output_file
        while os.path.exists(output_file):
            name, ext = os.path.splitext(original_output_file)
            output_file = f"{name}_{counter}{ext}"
            counter += 1
        
        # 保存合并后的Excel文件，不保存列名
        merged_df.to_excel(output_file, index=False, header=False)
        
        # 显示成功消息
        success_message = f"文件合并完成！\n保存位置: {output_file}\n\n"
        success_message += f"合并了 {len(file_paths)} 个文件:\n" + "\n".join(file_names)
        success_message += f"\n\n总数据行数: {len(merged_df)}"
        success_message += f"\n总列数: {len(merged_df.columns)}"
        
        messagebox.showinfo("成功", success_message)
        
        # 打开文件所在目录
        os.startfile(output_dir)  # Windows系统
        
    except Exception as e:
        messagebox.showerror("错误", f"合并过程中出现错误:\n{str(e)}")
    finally:
        root.destroy()

def merge_excel_files_with_source_column():
    """
    合并Excel文件，添加来源列但不使用原有列名
    """
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title="请选择要合并的Excel文件",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    
    # 如果没有选择文件，则退出
    if not file_paths:
        messagebox.showinfo("提示", "未选择任何文件")
        return
    
    # 检查是否选择了多个文件
    if len(file_paths) < 2:
        messagebox.showwarning("警告", "至少需要选择两个Excel文件进行合并")
        return
    
    try:
        # 存储所有数据框的列表
        dataframes = []
        file_names = []
        
        # 读取每个Excel文件
        for file_path in file_paths:
            # 读取Excel文件，将所有行都作为数据处理
            df = pd.read_excel(file_path, header=None)
            
            # 添加来源文件名列
            source_col_index = len(df.columns)
            df[source_col_index] = [os.path.basename(file_path)] * len(df)
            
            dataframes.append(df)
            file_names.append(os.path.basename(file_path))
        
        # 合并所有数据框
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # 获取第一个文件所在的目录作为输出目录
        output_dir = os.path.dirname(file_paths[0])
        
        # 生成输出文件路径
        output_file = os.path.join(output_dir, "合并结果.xlsx")
        
        # 如果文件已存在，添加序号避免覆盖
        counter = 1
        original_output_file = output_file
        while os.path.exists(output_file):
            name, ext = os.path.splitext(original_output_file)
            output_file = f"{name}_{counter}{ext}"
            counter += 1
        
        # 保存合并后的Excel文件
        merged_df.to_excel(output_file, index=False, header=False)
        
        # 显示成功消息
        success_message = f"文件合并完成！\n保存位置: {output_file}\n\n"
        success_message += f"合并了 {len(file_paths)} 个文件:\n" + "\n".join(file_names)
        success_message += f"\n\n总数据行数: {len(merged_df)}"
        success_message += f"\n总列数: {len(merged_df.columns)}"
        
        messagebox.showinfo("成功", success_message)
        
        # 打开文件所在目录
        os.startfile(output_dir)
        
    except Exception as e:
        messagebox.showerror("错误", f"合并过程中出现错误:\n{str(e)}")
    finally:
        root.destroy()

if __name__ == "__main__":
    # 运行简化版本，直接合并数据
    merge_excel_files_no_header()