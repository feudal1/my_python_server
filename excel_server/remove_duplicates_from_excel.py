import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
import os

def remove_duplicates_by_specified_column():
    """
    弹窗选择Excel文件，按指定列去重（默认B列）
    """
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()
    
    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择Excel文件",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("未选择文件")
        return
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"原始数据行数: {len(df)}")
        print("所有列名:", list(df.columns))
        
        # 默认使用B列（第二列）
        default_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # 让用户确认或更改列名
        column_name = simpledialog.askstring(
            "选择去重列",
            "请输入用于去重的列名:",
            initialvalue=default_column
        )
        
        if not column_name:
            print("未指定列名，操作取消")
            return
            
        # 检查列名是否存在
        if column_name not in df.columns:
            raise ValueError(f"列 '{column_name}' 不存在于文件中")
        
        print(f"按 '{column_name}' 列去重")
        
        # 按指定列去重
        df_deduplicated = df.drop_duplicates(subset=[column_name], keep='first')
        print(f"去重后数据行数: {len(df_deduplicated)}")
        print(f"共删除 {len(df) - len(df_deduplicated)} 行重复数据")
        
        # 保存文件
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(file_dir, f"{name}_去重后{ext}")
        
        df_deduplicated.to_excel(output_path, index=False)
        print(f"去重完成，文件已保存为: {output_path}")
        
        # 显示完成消息
        tk.messagebox.showinfo(
            "去重完成",
            f"按'{column_name}'列去重完成!\n\n"
            f"原始行数: {len(df)}\n"
            f"去重后行数: {len(df_deduplicated)}\n"
            f"删除重复行数: {len(df) - len(df_deduplicated)}"
        )
        
    except Exception as e:
        error_msg = f"处理文件时出错: {e}"
        print(error_msg)
        tk.messagebox.showerror("错误", error_msg)
    finally:
        root.destroy()

# 运行函数
if __name__ == "__main__":
    remove_duplicates_by_specified_column()