import pandas as pd
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil

def extract_drawing_number(part_number):
    """
    从零件号中提取图号（前四个数字段，如21085-3000-1001-0500）
    """
    # 匹配至少四个数字段的模式
    pattern = r'(\d+-\d+-\d+-\d+)'
    match = re.search(pattern, part_number)
    return match.group(1) if match else None

def select_excel_file():
    """
    弹窗选择Excel文件
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择Excel文件",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    return file_path

def select_folder():
    """
    弹窗选择文件夹
    """
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择包含PDF文件的文件夹")
    return folder_path

def get_sheet_names(excel_file):
    """
    获取Excel文件中的所有工作表名称
    """
    xl = pd.ExcelFile(excel_file)
    return xl.sheet_names

def select_sheet(sheet_names):
    """
    弹窗选择工作表
    """
    root = tk.Tk()
    root.title("选择工作表")
    
    selected_sheet = tk.StringVar()
    
    tk.Label(root, text="请选择工作表:").pack(pady=10)
    
    for sheet in sheet_names:
        tk.Radiobutton(root, text=sheet, variable=selected_sheet, value=sheet).pack(anchor=tk.W)
    
    selected_sheet.set(sheet_names[0])  # 默认选择第一个
    
    def confirm():
        root.quit()
        root.destroy()
    
    tk.Button(root, text="确定", command=confirm).pack(pady=10)
    
    root.mainloop()
    return selected_sheet.get()

def process_excel_and_find_pdfs():
    """
    主处理函数
    """
    try:
        # 1. 选择Excel文件
        excel_file = select_excel_file()
        if not excel_file:
            messagebox.showinfo("提示", "未选择Excel文件")
            return
        
        # 2. 选择工作表
        sheet_names = get_sheet_names(excel_file)
        if len(sheet_names) == 1:
            sheet_name = sheet_names[0]
        else:
            sheet_name = select_sheet(sheet_names)
        
        # 3. 读取Excel数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 检查是否存在"零件号"列
        if '零件号' not in df.columns:
            messagebox.showerror("错误", "未找到'零件号'列")
            return
        
        # 4. 选择PDF文件所在的文件夹
        pdf_folder = select_folder()
        if not pdf_folder:
            messagebox.showinfo("提示", "未选择PDF文件夹")
            return
        
        # 5. 创建保存PDF的目标文件夹（在Excel文件同目录下）
        excel_dir = os.path.dirname(excel_file)
        target_folder = os.path.join(excel_dir, "提取的PDF文件")
        os.makedirs(target_folder, exist_ok=True)
        
        # 6. 处理每个零件号，提取图号并查找PDF
        found_count = 0
        not_found_list = []
        
        for index, row in df.iterrows():
            part_number = str(row['零件号'])
            drawing_number = extract_drawing_number(part_number)
            
            if drawing_number:
                # 查找包含图号的PDF文件
                found_pdf = False
                for filename in os.listdir(pdf_folder):
                    if filename.lower().endswith('.pdf') and drawing_number in filename:
                        source_pdf_path = os.path.join(pdf_folder, filename)
                        target_pdf_path = os.path.join(target_folder, filename)
                        shutil.copy2(source_pdf_path, target_pdf_path)
                        found_count += 1
                        found_pdf = True
                        break
                
                if not found_pdf:
                    not_found_list.append(f"{part_number} (图号: {drawing_number})")
            else:
                not_found_list.append(f"{part_number} (无法提取图号)")
        
        # 7. 显示结果
        result_message = f"处理完成!\n\n成功找到并复制了 {found_count} 个PDF文件。\n"
        if not_found_list:
            result_message += f"\n以下 {len(not_found_list)} 个项目未找到对应的PDF文件:\n"
            result_message += "\n".join(not_found_list[:10])  # 只显示前10个
            if len(not_found_list) > 10:
                result_message += f"\n... 还有 {len(not_found_list)-10} 个"
        
        messagebox.showinfo("处理结果", result_message)
        
    except Exception as e:
        messagebox.showerror("错误", f"处理过程中出现错误:\n{str(e)}")

# 运行程序
if __name__ == "__main__":
    process_excel_and_find_pdfs()