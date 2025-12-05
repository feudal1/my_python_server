import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import shutil
from PyPDF2 import PdfMerger


def select_sheet_safe(parent, sheet_names):
    """弹窗选择工作表（使用 Toplevel）"""
    top = tk.Toplevel(parent)
    top.title("选择工作表")
    top.geometry("300x300")
    selected_sheet = tk.StringVar(value=sheet_names[0])

    tk.Label(top, text="请选择工作表:", font=("Arial", 10)).pack(pady=10)

    # 使用 Listbox 显示所有工作表
    listbox = tk.Listbox(top, exportselection=False)
    for name in sheet_names:
        listbox.insert(tk.END, name)
    listbox.selection_set(0)
    listbox.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    def confirm():
        idx = listbox.curselection()
        if idx:
            selected_sheet.set(sheet_names[idx[0]])
            top.quit()
            top.destroy()
        else:
            messagebox.showwarning("警告", "请选择一个工作表")

    tk.Button(top, text="确定", command=confirm, width=15).pack(pady=10)

    top.transient(parent)
    top.grab_set()
    parent.wait_window(top)
    return selected_sheet.get()


def select_column_safe(parent, columns):
    """弹窗选择列（使用 Toplevel）"""
    top = tk.Toplevel(parent)
    top.title("选择列")
    top.geometry("300x180")
    selected_column = tk.StringVar(value=columns[0])

    tk.Label(top, text="请选择包含零件号的列:", font=("Arial", 10)).pack(pady=10)

    combo = ttk.Combobox(top, textvariable=selected_column, values=columns, state="readonly", font=("Arial", 9))
    combo.pack(pady=10)
    combo.current(0)

    def confirm():
        val = selected_column.get()
        if val in columns:
            top.quit()
            top.destroy()
        else:
            messagebox.showwarning("警告", "请选择一个有效列")

    tk.Button(top, text="确定", command=confirm, width=15).pack(pady=10)

    top.transient(parent)
    top.grab_set()
    parent.wait_window(top)
    return selected_column.get()


def normalize_filename(name):
    """
    标准化文件名，去除扩展名并转换为小写
    """
    return os.path.splitext(name)[0].lower()


def process_excel_and_find_pdfs():
    # 创建主 Tk 实例（隐藏）
    root = tk.Tk()
  

    try:
        # 1. 选择Excel文件
        excel_file = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not excel_file:
            messagebox.showinfo("提示", "未选择Excel文件")
            return

        # 2. 获取所有工作表名称
        with pd.ExcelFile(excel_file) as xl:
            sheet_names = xl.sheet_names

        if not sheet_names:
            messagebox.showerror("错误", "Excel文件中没有工作表")
            return

        # 3. 选择工作表
        if len(sheet_names) == 1:
            sheet_name = sheet_names[0]
        else:
            sheet_name = select_sheet_safe(root, sheet_names)

        # 4. 读取选定工作表的数据
        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        if df.empty:
            messagebox.showerror("错误", f"工作表 '{sheet_name}' 为空")
            return

        # 5. 选择包含零件号的列
        columns = list(df.columns.astype(str))  # 确保列名是字符串
        if len(columns) == 1:
            part_number_column = columns[0]
        else:
            part_number_column = select_column_safe(root, columns)

        # 6. 选择PDF文件夹
        pdf_folder = filedialog.askdirectory(title="选择包含PDF文件的文件夹")
        if not pdf_folder:
            messagebox.showinfo("提示", "未选择PDF文件夹")
            return

        # 7. 创建目标文件夹
        excel_dir = os.path.dirname(excel_file)
        target_folder = os.path.join(excel_dir, "提取的PDF文件")
        os.makedirs(target_folder, exist_ok=True)

        # 8. 预处理PDF文件映射
        pdf_map = {}  # 零件号到PDF文件路径的映射
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith('.pdf'):
                normalized_name = normalize_filename(filename)
                pdf_map[normalized_name] = os.path.join(pdf_folder, filename)

        # 9. 处理每一行
        found_count = 0
        not_found_list = []
        pdf_files_in_order = []  # 存储 (目标路径, 源路径) 元组
        copied_files = set()  # 已复制的源文件路径

        for index, row in df.iterrows():
            raw_value = row[part_number_column]
            part_number = str(raw_value).strip() if pd.notna(raw_value) else ""

            if not part_number:
                not_found_list.append(f"空零件号 (行 {index+1})")
                continue

            # 精确匹配零件号
            normalized_part_number = part_number.lower()
            if normalized_part_number in pdf_map:
                src = pdf_map[normalized_part_number]
                filename = os.path.basename(src)
                dst = os.path.join(target_folder, filename)
                
                # 避免重复复制相同文件
                if src not in copied_files:
                    shutil.copy2(src, dst)
                    copied_files.add(src)
                
                # 保存目标文件和源文件的对应关系
                pdf_files_in_order.append((dst, src))
                found_count += 1
            else:
                not_found_list.append(f"{part_number}")

        # 10. 合并PDF（按Excel顺序）并打印信息
        merged_pdf_path = os.path.join(excel_dir, "合并的PDF文件.pdf")
        if pdf_files_in_order:
            print("即将合并以下PDF文件:")
            for i, (dst, src) in enumerate(pdf_files_in_order, 1):
                print(f"{i}. {os.path.basename(dst)} (来自: {src})")
            
            merger = PdfMerger()
            for dst, src in pdf_files_in_order:
                try:
                    merger.append(dst)  # 使用复制后的文件
                except Exception as e:
                    print(f"警告：跳过无效PDF {dst}: {e}")
            merger.write(merged_pdf_path)
            merger.close()
            print(f"\n合并完成，输出文件: {merged_pdf_path}")

        # 11. 显示结果
        msg = f"处理完成！\n\n成功找到并复制了 {found_count} 个PDF文件。\n"
        if pdf_files_in_order:
            msg += f"已生成合并PDF: 合并的PDF文件.pdf\n"
        if not_found_list:
            msg += f"\n{len(not_found_list)} 个项目未找到PDF:\n"
            msg += "\n".join(not_found_list[:10])
            if len(not_found_list) > 10:
                msg += f"\n... 还有 {len(not_found_list)-10} 个"

        messagebox.showinfo("处理结果", msg)


    finally:
        root.destroy()


if __name__ == "__main__":
    process_excel_and_find_pdfs()