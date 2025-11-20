import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PyPDF2 import PdfMerger
import subprocess
import platform

class PDFMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF合并工具")
        self.root.geometry("500x400")
        
        self.selected_files = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="PDF合并工具", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # 文件列表显示框
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(list_frame, text="已选择的PDF文件:").pack(anchor=tk.W)
        
        # 创建列表框和滚动条
        listbox_frame = tk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.listbox = tk.Listbox(listbox_frame)
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # 选择文件按钮
        select_btn = tk.Button(button_frame, text="选择PDF文件", command=self.select_files, 
                              bg="#4CAF50", fg="white", padx=10, pady=5)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # 合并按钮
        merge_btn = tk.Button(button_frame, text="合并PDF", command=self.merge_pdfs,
                             bg="#2196F3", fg="white", padx=10, pady=5)
        merge_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除列表按钮
        clear_btn = tk.Button(button_frame, text="清除列表", command=self.clear_list,
                             bg="#f44336", fg="white", padx=10, pady=5)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 上移/下移按钮
        move_frame = tk.Frame(self.root)
        move_frame.pack(pady=5)
        
        up_btn = tk.Button(move_frame, text="上移", command=self.move_up)
        up_btn.pack(side=tk.LEFT, padx=5)
        
        down_btn = tk.Button(move_frame, text="下移", command=self.move_down)
        down_btn.pack(side=tk.LEFT, padx=5)
    
    def select_files(self):
        """选择PDF文件"""
        files = filedialog.askopenfilenames(
            title="选择PDF文件",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                filename = os.path.basename(file)
                self.listbox.insert(tk.END, filename)
    
    def clear_list(self):
        """清除文件列表"""
        self.selected_files.clear()
        self.listbox.delete(0, tk.END)
    
    def move_up(self):
        """上移选中的文件"""
        try:
            selection = self.listbox.curselection()[0]
            if selection > 0:
                # 交换列表中的位置
                self.selected_files[selection], self.selected_files[selection-1] = \
                    self.selected_files[selection-1], self.selected_files[selection]
                
                # 更新列表框显示
                self.update_listbox()
                self.listbox.selection_set(selection-1)
        except IndexError:
            pass
    
    def move_down(self):
        """下移选中的文件"""
        try:
            selection = self.listbox.curselection()[0]
            if selection < len(self.selected_files) - 1:
                # 交换列表中的位置
                self.selected_files[selection], self.selected_files[selection+1] = \
                    self.selected_files[selection+1], self.selected_files[selection]
                
                # 更新列表框显示
                self.update_listbox()
                self.listbox.selection_set(selection+1)
        except IndexError:
            pass
    
    def update_listbox(self):
        """更新列表框显示"""
        self.listbox.delete(0, tk.END)
        for file in self.selected_files:
            filename = os.path.basename(file)
            self.listbox.insert(tk.END, filename)
    
    def merge_pdfs(self):
        """合并PDF文件"""
        if len(self.selected_files) < 2:
            messagebox.showwarning("警告", "请至少选择两个PDF文件进行合并！")
            return
        
        try:
            # 创建PDF合并对象
            merger = PdfMerger()
            
            # 添加所有选中的PDF文件
            for pdf_file in self.selected_files:
                merger.append(pdf_file)
            
            # 选择保存路径
            output_path = filedialog.asksaveasfilename(
                title="保存合并后的PDF",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if output_path:
                # 保存合并后的PDF
                with open(output_path, "wb") as output_file:
                    merger.write(output_file)
                
                merger.close()
                
                # 合并完成后打开文件
                self.open_file(output_path)
                
                messagebox.showinfo("成功", f"PDF合并完成！\n保存路径：{output_path}")
            
        except Exception as e: 
            messagebox.showerror("错误", f"合并过程中出现错误：\n{str(e)}")
    
    def open_file(self, file_path):
        """根据操作系统打开文件"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", file_path])
            elif system == "Linux":
                subprocess.call(["xdg-open", file_path])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件：\n{str(e)}")

def main():
    root = tk.Tk()
    app = PDFMergerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()