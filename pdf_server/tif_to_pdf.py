# tif_to_pdf.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import os

class TifToPdfConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("TIF转PDF工具")
        self.root.geometry("500x300")
        
        self.selected_files = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="TIF转PDF工具", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # 说明文本
        info_label = tk.Label(self.root, text="选择一个或多个TIF文件转换为PDF格式", 
                             font=("Arial", 10))
        info_label.pack(pady=5)
        
        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # 选择文件按钮
        select_btn = tk.Button(button_frame, text="选择TIF文件", command=self.select_files, 
                              bg="#4CAF50", fg="white", padx=10, pady=5)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # 转换按钮
        convert_btn = tk.Button(button_frame, text="转换为PDF", command=self.convert_to_pdf,
                               bg="#2196F3", fg="white", padx=10, pady=5)
        convert_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除列表按钮
        clear_btn = tk.Button(button_frame, text="清除列表", command=self.clear_list,
                             bg="#f44336", fg="white", padx=10, pady=5)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 文件数量标签
        self.count_label = tk.Label(self.root, text="已选择文件数: 0")
        self.count_label.pack(pady=5)
    
    def select_files(self):
        """选择TIF文件"""
        files = filedialog.askopenfilenames(
            title="选择TIF文件",
            filetypes=[("TIF files", "*.tif"), ("TIFF files", "*.tiff"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
        
        self.update_count()
    
    def clear_list(self):
        """清除文件列表"""
        self.selected_files.clear()
        self.update_count()
    
    def update_count(self):
        """更新文件计数显示"""
        self.count_label.config(text=f"已选择文件数: {len(self.selected_files)}")
    
    def convert_to_pdf(self):
        """将TIF文件转换为PDF"""
        if len(self.selected_files) == 0:
            messagebox.showwarning("警告", "请先选择TIF文件！")
            return
        
        try:
            converted_count = 0
            
            for tif_file in self.selected_files:
                # 获取文件名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(tif_file))[0]
                
                # 构建输出路径（与源文件同一目录）
                output_path = os.path.join(os.path.dirname(tif_file), f"{base_name}.pdf")
                
                # 打开TIF图像并保存为PDF
                image = Image.open(tif_file)
                image.save(output_path, "PDF", resolution=100.0)
                image.close()
                
                converted_count += 1
            
            # 显示结果消息
            messagebox.showinfo("成功", f"成功转换 {converted_count} 个文件！")
            
        except Exception as e:
            messagebox.showerror("错误", f"转换过程中出现错误：\n{str(e)}")

def main():
    root = tk.Tk()
    app = TifToPdfConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()