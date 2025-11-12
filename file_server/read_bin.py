import tkinter as tk
from tkinter import filedialog
import struct

def parse_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # 解析逻辑需要根据实际格式调整
    # 这里只是一个基础示例
    print("文件大小:", len(data), "字节")
    print("前100个字节:", data)
    
    # 可以在这里添加具体的解析逻辑
    # 例如查找"LwPolyline"标记并提取坐标点

def select_and_parse():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        title="选择BIN文件",
        filetypes=[("BIN files", "*.bin")]
    )
    
    if file_path:
        print(f"正在解析文件: {file_path}")
        parse_bin_file(file_path)
    else:
        print("未选择文件")

if __name__ == "__main__":
    select_and_parse()