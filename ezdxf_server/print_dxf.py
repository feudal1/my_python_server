import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import subprocess
import sys

def dxf_to_pdf(dxf_path, pdf_path):
    """将DXF文件转换为PDF，并将所有字体替换为gbcbig.shx"""
    # 读取 DXF 文件
    doc = ezdxf.readfile(dxf_path)
    
    # 替换所有文本样式中的字体为 gbcbig.shx
    for style in doc.styles:
        style.dxf.font = 'gbcbig.shx'
    
    # 创建渲染环境
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    
    # 创建后端渲染器
    backend = MatplotlibBackend(ax)
    context = RenderContext(doc)
    
    frontend = Frontend(context, backend)
    
    # 渲染图形
    frontend.draw_layout(doc.modelspace())
    backend.finalize()
    
    # 保存为 PDF
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def open_file(filepath):
    """跨平台打开文件"""
    if sys.platform.startswith('darwin'):  # macOS
        subprocess.call(['open', filepath])
    elif sys.platform.startswith('win'):   # Windows
        os.startfile(filepath)
    elif sys.platform.startswith('linux'): # Linux
        subprocess.call(['xdg-open', filepath])

def convert_dxf_full_gui():
    """完整的GUI交互式DXF转PDF功能"""
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 确保对话框置顶显示
    
    # 选择DXF文件
    dxf_path = filedialog.askopenfilename(
        title="选择DXF文件",
        filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
    )
    
    if not dxf_path:
        print("未选择DXF文件，程序退出")
        root.destroy()
        return
    
    # 弹窗选择保存路径和文件名
    pdf_path = filedialog.asksaveasfilename(
        title="保存PDF文件",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        initialfile=os.path.splitext(os.path.basename(dxf_path))[0] + ".pdf"  # 默认文件名为DXF文件名
    )
    
    # 销毁根窗口
    root.destroy()
    
    # 如果用户选择了保存路径，则执行转换
    if pdf_path:
        try:
            dxf_to_pdf(dxf_path, pdf_path)
            print(f"转换完成: {pdf_path}")
            
            # 转换完成后打开PDF文件
            open_file(pdf_path)
        except Exception as e:
            print(f"转换失败: {e}")
    else:
        print("操作已取消")

# 运行完整GUI版本
if __name__ == "__main__":
    convert_dxf_full_gui()