import os
import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageEnhance
import io

def invert_pdf_colors(input_pdf_path):
    doc = fitz.open(input_pdf_path)
    inverted_images = []

    print(f"共 {doc.page_count} 页，正在处理...")

    for page_num in range(doc.page_count):
        print(f"处理第 {page_num + 1} 页...")
        # ✅ 正确获取页面对象
        page = doc[page_num]
        # 渲染页面为图像（提高分辨率）
        mat = fitz.Matrix(4.0, 4.0)  # 4x zoom ≈ 288 DPI
        pix = page.get_pixmap(matrix=mat)

        # 转为 PIL 图像
        img_data = pix.tobytes("ppm")
        pil_img = Image.open(io.BytesIO(img_data))

        # 转灰度并增强对比度
        gray_img = pil_img.convert("L")
        # 增强对比度（1.5倍），可以根据需要调整这个值
        enhancer = ImageEnhance.Contrast(gray_img)
        contrast_enhanced = enhancer.enhance(20)
        
        # 反色处理
        inverted = ImageOps.invert(contrast_enhanced)
        inverted_images.append(inverted)

    # 保存为新的 PDF
    output_path = os.path.splitext(input_pdf_path)[0] + "_inverted.pdf"
    inverted_images[0].save(
        output_path,
        save_all=True,
        append_images=inverted_images[1:],
        resolution=300.0
    )
    print(f"✅ 反色完成！已保存至：{output_path}")
    doc.close()

def select_and_process():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="请选择一个 PDF 文件",
        filetypes=[("PDF 文件", "*.pdf")]
    )
    if not file_path:
        print("未选择文件。")
        return
    invert_pdf_colors(file_path)

if __name__ == "__main__":
    select_and_process()