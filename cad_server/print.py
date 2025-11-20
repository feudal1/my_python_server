from pyautocad import Autocad, APoint
import os
import subprocess
import time
import tkinter as tk
from tkinter import filedialog

acad = Autocad(create_if_not_exists=True)
acad.app.Visible = True

# 直接使用活动文档
doc = acad.doc

# 获取DXF文件所在目录作为默认PDF保存路径
dxf_path = doc.FullName
default_pdf_path = os.path.dirname(dxf_path)

# 提示用户选择第一个点
acad.prompt("请选择打印窗口的第一个角点:\n")
point1_raw = doc.Utility.GetPoint()
point1 = acad.aDouble(point1_raw[0], point1_raw[1])

# 提示用户选择第二个点
acad.prompt("请选择打印窗口的第二个角点:\n")
point2_raw = doc.Utility.GetPoint()  # 使用 point1 而不是 point1_raw
point2 = acad.aDouble(point2_raw[0], point2_raw[1])

# 设置打印配置
doc.ActiveLayout.ConfigName = "DWG To PDF.pc3"
# 在现有打印设置代码中添加以下行
doc.ActiveLayout.CenterPlot = True  # 启用居中打印
# 获取并打印可用的打印设备
plot_devices = acad.ActiveDocument.ModelSpace.Layout.GetPlotDeviceNames()
print("Available plot devices:")
for i, device in enumerate(plot_devices, 1):
    print(f"  {i}. {device}")

# 获取并打印可用的纸张尺寸
media_names = doc.ActiveLayout.GetCanonicalMediaNames()
print("Available media sizes:")
for i, media in enumerate(media_names, 1):
    print(f"  {i}. {media}")
    
doc.ActiveLayout.CanonicalMediaName = "ISO_A3_(297.00_x_420.00_MM)"
doc.ActiveLayout.PlotRotation = 1

# 设置打印窗口
doc.ActiveLayout.SetWindowToPlot(point1, point2)
doc.ActiveLayout.PlotType = 4

# 构造PDF文件路径 - 直接使用DXF文件所在目录
pdf_filename = os.path.splitext(doc.Name)[0] + ".pdf"
full_pdf_path = os.path.normpath(os.path.join(default_pdf_path, pdf_filename))

# 检查文件是否已存在，如果存在则添加序号
counter = 1
original_full_pdf_path = full_pdf_path
while os.path.exists(full_pdf_path):
    name_without_ext = os.path.splitext(doc.Name)[0]
    pdf_filename = f"{name_without_ext}_{counter}.pdf"
    full_pdf_path = os.path.normpath(os.path.join(default_pdf_path, pdf_filename))
    counter += 1

# 打印到文件
doc.Plot.PlotToFile(full_pdf_path)
print(f"Printed: {doc.Name} to {full_pdf_path}")

time.sleep(2)

if os.path.exists(full_pdf_path):
    subprocess.Popen([full_pdf_path], shell=True)
    print(f"Opened PDF: {pdf_filename}")
else:
    print(f"Warning: Could not find generated PDF file: {full_pdf_path}")

print("Document printed.")