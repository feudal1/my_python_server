from pyautocad import Autocad, APoint  
import os  
import subprocess  
import time  
import tkinter as tk  
from tkinter import filedialog  
  
acad = Autocad(create_if_not_exists=True)  
acad.app.Visible = True  
  
def choose_save_location():  
    root = tk.Tk()  
    root.withdraw()  
    folder_selected = filedialog.askdirectory(  
        title="请选择PDF保存位置",  
        
    )  
    root.destroy()  
    return folder_selected  
  
default_pdf_path = choose_save_location()  
  

# 直接使用活动文档  
doc = acad.doc  
  
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
doc.ActiveLayout.CanonicalMediaName = "ISO_A3_(420.00_x_297.00_MM)"  
doc.ActiveLayout.PlotRotation = 1  
  
# 设置打印窗口  
doc.ActiveLayout.SetWindowToPlot(point1, point2)  
doc.ActiveLayout.PlotType = 4  
  
# 构造PDF文件路径  
pdf_filename = os.path.splitext(doc.Name)[0] + ".pdf"  
full_pdf_path = os.path.normpath(os.path.join(default_pdf_path, pdf_filename))  
  
if os.path.exists(full_pdf_path):  
    os.remove(full_pdf_path)  
  
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