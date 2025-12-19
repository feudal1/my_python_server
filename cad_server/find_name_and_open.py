import os  
import tkinter.filedialog  
import tkinter.simpledialog  
from pyautocad import Autocad  
  
# 选择文件夹  
folder_path = tkinter.filedialog.askdirectory()  
  
# 弹窗输入要查找的文本  
search_text = tkinter.simpledialog.askstring("查找", "输入要查找的文本:")  
if not search_text:  
    print("未输入查找文本")  
    exit()  
  
# 递归遍历所有子文件夹查找DWG/DXF文件  
cad_files = []  
for root, dirs, files in os.walk(folder_path):  
    for file in files:  
        if file.lower().endswith(('.dwg', '.dxf')):  
            if search_text.lower() in file.lower():  
                cad_files.append(os.path.join(root, file))  
  
# 在AutoCAD中打开第一个匹配的文件并替换字体  
acad = Autocad(create_if_not_exists=True)  
if cad_files:  
    try:  
        file_to_open = cad_files[0]  
        print(f"尝试打开文件: {file_to_open}")  
          
        # 打开文件  
        acad.app.Documents.Open(file_to_open)  
        print(f"成功打开文件: {file_to_open}")  
          
        # 等待文档完全加载  
        import time  
        time.sleep(0.1)  
          
        # 使用 acad.doc 而不是返回的文档对象  
        doc = acad.doc  
          
        # 检查文档是否有 TextStyles 属性  
        if hasattr(doc, 'TextStyles'):  
            print("找到 TextStyles 集合")  
            for style in doc.TextStyles:  
                try:  
                    # 尝试多种方式更新字体
                    if hasattr(style, 'FontFile'):
                        style.FontFile = 'gbcbig.shx'
                        print(f"已更新样式 {style.Name} 的字体为 gbcbig.shx")
                    elif hasattr(style, 'font'):
                        style.font = 'gbcbig.shx'
                        print(f"已更新样式 {style.Name} 的字体为 gbcbig.shx")
                    else:
                        print(f"样式 {style.Name} 不支持字体修改")
                except Exception as e:  
                    print(f"更新样式 {style.Name} 失败: {e}")
                  
    except Exception as e:  
        print(f"打开文件时出错: {e}")  
        print(f"错误类型: {type(e).__name__}")  
else:  
    print("未找到文件名包含该文本的DWG或DXF文件")
try:
    import win32gui
    import win32con
    
    def find_autocad_window():
        """查找包含'AutoCAD'字样的窗口"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if 'autocad' in window_title.lower():
                    windows.append((hwnd, window_title))
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows
    
    # 查找AutoCAD窗口
    autocad_windows = find_autocad_window()
    
    if autocad_windows:
        # 获取第一个匹配的窗口
        hwnd, title = autocad_windows[0]
        print(f"找到AutoCAD窗口: {title}")
        
        # 将窗口带到前台并置顶
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        print("AutoCAD窗口已置顶")
    else:
        print("未找到AutoCAD窗口")
except ImportError:
    print("缺少win32gui库，请安装pywin32")
except Exception as e:
    print(f"设置窗口置顶失败: {e}")