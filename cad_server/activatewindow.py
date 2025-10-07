from pyautocad import Autocad
import win32gui

acad = Autocad()
# 获取AutoCAD窗口句柄并激活
hwnd = acad.app.HWND
win32gui.SetForegroundWindow(hwnd)