
import os
import subprocess
import time
import sys
def activate_ue_window():
    """
    激活ue窗口（精确匹配窗口标题），如果没有运行则启动ue
    """
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow未安装，请运行: pip install pygetwindow")
        return False
    
    try:
        # 获取所有窗口
        all_windows = gw.getAllWindows()
        ue_window = None
        
        for window in all_windows:
            window_title = window.title.strip()
            
            # 多种可能的ue标题格式
            if (window_title == 'ue' or  # 基础标题
                window_title.startswith('ue') and 
                not any(exclude in window_title.lower() for exclude in ['vscode', 'visual studio', 'code'])):
                
                # 额外检查确保不是VSCode或其他编辑器
                if ' - ' not in window_title or 'ue.exe' in window_title.lower():
                    ue_window = window
                    break
        
        if ue_window:
            print(f"激活窗口: {ue_window.title}")
            
            try:
                import win32gui
                import win32con
                
                hwnd = ue_window._hWnd
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                
                print(f"窗口已放到最前端: {ue_window.title}")
                return True
            except ImportError:
                print("pywin32未安装，请运行: pip install pywin32")
                if ue_window.isMinimized:
                    ue_window.restore()
                ue_window.activate()
                print("ue窗口已激活（使用pygetwindow方法）")
                return True
        else:
            print("未找到ue窗口，正在启动ue...")
            # 启动ue
            start_ue()
            return False

    except Exception as e:
        print(f"激活ue窗口时出错: {e}")
        return False
    
def execute_tool(tool_name, *args):
    """
    根据工具名称执行对应的ue操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    tool_functions = {

        "activate_ue": activate_ue_window,

    }
    
    
    if tool_name in tool_functions:
        if tool_name == "activate_ue":
            # 特殊处理：激活ue窗口不需要API调用
            result = tool_functions[tool_name]()
            return {"status": "success", "result": result} if result else {"status": "error", "message": "Failed to activate ue"}
        elif tool_name in ["delete_objects_by_name"] and args:
            # 处理带参数的删除功能
            return tool_functions[tool_name](args[0])
        else:
            # 其他工具调用API
            return tool_functions[tool_name]()
    else:
        print(f"错误: 未知的ue工具 '{tool_name}'")
        return None

def start_ue():
    """
    启动ue（在后台进程启动，不阻塞当前线程）
    """
    ue_path = r"D:\UE_4.26\Engine\Binaries\Win64\UE4Editor.exe"
    
    if not os.path.exists(ue_path):
        print(f"错误: 找不到ue可执行文件: {ue_path}")
        return False
    
    try:
        # 使用CREATE_NEW_CONSOLE标志启动ue，使其在独立的进程中运行
        subprocess.Popen([ue_path], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS)
        print(f"ue已在后台启动: {ue_path}")
        
        time.sleep(2)  # 短暂延迟确保进程已开始启动
        return True
    except Exception as e:
        print(f"启动ue时出错: {e}")
        return False
    
def main():
    # delete_objects_by_name("ObjectName")
    # return 
    tool_name = sys.argv[1]  
    
    # 检查是否有额外参数传递给工具
    args = sys.argv[2:]  # 获取工具名称之后的所有参数
    
    # 执行对应的工具
    response = execute_tool(tool_name, *args)
    



if __name__ == "__main__":
    main()