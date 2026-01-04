import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    # 设置工作目录
    work_dir = r"E:\code\my_python_server"
    
    # 切换到工作目录
    os.chdir(work_dir)
    
    # Python解释器路径
    python_executable = r"e:\code\my_python_server\micromambavenv\python.exe"
    
    # 脚本路径
    script_path = r"E:\code\my_python_server\llm_server\memory_llm.py"
    
    try:
        # 执行Python脚本
        result = subprocess.run([
            python_executable,
            script_path
        ], cwd=work_dir)
        
        # 模拟批处理中的pause命令
        print("按任意键继续. . . ", end="")
        try:
            # 尝试使用msvcrt（Windows）
            import msvcrt
            msvcrt.getch()
        except ImportError:
            # 对于跨平台兼容，使用input
            input()
        
    except FileNotFoundError:
        print(f"错误: 找不到Python解释器 {python_executable}")
        print("按任意键继续. . . ", end="")
        input()
    except Exception as e:
        print(f"执行出错: {e}")
        print("按任意键继续. . . ", end="")
        input()

if __name__ == "__main__":
    main()