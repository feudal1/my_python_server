import os
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

def select_file_via_dialog():
    """通过文件选择对话框选择要打包的Python文件"""
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 置于顶层
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择要打包的Python文件",
        filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
        initialdir=os.getcwd()
    )
    
    # 销毁根窗口
    root.destroy()
    
    return file_path

def build_exe_with_nuitka(script_path):
    """使用Nuitka打包指定的Python文件"""
    # 获取文件名和目录
    script_name = os.path.basename(script_path)
    script_dir = os.path.dirname(script_path)
    
    # 保存当前工作目录
    original_dir = os.getcwd()
    
    # 定义输出文件夹（使用绝对路径）
    output_folder = os.path.abspath(os.path.join(original_dir, "exe_output"))
    
    # 获取不带扩展名的文件名作为exe名称
    exe_name = os.path.splitext(script_name)[0]
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建文件夹: {output_folder}")
    
    # 构建exe文件
    print(f"正在使用Nuitka打包 {script_path}...")
    try:
        # 使用Nuitka打包，通过 python -m nuitka 调用
        python_executable = sys.executable
        venv_path = r"E:\code\my_python_server\micromambavenv"
        
        # 检查虚拟环境是否存在
        if not os.path.exists(venv_path):
            print(f"警告: 虚拟环境路径不存在: {venv_path}")
        
        # 检查 python 可执行文件是否存在
        if not os.path.exists(python_executable):
            print(f"错误: 找不到 Python 可执行文件: {python_executable}")
            return False
        
        # 检查 nuitka 是否可用
        cmd_check = [python_executable, "-m", "nuitka", "--version"]
        print(f"检查Nuitka: {' '.join(cmd_check)}")
        result_check = subprocess.run(cmd_check, capture_output=True, text=True, timeout=10)
        if result_check.returncode != 0:
            raise Exception("Nuitka not found")
        print(f"找到Nuitka版本: {result_check.stdout.strip()}")
        
        # 构建Nuitka命令，添加verbose选项以便看到更多输出
        cmd = [
            python_executable,
            "-m", 
            "nuitka",
            "--standalone",           # 创建独立的可执行文件
            "--onefile",              # 打包成单个文件
            "--output-dir=" + output_folder,  # 指定输出目录
            "--verbose",              # 显示详细输出
            script_path
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("=" * 50)
        print("Nuitka 输出开始:")
        print("=" * 50)
        
        # 实时显示Nuitka输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程结束并获取返回码
        result_code = process.poll()
        
        print("=" * 50)
        print("Nuitka 输出结束")
        print("=" * 50)
        
        if result_code == 0:
            print("打包成功!")
            exe_path = os.path.join(output_folder, f"{exe_name}.exe")
            
            # Nuitka生成的文件名可能与预期不同，需要查找实际生成的文件
            actual_exe = None
            for file in os.listdir(output_folder):
                if file.endswith(".exe") and exe_name in file:
                    actual_exe = os.path.join(output_folder, file)
                    break
            
            if actual_exe and actual_exe != exe_path:
                # 重命名文件为期望的名称
                try:
                    if os.path.exists(exe_path):
                        os.remove(exe_path)
                    os.rename(actual_exe, exe_path)
                    print(f"EXE文件已保存到: {exe_path}")
                except Exception as rename_error:
                    print(f"重命名文件时出错: {rename_error}")
                    exe_path = actual_exe  # 使用原始名称
                    print(f"EXE文件保存在: {exe_path}")
            elif os.path.exists(exe_path):
                print(f"EXE文件已保存到: {exe_path}")
            else:
                print("警告: EXE文件未在预期位置找到")
                
            return True
        else:
            print("打包失败!")
            return False
            
    except subprocess.TimeoutExpired:
        print("检查Nuitka超时，请确保Nuitka已正确安装")
        return False
    except Exception as e:
        print(f"打包过程中出现错误: {e}")
        return False
    finally:
        # 清理临时文件夹和文件
        temp_patterns = ["build", "__pycache__"]
        script_basename = os.path.splitext(os.path.basename(script_path))[0]
        temp_patterns.append(f"{script_basename}.dist")
        temp_patterns.append(f"{script_basename}.build")
        temp_patterns.append(f"{script_basename}.onefile-dist")
        temp_patterns.append(f"{script_basename}.onefile-build")
        
        for pattern in temp_patterns:
            temp_path = os.path.join(original_dir, pattern)
            if os.path.exists(temp_path):
                try:
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                    else:
                        os.remove(temp_path)
                    print(f"已清理临时项: {pattern}")
                except Exception as e:
                    print(f"清理临时项 {pattern} 失败: {e}")

def main():
    print("Python转EXE打包工具 (使用Nuitka)")
    print("=" * 35)
    
    # 通过对话框选择文件
    selected_file = select_file_via_dialog()
    
    if not selected_file:
        print("未选择文件，程序退出")
        return
    
    # 检查是否为Python文件
    if not selected_file.endswith('.py'):
        print("错误: 请选择Python文件(.py)")
        return
    
    # 检查文件是否存在
    if not os.path.exists(selected_file):
        print(f"错误: 文件 {selected_file} 不存在")
        return
    
    # 检查是否安装了Nuitka
    try:
        python_executable = sys.executable
        venv_path = r"E:\code\my_python_server\micromambavenv"
        
        # 检查虚拟环境是否存在
        if not os.path.exists(venv_path):
            print(f"警告: 虚拟环境路径不存在: {venv_path}")
            
        # 检查 python 可执行文件是否存在
        if not os.path.exists(python_executable):
            raise Exception("Python executable not found")
            
        cmd = [python_executable, "-m", "nuitka", "--version"]
        print(f"检查Nuitka: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception("Nuitka not found")
        print(f"找到Nuitka版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print("错误: 未检测到Nuitka，请先安装Nuitka")
        print("可以通过以下命令安装: pip install nuitka")
        print(f"错误详情: {e}")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("缺少依赖", "未检测到Nuitka，请先安装Nuitka\n\n可以通过以下命令安装:\npip install nuitka")
        root.destroy()
        return
    
    # 打包选定的文件
    print(f"\n开始打包: {selected_file}")
    success = build_exe_with_nuitka(selected_file)
    
    # 显示结果
    if success:
        print("\n打包完成!")
        # 显示成功消息框
        root = tk.Tk()
        root.withdraw()
        output_folder = os.path.abspath("exe_output")
        exe_name = os.path.splitext(os.path.basename(selected_file))[0]
        messagebox.showinfo("打包完成", f"EXE文件已成功生成!\n\n保存位置:\n{output_folder}\\{exe_name}.exe")
        root.destroy()
    else:
        print("\n打包失败!")
        # 显示错误消息框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("打包失败", "打包过程中出现错误，请查看控制台输出")
        root.destroy()

if __name__ == "__main__":
    main()