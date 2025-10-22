import subprocess
import sys
import os
import threading
import time
from pathlib import Path

def clear_log_file(log_path):
    '''清空日志文件'''
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('')
        print(f"已清空日志文件: {log_path}")
    except Exception as e:
        print(f"清空日志文件 {log_path} 时出错: {str(e)}")
        return False
    return True

def tail_file(log_path, name):
    '''实时读取日志文件内容并输出到终端'''
    while True:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # 移到文件末尾
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)  # 短暂休眠以降低 CPU 使用
                        continue
                    # 为每个服务端的输出添加前缀标识
                    print(f"[{name}]: {line.strip()}")
        except FileNotFoundError:
            time.sleep(0.1)  # 文件可能尚未创建，等待重试
        except Exception as e:
            print(f"读取日志文件 {log_path} 时出错: {str(e)}")
            time.sleep(1)

def select_servers(excel_servers, cad_servers, dino_servers,llm_servers):
    '''让用户选择要启动的服务器'''
    print("请选择要启动的服务器:")
    print("1. Excel Servers (默认)")
    print("2. CAD Servers")
    print("3. DINO Servers")
    print("4. llm Servers")
    print("5. 所有服务器")
    
    choice = input("请输入选项编号 (默认为1): ").strip()
    
    if choice == "" or choice == "1":
        print("Server URL: http://localhost:5001/process_excel?filepath=&sheet_name=Sheet1")
        return excel_servers
    elif choice == "2":
        print("Server URL: http://localhost:5301/view_dxf?dxf_path=")
        return cad_servers
    elif choice == "3":
        print("Server URL: http://localhost:5200/process_image?image_path=")
        return dino_servers
    elif choice == "4":
        print('Server URL: http://localhost:5003/vlm/chat?messages=[{"role":"user","content":"描述图片"}]&image_source=/path/to/local/image.jpg')
        return llm_servers
    elif choice == "5":
        return excel_servers + cad_servers + dino_servers+ llm_servers
    else:
        print("无效选项，使用默认的Excel Servers")
        return excel_servers

def start_servers():
    # 定义服务器配置
    excel_servers = [
        {
            "name": "Python SSE Server",
            "command": "micromambavenv\\python excel_server\\sse_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/sse.log"
        },
        {
            "name": "Excel MCP Server",
            "command": "uvx excel-mcp-server sse",
            "cwd": "E:\\code\\excel-mcp-server",
            "log_file": "logs/mcp.log"
        },
        {
            "name": "LLM Server",
            "command": "micromambavenv\\python llm_server\llm_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/llm.log"
        },
        {
            "name": "Thick Part JSON Server",
            "command": "micromambavenv\\python excel_server\\get_thick_part_json.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/thick_part.log"
            
        },
            {
            "name": "json2excel Server",
            "command": "micromambavenv\\python excel_server\\json2excel.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/json2excel.log"
            
        }
        
    ]
    cad_servers = [
        {
            "name": "cad Server",
            "command": "micromambavenv\\python cad_server\\server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad_server.log"
        },
        {
            "name": "base Server",
            "command": "micromambavenv\\python ezdxf_server\\base_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/base_server.log"
        }, 
        {
            "name": "cad2excel Server",
            "command": "micromambavenv\\python cad_server\\extract_parts_from_drawing.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad2excel_server.log"
        },
        {
            "name": "LLM Server",
            "command": "micromambavenv\\python llm_server\llm_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/llm.log"
        },
            {
            "name": "json2excel Server",
            "command": "micromambavenv\\python excel_server\\json2excel.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/json2excel.log"
            
        }
    ]
    dino_servers = [
        {
            "name": "dino Server",
            "command": "micromambavenv\\python dino_server\dino_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/dino_server.log"
        }
    ]
    llm_servers = [
         {
            "name": "LLM Server",
            "command": "micromambavenv\\python llm_server\llm_server.py",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/llm.log"
        }
    ]
    
    # 让用户选择要启动的服务器
    start_servers_list = select_servers(excel_servers, cad_servers, dino_servers, llm_servers)
    
    processes = []
    
    print("正在启动所有服务器...")
    
    # 启动每个服务器
    for server in start_servers_list:
        # 确保日志目录存在
        log_path = Path(server["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 清空日志文件
        if not clear_log_file(log_path):
            cleanup(processes)
            sys.exit(1)
            
        try:
            print(f"启动 {server['name']}...")
            # 打开日志文件(追加模式)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                # 启动进程，重定向输出到日志文件
                process = subprocess.Popen(
                    server["command"],
                    cwd=server["cwd"],
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # 将错误输出也重定向到日志
                    text=True,
                    bufsize=1
                )
                
                # 等待一会检查进程是否仍在运行
                try:
                    return_code = process.wait(timeout=5)
                    if return_code != 0:
                        print(f"错误: {server['name']} 启动失败，返回码: {return_code}")
                        print(f"请检查日志文件: {log_path}")
                        cleanup(processes)
                        sys.exit(1)
                except subprocess.TimeoutExpired:
                    # 如果运行正常，进程仍在运行
                    processes.append((server["name"], process))
                    print(f"{server['name']} 启动成功，日志输出到: {log_path}")
                    # 启动线程实时读取日志文件
                    threading.Thread(target=tail_file, args=(log_path, server["name"]), daemon=True).start()
                    
        except Exception as e:
            print(f"启动 {server['name']} 时发生异常: {str(e)}")
            cleanup(processes)
            sys.exit(1)
    
    print("\n所有服务器已启动，日志实时显示在上方")
    print("按 Ctrl+C 停止所有服务器...")
    
    try:
        # 主线程等待，直到被中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止所有服务器...")
        cleanup(processes)

def cleanup(processes):
    '''停止所有正在运行的进程'''
    for name, process in processes:
        try:
            print(f"正在停止 {name}...")
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass

if __name__ == "__main__":
    print("===== 服务器启动脚本 =====")
    start_servers()