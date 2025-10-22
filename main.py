import subprocess
import sys
import os
import threading
import time
from pathlib import Path
import requests
import json

# 统一端口配置
PORT_CONFIG = {
    # Excel 相关服务端口 (5000-5099)
    "excel_sse": 5001,
    "excel_mcp": 5002,
    "excel_llm": 5003,
    "excel_thick_part": 5004,
    "excel_json2excel": 5005,
    
    # CAD 相关服务端口 (5300-5399)
    "cad_main": 5301,
    "cad_base": 5302,
    "cad_cad2excel": 5303,
    "cad_llm": 5304,
    "cad_json2excel": 5305,
    
    # DINO 相关服务端口 (5200-5299)
    "dino_main": 5200,
    
    # LLM 独立服务端口 (5100-5199)
    "llm_standalone": 5100
}

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

def get_service_routes(port):
    """
    获取指定端口服务的路由信息
    """
    try:
        response = requests.get(f"http://localhost:{port}/routes", timeout=3)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"无法获取端口 {port} 的路由信息: {str(e)}")
        return None

def print_service_routes(service_name, port):
    """
    打印特定服务的路由信息
    """
    print(f"\n=== {service_name} 路由信息 (端口: {port}) ===")
    routes_info = get_service_routes(port)
    if routes_info and 'routes' in routes_info:
        for route in routes_info['routes']:
            methods = ', '.join([m for m in route['methods'] if m != 'HEAD'])
            # 只显示第一行描述
            description = route['description'].split('\n')[0] if route['description'] else ''
            print(f"   http://localhost:{port}{route['rule']} [{methods}] - {description}")
         
    else:
        print(f"  无法获取路由信息或服务尚未启动")
    print("=" * 50)
def select_servers(excel_servers, cad_servers, dino_servers, llm_servers):
    '''让用户选择要启动的服务器'''
    print("请选择要启动的服务器:")
    print("1. Excel Servers (默认)")
    print("2. CAD Servers")
    print("3. DINO Servers")
    print("4. LLM Servers")
    print("5. 所有服务器")
    
    choice = input("请输入选项编号 (默认为1): ").strip()
    
    if choice == "" or choice == "1":
     
        return excel_servers, "Excel"
    elif choice == "2":
 
        # 启动后再打印路由信息
        return cad_servers, "CAD"
    elif choice == "3":
       
        return dino_servers, "DINO"
    elif choice == "4":
      
        return llm_servers, "LLM"
    elif choice == "5":
        return excel_servers + cad_servers + dino_servers + llm_servers, "All"
    else:
      
        return excel_servers, "Excel"

def start_servers():
    # 定义服务器配置
    excel_servers = [
        {
            "name": "Python SSE Server",
            "command": f"micromambavenv\\python excel_server\\sse_server.py --port {PORT_CONFIG['excel_sse']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/sse.log"
        },
        {
            "name": "Thick Part JSON Server",
            "command": f"micromambavenv\\python excel_server\\get_thick_part_json.py --port {PORT_CONFIG['excel_thick_part']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/thick_part.log"
        },
        {
            "name": "json2excel Server",
            "command": f"micromambavenv\\python excel_server\\json2excel.py --port {PORT_CONFIG['excel_json2excel']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/json2excel.log"
        }
    ]
    
    cad_servers = [
        {
            "name": "cad Server",
            "command": f"micromambavenv\\python cad_server\\server.py --port {PORT_CONFIG['cad_main']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad_server.log"
        },
        {
            "name": "base Server",
            "command": f"micromambavenv\\python ezdxf_server\\base_server.py --port {PORT_CONFIG['cad_base']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/base_server.log"
        }, 
        {
            "name": "cad2excel Server",
            "command": f"micromambavenv\\python cad_server\\extract_parts_from_drawing.py --port {PORT_CONFIG['cad_cad2excel']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad2excel_server.log"
        },
        {
            "name": "CAD LLM Server",
            "command": f"micromambavenv\\python llm_server\\llm_server.py --port {PORT_CONFIG['cad_llm']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad_llm.log"
        },
        {
            "name": "CAD json2excel Server",
            "command": f"micromambavenv\\python excel_server\\json2excel.py --port {PORT_CONFIG['cad_json2excel']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/cad_json2excel.log"
        }
    ]
    
    dino_servers = [
        {
            "name": "dino Server",
            "command": f"micromambavenv\\python dino_server\\dino_server.py --port {PORT_CONFIG['dino_main']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/dino_server.log"
        }
    ]
    
    llm_servers = [
        {
            "name": "Standalone LLM Server",
            "command": f"micromambavenv\\python llm_server\\llm_server.py --port {PORT_CONFIG['llm_standalone']}",
            "cwd": "E:\\code\\my_python_server",
            "log_file": "logs/standalone_llm.log"
        }
    ]
    
    # 让用户选择要启动的服务器
    servers_result, server_type = select_servers(excel_servers, cad_servers, dino_servers, llm_servers)
    
    # 根据选择的服务类型准备路由信息打印计划
    route_print_plan = []
    if server_type == "Excel":
        route_print_plan = [
            ("Excel SSE Server", PORT_CONFIG['excel_sse']),
            ("Excel Thick Part JSON Server", PORT_CONFIG['excel_thick_part']),
            ("Excel JSON2Excel Server", PORT_CONFIG['excel_json2excel'])
        ]
    elif server_type == "CAD":
        route_print_plan = [
            ("CAD Server", PORT_CONFIG['cad_main']),
            ("CAD Base Server", PORT_CONFIG['cad_base']),
            ("CAD CAD2Excel Server", PORT_CONFIG['cad_cad2excel']),
            ("CAD LLM Server", PORT_CONFIG['cad_llm']),
            ("CAD JSON2Excel Server", PORT_CONFIG['cad_json2excel'])
        ]
    elif server_type == "DINO":
        route_print_plan = [("DINO Server", PORT_CONFIG['dino_main'])]
    elif server_type == "LLM":
        route_print_plan = [("Standalone LLM Server", PORT_CONFIG['llm_standalone'])]
    
    processes = []
    
    print("正在启动所有服务器...")
    
    # 启动每个服务器
    for server in servers_result:
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
    
    # 等待几秒钟让服务完全启动后再获取路由信息
    print("正在获取服务路由信息...")
    time.sleep(3)
    
    # 打印路由信息
    for service_name, port in route_print_plan:
        print_service_routes(service_name, port)
    
    print("\n按 Ctrl+C 停止所有服务器...")
    
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