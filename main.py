# main.py
import subprocess
import sys
import os
import threading
import time
from pathlib import Path
import requests
import json
import re
import ast
import fnmatch
from typing import List, Dict, Any
import concurrent.futures
from functools import partial
import signal

# 添加新的常量
AUTO_PORT_START = 5000  # 自动分配端口起始值
AUTO_PORT_RANGE = 1000  # 自动分配端口范围

class ServiceManager:
    def __init__(self):
        self.processes = {}  # 存储运行中的服务进程
        self.services_config = {}  # 存储所有服务配置
        self.running = True
        
    def scan_single_file(self, file_path, directory):
        """
        扫描单个文件是否包含Flask路由
        """
        try:
            file_name = os.path.basename(file_path)
            # 排除main.py自身
            if file_name == 'main.py':
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查文件是否包含Flask应用和路由定义
            if ('@app.route' in content or 'Flask(' in content) and 'if __name__ == \'__main__\':' in content:
                # 尝试解析Python代码以确认其有效性
                ast.parse(content)
                return {
                    'file_path': file_path,
                    'file_name': file_name,
                    'relative_path': os.path.relpath(file_path, directory)
                }
        except (SyntaxError, UnicodeDecodeError, Exception):
            # 忽略无法解析或编码错误的文件
            pass
        return None

    def scan_python_files_with_routes(self, directory="E:\\code\\my_python_server"):
        """
        扫描目录下所有包含Flask路由的Python文件（优化版）
        """
        print("正在扫描服务文件...")
        route_files = []
        
        # 收集所有Python文件路径
        python_files = []
        for root, dirs, files in os.walk(directory):
            # 排除一些常见的不需要扫描的目录
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env', 'logs']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        # 使用线程池并行处理文件扫描
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            scan_func = partial(self.scan_single_file, directory=directory)
            results = executor.map(scan_func, python_files)
            route_files = [result for result in results if result is not None]
        
        print(f"扫描完成，共发现 {len(route_files)} 个服务文件")
        return route_files

    def create_auto_servers(self, route_files):
        """
        为找到的路由文件创建服务器配置
        """
        auto_servers = {}
        port_counter = AUTO_PORT_START
        
        for route_file in route_files:
            # 生成服务名称
            file_name = os.path.splitext(route_file['file_name'])[0]
            relative_dir = os.path.dirname(route_file['relative_path'])
            if relative_dir:
                # 清理路径分隔符，生成更友好的服务名称
                dir_name = relative_dir.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
                service_name = f"{dir_name}_{file_name}".title().replace('_', ' ')
            else:
                service_name = f"{file_name.title()} Server"
                
            # 创建服务器配置
            server_config = {
                "name": service_name,
                "command": f"micromambavenv\\python {route_file['relative_path'].replace(os.sep, '/')}",
                "cwd": "E:\\code\\my_python_server",
                "log_file": f"logs/{file_name}.log",
                "port": port_counter,
                "file_path": route_file['file_path']
            }
            
            auto_servers[service_name] = server_config
            port_counter += 1
            
            # 防止超出分配的端口范围
            if port_counter >= AUTO_PORT_START + AUTO_PORT_RANGE:
                print(f"警告: 达到端口分配上限 ({AUTO_PORT_RANGE}个服务)")
                break
                
        return auto_servers

    def select_individual_services(self, all_services):
        """
        允许用户逐个选择要启动的服务
        """
        if not all_services:
            print("没有发现可启动的服务")
            return []
        
        service_list = list(all_services.values())
        print("\n发现以下服务:")
        for i, (name, service) in enumerate(all_services.items(), 1):
            status = "运行中" if name in self.processes else "已停止"
            print(f"{i}. {service['name']} (端口: {service['port']}, 状态: {status})")
        
        print(f"{len(all_services) + 1}. 返回主菜单")
        
        selected_indices = input("\n请输入要操作的服务编号(用逗号分隔，如: 1,3,5): ").strip()
        
        if not selected_indices:
            return []
        
        if selected_indices == str(len(all_services) + 1):
            return []
        
        try:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            selected_services = []
            for idx in indices:
                if 1 <= idx <= len(service_list):
                    selected_services.append(list(all_services.keys())[idx-1])
                else:
                    print(f"警告: 无效的服务编号 {idx}")
            return selected_services
        except ValueError:
            print("输入格式错误，应为数字并用逗号分隔")
            return []

    def clear_log_file(self, log_path):
        '''清空日志文件'''
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('')
            print(f"已清空日志文件: {log_path}")
        except Exception as e:
            print(f"清空日志文件 {log_path} 时出错: {str(e)}")
            return False
        return True

    def tail_file(self, log_path, name):
        '''实时读取日志文件内容并输出到终端'''
        while name in self.processes and self.running:
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(0, 2)  # 移到文件末尾
                    while name in self.processes and self.running:
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

    def get_service_routes(self, port):
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
            # print(f"无法获取端口 {port} 的路由信息: {str(e)}")
            return None

    def print_service_routes(self, service_name, port):
        """
        打印特定服务的路由信息
        """
        print(f"\n=== {service_name} 路由信息 (端口: {port}) ===")
        routes_info = self.get_service_routes(port)
        if routes_info and 'routes' in routes_info:
            for idx, route in enumerate(routes_info['routes'], 1):
                methods = ', '.join([m for m in route['methods'] if m != 'HEAD'])
                # 只显示第一行描述
                description = route['description'].split('\n')[0] if route['description'] else ''
                
                # 默认显示基本URL
                base_url = f"http://localhost:{port}{route['rule']}"
                
                # 检查是否有查询参数
                has_query_params = False
                if 'GET' in route['methods'] and route['description']:
                    lines = route['description'].split('\n')
                    params = []
                    for line in lines:
                        if 'name:' in line:
                            has_query_params = True
                            # 提取参数名
                            param_name_match = re.search(r'name:\s*(\w+)', line)
                            if param_name_match:
                                param_name = param_name_match.group(1)
                                # 检查是否有默认值
                                default_match = re.search(r'default:\s*([^,\n]+)', line)
                                if default_match:
                                    default_val = default_match.group(1).strip()
                                    params.append(f"{param_name}={default_val}")
                                else:
                                    params.append(f"{param_name}=")
                    
                    # 如果找到查询参数，则构建带参数的URL示例
                    if params:
                        query_string = '&'.join(params)
                        print(f"{idx}. http://localhost:{port}{route['rule']}?{query_string} [{methods}] - {description}")
                    else:
                        # 有参数定义但未解析出具体参数的情况
                        print(f"{idx}. {base_url} [{methods}] - {description}")
                else:
                    # 非GET请求或没有描述的情况
                    print(f"{idx}. {base_url} [{methods}] - {description}")
        else:
            print(f"  无法获取路由信息或服务尚未启动")
        print("=" * 50)

    def start_service(self, service_name):
        """启动指定服务"""
        if service_name in self.processes:
            print(f"服务 {service_name} 已经在运行中")
            return False
            
        if service_name not in self.services_config:
            print(f"未找到服务 {service_name}")
            return False
            
        service = self.services_config[service_name]
        
        # 为服务添加端口参数
        command = service['command']
        if '--port' not in command:
            command += f" --port {service['port']}"
            
        # 确保日志目录存在
        log_path = Path(service["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 清空日志文件
        if not self.clear_log_file(log_path):
            return False
            
        try:
            print(f"启动 {service['name']} (端口: {service['port']})...")
            # 打开日志文件(追加模式)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                # 启动进程，重定向输出到日志文件
                process = subprocess.Popen(
                    command,
                    cwd=service["cwd"],
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
                        print(f"错误: {service['name']} 启动失败，返回码: {return_code}")
                        print(f"请检查日志文件: {log_path}")
                        return False
                except subprocess.TimeoutExpired:
                    # 如果运行正常，进程仍在运行
                    self.processes[service_name] = process
                    print(f"{service['name']} 启动成功，日志输出到: {log_path}")
                    # 启动线程实时读取日志文件
                    threading.Thread(target=self.tail_file, args=(log_path, service_name), daemon=True).start()
                    # 自动显示路由信息
                    self.print_service_routes(service_name, service['port'])
                    return True
                    
        except Exception as e:
            print(f"启动 {service['name']} 时发生异常: {str(e)}")
            return False

    def stop_service(self, service_name):
        """停止指定服务"""
        if service_name not in self.processes:
            print(f"服务 {service_name} 未在运行")
            return False
            
        try:
            process = self.processes[service_name]
            print(f"正在停止 {service_name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
            del self.processes[service_name]
            print(f"服务 {service_name} 已停止")
            return True
        except Exception as e:
            print(f"停止服务 {service_name} 时出错: {str(e)}")
            return False

    def list_services(self):
        """列出所有服务及其状态"""
        if not self.services_config:
            print("未发现任何服务")
            return
            
        print("\n=== 服务列表 ===")
        for i, (name, service) in enumerate(self.services_config.items(), 1):
            status = "运行中" if name in self.processes else "已停止"
            print(f"{i}. {service['name']} (端口: {service['port']}, 状态: {status})")
        print("=" * 30)

    def execute_get_request(self, port, route_rule, service_name, route_index, route_info):
        """
        执行GET请求，支持参数输入
        """
        try:
            # 解析路由参数
            params = {}
            if route_info.get('description'):
                lines = route_info['description'].split('\n')
                required_params = []
                optional_params = []
                
                for line in lines:
                    if 'name:' in line:
                        # 提取参数信息
                        param_name_match = re.search(r'name:\s*(\w+)', line)
                        if param_name_match:
                            param_name = param_name_match.group(1)
                            
                            # 检查是否必需
                            required = 'required:' in line and 'required: true' in line
                            
                            # 检查是否有默认值
                            default_match = re.search(r'default:\s*([^,\n]+)', line)
                            default_val = default_match.group(1).strip() if default_match else None
                            
                            if default_val is not None:
                                optional_params.append((param_name, default_val))
                            elif required:
                                required_params.append(param_name)
                            else:
                                optional_params.append((param_name, None))
                
                # 获取用户输入的必需参数
                for param_name in required_params:
                    value = input(f"请输入参数 '{param_name}' 的值: ").strip()
                    if value:
                        params[param_name] = value
                    else:
                        print(f"参数 '{param_name}' 是必需的")
                        return
                
                # 询问可选参数
                for param_name, default_val in optional_params:
                    prompt = f"请输入参数 '{param_name}' 的值"
                    if default_val is not None:
                        prompt += f" (默认: {default_val})"
                    prompt += " (直接回车跳过): "
                    
                    value = input(prompt).strip()
                    if value:
                        params[param_name] = value
                    elif default_val is not None:
                        params[param_name] = default_val
        
            url = f"http://localhost:{port}{route_rule}"
            print(f"正在执行请求: {url}")
            if params:
                print(f"参数: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            print(f"\n=== 响应结果 (服务: {service_name}, 路由 #{route_index}) ===")
            print(f"状态码: {response.status_code}")
            print(f"请求URL: {response.url}")
            print(f"响应头: {dict(response.headers)}")
            
            # 尝试解析JSON响应
            try:
                json_data = response.json()
                print("响应内容 (JSON格式):")
                print(json.dumps(json_data, indent=2, ensure_ascii=False))
            except:
                # 如果不是JSON格式，则按文本处理
                print("响应内容 (文本格式):")
                print(response.text)
            print("=" * 50)
            
        except Exception as e:
            print(f"执行请求时出错: {str(e)}")

    def manage_services_menu(self):
        """服务管理菜单"""
        # 只在进入菜单时显示一次服务列表
      
        
        while True:
            # 显示可用命令
            print("\n可用命令:")
            print("start <编号> - 启动服务")
            print("stop <编号> - 停止服务")
            print("routes <编号> - 显示服务路由")
            print("get <编号> <路由序号> - 执行GET请求")
            print("list - 显示服务列表")
            print("exit - 退出程序")
            
            # 获取用户输入
            command = input("\n请输入命令: ").strip().lower()
            
            if command.startswith("start"):
                try:
                    service_index = int(command.split()[1])
                    if 1 <= service_index <= len(self.services_config):
                        service_name = list(self.services_config.keys())[service_index-1]
                        self.start_service(service_name)
                    else:
                        print("无效的服务编号")
                except (ValueError, IndexError):
                    print("请输入有效的命令格式: start <编号>")
                    
            elif command.startswith("stop"):
                try:
                    service_index = int(command.split()[1])
                    if 1 <= service_index <= len(self.services_config):
                        service_name = list(self.services_config.keys())[service_index-1]
                        self.stop_service(service_name)
                    else:
                        print("无效的服务编号")
                except (ValueError, IndexError):
                    print("请输入有效的命令格式: stop <编号>")
                    
            elif command.startswith("routes"):
                try:
                    service_index = int(command.split()[1])
                    if 1 <= service_index <= len(self.services_config):
                        service_name = list(self.services_config.keys())[service_index-1]
                        service = self.services_config[service_name]
                        self.print_service_routes(service_name, service['port'])
                    else:
                        print("无效的服务编号")
                except (ValueError, IndexError):
                    print("请输入有效的命令格式: routes <编号>")
                    
            elif command.startswith("get"):
                try:
                    parts = command.split()
                    if len(parts) < 3:
                        print("请输入有效的命令格式: get <编号> <路由序号>")
                        continue
                        
                    service_index = int(parts[1])
                    route_index = int(parts[2])
                    
                    if 1 <= service_index <= len(self.services_config):
                        service_name = list(self.services_config.keys())[service_index-1]
                        service = self.services_config[service_name]
                        
                        # 获取路由信息
                        routes_info = self.get_service_routes(service['port'])
                        if routes_info and 'routes' in routes_info:
                            if 1 <= route_index <= len(routes_info['routes']):
                                route = routes_info['routes'][route_index-1]
                                if 'GET' in route['methods']:
                                    self.execute_get_request(
                                        service['port'], 
                                        route['rule'], 
                                        service_name, 
                                        route_index,
                                        route
                                    )
                                else:
                                    print("该路由不支持GET方法")
                            else:
                                print("无效的路由序号")
                        else:
                            print("无法获取路由信息")
                    else:
                        print("无效的服务编号")
                except (ValueError, IndexError):
                    print("请输入有效的命令格式: get <编号> <路由序号>")
                    
            elif command == "list":
                self.list_services()
                
            elif command == "exit":
                break
                
            else:
                print("无效命令，请重新输入")

    def initialize_services(self):
        """初始化服务配置"""
        route_files = self.scan_python_files_with_routes()
        self.services_config = self.create_auto_servers(route_files)
        
        if not self.services_config:
            print("未发现任何服务")
            return False
        return True

    def start_initial_services(self):
        """启动初始服务"""
        print("\n请选择要启动的服务:")
        selected = self.select_individual_services(self.services_config)
        
        started_count = 0
        for service_name in selected:
            if self.start_service(service_name):
                started_count += 1
                # 注意：这里不再需要手动调用 print_service_routes，因为 start_service 内部已经自动调用了
                
        print(f"\n成功启动 {started_count} 个服务")
        return started_count > 0

    def cleanup(self):
        """清理所有运行中的服务"""
        print("\n正在停止所有服务...")
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        self.running = False

    def main_menu(self):
        """主菜单"""
        print("===== 服务器启动脚本 =====")
        
        # 初始化服务
        if not self.initialize_services():
            return
            
        # 启动初始服务
        self.start_initial_services()
        
        # 进入管理循环
        try:
            while self.running:
                # 直接进入服务管理菜单，不显示主菜单
                self.manage_services_menu()
        except KeyboardInterrupt:
            print("\n收到中断信号...")
            self.cleanup()

# 全局服务管理器实例
service_manager = ServiceManager()

def signal_handler(sig, frame):
    """处理信号中断"""
    print("\n收到终止信号，正在清理...")
    service_manager.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行主菜单
    service_manager.main_menu()