# start_nginx.py
import subprocess
import sys
import os
import argparse
import time

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('nginx_path', help='nginx可执行文件的路径')
    parser.add_argument('--log-file', help='日志文件路径')
    parser.add_argument('--config', help='nginx配置文件路径')
    
    # 解析已知参数
    args, unknown_args = parser.parse_known_args()
    
    nginx_path = args.nginx_path
    
    # 检查nginx文件是否存在
    if not os.path.exists(nginx_path):
        print(f"指定的nginx路径不存在: {nginx_path}")
        sys.exit(1)
    
    # 构建nginx的参数
    nginx_args = []
    
    # 如果指定了配置文件
    if args.config:
        nginx_args.extend(['-c', args.config])
    
    # 添加其他未知参数
    nginx_args.extend(unknown_args)
    
    # 执行nginx并处理日志
    try:
        if args.log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(args.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 打开日志文件
            with open(args.log_file, 'w', encoding='utf-8') as log_file:
                print(f"正在启动nginx: {nginx_path}")
                print(f"日志文件: {args.log_file}")
                if args.config:
                    print(f"配置文件: {args.config}")
                
                result = subprocess.Popen(
                    [nginx_path] + nginx_args,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                
                # 等待一小段时间以确保nginx启动
                time.sleep(2)
                
                # 检查进程是否仍在运行
                if result.poll() is None:
                    print("nginx已成功启动")
                    print("按 Ctrl+C 停止 nginx 或关闭此窗口以保持后台运行")
                    try:
                        # 保持脚本运行，直到用户中断
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n正在停止 nginx...")
                        # 尝试优雅地停止 nginx
                        stop_result = subprocess.run([nginx_path, '-s', 'quit'], 
                                                   capture_output=True, text=True)
                        if stop_result.returncode == 0:
                            print("nginx 已停止")
                        else:
                            print("停止 nginx 失败，可能需要手动停止")
                else:
                    print("nginx启动失败")
                    sys.exit(1)
        else:
            # 不记录日志，直接运行
            print(f"正在启动nginx: {nginx_path}")
            result = subprocess.Popen([nginx_path] + nginx_args)
            
            # 等待一小段时间以确保nginx启动
            time.sleep(2)
            
            # 检查进程是否仍在运行
            if result.poll() is None:
                print("nginx已成功启动")
                print("按 Ctrl+C 停止 nginx 或关闭此窗口以保持后台运行")
                try:
                    # 保持脚本运行，直到用户中断
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n正在停止 nginx...")
                    # 尝试优雅地停止 nginx
                    stop_result = subprocess.run([nginx_path, '-s', 'quit'], 
                                               capture_output=True, text=True)
                    if stop_result.returncode == 0:
                        print("nginx 已停止")
                    else:
                        print("停止 nginx 失败，可能需要手动停止")
            else:
                print("nginx启动失败")
                sys.exit(1)
                
    except Exception as e:
        print(f"执行nginx时出错: {e}")
        if args.log_file:
            with open(args.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"执行nginx时出错: {e}\n")
        sys.exit(1)