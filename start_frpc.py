# start_frpc.py
import subprocess
import sys
import os
import argparse

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('frpc_path', help='frpc可执行文件的路径')
    parser.add_argument('--log-file', help='日志文件路径')
    
    # 解析已知参数，保留未知参数传递给frpc
    args, unknown_args = parser.parse_known_args()
    
    frpc_path = args.frpc_path
    
    # 检查frpc文件是否存在
    if not os.path.exists(frpc_path):
        print(f"指定的frpc路径不存在: {frpc_path}")
        sys.exit(1)
    
    # 组合frpc的参数
    frpc_args = unknown_args
    
    # 执行frpc并处理日志
    try:
        if args.log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(args.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 打开日志文件
            with open(args.log_file, 'w', encoding='utf-8') as log_file:
                result = subprocess.run(
                    [frpc_path] + frpc_args,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                sys.exit(result.returncode)
        else:
            # 不记录日志，直接运行
            result = subprocess.run([frpc_path] + frpc_args)
            sys.exit(result.returncode)
    except Exception as e:
        print(f"执行frpc时出错: {e}")
        if args.log_file:
            with open(args.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"执行frpc时出错: {e}\n")
        sys.exit(1)