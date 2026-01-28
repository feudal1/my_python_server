#!/usr/bin/env python3
"""
打开浏览器脚本
参数: URL
"""
import sys
import webbrowser
import time


def openURL(url):
    """
    标准化URL格式，处理可能的错误格式
    """
    # 如果URL包含"=",可能是在"="后面才是真正的URL
    if '=' in url:
        parts = url.split('=', 1)
        if len(parts) > 1:
            url = parts[1]  # 取"="后面的部分作为URL
    
    # 移除可能的协议前缀重复
    url = url.replace('http://http://', 'http://').replace('https://http://', 'https://')
    url = url.replace('http://https://', 'https://').replace('https://https://', 'https://')
    
    # 如果URL没有协议，添加https://
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    webbrowser.open(url)
    
    result = f"成功 打开网页: {url}"
    print(result)
    return result

if __name__ == '__main__':
    print("打开浏览器工具")
    print("可用函数:")
    print("- openURL(url)")
    # 测试
    if len(sys.argv) > 1:
        url = sys.argv[1]
        openURL(url)