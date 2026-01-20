#!/usr/bin/env python3
"""
MCP客户端示例脚本 - 调用MCP服务器中的工具

此脚本演示如何使用mcp库连接到已配置的MCP服务器并调用其工具。
支持的服务器包括：blender-tool, ue-tool, browser-tool, computer-tool, ocr-tool, likefavarite-tools
"""

import asyncio
import sys
import os

# 导入mcp客户端相关模块
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP客户端类，用于连接和调用MCP服务器工具"""

    def __init__(self):
        self.servers = {
            'blender-tool': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'blender_cline', 'blender_api_tool.py')],
                'description': 'Blender工具服务器'
            },
            'ue-tool': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'ue_cline', 'ue_api_tool.py')],
                'description': 'Unreal Engine工具服务器'
            },
            'browser-tool': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'browser_cline', 'browser_api_tool.py')],
                'description': '浏览器工具服务器'
            },
            'computer-tool': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'computer_cline', 'computer_api_tool.py')],
                'description': '计算机控制工具服务器'
            },
            'ocr-tool': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'ocr', 'ocr.py')],
                'description': 'OCR工具服务器'
            },
            'likefavarite-tools': {
                'command': sys.executable,
                'args': [os.path.join('my_python_server', 'yolo', 'detect_like_favorite.py')],
                'description': '点赞收藏检测工具服务器'
            }
        }

    async def list_tools(self, server_name):
        """列出指定服务器的所有可用工具"""
        if server_name not in self.servers:
            print(f"错误：未知的服务器 '{server_name}'")
            return []

        server_config = self.servers[server_name]
        server_params = StdioServerParameters(
            command=server_config['command'],
            args=server_config['args'],
            env=None
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化会话
                    await session.initialize()

                    # 获取可用工具
                    tools_result = await session.list_tools()
                    tools = tools_result.tools

                    print(f"\n{server_config['description']} ({server_name}) 可用工具：")
                    for tool in tools:
                        print(f"- {tool.name}: {tool.description}")

                    return tools

        except Exception as e:
            print(f"连接到 {server_name} 时出错: {e}")
            return []

    async def call_tool(self, server_name, tool_name, arguments=None):
        """调用指定服务器的指定工具"""
        if server_name not in self.servers:
            print(f"错误：未知的服务器 '{server_name}'")
            return None

        if arguments is None:
            arguments = {}

        server_config = self.servers[server_name]
        server_params = StdioServerParameters(
            command=server_config['command'],
            args=server_config['args'],
            env=None
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化会话
                    await session.initialize()

                    # 调用工具
                    result = await session.call_tool(tool_name, arguments)

                    print(f"\n调用 {server_name} 的 {tool_name} 工具结果：")
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                        elif hasattr(content, 'type'):
                            print(f"[{content.type}]: {content}")

                    return result

        except Exception as e:
            print(f"调用 {server_name} 的 {tool_name} 时出错: {e}")
            return None


