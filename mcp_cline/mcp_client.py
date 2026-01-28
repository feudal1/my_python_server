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
        # 获取项目根目录的绝对路径
        project_root = "E:\code"
        
        self.servers = {
            'blender-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server', 'tools', 'blender_cline', 'blender_api_tool.py')],
                'description': 'Blender工具服务器'
            },
            'ue-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server',  'tools', 'ue_cline', 'ue_api_tool.py')],
                'description': 'Unreal Engine工具服务器'
            },
            'browser-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server',  'tools', 'browser_cline', 'browser_api_tool.py')],
                'description': '浏览器工具服务器'
            },
            'computer-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server',  'tools', 'computer_cline', 'computer_api_tool.py')],
                'description': '计算机控制工具服务器'
            },
            'ocr-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server', 'tools',  'ocr', 'ocr.py')],
                'description': 'OCR工具服务器'
            },
            'likefavarite-tools': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server',  'tools', 'yolo', 'detect_like_favorite.py')],
                'description': '点赞收藏检测工具服务器'
            },
            'memory-tool': {
                'command': sys.executable,
                'args': [os.path.join(project_root, 'my_python_server',  'memory', 'memory_api_tool.py')],
                'description': '记忆管理工具服务器'
            }
        }
        
        # 会话池，用于保持长连接
        self.session_pool = {}
        self.session_lock = asyncio.Lock()

    async def list_tools(self, server_name):
        """列出指定服务器的所有可用工具"""
        if server_name not in self.servers:
            print(f"错误：未知的服务器 '{server_name}'")
            return []

        server_config = self.servers[server_name]
        
        # 检查脚本文件是否存在
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *server_config['args'])
        if not os.path.exists(script_path):
            print(f"错误：找不到服务器脚本 '{script_path}'")
            return []

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
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return []

    async def _get_or_create_session(self, server_name):
        """获取或创建服务器会话"""
        async with self.session_lock:
            if server_name in self.session_pool:
                session, read, write = self.session_pool[server_name]
                try:
                    # 测试会话是否仍然有效
                    await session.list_tools()
                    return session
                except Exception as e:
                    print(f"会话已失效，创建新会话: {e}")
                    # 清理失效会话
                    if server_name in self.session_pool:
                        del self.session_pool[server_name]

            # 创建新会话
            server_config = self.servers[server_name]
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *server_config['args'])
            if not os.path.exists(script_path):
                print(f"错误：找不到服务器脚本 '{script_path}'")
                return None

            server_params = StdioServerParameters(
                command=server_config['command'],
                args=server_config['args'],
                env=None
            )

            try:
                read, write = await stdio_client(server_params)
                session = ClientSession(read, write)
                await session.initialize()
                self.session_pool[server_name] = (session, read, write)
                print(f"已创建新会话: {server_name}")
                return session
            except Exception as e:
                print(f"创建会话时出错: {e}")
                return None

    async def call_tool(self, server_name, tool_name, arguments=None):
        """调用指定服务器的指定工具"""
        if server_name not in self.servers:
            print(f"错误：未知的服务器 '{server_name}'")
            return None

        if arguments is None:
            arguments = {}

        try:
            # 获取或创建会话
            session = await self._get_or_create_session(server_name)
            if not session:
                return None

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
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            
            # 会话可能已失效，从会话池中移除
            async with self.session_lock:
                if server_name in self.session_pool:
                    del self.session_pool[server_name]
            
            return None