import asyncio
import time
from typing import Dict
from PyQt6.QtCore import QObject, pyqtSignal


class ToolLoader(QObject):
    """异步加载MCP工具的工作者类"""

    tools_loaded = pyqtSignal(object)
    loading_failed = pyqtSignal(str)

    def __init__(self, mcp_client):
        """
        初始化工具加载器

        Args:
            mcp_client: MCP客户端实例
        """
        super().__init__()
        self.mcp_client = mcp_client

    def load_tools(self):
        """异步加载所有MCP工具"""
        start_time = time.time()
        print(f"[工具加载] 开始加载工具，开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

        try:
            tools_by_server = {}
            all_tools_mapping = {}
            tool_counter = 1

            # 并行加载所有服务器的工具列表
            async def load_server_tools(server_name):
                """加载单个服务器的工具列表"""
                try:
                    print(f"[工具加载] 正在连接服务器: {server_name}")
                    tools = await self.mcp_client.list_tools(server_name)
                    print(f"[工具加载] 服务器 {server_name} 返回了 {len(tools) if tools else 0} 个工具")
                    server_result = {'server_name': server_name, 'tools': tools}
                    return server_result
                except Exception as e:
                    print(f"[工具加载] 获取服务器 {server_name} 的工具列表失败: {str(e)}")
                    import traceback
                    print(f"[工具加载] 详细错误: {traceback.format_exc()}")
                    return {'server_name': server_name, 'tools': None}

            # 创建一个事件循环并并行加载所有服务器的工具列表
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 创建所有服务器的加载任务
                tasks = []
                for server_name in self.mcp_client.servers.keys():
                    tasks.append(load_server_tools(server_name))

                # 并行执行所有任务
                results = loop.run_until_complete(asyncio.gather(*tasks))

                print(f"[工具加载] 收到 {len(results)} 个服务器的结果")

                # 处理结果
                for result in results:
                    server_name = result['server_name']
                    tools = result['tools']

                    if tools:
                        server_tools = {}
                        for tool in tools:
                            server_tools[tool.name] = tool.description
                            all_tools_mapping[str(tool_counter)] = {
                                'name': tool.name,
                                'description': tool.description,
                                'server': server_name,
                                'input_schema': getattr(tool, 'inputSchema', {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }),
                            }
                            tool_counter += 1
                            print(f"[工具加载] 添加工具 {tool_counter-1}: {tool.name} ({tool.description[:30] if tool.description else '无描述'}...)")

                        display_server_name = server_name.replace('-tool', '') \
                            .replace('-', ' ').title() + " 工具服务器"
                        tools_by_server[display_server_name] = server_tools
                        print(f"[工具加载] 服务器 {server_name} 的工具已添加到 tools_by_server，共 {len(server_tools)} 个工具")
                    else:
                        display_server_name = server_name.replace('-tool', '') \
                            .replace('-', ' ').title() + " 工具服务器"
                        tools_by_server[display_server_name] = {
                            server_name: self.mcp_client.servers[server_name]['description']
                        }
                        all_tools_mapping[str(tool_counter)] = {
                            'name': server_name,
                            'description': self.mcp_client.servers[server_name]['description'],
                            'server': server_name,
                            'input_schema': {"type": "object", "properties": {}, "required": []},
                        }
                        tool_counter += 1

                print(f"[工具加载] all_tools_mapping 总共 {len(all_tools_mapping)} 个工具")
                print(f"[工具加载] tools_by_server 总共 {len(tools_by_server)} 个服务器")
            finally:
                loop.close()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[工具加载] 工具加载完成，总耗时: {elapsed_time:.2f} 秒")
            print(f"[工具加载] 工具加载完成，结束时间: {time.strftime('%H:%M:%S', time.localtime(end_time))}")

            self.tools_loaded.emit((all_tools_mapping, tools_by_server))
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[工具加载] 加载失败，耗时: {elapsed_time:.2f} 秒")
            import traceback
            traceback.print_exc()
            self.loading_failed.emit(f"加载工具列表失败: {str(e)}")
