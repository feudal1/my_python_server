from mcp import ClientSession  
from mcp.client.sse import sse_client  
import asyncio  
  
async def connect_to_mcp_sse():  
    """通过 SSE 连接到 MCP 服务器"""  
    url = "http://localhost:8017/sse"  
      
    async with sse_client(url) as (read, write):  
        async with ClientSession(read, write) as session:  
            # 初始化连接  
            await session.initialize()  
            print("已连接到 MCP 服务器")  
              
            # 列出所有可用工具  
            tools = await session.list_tools()  
            print("\n可用工具:")  
            for tool in tools.tools:  
                print(f"- {tool.name}: {tool.description}")  
              
            # 调用工具示例  
            result = await session.call_tool(  
                "create_workbook",  
                arguments={"filepath": "test.xlsx"}  
            )  
            print(f"\n工具调用结果: {result}")  
              
            # 读取数据示例  
            result = await session.call_tool(  
                "read_data_from_excel",  
                arguments={  
                    "filepath": "test.xlsx",  
                    "sheet_name": "Sheet1",  
                    "start_cell": "A1",  
                    "preview_only": True  
                }  
            )  
            print(f"\n读取数据结果: {result}")  
  
# 运行客户端  
asyncio.run(connect_to_mcp_sse())