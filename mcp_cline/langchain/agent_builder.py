from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from typing import List, Dict, Any
import json


class AgentBuilder:
    """构建LangChain Agent的类"""

    def __init__(self, llm, mcp_client):
        """初始化Agent构建器"""
        self.llm = llm
        self.mcp_client = mcp_client
        self.tools = []

    def build_agent(self, tools_list: Dict[str, Dict[str, Any]], tools_by_server: Dict[str, List[Dict[str, Any]]]):
        """构建Agent
        
        Args:
            tools_list: 工具列表，格式为 {"1": {...}, "2": {...}, ...}
            tools_by_server: 按服务器分组的工具列表
            
        Returns:
            构建好的Agent
        """
        # 构建工具
        self._build_tools(tools_list, tools_by_server)

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，可以调用各种工具来帮助用户完成任务。请根据用户的请求，决定是直接回答还是调用工具。"),
            ("user", "{input}"),
        ])

        # 创建简单的Agent
        def agent(input_text, chat_history=None):
            """简单的Agent函数"""
            # 构建消息
            messages = [
                ("system", "你是一个智能助手，可以调用各种工具来帮助用户完成任务。请根据用户的请求，决定是直接回答还是调用工具。"),
                ("user", input_text)
            ]

            # 添加聊天历史
            if chat_history:
                for msg in chat_history:
                    messages.append((msg["role"], msg["content"]))

            # 调用LLM
            result = self.llm(messages)
            return result.content

        return agent

    def _build_tools(self, tools_list: Dict[str, Dict[str, Any]], tools_by_server: Dict[str, List[Dict[str, Any]]]):
        """构建工具列表
        
        Args:
            tools_list: 工具列表
            tools_by_server: 按服务器分组的工具列表
        """
        self.tools = []

        for tool_id, tool_info in tools_list.items():
            tool_name = tool_info['name']
            tool_description = tool_info['description']
            tool_server = tool_info['server']
            input_schema = tool_info.get('input_schema', {
                "type": "object",
                "properties": {},
                "required": []
            })

            # 创建工具函数
            def create_tool_func(server_name, func_name):
                async def tool_func(**kwargs):
                    """工具函数"""
                    try:
                        result = await self.mcp_client.call_tool(server_name, func_name, kwargs)
                        return str(result)
                    except Exception as e:
                        return f"错误: {str(e)}"
                return tool_func

            # 创建工具
            tool_func = create_tool_func(tool_server, tool_name)
            tool = Tool(
                name=tool_name,
                func=tool_func,
                description=tool_description,
                args_schema=input_schema
            )

            self.tools.append(tool)

        print(f"已构建 {len(self.tools)} 个工具")
