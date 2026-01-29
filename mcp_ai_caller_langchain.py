#!/usr/bin/env python3
"""
MCP AI Caller (LangChain)

这是系统的核心模块，负责处理用户输入、调用工具和生成响应。
"""

import sys
import asyncio
import os
import json
import threading
import queue
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 添加当前目录和langchain目录到路径,支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_dir = os.path.join(current_dir, "langchain")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if langchain_dir not in sys.path:
    sys.path.insert(0, langchain_dir)

# 动态导入窗口管理器
import importlib.util
window_manager_path = os.path.join(current_dir, "ui", "window_manager.py")
spec = importlib.util.spec_from_file_location("window_manager", window_manager_path)
window_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window_manager_module)
WindowManager = window_manager_module.WindowManager

# 导入LangChain相关模块
from langchain_community.llms import VLLMOpenAI,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool


class MCPAICallerLangChain:
    """
    使用LangChain的MCP AI调用器
    """

    def __init__(self):
        """
        初始化MCP AI调用器
        """
        # 初始化变量
        self._initialize_variables()

        # 初始化服务（只初始化LLM）
        self._initialize_services()

        # 初始化窗口管理器，并传递process_user_input方法
        self.window_manager = WindowManager()

        # 连接用户输入信号到处理函数
        self.window_manager.input_submitted_signal.connect(self.process_user_input)

        # 初始化工具
        self._initialize_tools()

        # 创建对话链（需要工具列表）
        self._create_conversation_chain()

        # 设置信号连接
        self._setup_connections()

        print("[初始化] 完成!")
        print("[启动] MCP AI Caller (LangChain) 已启动")

    def _initialize_variables(self):
        """
        初始化变量
        """
        print("[初始化] 设置初始变量...")

        # 存储VLM分析结果
        self.vlm_results = []

        # 存储LLM回复
        self.llm_responses = []

        # 存储用户输入历史
        self.user_input_history = []

        # 存储工具列表
        self.tools_list = {}
        # 存储按服务器分组的工具列表
        self.tools_by_server = {}

        # 存储对话历史，用于API调用
        self.conversation_history = []

        # 工具加载状态
        self.tools_loading = False

        # 工具是否已经加载过
        self.tools_loaded_once = False

        # 最后一次工具加载时间
        self.last_tools_load_time = 0

        # LangChain Agent
        self.agent = None

    def _initialize_services(self):
        """
        初始化LLM和VLM服务
        """
        print("[初始化] 初始化服务...")
        ZHIPUAI_CONFIG = {
            "openai_api_key": "d121416576624afca2902462fda1baff.Va2AyE6qDFzYTmZc",
            "openai_api_base": "https://open.bigmodel.cn/api/paas/v4",
            "model_name": "glm-4.6v-flash"
        }
        # 初始化VLLMOpenAI
        self.llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8001/v1",
            model_name="/root/my_python_server/wsl/models/OpenBMB_MiniCPM-V-2_6-int4",
            temperature=0.7
        )
        self.llm = ChatOpenAI(
            api_key=ZHIPUAI_CONFIG["openai_api_key"],
            model_name=ZHIPUAI_CONFIG["model_name"],
            openai_api_base=ZHIPUAI_CONFIG["openai_api_base"],
            temperature=0.7,
            max_tokens=1000
        )

        # 初始化对话记忆，限制只保存最近的3轮对话（6条消息）
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            max_token_limit=1000  # 限制token数量，避免历史过长
        )

    def _setup_connections(self):
        """
        设置信号连接
        """
        print("[初始化] 设置信号连接...")

        # 注意：WindowManager类中没有input_submitted信号
        # 移除信号连接，因为WindowManager类使用的是直接方法调用

    def _initialize_tools(self):
        """
        初始化工具
        """
        print("[初始化] 初始化工具...")
        
        # 导入所有工具
        from tools.memory_tools import search_memory, write_memory, get_memory_stats
        from tools.web_tools import (
            open_webpage, ocr_recognize, click_position, scroll_down, 
            yolo_detect, check_download_bar, wait_for_download
        )
        from tools.blender_tools import (
            activate_blender_window, delete_all_objects, import_pmx, fix_model,
            set_scale, import_psk, scale_to_object_name, set_parent_bone,
            switch_pose_mode, add_vertex_group_transfer, delete_object, open_blender_folder
        )
        from tools.ue_tools import activate_ue_window, import_fbx, build_sifu_mod
        from skills.skill import load_skill, list_skills
        
        # 将所有工具添加到列表
        self.tools = [
            # Memory工具
            search_memory, write_memory, get_memory_stats,
            # Web工具
            open_webpage, ocr_recognize, click_position, scroll_down,
            yolo_detect, check_download_bar, wait_for_download,
            # Blender工具
            activate_blender_window, delete_all_objects, import_pmx, fix_model,
            set_scale, import_psk, scale_to_object_name, set_parent_bone,
            switch_pose_mode, add_vertex_group_transfer, delete_object, open_blender_folder,
            # UE工具
            activate_ue_window, import_fbx, build_sifu_mod,
            # Skill工具
            load_skill, list_skills
        ]
        print(f"[初始化] 成功创建 {len(self.tools)} 个工具")

    def _generate_tools_description(self):
        """
        动态生成工具描述
        
        Returns:
            str: 工具描述文本
        """
        tools_desc = []
        for tool_item in self.tools:
            tool_name = tool_item.name
            tool_desc = tool_item.description
            tools_desc.append(f"- {tool_name}: {tool_desc}")
        
        return "\n".join(tools_desc)

    def _create_conversation_chain(self):
        """
        创建对话链，动态绑定工具
        """
        print("[初始化] 创建对话链...")

        # 动态生成工具描述
        self.tools_description = self._generate_tools_description()
        print(f"[初始化] 已加载 {len(self.tools)} 个工具")

        # 创建简洁的系统提示
        system_prompt = """你是一个智能AI助手，能够帮助用户完成各种任务。

【重要规则 - 必须严格遵守】：
1. 绝对不要输出任何示例对话、虚构对话或训练数据中的对话内容
2. 绝对不要输出 "Human:"、"ChatGPT:"、"AI:" 等对话格式标记
3. 绝对不要重复或复述用户的问题
4. 只针对用户的实际输入内容进行回复
5. 回复要简洁直接，一句话或几句话即可
6. 如果用户只是打招呼，简单回应即可，例如："你好！有什么我可以帮助你的吗？"
7. 不要编造任何示例场景或对话

记住：你的用户是正在使用这个AI助手的人，而不是训练数据中的虚拟人物。只针对真实用户的输入进行回复！"""

        # 更新系统提示 - 不使用对话历史，避免历史干扰
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # 创建对话链（不包含对话历史）
        self.conversation_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("[初始化] 对话链创建完成")

    def process_user_input(self, input_text: str):
        """
        处理用户输入
        
        Args:
            input_text: 用户输入的文本
        """
        print(f"[用户输入] 收到输入: {input_text}")

        # 存储用户输入
        self.user_input_history.append(input_text)

        # 处理输入
        self.process_input(input_text)

    def _parse_tool_call(self, response):
        """
        解析工具调用请求
        
        Args:
            response: LLM的响应文本
            
        Returns:
            tuple: (tool_name, tool_args) 或 (None, None)
        """
        import re
        
        # 匹配工具调用格式
        tool_call_pattern = re.compile(r'```tool_call\n(.*?)\n(.*?)```', re.DOTALL)
        match = tool_call_pattern.search(response)
        
        if not match:
            return None, None
        
        tool_name = match.group(1).strip()
        args_text = match.group(2).strip()
        
        # 解析参数
        tool_args = {}
        if args_text:
            for line in args_text.split('\n'):
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    tool_args[key.strip()] = value.strip()
        
        return tool_name, tool_args
    
    def _execute_tool(self, tool_name, tool_args):
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            str: 工具执行结果
        """
        for tool_item in self.tools:
            if tool_item.name == tool_name:
                try:
                    result = tool_item(**tool_args)
                    return f"工具执行成功: {result}"
                except Exception as e:
                    return f"工具执行失败: {str(e)}"
        
        return f"未找到工具: {tool_name}"
    
    def clear_conversation_history(self):
        """
        清除对话历史
        """
        self.memory.clear()
        print("[对话历史] 已清除所有对话历史")

    def process_input(self, input_text: str):
        """
        处理输入文本

        Args:
            input_text: 输入文本
        """
        try:
            # 检查是否需要清除历史（用户开始新的对话）
            clear_keywords = ['你好', 'hi', 'hello', '开始', '新的对话']
            should_clear = any(keyword in input_text.lower() for keyword in clear_keywords)

            if should_clear:
                self.clear_conversation_history()

            # 检查用户是否提到需要工具
            need_tool_keywords = ['工具', '打开网页', '搜索', '记忆', 'blender', 'ue', 'skill']
            need_tool = any(keyword in input_text.lower() for keyword in need_tool_keywords)
            
            if need_tool:
                # 当用户需要工具时，提供工具信息
                tool_info_prompt = f"""用户需要使用工具。以下是可用工具：

{self.tools_description}

使用工具时，请使用以下格式：
```tool_call
工具名称
参数1: 值1
参数2: 值2
```

请根据用户需求选择合适的工具并调用。"""
                
                # 组合输入
                combined_input = f"{input_text}\n\n{tool_info_prompt}"
                response = self.conversation_chain.invoke({"input": combined_input})
            else:
                # 普通对话
                response = self.conversation_chain.invoke({"input": input_text})
            
            # 解析工具调用
            tool_name, tool_args = self._parse_tool_call(response)
            
            if tool_name:
                print(f"[工具调用] 检测到工具调用: {tool_name}, 参数: {tool_args}")
                # 执行工具
                tool_result = self._execute_tool(tool_name, tool_args)
                print(f"[工具调用] 工具执行结果: {tool_result}")
                
                # 将工具执行结果返回给LLM
                tool_response = f"工具执行结果: {tool_result}"
                follow_up_response = self.conversation_chain.invoke({"input": tool_response})
                
                # 显示最终回复
                print(f"[LangChain] 最终响应: {follow_up_response}")
                self.window_manager.add_caption_line(follow_up_response)
                
                # 存储对话历史
                self.memory.save_context({"input": input_text}, {"output": follow_up_response})
            else:
                # 没有工具调用，直接显示回复
                print(f"[LangChain] 对话链响应: {response}")
                self.window_manager.add_caption_line(response)
                
                # 存储对话历史
                self.memory.save_context({"input": input_text}, {"output": response})
                
        except Exception as e:
            print(f"[LangChain] 处理输入时出错: {str(e)}")
            self.window_manager.add_caption_line(f"处理失败: {str(e)}")

    def start(self):
        """
        启动系统
        """
        print("[系统] 启动中...")
        # 系统已经在初始化时启动

    def run(self):
        """
        运行系统
        """
        # 显示窗口
        self.window_manager.show()


if __name__ == "__main__":
    # 创建应用程序实例
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    # 创建主窗口实例
    caller = MCPAICallerLangChain()
    caller.run()

    # 进入应用程序主循环
    sys.exit(app.exec())
