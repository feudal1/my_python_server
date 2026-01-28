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
from langchain_community.llms import VLLMOpenAI
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

        # 初始化服务
        self._initialize_services()

        # 初始化窗口管理器，并传递process_user_input方法
        self.window_manager = WindowManager()

        # 连接用户输入信号到处理函数
        self.window_manager.input_submitted_signal.connect(self.process_user_input)

        # 设置信号连接
        self._setup_connections()

        # 初始化工具
        self._initialize_tools()

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

        # 初始化VLLMOpenAI
        self.llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8001/v1",
            model_name="/root/my_python_server/wsl/models/OpenBMB_MiniCPM-V-2_6-int4",
            temperature=0.7
        )

        # 初始化对话记忆
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # 创建对话提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能AI助手，能够帮助用户完成各种任务。你可以使用工具来获取信息和执行操作。当你需要处理SQL编写或法律文档审查时，必须先调用load_skill工具加载对应技能。可用技能：write_sql（SQL工程师）、review_legal_doc（法律顾问）。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # 创建对话链
        self.conversation_chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({}).get("chat_history", [])
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
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
        from tools.skill import load_skill, list_skills
        
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

    def process_input(self, input_text: str):
        """
        处理输入文本
        
        Args:
            input_text: 输入文本
        """
        try:
            # 使用对话链处理输入
            print("[LangChain] 使用对话链处理输入...")
            response = self.conversation_chain.invoke({"input": input_text})
            
            # 存储对话历史
            self.memory.save_context({"input": input_text}, {"output": response})
            
            # 显示回复
            print(f"[LangChain] 对话链响应: {response}")
            self.window_manager.add_caption_line(response)
                
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
