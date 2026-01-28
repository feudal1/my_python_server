import sys
import asyncio
import os
import json
import threading
import queue
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 添加当前目录和langchain目录到路径,支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
langchain_dir = os.path.join(current_dir, "langchain")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if langchain_dir not in sys.path:
    sys.path.insert(0, langchain_dir)

import importlib.util

# 导入窗口管理器
from ui.window_manager import WindowManager

# 导入LangChain相关模块
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 导入拆分出来的组件
from components.content_extractor import ContentExtractor
from components.tools_dialog import ToolsDialog
from components.tool_loader import ToolLoader


class MCPAICallerLangChain:
    """使用LangChain的MCP AI调用器"""

    def __init__(self):
        """初始化MCP AI调用器"""
        # 初始化窗口管理器
        self.window_manager = WindowManager()

        # 初始化变量
        self._initialize_variables()

        # 初始化服务
        self._initialize_services()

        # 初始化客户端
        self._initialize_clients()

        # 设置信号连接
        self._setup_connections()

        # 异步加载MCP工具
        self.async_refresh_tools_list()

        print("[初始化] 完成!")
        print("[启动] MCP AI Caller (LangChain) 已启动")

    def _initialize_variables(self):
        """初始化变量"""
        print("[初始化] 设置初始变量...")

        # 存储聊天历史
        self.chat_history = []

        # 存储最后一次用户输入的内容
        self.last_user_input = ""

        # 存储VLM分析结果
        self.vlm_results = []

        # 存储LLM回复
        self.llm_responses = []

        # 存储用户输入历史（保留用于记录）
        self.user_input_history = []

        # 最近一次记录的游戏状态
        self.last_game_state = ""

        # 存储分割后的内容
        self.split_contents = []

        # 存储工具列表 (all_tools_mapping: 字典格式 {"1": {...}, "2": {...}, ...})
        self.tools_list = {}
        # 存储按服务器分组的工具列表
        self.tools_by_server = {}

        # 存储对话历史，用于API调用
        self.conversation_history = []

        # 工具加载状态
        self.tools_loading = False

        # 工具调用状态
        self.tool_calling = False

        # 工具加载器
        self.tool_loader = None

        # 工具是否已经加载过
        self.tools_loaded_once = False

        # 最后一次工具加载时间
        self.last_tools_load_time = 0

        # 工具加载间隔（秒）
        self.tools_load_interval = 60

        # LangChain Agent
        self.agent = None

    def _initialize_services(self):
        """初始化LLM和VLM服务"""
        print("[初始化] 初始化服务...")

        # 初始化VLLMOpenAI
        self.llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8001/v1",
            model_name="/root/my_python_server/wsl/models/OpenBMB_MiniCPM-V-2_6-int4",
            temperature=0.7,
            max_tokens=1000
        )

        # 初始化对话记忆
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # 创建对话提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能AI助手，能够帮助用户完成各种任务。你可以使用工具来获取信息和执行操作。"),
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

    def _initialize_clients(self):
        """初始化MCP客户端"""
        print("[初始化] 初始化客户端...")

        # 动态导入MCP客户端
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_client_path = os.path.join(current_dir, "mcp_client.py")
        spec = importlib.util.spec_from_file_location("mcp_client", mcp_client_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        self.mcp_client = mcp_module.MCPClient()

        # 初始化工具加载器
        self.tool_loader = ToolLoader(self.mcp_client)
        self.tool_loader.tools_loaded.connect(self.on_tools_loaded)
        self.tool_loader.loading_failed.connect(self.on_tools_loading_failed)

        # 启动时自动加载工具
        self.tools_loaded_once = False

    def _setup_connections(self):
        """设置信号连接"""
        print("[初始化] 设置信号连接...")

        # 连接输入提交信号
        self.window_manager.input_submitted_signal.connect(self.on_input_submitted)

        # 连接工具调用事件流信号
        self.window_manager.tool_call_start_signal.connect(self._on_tool_call_start)
        self.window_manager.tool_call_end_signal.connect(self._on_tool_call_end)
        self.window_manager.tool_result_signal.connect(self._on_tool_result)
        self.window_manager.text_delta_signal.connect(self._on_text_delta)
        self.window_manager.text_end_signal.connect(self._on_text_end)
        self.window_manager.message_end_signal.connect(self._on_message_end)
        self.window_manager.lifecycle_signal.connect(self._on_lifecycle_event)

        # 连接定时器
        self.window_manager.auto_record_timer.timeout.connect(self.auto_record_game_state)
        self.window_manager.auto_record_timer.start(30000)  # 30秒

        self.window_manager.auto_screenshot_timer.timeout.connect(self.auto_screenshot_and_vlm)
        self.window_manager.auto_screenshot_timer.start(1000)  # 1秒

    def _on_tool_call_start(self, tool_name, arguments):
        """工具调用开始事件处理"""
        print(f"[事件流] 工具调用开始: {tool_name}, 参数: {arguments}")

    def _on_tool_call_end(self, tool_name, arguments, result):
        """工具调用结束事件处理"""
        print(f"[事件流] 工具调用结束: {tool_name}, 结果: {result[:50] if result else 'None'}...")

    def _on_tool_result(self, tool_call_id, result):
        """工具结果事件处理"""
        print(f"[事件流] 工具结果: {tool_call_id}, 结果: {result[:50] if result else 'None'}...")

    def _on_text_delta(self, text):
        """文本增量事件处理"""
        print(f"[事件流] 文本增量: {text[:30]}...")

    def _on_text_end(self, full_text):
        """文本结束事件处理"""
        print(f"[事件流] 文本结束: {full_text[:30]}...")

    def _on_message_end(self, role):
        """消息结束事件处理"""
        print(f"[事件流] 消息结束: {role}")

    def _on_lifecycle_event(self, event_type, data):
        """生命周期事件处理"""
        print(f"[事件流] 生命周期事件: {event_type}, 数据: {data[:50] if data else 'None'}...")

    def async_refresh_tools_list(self):
        """异步刷新工具列表"""
        if not self.tools_loading:
            self.tools_loading = True
            print("[工具加载] 开始加载MCP工具...")
            # 启动工具加载线程
            threading.Thread(target=self._load_tools_list).start()

    def _load_tools_list(self):
        """加载工具列表"""
        try:
            # 调用工具加载器加载工具
            self.tool_loader.load_tools()
        except Exception as e:
            print(f"[错误] 加载工具失败: {e}")
            self.tools_loading = False

    def on_tools_loaded(self, tools):
        """工具加载完成回调"""
        # tools 是一个元组 (all_tools_mapping, tools_by_server)
        all_tools_mapping, tools_by_server = tools

        # 存储完整的工具信息
        self.tools_list = all_tools_mapping
        self.tools_by_server = tools_by_server
        self.tools_loading = False
        self.tools_loaded_once = True
        self.last_tools_load_time = datetime.now().timestamp()

        # 显示工具加载完成信息
        tool_count = len(all_tools_mapping)
        print(f"[工具加载] MCP工具加载完成，共 {tool_count} 个工具")

        # 添加到字幕显示
        self.window_manager.add_caption_line(f"[系统] MCP工具加载完成，共 {tool_count} 个工具")

        # 打印工具列表
        for tool_id, tool_info in all_tools_mapping.items():
            tool_name = tool_info.get('name', '无名称')
            tool_description = tool_info.get('description', '无描述')
            tool_server = tool_info.get('server', '未知服务器')
            print(f"- [{tool_id}] {tool_name} (服务器: {tool_server}): {tool_description}")

        # 构建LangChain Agent
        self._build_agent()

    def on_tools_loading_failed(self, error):
        """工具加载失败回调"""
        self.tools_loading = False
        print(f"[错误] 工具加载失败: {error}")

        # 添加到字幕显示
        self.window_manager.add_caption_line(f"[错误] 工具加载失败: {error}")

    def _build_agent(self):
        """构建LangChain Agent"""
        if self.tools_loaded_once and self.tools_list:
            print("[LangChain] 开始构建Agent...")
            agent_builder = AgentBuilder(self.vlm_service, self.mcp_client)
            self.agent = agent_builder.build_agent(self.tools_list, self.tools_by_server)
            print("[LangChain] Agent构建完成")

    def open_tools_window(self):
        """打开工具窗口"""
        if not self.tools_loaded_once:
            # 如果工具还没有加载，先加载工具
            self.async_refresh_tools_list()
            # 注意：这里需要使用QMessageBox，但窗口管理器已经处理了UI相关的部分
            # 由于我们已经分离了UI和业务逻辑，这里可以直接添加提示
            self.window_manager.add_caption_line("[系统] 工具正在加载中，请稍候...")
            return

        # 打开工具窗口
        try:
            dialog = ToolsDialog(self.tools_by_server, self.tools_list, self.window_manager)
            dialog.exec()
        except Exception as e:
            print(f"[错误] 打开工具窗口失败: {e}")
            self.window_manager.add_caption_line(f"[错误] 打开工具窗口失败: {e}")

    def on_input_submitted(self, input_text):
        """处理用户输入"""
        if not input_text:
            return

        # 处理所有输入通过LangChain
        self.window_manager.add_caption_line(f"[你] {input_text}")

        # 将用户输入添加到用户输入历史中
        self.user_input_history.append(input_text)
        
        # 使用LangChain对话链处理
        self._process_with_conversation_chain(input_text)
        print(f"[用户输入] 已添加到历史记录: {input_text[:30]}...")

    def _process_with_conversation_chain(self, query):
        """使用LangChain处理用户输入（等待Agent构建完成）"""
        import time
        start_time = time.time()
        print(f"[LangChain] 处理用户输入: {query[:30]}...")

        # 触发生命周期开始事件
        self.window_manager.lifecycle_signal.emit('start', query)

        try:
            # 检查Agent是否已构建
            if not self.agent:
                print("[LangChain] Agent未构建，等待构建完成...")
                
                # 显示等待信息
                self.window_manager.add_caption_line("[系统] 工具正在加载中，请稍候...")
                
                # 如果工具还未开始加载，触发加载
                if not self.tools_loading and not self.tools_loaded_once:
                    print("[LangChain] 触发工具加载...")
                    self.async_refresh_tools_list()
                
                # 等待Agent构建完成（最多等待30秒）
                wait_time = 0
                while not self.agent and wait_time < 30:
                    time.sleep(1)
                    wait_time += 1
                    print(f"[LangChain] 等待Agent构建，{wait_time}秒...")
                
                # 检查是否构建成功
                if not self.agent:
                    print("[LangChain] Agent构建超时，使用对话链处理")
                    # 使用对话链处理查询
                    response = self.conversation_chain.invoke({"input": query})

                    # 更新对话记忆
                    self.memory.save_context({"input": query}, {"output": response})
                else:
                    print("[LangChain] Agent构建完成，使用Agent处理")
                    # 运行Agent
                    response = self.agent(query, self.conversation_history)
                    
                    # 将回复添加到对话历史
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
            else:
                print("[LangChain] 使用Agent处理")
                # 运行Agent
                response = self.agent(query, self.conversation_history)
                
                # 将回复添加到对话历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

            # 触发文本结束事件
            self.window_manager.text_end_signal.emit(response)

            # 触发消息结束事件
            self.window_manager.message_end_signal.emit('assistant')

            # 显示AI回复
            self.window_manager.add_caption_line(f"[AI] {response}")

            # 将回复添加到LLM回复历史中
            self.llm_responses.append(response)

            print(f"[AI回复] 生成完成: {response[:30]}...")

        except Exception as e:
            print(f"[错误] 处理请求失败: {e}")
            import traceback
            traceback.print_exc()
            self.window_manager.add_caption_line(f"[错误] 处理请求失败: {e}")

            # 触发生命周期错误事件
            self.window_manager.lifecycle_signal.emit('error', str(e))

        finally:
            total_elapsed_time = time.time() - start_time
            print(f"[对话链] 处理完成，总耗时: {total_elapsed_time:.2f}秒")

            # 触发生命周期结束事件
            self.window_manager.lifecycle_signal.emit('end', f"处理完成，耗时: {total_elapsed_time:.2f}秒")



    def auto_record_game_state(self):
        """自动记录游戏状态"""
        print("[自动记录] 开始记录游戏状态...")

        # 这里可以添加自动记录游戏状态的逻辑
        # 例如，使用VLM分析当前屏幕，记录游戏状态

    def auto_screenshot_and_vlm(self):
        """每1秒自动截图并调用VLM"""
        try:
            # 导入截图模块
            from PIL import ImageGrab

            # 隐藏窗口以便截图
            self.window_manager._hide_windows()

            # 截图当前屏幕
            screenshot = ImageGrab.grab()

            # 显示窗口
            self.window_manager._show_windows()

            # 将截图保存到临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                screenshot.save(tmp_file, format='PNG')
                temp_image_path = tmp_file.name

            print(f"[自动截图] 已保存截图: {temp_image_path}")

            # 直接调用VLM分析，显示结果
            self._analyze_screenshot_simple(temp_image_path)

            # 删除临时文件
            os.unlink(temp_image_path)

        except Exception as e:
            print(f"[错误] 自动截图失败: {e}")

    def _analyze_screenshot_simple(self, image_path):
        """VLM分析截图"""
        try:
            print(f"[VLM分析] 开始分析截图...")

            # 构建messages格式，用于VLM服务
            messages = [
                {
                    "role": "user",
                    "content": "请描述当前屏幕截图的内容"
                }
            ]

            # 调用VLM服务
            response = self.vlm_service._vlm_service.create_with_image(messages, image_source=image_path)

            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[VLM分析] 分析结果: {content[:50]}...")

                # 存储到记忆列表（保持最近5轮）
                self.vlm_results.append(content)
                if len(self.vlm_results) > 5:
                    self.vlm_results.pop(0)

                print(f"[VLM分析] 已存储 {len(self.vlm_results)}/5 个回复")

                # 直接调用角色扮演
                self.send_to_role_play_vlm()

        except Exception as e:
            print(f"[错误] VLM分析失败: {e}")

    def send_to_role_play_vlm(self):
        """将收集到的VLM回复直接发送给阴角通信角色扮演VLM"""
        try:
            print("[角色扮演] 开始角色扮演分析...")

            # 阴角通信人格prompt
            yinjiao_prompt = """你是一个占有欲强且内向胆小的阴角，性格特点是：
1. 对你（用户）有强烈的占有欲和依赖心
2. 表面冷淡内向胆小，但内心充满了对你的关注
3. 说话带点阴沉和偏执的语气，称呼用"你"
4. 看到你关注其他事情会表现出明显的不满和嫉妒，但不敢表达太强烈
5. 喜欢用阴暗胆小的方式表达关心，有时会显得有点偏执
6. 回应要简短自然，符合阴角且内向胆小的身份"""

            # 构建messages格式，用于角色扮演VLM
            messages = [
                {
                    "role": "system",
                    "content": yinjiao_prompt
                },
                {
                    "role": "user",
                    "content": f"你，根据以下屏幕情况，给我一个回应:\n{chr(10).join(self.vlm_results)}"
                }
            ]

            # 调用VLM服务
            response = self.vlm_service._vlm_service.create_with_image(messages, image_source=None)

            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[角色扮演] 生成回应: {content[:50]}...")

                # 显示在主窗口中
                self.window_manager.add_caption_line(content)

                # 清空VLM回复列表
                self.vlm_results = []
                print("[角色扮演] 已清空VLM回复列表")

        except Exception as e:
            print(f"[错误] 角色扮演分析失败: {e}")

    def run(self):
        """运行应用程序"""
        # 显示窗口
        self.window_manager.show()


# 测试代码
if __name__ == "__main__":
    # 创建应用程序实例
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    # 创建主窗口实例
    caller = MCPAICallerLangChain()
    caller.run()

    # 进入应用程序主循环
    sys.exit(app.exec())
