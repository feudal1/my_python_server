"""
MCP AI Caller - MCP协议AI调用器

这是一个基于PyQt6的图形界面应用，用于与MCP服务器进行交互，并提供AI驱动的工具调用、
记忆管理和战术分析功能。
"""

import sys
import asyncio
import os
import json
import threading
import queue
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout, 
    QWidget, QLabel, QDialog, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QKeySequence, QShortcut
import importlib.util


class ContentExtractor:
    """从LLM响应中提取内容的工具类"""
    
    @staticmethod
    def extract_content_from_response(result: Any) -> str:
        """
        从LLM响应中安全提取纯文本内容，兼容多种格式
        
        Args:
            result: LLM返回的结果，可能是字符串或字典
            
        Returns:
            提取的文本内容
        """
        if isinstance(result, str):
            return result.strip()
        
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                
                # 尝试从message中提取内容
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    elif content is None:
                        return ""
                
                # 尝试从delta中提取内容（流式响应）
                delta = first_choice.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        return content.strip()
                
                # 如果有finish_reason但没有内容，返回空字符串
                if "finish_reason" in first_choice:
                    return ""
        
        return ""


class ToolsDialog(QDialog):
    """显示可用MCP工具的对话框"""
    
    def __init__(self, tools_by_server: Dict[str, Dict[str, str]], 
                 all_tools_mapping: Optional[Dict[str, Any]] = None, 
                 parent=None):
        """
        初始化工具对话框
        
        Args:
            tools_by_server: 按服务器分组的工具映射
            all_tools_mapping: 所有工具的详细信息映射
            parent: 父窗口
        """
        super().__init__(parent)
        self.all_tools_mapping = all_tools_mapping or {}
        self.tools_by_server = tools_by_server
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("可用的MCP工具")
        self.setGeometry(300, 300, 800, 600)
        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.WindowMinimizeButtonHint | 
            Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea { 
                border: 1px solid #555; 
                background-color: #f0f0f0; 
            }
        """)
        
        content_widget = QWidget()
        scroll_layout = QVBoxLayout(content_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        for server_name, tools in self.tools_by_server.items():
            server_title = QLabel(f"<b>{server_name}</b>")
            server_title.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #333;
                    margin-top: 10px;
                    margin-bottom: 10px;
                    padding: 5px;
                    border-bottom: 1px solid #555;
                }
            """)
            scroll_layout.addWidget(server_title)
            
            tools_vbox = QVBoxLayout()
            tools_vbox.setSpacing(8)
            tools_vbox.setContentsMargins(10, 0, 0, 0)
            
            for tool_name, tool_desc in tools.items():
                tool_number = self.find_tool_number(tool_name, server_name)
                if tool_number:
                    name_label = QLabel(f"• [{tool_number}] {tool_name}:")
                else:
                    name_label = QLabel(f"• {tool_name}:")
                    
                name_label.setStyleSheet("""
                    QLabel { 
                        font-weight: bold; 
                        color: #333; 
                        margin-left: 10px; 
                    }
                """)
                
                desc_label = QLabel(tool_desc)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("""
                    QLabel { 
                        color: #555; 
                        margin-left: 25px; 
                        margin-bottom: 8px; 
                    }
                """)
                
                tools_vbox.addWidget(name_label)
                tools_vbox.addWidget(desc_label)
                
            scroll_layout.addLayout(tools_vbox)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def find_tool_number(self, tool_name: str, display_server_name: str) -> Optional[str]:
        """
        根据工具名称和服务器名称查找工具编号
        
        Args:
            tool_name: 工具名称
            display_server_name: 显示的服务器名称
            
        Returns:
            工具编号，如果找不到则返回None
        """
        original_server_name = self.get_original_server_name(display_server_name)
        
        for num, info in self.all_tools_mapping.items():
            if (info['name'] == tool_name and 
                (info['server'] == original_server_name or 
                 self.format_server_name(info['server']) == display_server_name)):
                return num
        return None

    def get_original_server_name(self, display_server_name: str) -> str:
        """
        从显示的服务器名称获取原始服务器名称
        
        Args:
            display_server_name: 显示的服务器名称
            
        Returns:
            原始服务器名称
        """
        if display_server_name.endswith(" 工具服务器"):
            formatted_name = display_server_name[:-5].strip()
            original = formatted_name.replace(' ', '-').lower()
            for candidate in [original, original + "-tool"]:
                for _, info in self.all_tools_mapping.items():
                    if info['server'] == candidate:
                        return candidate
            return original
        return display_server_name

    def format_server_name(self, original_server_name: str) -> str:
        """
        格式化服务器名称以便显示
        
        Args:
            original_server_name: 原始服务器名称
            
        Returns:
            格式化后的服务器名称
        """
        display_name = original_server_name.replace('-tool', '').replace('-', ' ').title()
        display_name += " 工具服务器"
        return display_name


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
        try:
            tools_by_server = {}
            all_tools_mapping = {}
            tool_counter = 1

            for server_name in self.mcp_client.servers.keys():
                try:
                    # 创建新的事件循环来获取工具列表
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        tools = loop.run_until_complete(self.mcp_client.list_tools(server_name))
                        
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
                            
                            display_server_name = server_name.replace('-tool', '') \
                                .replace('-', ' ').title() + " 工具服务器"
                            tools_by_server[display_server_name] = server_tools
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
                    finally:
                        loop.close()
                except Exception as e:
                    print(f"获取服务器 {server_name} 的工具列表失败: {str(e)}")
                    continue

            self.tools_loaded.emit((all_tools_mapping, tools_by_server))
        except Exception as e:
            self.loading_failed.emit(f"加载工具列表失败: {str(e)}")


class MCPAICaller(QMainWindow):
    """MCP AI调用器主窗口类"""
    
    def __init__(self):
        """初始化MCP AI调用器"""
        super().__init__()
        self._setup_initial_variables()
        self._setup_memory_system()
        self._setup_timers()
        self._setup_window()
        self._initialize_services()
        self._initialize_clients()
        self._setup_ui()
        self._setup_shortcuts()

    def _setup_initial_variables(self):
        """设置初始变量"""
        self.output_buffer = ""
        self.worker_thread = None
        self.worker = None
        self.tool_loader_thread = None
        self.tool_loader = None
        self.is_loading_tools = False
        self.all_tools_mapping = {}
        self.tools_by_server = {}
        self.loading_dialog = None
        self.pending_show_tools = False
        
        # 战术分析师模式相关
        self.tactical_analyzer_mode = False
        self.tactical_analyzer_timer = None
        self.last_recorded_action = ""

    def _setup_memory_system(self):
        """设置记忆系统"""
        memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_memory.json")
        from memory_module import GameMemory, MemoryPromptInjector
        
        self.game_memory = GameMemory(memory_file)
        self.memory_injector = MemoryPromptInjector(self.game_memory)

    def _setup_timers(self):
        """设置定时器"""
        # 设置定时器，每30秒自动记录游戏状态
        self.auto_record_timer = QTimer(self)
        self.auto_record_timer.timeout.connect(self.auto_record_game_state)
        self.auto_record_timer.start(30000)  # 30秒

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(200, 200, 300, 120)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.transparent)
        self.setPalette(palette)
        self.min_height = 120
        self.max_height = 500

    def _initialize_services(self):
        """初始化LLM和VLM服务"""
        llm_spec = importlib.util.spec_from_file_location(
            "llm_class",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_server", "llm_class.py")
        )
        llm_module = importlib.util.module_from_spec(llm_spec)
        llm_spec.loader.exec_module(llm_module)
        self.LLMService = llm_module.LLMService
        self.llm_service = self.LLMService()

        # 初始化VLM服务（用于战术分析师模式的视觉分析）
        self.VLMService = llm_module.VLMService
        self.vlm_service = self.VLMService()

    def _initialize_clients(self):
        """初始化MCP客户端"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_client_path = os.path.join(current_dir, "mcp_client.py")
        spec = importlib.util.spec_from_file_location("mcp_client", mcp_client_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        self.mcp_client = mcp_module.MCPClient()

        self.tool_loader = ToolLoader(self.mcp_client)
        self.tool_loader.tools_loaded.connect(self.on_tools_loaded)
        self.tool_loader.loading_failed.connect(self.on_tools_loading_failed)
        
        # 不在启动时加载工具，改为懒加载
        self.tools_loaded_once = False

    def on_tools_loaded(self, data_tuple: Tuple[Dict, Dict]):
        """
        工具加载完成回调
        
        Args:
            data_tuple: 包含所有工具映射和按服务器分组的工具的元组
        """
        all_tools_mapping, tools_by_server = data_tuple
        self.all_tools_mapping = all_tools_mapping
        self.tools_by_server = tools_by_server
        self.is_loading_tools = False
        
        if self.loading_dialog:
            self.loading_dialog.accept()
            self.loading_dialog = None
        if self.pending_show_tools:
            self.pending_show_tools = False
            self._show_tools_dialog_now()

    def on_tools_loading_failed(self, error_msg: str):
        """
        工具加载失败回调
        
        Args:
            error_msg: 错误信息
        """
        self.is_loading_tools = False
        if self.loading_dialog:
            self.loading_dialog.reject()
            self.loading_dialog = None
            
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("错误")
        error_dialog.setText(f"加载工具列表失败: {error_msg}")
        error_dialog.setIcon(QMessageBox.Icon.Warning)
        error_dialog.exec()
        self.pending_show_tools = False

    def async_refresh_tools_list(self):
        """异步刷新工具列表"""
        if self.tools_loaded_once:
            return  # 已经加载过了，不再重复加载
            
        if self.is_loading_tools and self.tool_loader_thread and self.tool_loader_thread.isRunning():
            return
            
        self.is_loading_tools = True
        self.tools_loaded_once = True  # 标记为已加载
        
        if self.tool_loader_thread and self.tool_loader_thread.isRunning():
            self.tool_loader_thread.quit()
            self.tool_loader_thread.wait(2000)
            
        self.tool_loader_thread = QThread()
        self.tool_loader.moveToThread(self.tool_loader_thread)
        self.tool_loader_thread.started.connect(self.tool_loader.load_tools)
        self.tool_loader_thread.finished.connect(self.tool_loader_thread.deleteLater)
        self.tool_loader_thread.start()

    def _setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.caption_text = QTextEdit()
        self.caption_text.setReadOnly(True)
        self.caption_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0);
                color: #00ff41;
                border: none;
                font-family: Consolas, monospace;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.caption_text.setMaximumHeight(100)
        layout.addWidget(self.caption_text)

        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("输入消息...")
        self.input_text.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 180);
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #00ff41;
            }
        """)
        self.input_text.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_text)
        self.input_text.setFocus()

    def _setup_shortcuts(self):
        """设置快捷键"""
        quit_shortcut = QShortcut(QKeySequence('Ctrl+Q'), self)
        quit_shortcut.activated.connect(self.close)
        
        up_shortcut = QShortcut(QKeySequence('Alt+Up'), self)
        up_shortcut.activated.connect(lambda: self.move_window(0, -10))
        down_shortcut = QShortcut(QKeySequence('Alt+Down'), self)
        down_shortcut.activated.connect(lambda: self.move_window(0, 10))
        left_shortcut = QShortcut(QKeySequence('Alt+Left'), self)
        left_shortcut.activated.connect(lambda: self.move_window(-10, 0))
        right_shortcut = QShortcut(QKeySequence('Alt+Right'), self)
        right_shortcut.activated.connect(lambda: self.move_window(10, 0))

    def move_window(self, dx: int, dy: int):
        """
        移动窗口
        
        Args:
            dx: X轴移动距离
            dy: Y轴移动距离
        """
        current_pos = self.pos()
        new_x = max(0, current_pos.x() + dx)
        new_y = max(0, current_pos.y() + dy)
        self.move(new_x, new_y)

    def clear_captions(self):
        """清除字幕文本"""
        self.caption_text.clear()

    def add_caption_line(self, text: str):
        """
        添加一行字幕文本
        
        Args:
            text: 要添加的文本
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return
            
        current = self.caption_text.toPlainText()
        new = current + "\n" + text if current else text
        lines = new.split('\n')[-20:]
        self.caption_text.setPlainText('\n'.join(lines))
        self.caption_text.moveCursor(self.caption_text.textCursor().MoveOperation.End)
        QTimer.singleShot(30000, self.clear_captions)

    def send_message(self):
        """发送消息处理"""
        user_input = self.input_text.text().strip()
        if not user_input:
            return
        self.input_text.clear()

        # 处理特殊命令
        if user_input == '/h':
            self._handle_help_command()
        elif user_input == '/m':
            self.show_memory_summary()
        elif user_input == '/mc':
            self.clear_memory()
        elif user_input == '/g':
            self.toggle_tactical_analyzer_mode()
        elif user_input.startswith('/r ') and len(user_input) > 3:
            self._handle_run_command(user_input[3:].strip())
        else:
            self.process_message_with_function_call(user_input)

    def _handle_help_command(self):
        """处理帮助命令"""
        # 第一次使用/h时加载工具
        if not self.tools_loaded_once:
            self.add_caption_line("[系统] 首次加载MCP工具列表...")
            self.async_refresh_tools_list()
        self.show_tools_dialog()

    def _handle_run_command(self, command: str):
        """处理运行命令
        
        Args:
            command: 运行命令参数
        """
        # 第一次使用/r时加载工具
        if not self.tools_loaded_once:
            self.add_caption_line("[系统] 首次加载MCP工具列表...")
            self.async_refresh_tools_list()
            # 等待工具加载完成
            while self.is_loading_tools:
                QApplication.processEvents()
        
        if command.isdigit():
            self.handle_run_command_by_index(int(command))
        else:
            self.process_message_with_function_call(command)

    def get_mcp_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取MCP工具的JSON Schema
        
        Returns:
            工具的JSON Schema列表
        """
        if not self.all_tools_mapping:
            return []
            
        tools = []
        for tool_id, tool_info in self.all_tools_mapping.items():
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_info['name'],
                    "description": tool_info['description'],
                    "parameters": tool_info.get('input_schema', {
                        "type": "object", 
                        "properties": {}, 
                        "required": []
                    })
                }
            }
            tools.append(tool_schema)
        return tools

    def handle_run_command_by_index(self, index: int):
        """
        通过索引号处理运行命令
        
        Args:
            index: 工具索引号
        """
        if str(index) not in self.all_tools_mapping:
            self.add_caption_line(f"错误：没有找到编号为 {index} 的工具")
            return

        tool_info = self.all_tools_mapping[str(index)]
        tools_schema = self.get_mcp_tools_schema()
        messages = [{"role": "user", "content": f"请立即调用工具 '{tool_info['name']}'。"}]

        try:
            result = self.llm_service.create(messages, tools=tools_schema)
            self.process_function_call_response(result, messages)
        except Exception as e:
            self.add_caption_line(f"调用失败: {str(e)}")

    def process_function_call_response(self, result: Dict[str, Any], original_messages: List[Dict[str, str]]):
        """
        处理函数调用响应
        
        Args:
            result: LLM返回的结果
            original_messages: 原始消息列表
        """
        try:
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                tool_calls = choice.get('message', {}).get('tool_calls', [])
                
                if tool_calls:
                    for tool_call in tool_calls:
                        function_name = tool_call['function']['name']
                        arguments = json.loads(tool_call['function']['arguments'])

                        server_name = None
                        for tool_id, info in self.all_tools_mapping.items():
                            if info['name'] == function_name:
                                server_name = info['server']
                                break

                        if not server_name:
                            self.add_caption_line(f"错误：找不到工具 {function_name} 对应的服务器")
                            continue

                        self.add_caption_line(f"[调用工具] {function_name}")
                        print(f"【MCP CALL】Calling {server_name}.{function_name} with args: {arguments}")

                        # 执行 MCP 调用
                        mcp_result = self.execute_mcp_call_sync(server_name, function_name, arguments)
                        print(f"【MCP RESULT】{mcp_result}")

                        # 显示错误（如果存在）
                        if isinstance(mcp_result, dict) and "error" in mcp_result:
                            self.add_caption_line(f"[错误] {mcp_result['error']}")

                        # 构造 tool response 消息
                        updated_messages = original_messages.copy()
                        updated_messages.append({
                            "role": "assistant",
                            "tool_calls": [tool_call]
                        })
                        updated_messages.append({
                            "role": "tool",
                            "content": json.dumps(mcp_result, ensure_ascii=False),
                            "tool_call_id": tool_call.get('id', '')
                        })

                        # 获取最终自然语言回复
                        final_result = self.llm_service.create(updated_messages)
                        final_content = ContentExtractor.extract_content_from_response(final_result)
                        self.add_caption_line(final_content if final_content else "[AI未返回内容]")
                else:
                    content = ContentExtractor.extract_content_from_response(result)
                    self.add_caption_line(content if content else "[AI未返回内容]")
            else:
                content = ContentExtractor.extract_content_from_response(result)
                self.add_caption_line(content if content else "[AI未返回内容]")
        except Exception as e:
            import traceback
            error_msg = f"处理函数调用时出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.add_caption_line(f"处理函数调用时出错: {str(e)}")

    def show_tools_dialog(self):
        """显示工具对话框"""
        if self.tools_by_server:
            self._show_tools_dialog_now()
        else:
            self.pending_show_tools = True
            if not self.is_loading_tools:
                self.async_refresh_tools_list()
            if not self.loading_dialog:
                self.loading_dialog = QMessageBox(self)
                self.loading_dialog.setWindowTitle("加载中")
                self.loading_dialog.setText("正在加载MCP工具列表，请稍候...")
                self.loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
                self.loading_dialog.show()

    def _show_tools_dialog_now(self):
        """立即显示工具对话框"""
        if self.tools_by_server:
            dialog = ToolsDialog(self.tools_by_server, self.all_tools_mapping, self)
            dialog.show()  # 使用 show() 替代 exec()，不阻塞主线程
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("暂无可用的MCP工具")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.show()  # 使用 show() 替代 exec()，不阻塞主线程

    def process_message_with_function_call(self, user_input: str):
        """
        处理带有函数调用的消息
        
        Args:
            user_input: 用户输入的消息
        """
        # 判断是否需要注入记忆
        if self.memory_injector.should_inject_memory(user_input):
            # 普通对话，注入记忆
            user_input_with_memory = self.memory_injector.inject_memory_to_prompt(user_input)
            messages = [{"role": "user", "content": user_input_with_memory}]
        else:
            # 工具调用，不注入记忆
            messages = [{"role": "user", "content": user_input}]
        
        tools_schema = self.get_mcp_tools_schema() if self.tools_by_server else None

        try:
            result = self.llm_service.create(messages, tools=tools_schema)
            choices = result.get("choices", [])
            if choices and "tool_calls" in choices[0].get("message", {}):
                self.process_function_call_response(result, messages)
            else:
                content = ContentExtractor.extract_content_from_response(result)
                if content:
                    self.add_caption_line(content)
                    # 如果是普通对话，尝试解析AI回复并记录到记忆
                    if not user_input.startswith('/r'):
                        self.record_ai_response_to_memory(user_input, content)
        except Exception as e:
            self.add_caption_line(f"处理消息时出错: {str(e)}")

    def execute_mcp_call_sync(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步执行 MCP 调用，使用 Queue 获取子线程结果
        
        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 调用参数
            
        Returns:
            调用结果
        """
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.mcp_client.call_tool(server_name, tool_name, arguments)
                )
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
            finally:
                # 安全关闭事件循环
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=15)  # 给 Blender 等慢启动工具留足时间

        if thread.is_alive():
            return {"error": "MCP 调用超时（15秒）"}

        if not exception_queue.empty():
            exc = exception_queue.get()
            return {"error": f"MCP 执行异常: {str(exc)}"}

        if not result_queue.empty():
            result = result_queue.get()
            # 尝试从 MCP Result 提取文本内容
            try:
                texts = []
                for item in getattr(result, 'content', []):
                    if hasattr(item, 'text') and isinstance(item.text, str):
                        texts.append(item.text.strip())
                if texts:
                    return {"result": "\n".join(texts)}
                else:
                    return {"result": str(result)}
            except Exception as parse_err:
                return {"result": str(result), "warning": f"解析结果时出错: {parse_err}"}
        else:
            return {"error": "MCP 调用无返回结果"}

    def closeEvent(self, event):
        """
        关闭事件处理
        
        Args:
            event: 关闭事件
        """
        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
        
        if self.tool_loader_thread and self.tool_loader_thread.isRunning():
            self.tool_loader_thread.quit()
            self.tool_loader_thread.wait(3000)
        self.is_loading_tools = False
        
        # 停止自动记录定时器
        self.auto_record_timer.stop()
        
        # 保存记忆
        self.game_memory.save_memory()
        
        event.accept()

    def record_ai_response_to_memory(self, user_input: str, ai_response: str):
        """
        将AI响应记录到记忆中
        
        Args:
            user_input: 用户输入
            ai_response: AI响应
        """
        try:
            action, analysis = self.memory_injector.parse_ai_response(ai_response)
            self.last_recorded_action = action
            self.game_memory.add_memory(
                action=action,
                context=f"玩家提问: {user_input}",
                analysis=analysis
            )
            print(f"[记忆已记录] 行动: {action}")
        except Exception as e:
            print(f"记录记忆失败: {e}")

    def auto_record_game_state(self):
        """自动记录游戏状态（每30秒调用一次）"""
        # 只有当有记录的行动时才自动记录
        if self.last_recorded_action:
            self.add_caption_line("[系统] 自动记录游戏状态...")
            self.game_memory.add_memory(
                action=f"持续执行: {self.last_recorded_action}",
                context="30秒时间节点记录",
                analysis="等待玩家下一步指令"
            )
            print(f"[自动记录] 时间: {datetime.now().strftime('%H:%M:%S')}")

    def show_memory_summary(self):
        """显示记忆摘要"""
        summary = self.game_memory.analyze_memories()
        self.add_caption_line(summary)

    def clear_memory(self):
        """清空当前会话记忆"""
        self.game_memory.clear_current_session()
        self.add_caption_line("[系统] 当前会话记忆已清空")
    
    def toggle_tactical_analyzer_mode(self):
        """切换战术分析师模式"""
        self.tactical_analyzer_mode = not self.tactical_analyzer_mode

        if self.tactical_analyzer_mode:
            # 初始化战术分析计时器
            if self.tactical_analyzer_timer is None:
                self.tactical_analyzer_timer = QTimer(self)
                self.tactical_analyzer_timer.timeout.connect(self.auto_tactical_analysis)
            
            self.tactical_analyzer_timer.start(30000)  # 30秒
            self.add_caption_line("[战术分析师] 模式已启动 - 每30秒自动分析")
            self.add_caption_line("[战术分析师] 开始首次分析...")
            self.caption_text.update()  # 强制刷新显示
            QApplication.processEvents()  # 处理事件队列
            # 立即执行一次分析
            self.auto_tactical_analysis()
        else:
            if self.tactical_analyzer_timer:
                self.tactical_analyzer_timer.stop()
            self.add_caption_line("[战术分析师] 模式已停止")
            self.caption_text.update()  # 强制刷新显示
            QApplication.processEvents()  # 处理事件队列
    
    def get_latest_screenshot(self) -> Optional[str]:
        """
        获取最新的游戏截图
        
        Returns:
            最新截图路径，如果没有找到则返回None
        """
        # 定义可能的截图目录
        screenshot_dirs = [
            os.path.join(os.path.expanduser('~'), 'Desktop', 'screenshots'),
            os.path.join(os.path.expanduser('~'), 'Pictures', 'Screenshots'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screenshots'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        ]

        # 支持的截图文件格式
        screenshot_extensions = ['.png', '.jpg', '.jpeg']

        latest_file = None
        latest_time = 0

        for dir_path in screenshot_dirs:
            if not os.path.exists(dir_path):
                continue

            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    # 检查文件扩展名
                    _, ext = os.path.splitext(file)
                    if ext.lower() in screenshot_extensions:
                        # 检查文件修改时间
                        file_time = os.path.getmtime(file_path)
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_file = file_path

        return latest_file

    def auto_tactical_analysis(self):
        """自动战术分析（每30秒调用一次）"""
        if not self.tactical_analyzer_mode:
            return

        self.add_caption_line(f"[AI指挥] {datetime.now().strftime('%H:%M:%S')} 正在分析...")

        # 获取历史记录
        recent_memories = self.game_memory.get_recent_memories(10)
        context_lines = ["历史记录:"]
        for i, memory in enumerate(recent_memories, 1):
            context_lines.append(f"{i}. {memory['action']}")
            if memory['context']:
                context_lines.append(f"   情况: {memory['context']}")
            if memory['analysis']:
                context_lines.append(f"   分析: {memory['analysis']}")
            context_lines.append("")

        # 获取截图
        screenshot_path = self.get_latest_screenshot()
        visual_analysis = ""
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                vlm_result = self.vlm_service.create_with_image(
                    [{"role": "user", "content": "一句话描述当前角色意图"}],
                    image_source=screenshot_path
                )
                visual_analysis = ContentExtractor.extract_content_from_response(vlm_result)
                self.add_caption_line(f"[画面] {visual_analysis}")
            except Exception as e:
                print(f"VLM失败: {e}")

        # 构建分析提示
        analysis_prompt = f"""{chr(10).join(context_lines)}

画面意图: {visual_analysis if visual_analysis else "无"}

你是作战辅助师，用三句话回复：
1. 一句话描述当前角色意图
2. 一句话指导下一步操作
3. 一句话解释为什么要走这一步"""

        try:
            # 异步执行分析，避免阻塞UI
            result = self.llm_service.create([{"role": "user", "content": analysis_prompt}])
            content = ContentExtractor.extract_content_from_response(result)

            if content:
                self.add_caption_line(f"--- 作战辅助 ---")
                self.add_caption_line(content)

                # 记录到记忆
                self.game_memory.add_memory(
                    action=content,
                    context=f"自动战术分析 - {datetime.now().strftime('%H:%M:%S')}",
                    analysis=""
                )
            else:
                self.add_caption_line("[AI] 未能获取分析结果")
                
        except Exception as e:
            error_msg = f"战术分析出错: {str(e)}"
            self.add_caption_line(error_msg)
            print(error_msg)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("QMainWindow { background-color: transparent; }")
    window = MCPAICaller()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()