import sys
import asyncio
import os
import json
import threading
import queue
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 添加当前目录到路径,支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout,
    QWidget, QLabel, QDialog, QScrollArea, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QKeySequence, QShortcut
import importlib.util

# 导入记忆窗口 (使用绝对导入)
try:
    from memory_window import MemoryWindow
except ImportError:
    MemoryWindow = None


class MonitoringWindow(QMainWindow):
    """监控显示窗口 - 用于显示VLM分析或吐槽内容"""

    def __init__(self, title: str, window_type: str):
        """
        初始化监控窗口

        Args:
            title: 窗口标题
            window_type: 窗口类型 ('analysis' 或 'commentary')
        """
        super().__init__()
        self.window_type = window_type
        self._setup_window(title)
        self._setup_ui()

    def _setup_window(self, title: str):
        """设置窗口属性"""
        self.setWindowTitle(title)

        # 吐槽窗口使用无边框透明窗口
        if self.window_type == 'commentary':
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        else:
            self.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.WindowStaysOnTopHint
            )

        # 设置窗口位置和大小
        if self.window_type == 'commentary':
            # 吐槽窗口在上层，位置更靠上，与主窗口x坐标对齐
            self.setGeometry(200, 50, 500, 200)

    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)

        # 根据窗口类型设置样式
        if self.window_type == 'analysis':
            color = '#00aaff'  # VLM分析窗口用蓝色
            bg_alpha = 220
        else:
            color = '#ffaa00'  # 吐槽窗口用橙色
            bg_alpha = 100  # 吐槽窗口更透明

        self.text_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(30, 30, 30, {bg_alpha});
                color: {color};
                border: 2px solid {color};
                border-radius: 5px;
                font-family: Consolas, monospace;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }}
        """)

        layout.addWidget(self.text_display)

    def add_text(self, text: str):
        """添加文本到窗口"""
        self.text_display.append(text)
        # 自动滚动到底部
        scrollbar = self.text_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_text(self):
        """清空窗口内容"""
        self.text_display.clear()


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

    # 定义信号用于线程间通信
    vlm_result_ready = pyqtSignal(str)
    vlm_error_ready = pyqtSignal(str)
    add_caption_signal = pyqtSignal(str)  # 添加字幕信号

    def __init__(self):
        """初始化MCP AI调用器"""
        super().__init__()
        try:
            print("[初始化] 设置初始变量...")
            self._setup_initial_variables()
            
            print("[初始化] 设置记忆系统...")
            self._setup_memory_system()
            
            print("[初始化] 设置定时器...")
            self._setup_timers()
            
            print("[初始化] 设置窗口...")
            self._setup_window()
            
            print("[初始化] 初始化服务...")
            self._initialize_services()
            
            print("[初始化] 初始化客户端...")
            self._initialize_clients()
            
            print("[初始化] 设置知识库...")
            self._setup_knowledge_base()  # 新增知识库功能

            print("[初始化] 设置监控窗口...")
            self._setup_monitoring_windows()  # 新增监控窗口

            print("[初始化] 设置UI...")
            self._setup_ui()

            print("[初始化] 设置快捷键...")
            self._setup_shortcuts()

            print("[初始化] 设置信号连接...")
            self._setup_signal_connections()

            print("[初始化] 初始化自我监控...")
            self._initialize_self_monitoring()

            # 默认启动工作分析模式
            self.work_analysis_mode = True
            self.tool_activation_pending = False
            self.tool_activation_start_time = None
            
            print("[初始化] 异步加载MCP工具...")
            self.async_refresh_tools_list()  # 默认寻找MCP工具
            
            print("[初始化] 完成!")
        except Exception as e:
            import traceback
            print(f"[错误] 初始化失败: {e}")
            traceback.print_exc()
            raise

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
        self.screenshot_timer = None
        self.screenshot_buffer = []
        self.last_recorded_action = ""
        self.last_analysis_result = ""  # 记录上一次的分析结果，用于去重

        # 多轮对话历史
        self.conversation_history = []
        self.is_processing_message = False  # 防止消息并发处理
        self.pending_messages = []  # 待处理消息队列

        # VLM历史记录（用于发送给LLM吐槽）
        self.vlm_history = []
        self.vlm_commentary_timer = None

        # VLM请求队列（确保按顺序执行）
        self.vlm_request_queue = []
        self.vlm_processing = False  # 标记是否正在处理VLM请求

        # 吐槽控制：每3个VLM消息触发一次吐槽
        self.vlm_commentary_threshold = 3  # 吐槽阈值
        self.vlm_count_since_last_commentary = 0  # 距离上次吐槽的VLM数量

        # 工具激活相关
        self.tool_activation_pending = False
        self.tool_activation_start_time = None

        # 知识库
        self.knowledge_base = []

        # 自我监控线程
        self.self_monitoring_thread = None

    def _setup_memory_system(self):
        """设置记忆系统"""
        memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_memory.json")
        from memory_module import GameMemory, MemoryPromptInjector
        
        # 删除旧的记忆文件以强制重新创建
        if os.path.exists(memory_file):
            os.remove(memory_file)
        
        # 创建默认格式的记忆文件
        default_data = {
            "memories": [],
            "session_start_time": datetime.now().isoformat(),
            "total_entries": 0
        }
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        
        self.game_memory = GameMemory(memory_file)
        self.memory_injector = MemoryPromptInjector(self.game_memory)
        self.game_memory.load_memory()

    def _setup_knowledge_base(self):
        """设置知识库"""
        knowledge_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.txt")
        self.knowledge_base = []
        
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 按行分割知识库内容
                    self.knowledge_base = [line.strip() for line in content.split('\n') if line.strip()]
            except Exception as e:
                print(f"加载知识库失败: {e}")
                self.knowledge_base = []
        else:
            # 创建默认知识库文件
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                f.write("# 工作流程知识库\n")
                f.write("建模步骤: 设计草图 -> 建立基础模型 -> 添加细节 -> 渲染输出\n")
                f.write("日常任务: 查看邮件 -> 回复重要信息 -> 执行项目任务 -> 总结报告\n")
                f.write("会议准备: 准备材料 -> 检查设备 -> 预演内容 -> 发送提醒\n")
            self.knowledge_base = [
                "建模步骤: 设计草图 -> 建立基础模型 -> 添加细节 -> 渲染输出",
                "日常任务: 查看邮件 -> 回复重要信息 -> 执行项目任务 -> 总结报告",
                "会议准备: 准备材料 -> 检查设备 -> 预演内容 -> 发送提醒"
            ]

    def _setup_timers(self):
        """设置定时器"""
        # 设置定时器，每30秒自动记录游戏状态
        self.auto_record_timer = QTimer(self)
        self.auto_record_timer.timeout.connect(self.auto_record_game_state)
        self.auto_record_timer.start(30000)  # 30秒

        # 设置定时器，每30秒将VLM历史发送给LLM吐槽（已改为基于VLM数量触发）
        # 保留定时器作为备选，但主要逻辑改为每3个VLM消息触发一次
        self.vlm_commentary_timer = QTimer(self)
        self.vlm_commentary_timer.timeout.connect(self._send_vlm_history_to_llm)
        self.vlm_commentary_timer.start(30000)  # 30秒（备用定时器）

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

    def _setup_monitoring_windows(self):
        """设置监控显示窗口（吐槽窗口和记忆窗口）"""
        # 创建吐槽窗口
        self.commentary_window = MonitoringWindow("吐槽窗口", "commentary")
        self.commentary_window.hide()

        # 创建记忆窗口 (如果可用)
        if MemoryWindow is not None:
            self.memory_window = MemoryWindow()
            self.memory_window.hide()
        else:
            self.memory_window = None
            print("[警告] 记忆窗口不可用")

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

        # 启动时自动加载工具
        self.tools_loaded_once = False
        self.async_refresh_tools_list()  # 默认寻找MCP工具

    def _initialize_self_monitoring(self):
        """初始化自我监控线程"""
        from self_monitoring import SelfMonitoringThread

        # 创建自我监控线程（verbose=False，减少控制台输出）
        self.self_monitoring_thread = SelfMonitoringThread(
            vlm_service=self.vlm_service,
            llm_service=self.llm_service,
            callback_analysis=self._on_self_monitoring_analysis,
            callback_commentary=self._on_self_monitoring_commentary,
            verbose=False,  # 不输出详细日志到控制台
            enable_memory=True,  # 启用向量记忆系统
            callback_memory_retrieved=self._on_memory_retrieved,  # 记忆检索回调
            callback_memory_saved=self._on_memory_saved  # 记忆保存回调
        )
        print("[自我监控] 线程已创建（未启动，已启用向量记忆系统）")

    def _on_self_monitoring_analysis(self, analysis: str):
        """
        自我监控VLM分析结果回调 - 显示在主窗口字幕区

        Args:
            analysis: VLM分析结果
        """
        # 添加到主窗口字幕区
        self.add_caption_line(f"[分析] {analysis}")

    def _on_self_monitoring_commentary(self, commentary: str):
        """
        自我监控吐槽结果回调 - 显示在吐槽窗口

        Args:
            commentary: 吐槽文本
        """
        # 添加到吐槽窗口
        try:
            # 显示窗口（如果还没显示）
            self.commentary_window.show()
            # 添加文本
            self.commentary_window.add_text(f"{commentary}")
            print(f"[回调] 吐槽已添加到吐槽窗口: {commentary[:30]}...")
        except Exception as e:
            print(f"[错误] 添加吐槽到窗口失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_memory_retrieved(self, query_text: str, results: List):
        """
        记忆检索回调 - 显示在记忆窗口

        Args:
            query_text: 查询文本
            results: 检索结果
        """
        if hasattr(self, 'memory_window') and self.memory_window:
            self.memory_window.log_retrieval(query_text, results)

    def _on_memory_saved(self, memory_id: str, vlm_analysis: str, llm_commentary: str):
        """
        记忆保存回调 - 显示在记忆窗口

        Args:
            memory_id: 记忆ID
            vlm_analysis: VLM分析结果
            llm_commentary: LLM吐槽
        """
        if hasattr(self, 'memory_window') and self.memory_window:
            self.memory_window.log_save(memory_id, vlm_analysis, llm_commentary)
            # 更新统计信息
            if hasattr(self, 'self_monitoring_thread') and self.self_monitoring_thread.vector_memory:
                stats = self.self_monitoring_thread.vector_memory.get_stats()
                self.memory_window.update_stats(stats.get('total_memories', 0))

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

        # 自动启动自我监控
        self.add_caption_line("[系统] MCP工具加载完成")
        QTimer.singleShot(1000, self.start_self_monitoring)  # 延迟1秒启动

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
                font-size: 18px;
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
    
    def _setup_signal_connections(self):
        """设置信号连接"""
        # 连接 VLM 结果信号
        self.vlm_result_ready.connect(self._display_vlm_analysis)
        self.vlm_error_ready.connect(self._handle_vlm_analysis_error)
        # 连接添加字幕信号
        self.add_caption_signal.connect(self._add_caption_line_via_signal)

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
        添加一行字幕文本（线程安全）

        Args:
            text: 要添加的文本
        """
        print(f"[UI DEBUG] 尝试添加文本: {text[:50]}...")  # 调试输出
        print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}")  # 线程检查

        # 检查当前线程，确保在主线程中执行UI操作
        if QThread.currentThread() != self.thread():
            print("[UI DEBUG] 不在主线程，使用信号转发")
            # 如果不在主线程，使用信号转发
            self.add_caption_signal.emit(text)
        else:
            print("[UI DEBUG] 在主线程，直接添加")
            # 在主线程中直接执行
            self._add_caption_line_impl(text)

    def _add_caption_line_via_signal(self, text: str):
        """通过信号添加文本（在主线程中执行）"""
        print(f"[UI DEBUG] 信号触发，添加文本: {text[:30]}...")
        self._add_caption_line_impl(text)
    
    def _add_caption_line_impl(self, text: str):
        """实际执行添加字幕文本的实现（必须在主线程中调用）"""
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
        print(f"[UI DEBUG] 文本已添加，当前行数: {len(lines)}")  # 调试输出

        # 不再自动清空，保留所有历史记录，向上滚动查看

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
        elif user_input == '/t':  # 处理工具激活确认
            self.confirm_tool_activation()
        elif user_input.startswith('/r ') and len(user_input) > 3:
            self._handle_run_command(user_input[3:].strip())
        elif user_input == '/clear_conv':
            self.clear_conversation_history()
        elif user_input.startswith('/work '):  # 工作相关命令
            work_command = user_input[6:].strip()
            self.process_work_command(work_command)
        elif user_input == '/sm_start':  # 启动自我监控
            self.start_self_monitoring()
        elif user_input == '/sm_stop':  # 停止自我监控
            self.stop_self_monitoring()
        elif user_input == '/sm_pause':  # 暂停自我监控
            self.pause_self_monitoring()
        elif user_input == '/sm_resume':  # 恢复自我监控
            self.resume_self_monitoring()
        else:
            # 默认处理用户输入，包括工作分析
            self.process_message_with_function_call(user_input)

    def confirm_tool_activation(self):
        """确认工具激活"""
        if self.tool_activation_pending and self.tool_activation_start_time:
            elapsed = (datetime.now() - self.tool_activation_start_time).seconds
            if elapsed <= 20:
                self.tool_activation_pending = False
                self.tool_activation_start_time = None
                self.add_caption_line("[系统] 工具激活确认，开始执行...")
                
                # 执行工具推荐
                self.recommend_tools_based_on_context()
            else:
                self.tool_activation_pending = False
                self.tool_activation_start_time = None
                self.add_caption_line("[系统] 工具激活已超时")
        else:
            self.add_caption_line("[系统] 没有待确认的工具激活")

    def recommend_tools_based_on_context(self):
        """根据当前上下文推荐工具"""
        # 分析当前工作状态，推荐合适的工具
        context_analysis = self.analyze_current_context()
        
        # 生成工具推荐提示
        recommendation_prompt = f"""当前工作环境分析：{context_analysis}

基于当前工作状态，从以下可用工具中推荐最合适的：
{json.dumps(self.all_tools_mapping, ensure_ascii=False, indent=2)}

请推荐最适合当前工作的工具，并说明原因。"""
        
        try:
            result = self.llm_service.create([{"role": "user", "content": recommendation_prompt}])
            recommendation = ContentExtractor.extract_content_from_response(result)
            
            if recommendation:
                self.add_caption_line(f"[工具推荐] {recommendation}")
        except Exception as e:
            self.add_caption_line(f"工具推荐失败: {str(e)}")

    def analyze_current_context(self):
        """分析当前工作上下文"""
        # 结合记忆、当前输入和知识库分析上下文
        recent_memory = self.game_memory.get_recent_memories(5)
        recent_conversations = self.conversation_history[-5:] if self.conversation_history else []
        
        context_info = {
            "recent_memories": [mem.get("action", "") for mem in recent_memory],
            "recent_conversations": [conv.get("content", "") for conv in recent_conversations],
            "knowledge_base": self.knowledge_base,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(context_info, ensure_ascii=False)

    def process_work_command(self, command: str):
        """处理工作相关命令"""
        # 将工作命令加入对话历史
        self.conversation_history.append({"role": "user", "content": f"工作命令: {command}"})

        # 加入知识库信息
        knowledge_context = "\n".join(self.knowledge_base)
        enhanced_command = f"""工作知识库:
{knowledge_context}

当前工作命令: {command}

请分析这个工作命令并提供建议或执行方案。"""

        messages = [{"role": "user", "content": enhanced_command}]

        try:
            result = self.llm_service.create(messages, tools=self.get_mcp_tools_schema() if self.tools_by_server else None)
            response = ContentExtractor.extract_content_from_response(result)

            if response:
                self.add_caption_line(response)
                # 将AI回复添加到对话历史
                self.conversation_history.append({"role": "assistant", "content": response})

                # 检查是否需要激活工具
                if self.should_activate_tools_for_response(response):
                    self.request_tool_activation()

        except Exception as e:
            self.add_caption_line(f"处理工作命令时出错: {str(e)}")

    def start_self_monitoring(self):
        """启动自我监控"""
        if self.self_monitoring_thread:
            if not self.self_monitoring_thread.running:
                self.self_monitoring_thread.start_monitoring()
                # 显示吐槽窗口
                self.commentary_window.show()
                # 显示记忆窗口 (如果可用)
                if self.memory_window:
                    self.memory_window.show()
                self.add_caption_line("[系统] 自我监控已启动：每10秒截5张图，VLM分析后立即生成吐槽")
            else:
                self.add_caption_line("[系统] 自我监控已在运行中")
        else:
            self.add_caption_line("[错误] 自我监控线程未初始化")

    def stop_self_monitoring(self):
        """停止自我监控"""
        if self.self_monitoring_thread:
            self.self_monitoring_thread.stop_monitoring()
            self.add_caption_line("[系统] 自我监控已停止")
        else:
            self.add_caption_line("[错误] 自我监控线程未初始化")

    def pause_self_monitoring(self):
        """暂停自我监控"""
        if self.self_monitoring_thread:
            self.self_monitoring_thread.pause_monitoring()
            self.add_caption_line("[系统] 自我监控已暂停")
        else:
            self.add_caption_line("[错误] 自我监控线程未初始化")

    def resume_self_monitoring(self):
        """恢复自我监控"""
        if self.self_monitoring_thread:
            self.self_monitoring_thread.resume_monitoring()
            self.add_caption_line("[系统] 自我监控已恢复")
        else:
            self.add_caption_line("[错误] 自我监控线程未初始化")

    def should_activate_tools_for_response(self, response: str) -> bool:
        """判断响应是否需要激活工具"""
        # 如果响应中包含"建议使用"、"推荐使用"、"可以使用"等词汇，则提示激活工具
        activation_keywords = ["建议使用", "推荐使用", "可以使用", "应该使用", "需要使用", "考虑使用"]
        return any(keyword in response for keyword in activation_keywords)

    def request_tool_activation(self):
        """请求工具激活"""
        self.tool_activation_pending = True
        self.tool_activation_start_time = datetime.now()
        self.add_caption_line("[系统] 检测到工具建议，输入 /t 确认激活工具 (20秒内有效)")

    def clear_conversation_history(self):
        """清除对话历史"""
        self.conversation_history = []
        self.add_caption_line("[系统] 对话历史已清除")

    def _handle_help_command(self):
        """处理帮助命令"""
        # 第一次使用/h时加载工具
        if not self.tools_loaded_once:
            self.add_caption_line("[系统] 首次加载MCP工具列表...")
            self.async_refresh_tools_list()
        self.show_tools_dialog()

        # 显示帮助信息
        help_text = """
【可用命令】
/h - 显示MCP工具列表
/m - 显示记忆摘要
/mc - 清空当前会话记忆
/t - 确认工具激活
/r <命令> - 运行工具命令
/work <命令> - 处理工作相关命令
/clear_conv - 清除对话历史
/sm_start - 启动自我监控（每10秒截5张图）
/sm_stop - 停止自我监控
/sm_pause - 暂停自我监控
/sm_resume - 恢复自我监控
"""
        self.add_caption_line(help_text)

    def _handle_run_command(self, command: str):
        """处理运行命令 - 通过LLM调用function call

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

        # 构建提示词，让LLM理解用户意图并调用相应工具
        tools_info = ""
        for tool_id, tool_info in self.all_tools_mapping.items():
            tools_info += f"\n[{tool_id}] {tool_info['name']}: {tool_info['description']}"

        enhanced_prompt = f"""请执行以下任务，从可用工具中选择合适的工具并调用：

可用工具：
{tools_info}

用户请求：{command}

要求：
1. 理解用户意图，选择最合适的工具
2. 根据用户请求确定工具参数
3. 直接调用工具，不要询问用户
4. 如果用户只提供了工具编号或名称，请根据上下文推断参数
"""

        messages = [{"role": "user", "content": enhanced_prompt}]
        tools_schema = self.get_mcp_tools_schema()

        try:
            result = self.llm_service.create(messages, tools=tools_schema)
            self.process_function_call_response(result, messages)
        except Exception as e:
            self.add_caption_line(f"调用失败: {str(e)}")

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
        # 添加当前用户输入到对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # 判断是否需要注入记忆
        if self.memory_injector.should_inject_memory(user_input):
            # 普通对话，注入记忆
            user_input_with_memory = self.memory_injector.inject_memory_to_prompt(user_input)
            messages = self.conversation_history[-10:]  # 只使用最近10条对话
            messages[-1]["content"] = user_input_with_memory  # 更新最新消息内容
        else:
            # 工具调用，不注入记忆
            messages = self.conversation_history[-10:]  # 只使用最近10条对话
        
        # 加入知识库信息
        knowledge_context = "\n".join(self.knowledge_base)
        if knowledge_context:
            messages[0]["content"] = f"工作知识库:\n{knowledge_context}\n\n{messages[0].get('content', '')}"

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
                    # 将AI回复添加到对话历史
                    self.conversation_history.append({"role": "assistant", "content": content})
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
        thread.join(timeout=5)  # 减少超时时间

        if thread.is_alive():
            return {"error": "MCP 调用超时（5秒）"}

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

        # 停止自我监控线程
        if self.self_monitoring_thread:
            self.self_monitoring_thread.stop_monitoring()
            # 等待线程结束
            if self.self_monitoring_thread.is_alive():
                self.self_monitoring_thread.join(timeout=2)

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
                context=f"用户提问: {user_input}",
                analysis=analysis
            )
            print(f"[记忆已记录] 行动: {action}")
        except Exception as e:
            print(f"记录记忆失败: {e}")

    def auto_record_game_state(self):
        """自动记录游戏状态（每30秒调用一次）"""
        # 只有当有记录的行动时才自动记录
        if self.last_recorded_action:
            self.add_caption_line("[系统] 自动记录工作状态...")
            self.game_memory.add_memory(
                action=f"持续执行: {self.last_recorded_action}",
                context="30秒时间节点记录",
                analysis="等待用户下一步指令"
            )
            print(f"[自动记录] 时间: {datetime.now().strftime('%H:%M:%S')}")

    def _send_vlm_history_to_llm(self):
        """将VLM历史记录发送给LLM进行吐槽，支持工具调用建议"""
        if not self.vlm_history:
            return

        # 构建历史记录摘要
        history_summary = "\n".join([f"[{item['time']}] {item['action']}" for item in self.vlm_history])

        # 根据工作模式生成不同的吐槽提示
        if self.work_analysis_mode:
            commentary_prompt = f"""我正在工作，这是我最近的操作：

{history_summary}

你是我的工作助手，分析我的工作状态并提供建议：
1. 如果我在建模或开发，给出专业建议
2. 如果工作效率不高，提出改进意见
3. 如有必要，推荐可使用的工具来提高效率
4. 语气友善但专业，40-70字之间
5. 如果检测到适合使用工具的情况，明确指出
6. 推荐具体的MCP工具（如果有）"""
        else:
            commentary_prompt = f"""我哥哥在玩游戏，这是我看到的他最近的游戏操作：

{history_summary}

你是傲娇妹妹，用傲娇的语气吐槽一下，要求：
1. 典型的傲娇性格：表面嫌弃、嘴硬心软、爱逞强
2. 经典傲娇口癖："哼"、"才不是"、"笨蛋"、"笨-蛋"、"啰嗦"、"谁、谁在乎啊"、"才没有"
3. 明明关心却装作不在意：用"我不过是随便看看"、"才不是为了你"等
4. 看到好操作要傲娇地夸奖："勉勉强强还可以啦"、"一般般吧"、"也就那样"
5. 看到差操作要毒舌吐槽："笨蛋哥哥"、"真拿你没办法"、"啧，真是的"
6. 40-70字之间
7. 语气要像傲娇妹妹一边嫌弃一边偷偷关注哥哥玩游戏

此外，如果发现可以使用工具改善游戏体验，推荐相应的MCP工具。"""

        try:
            # 包含可用工具信息
            tools_info = json.dumps(self.all_tools_mapping, ensure_ascii=False) if self.all_tools_mapping else "无可用工具"
            complete_prompt = f"{commentary_prompt}\n\n可用MCP工具列表：\n{tools_info}"
            
            commentary = self.llm_service.create([{"role": "user", "content": complete_prompt}])
            commentary_text = ContentExtractor.extract_content_from_response(commentary)

            if commentary_text:
                self.add_caption_line(f"[吐槽] {commentary_text}")
                
                # 检查吐槽内容是否包含工具推荐
                if self.contains_tool_suggestion(commentary_text):
                    self.request_tool_activation()
        except Exception as e:
            print(f"获取LLM吐槽失败: {e}")

        # 清空历史记录
        self.vlm_history.clear()

    def contains_tool_suggestion(self, text: str) -> bool:
        """检查文本是否包含工具建议"""
        tool_keywords = ["工具", "使用", "推荐", "建议", "MCP", "执行", "运行", "调用"]
        return any(keyword in text for keyword in tool_keywords)

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
            # 清空截图缓冲区
            self.screenshot_buffer = []
            
            # 初始化截图计时器 - 每2秒截一张图
            if self.screenshot_timer is None:
                self.screenshot_timer = QTimer(self)
                self.screenshot_timer.timeout.connect(self.capture_screenshot)
            
            # 初始化战术分析计时器 - 每10秒分析一次
            if self.tactical_analyzer_timer is None:
                self.tactical_analyzer_timer = QTimer(self)
                self.tactical_analyzer_timer.timeout.connect(self.auto_tactical_analysis)
            
            self.screenshot_timer.start(2000)  # 2秒
            self.tactical_analyzer_timer.start(10000)  # 10秒
            
            self.add_caption_line("[工作分析师] 模式已启动")
            self.add_caption_line("[工作分析师] 每2秒截图，每10秒分析5张图")
            self.caption_text.update()
            QApplication.processEvents()
        else:
            if self.screenshot_timer:
                self.screenshot_timer.stop()
            if self.tactical_analyzer_timer:
                self.tactical_analyzer_timer.stop()
            self.screenshot_buffer = []
            self.add_caption_line("[工作分析师] 模式已停止")
            self.caption_text.update()
            QApplication.processEvents()
    
    def capture_screenshot(self):
        """每2秒捕获一张截图"""
        screenshot_path = self.take_immediate_screenshot()
        if screenshot_path and os.path.exists(screenshot_path):
            validated_path = self.validate_and_prepare_image(screenshot_path)
            if validated_path:
                self.screenshot_buffer.append(validated_path)
                # 保持最多5张截图
                if len(self.screenshot_buffer) > 5:
                    self.screenshot_buffer.pop(0)
                print(f"[截图] 已捕获 {len(self.screenshot_buffer)}/5 张截图")
    
    def take_immediate_screenshot(self) -> Optional[str]:
        """
        立即截取屏幕截图，支持区域选择和缩放
        """
        try:
            from PIL import ImageGrab
            import time
            
            # 截取整个屏幕
            screenshot = ImageGrab.grab()
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.png"
            
            # 保存截图到项目目录下的screenshots文件夹
            screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            
            filepath = os.path.join(screenshots_dir, filename)
            screenshot.save(filepath, "PNG")
            
            print(f"截图已保存到: {filepath}")
            return filepath
        except Exception as e:
            print(f"截图失败: {e}")
            return None

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

    def validate_and_prepare_image(self, image_path: str) -> Optional[str]:
        """
        验证并准备图像用于VLM处理，支持图像缩放
        """
        try:
            from PIL import Image
            
            # 打开并验证图像
            with Image.open(image_path) as img:
                # 检查图像大小，如果过大则缩放到最大 800x800
                max_size = (800, 800)
                if img.width > max_size[0] or img.height > max_size[1]:
                    print(f"图像尺寸 ({img.width}x{img.height}) 超过最大尺寸 {max_size}，开始缩放...")
                    
                    # 计算缩放比例
                    ratio = min(max_size[0]/img.width, max_size[1]/img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    
                    print(f"缩放到: {new_width}x{new_height}")
                    
                    # 缩放图像并保存到临时位置
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 生成缩放后的临时文件
                    import tempfile
                    import os
                    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 使用原文件名添加_resized后缀
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    ext = os.path.splitext(os.path.basename(image_path))[1]
                    temp_path = os.path.join(temp_dir, f"{base_name}_resized{ext}")
                    
                    img_resized.save(temp_path, quality=85)
                    print(f"缩放图像已保存到: {temp_path}")
                    return temp_path
                else:
                    # 图像尺寸合适，直接返回原路径
                    print(f"图像尺寸 ({img.width}x{img.height}) 无需缩放")
                    return image_path
        except Exception as e:
            print(f"图像验证失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def auto_tactical_analysis(self):
        """自动工作/游戏环境分析（每10秒调用一次）- 使用5张截图一起分析"""
        if not self.work_analysis_mode:
            return

        # 检查是否有足够的截图
        if len(self.screenshot_buffer) < 5:
            print(f"[分析] 截图数量不足 {len(self.screenshot_buffer)}/5，等待下一次...")
            return

        # 获取缓冲区中的5张截图
        screenshots_to_analyze = self.screenshot_buffer.copy()
        self.screenshot_buffer = []  # 清空缓冲区

        # 将请求加入队列
        vlm_request = {
            'type': 'environment_analysis',
            'screenshots': screenshots_to_analyze
        }
        self.vlm_request_queue.append(vlm_request)
        print(f"[队列] VLM请求已加入队列，当前队列长度: {len(self.vlm_request_queue)}")

        # 尝试处理队列
        self._process_vlm_queue()

    def _process_vlm_queue(self):
        """处理VLM请求队列（确保按顺序执行）"""
        # 如果正在处理或队列为空，直接返回
        if self.vlm_processing or not self.vlm_request_queue:
            return

        self.vlm_processing = True
        print(f"[队列] 开始处理VLM请求，剩余: {len(self.vlm_request_queue)}")

        # 取出第一个请求
        request = self.vlm_request_queue.pop(0)

        # 根据工作模式构建不同提示词
        if self.work_analysis_mode:
            full_prompt = """这5张是连续2秒间隔的桌面截图，请综合分析当前工作环境和正在进行的操作。

重点关注：
1. 建模软件：CAD、Blender、Maya、SolidWorks等
2. 办公软件：文档编辑、表格处理、演示文稿
3. 开发环境：代码编辑器、IDE、终端
4. 工作进度：当前操作、遇到的问题、下一步计划

要求：
- 识别当前使用的软件和工具
- 描述正在进行的工作内容
- 用50字以内概括当前工作状态
- 提供可能的改进建议"""
        else:
            full_prompt = """这5张是连续2秒间隔的游戏截图，请综合分析游戏角色在这10秒内的行为变化和动作序列。

重点关注：
1. 击杀瞬间：角色击杀敌人的画面（血条消失、敌人倒下等）
2. 死亡瞬间：角色被击杀的画面（屏幕变灰、倒地、阵亡提示等）
3. 战斗状态：交火、受伤、瞄准、逃跑等

要求：
- 如果出现击杀或死亡，明确标注"击杀"或"死亡"
- 用50字以内描述
- 只描述动作和场景，不要剧情描写

例如："角色开枪击杀一名敌人，继续前进" 或 "角色被击杀，屏幕变灰"""

        # 在后台线程中执行VLM调用
        def vlm_analysis_task():
            try:
                vlm_messages = [
                    {"role": "user", "content": full_prompt}
                ]

                # 使用5张截图一起分析
                vlm_result = self.vlm_service.create_with_multiple_images(
                    vlm_messages,
                    image_sources=request['screenshots']
                )

                analysis_result = ContentExtractor.extract_content_from_response(vlm_result)

                # 将分析结果添加到历史记录（带时间戳）
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.vlm_history.append({
                    "time": timestamp,
                    "action": analysis_result
                })

                # 检查是否是重复内容
                if self._is_duplicate_analysis(analysis_result):
                    print("[队列] 重复内容，跳过")
                else:
                    # 直接调用处理函数（PyQt6 支持从任何线程调用UI方法）
                    self._handle_vlm_analysis_result(analysis_result)

                # 检查是否达到吐槽阈值
                self.vlm_count_since_last_commentary += 1
                if self.vlm_count_since_last_commentary >= self.vlm_commentary_threshold:
                    print(f"[吐槽] 已收集{self.vlm_count_since_last_commentary}个VLM分析，触发吐槽")
                    self._send_vlm_history_to_llm()
                    self.vlm_count_since_last_commentary = 0

            except AttributeError:
                # 如果VLM服务不支持多图分析，fallback到单图分析
                print("[分析] VLM不支持多图分析，使用单图模式")
                self._fallback_single_image_analysis(request['screenshots'][-1])
            except Exception as vlm_error:
                error_msg = f"VLM分析失败: {vlm_error}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # 使用信号发送错误到主线程
                self.vlm_error_ready.emit(f"[画面] VLM分析失败: {str(vlm_error)}")
            finally:
                # 请求完成，标记为未处理状态，并处理下一个请求
                self.vlm_processing = False
                print(f"[队列] 当前请求完成，继续处理下一个...")
                self._process_vlm_queue()

        # 在后台线程中执行VLM调用
        analysis_thread = threading.Thread(target=vlm_analysis_task, daemon=True)
        analysis_thread.start()

    def _fallback_single_image_analysis(self, image_path: str):
        """fallback到单图分析"""
        try:
            full_prompt = """用20字以内简短描述当前正在做什么。

重点关注：
1. 建模：CAD、Blender、Maya、SolidWorks等
2. 办公：文档编辑、表格处理、演示文稿
3. 开发：代码编辑器、IDE、终端
4. 工作状态：编辑、设计、调试、浏览等

只描述动作和场景，不要剧情描写。"""
            
            vlm_messages = [{"role": "user", "content": full_prompt}]
            vlm_result = self.vlm_service.create_with_image(
                vlm_messages,
                image_source=image_path
            )
            
            analysis_result = ContentExtractor.extract_content_from_response(vlm_result)
            
            # 检查是否是重复内容
            if not self._is_duplicate_analysis(analysis_result):
                self._handle_vlm_analysis_result(analysis_result)
        except Exception as e:
            print(f"fallback单图分析失败: {e}")
            self.vlm_error_ready.emit(f"[画面] 单图分析失败: {str(e)}")

    def _handle_vlm_analysis_result(self, analysis_result: str):
        """处理VLM分析结果（线程安全）"""
        # 使用信号发送到主线程
        self.vlm_result_ready.emit(analysis_result)
    
    def _display_vlm_analysis(self, analysis_result: str):
        """在主线程中显示VLM分析结果"""
        if analysis_result and analysis_result.strip():
            # 清理输出
            cleaned_result = analysis_result.strip()

            # 更新最后一次分析结果
            self.last_analysis_result = cleaned_result

            # 直接显示结果
            self.add_caption_line(cleaned_result)

            # 强制刷新UI
            self.caption_text.update()
            QApplication.processEvents()

            # 记录到记忆
            self.game_memory.add_memory(
                action=cleaned_result,
                context=f"自动分析 - {datetime.now().strftime('%H:%M:%S')}",
                analysis=""
            )
    
    def _is_duplicate_analysis(self, new_analysis: str) -> bool:
        """检查分析结果是否重复"""
        if not self.last_analysis_result or not new_analysis:
            return False
        
        # 简单的相似度检查：如果长度差异超过50%则认为不重复
        len_diff = abs(len(new_analysis) - len(self.last_analysis_result))
        max_len = max(len(new_analysis), len(self.last_analysis_result))
        if len_diff / max_len > 0.5:
            return False
        
        # 检查是否完全相同
        if new_analysis == self.last_analysis_result:
            return True
        
        # 检查相似度（简单的前100个字符匹配）
        min_len = min(len(new_analysis), len(self.last_analysis_result))
        if min_len > 50:
            match_chars = sum(1 for a, b in zip(new_analysis[:100], self.last_analysis_result[:100]) if a == b)
            if match_chars / 100 > 0.9:  # 90%以上相似则认为是重复
                return True
        
        return False

    def _handle_vlm_analysis_error(self, error_message: str):
        """在主线程中处理VLM分析错误"""
        print(f"[UI DEBUG] _handle_vlm_analysis_error 被调用: {error_message}")
        self.add_caption_line(error_message)


def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        app.setStyleSheet("QMainWindow { background-color: transparent; }")
        
        print("[启动] 创建主窗口...")
        window = MCPAICaller()
        
        # 修改窗口大小，使其更大
        window.setGeometry(200, 200, 500, 300)  # 增加宽度和高度
        window.show()
        
        print("[启动] 进入消息循环...")
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        print(f"[错误] 启动失败: {e}")
        traceback.print_exc()
        input("按回车键退出...")


if __name__ == "__main__":
    main()