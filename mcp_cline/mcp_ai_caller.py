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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QMetaObject, pyqtSlot, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut, QColor
import importlib.util

# 导入记忆窗口 (使用绝对导入)
try:
    from memory_window import MemoryWindow
except ImportError:
    MemoryWindow = None

# 导入拆分出来的组件
from components.monitoring_window import MonitoringWindow
from components.content_extractor import ContentExtractor
from components.tools_dialog import ToolsDialog
from components.tool_loader import ToolLoader


# 以下是MCPAICaller类的定义


class MCPAICaller(QMainWindow):
    """MCP AI调用器主窗口类"""

    # 定义信号用于线程间通信
    vlm_result_ready = pyqtSignal(str)
    vlm_error_ready = pyqtSignal(str)
    add_caption_signal = pyqtSignal(str)  # 添加字幕信号

    def __init__(self):
        """初始化MCP AI调用器"""
        super().__init__()

        # 初始化变量
        self._initialize_variables()

        # 设置记忆系统
        self._setup_memory_system()

        # 设置定时器
        self._setup_timers()

        # 设置窗口
        self._setup_window()

        # 初始化服务
        self._initialize_services()

        # 初始化客户端
        self._initialize_clients()

        # 设置知识库
        self._setup_knowledge_base()

        # 设置监控窗口
        self._setup_monitoring_windows()

        # 设置UI
        self._setup_ui()

        # 设置快捷键
        self._setup_shortcuts()

        # 设置信号连接
        self._setup_connections()

        # 初始化自我监控
        self._initialize_self_monitoring()

        # 异步加载MCP工具
        self.async_refresh_tools_list()

        print("[初始化] 完成!")
        print("[启动] MCP AI Caller 已启动")

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

        # 记忆窗口对象
        self.memory_window = None

        # 吐槽窗口对象
        self.commentary_window = None

        # 记忆系统
        self.vector_memory = None

        # 知识库
        self.knowledge_base = None

        # 工具加载器
        self.tool_loader = None

        # 自我监控线程
        self.self_monitoring_thread = None

        # 工具是否已经加载过
        self.tools_loaded_once = False

        # 最后一次工具加载时间
        self.last_tools_load_time = 0

        # 工具加载间隔（秒）
        self.tools_load_interval = 60

    def _setup_memory_system(self):
        """设置记忆系统"""
        print("[初始化] 设置记忆系统...")

        # 尝试导入向量记忆系统
        try:
            from vector_memory import VectorMemory
            self.vector_memory = VectorMemory()
            print("[初始化] 向量记忆系统已加载")
        except ImportError:
            print("[警告] 向量记忆系统未找到，将使用简单记忆模式")
            self.vector_memory = None
        except Exception as e:
            print(f"[错误] 初始化向量记忆系统失败: {e}")
            self.vector_memory = None

    def _setup_timers(self):
        """设置定时器"""
        print("[初始化] 设置定时器...")

        # 设置定时器，每30秒自动记录游戏状态
        self.auto_record_timer = QTimer(self)
        self.auto_record_timer.timeout.connect(self.auto_record_game_state)
        self.auto_record_timer.start(30000)  # 30秒

    def _setup_window(self):
        """设置窗口属性"""
        print("[初始化] 设置窗口...")

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        # 设置窗口背景完全透明
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # 主窗口放在屏幕中间偏上的位置，避免与游戏UI重叠
        self.setGeometry(800, 100, 300, 180)  # 增加高度以容纳输入栏
        self.min_height = 180  # 增加最小高度
        self.max_height = 500

    def _setup_monitoring_windows(self):
        """设置监控显示窗口（吐槽窗口和记忆窗口）"""
        print("[初始化] 设置监控窗口...")

        # 创建吐槽窗口
        self.commentary_window = MonitoringWindow("吐槽窗口", "commentary")
        # 吐槽窗口放在屏幕中间偏左的位置，避免与游戏UI重叠
        self.commentary_window.setGeometry(300, 100, 500, 200)
        self.commentary_window.show()  # 初始显示

        # 创建记忆窗口 (如果可用)
        if MemoryWindow is not None:
            self.memory_window = MemoryWindow()
            # 记忆窗口放在屏幕中间偏下的位置，变小一点，避免与游戏UI重叠
            self.memory_window.setGeometry(300, 350, 800, 180)
            self.memory_window.show()  # 初始显示
        else:
            self.memory_window = None
            print("[警告] 记忆窗口不可用")

    def _initialize_services(self):
        """初始化LLM和VLM服务"""
        print("[初始化] 初始化服务...")

        # 动态导入LLM服务
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

    def _setup_knowledge_base(self):
        """设置知识库"""
        print("[初始化] 设置知识库...")

        # 尝试导入知识库
        try:
            from knowledge_base import KnowledgeBase
            self.knowledge_base = KnowledgeBase()
            print("[初始化] 知识库已加载")
        except ImportError:
            print("[警告] 知识库未找到，将使用简单模式")
            self.knowledge_base = None
        except Exception as e:
            print(f"[错误] 初始化知识库失败: {e}")
            self.knowledge_base = None

    def _setup_ui(self):
        """设置UI"""
        print("[初始化] 设置UI...")

        # 创建主布局（仅包含字幕显示）
        main_layout = QVBoxLayout()

        # 创建字幕显示区域
        self.caption_display = QTextEdit(self)
        self.caption_display.setReadOnly(True)
        self.caption_display.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setStyleSheet(
            "QTextEdit { background-color: transparent; color: #ff0000; font-size: 20px; padding: 10px; }"
        )
        main_layout.addWidget(self.caption_display)

        # 创建主窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 创建独立的输入窗口
        self.input_window = QWidget()
        self.input_window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.input_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置输入窗口布局
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)

        # 创建输入栏
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("输入命令，例如：/h 打开工具窗口")
        self.input_line.setFixedHeight(30)
        self.input_line.setStyleSheet(
            "QLineEdit { background-color: rgba(30, 30, 30, 200); color: white; font-size: 14px; padding: 5px; border: 1px solid rgba(100, 100, 100, 150); border-radius: 4px; }"
        )
        self.input_line.returnPressed.connect(self.on_input_submitted)
        input_layout.addWidget(self.input_line)

        self.input_window.setLayout(input_layout)

        # 设置输入窗口在屏幕左下角（任务栏上方）
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        input_window_width = 300
        input_window_height = 40
        self.input_window.setGeometry(
            screen_geometry.left() + 10,  # 左边距 10px
            screen_geometry.bottom() - input_window_height - 10,  # 距离底部 10px
            input_window_width,
            input_window_height
        )
        self.input_window.show()

        # 初始显示欢迎信息
        self.caption_display.setHtml(
            "<div style='color: #ff0000; font-size: 14px;'>"\
            "<p>欢迎使用MCP AI Caller！</p>"\
            "<p>输入 /h 打开工具窗口</p>"\
            "</div>"
        )

    def _setup_shortcuts(self):
        """设置快捷键（已禁用）"""
        print("[初始化] 设置快捷键...")
        # 禁用所有键盘快捷键
        pass

    def _setup_connections(self):
        """设置信号连接"""
        print("[初始化] 设置信号连接...")

        # 连接VLM结果信号
        self.vlm_result_ready.connect(self.on_vlm_result_ready)
        self.vlm_error_ready.connect(self.on_vlm_error_ready)

        # 连接添加字幕信号
        self.add_caption_signal.connect(self.add_caption_line)

    def _initialize_self_monitoring(self):
        """初始化自我监控线程"""
        print("[初始化] 初始化自我监控...")

        # 动态导入自我监控线程
        try:
            from self_monitoring import SelfMonitoringThread

            # 创建自我监控线程（verbose=True，输出详细日志以便调试）
            self.self_monitoring_thread = SelfMonitoringThread(
                vlm_service=self.vlm_service,
                llm_service=self.llm_service,
                callback_analysis=self._on_self_monitoring_analysis,
                callback_commentary=self._on_self_monitoring_commentary,
                verbose=True,  # 输出详细日志以便调试
                enable_memory=True,  # 启用向量记忆系统
                enable_memory_retrieval=True,  # 启用记忆检索功能
                callback_memory_retrieved=self._on_memory_retrieved,  # 记忆检索回调
                callback_memory_saved=self._on_memory_saved,  # 记忆保存回调
                callback_hide_windows=self._hide_windows,  # 截图前隐藏窗口回调
                callback_show_windows=self._show_windows,  # 截图后显示窗口回调
                blocked_windows=["MCP AI Caller", "吐槽窗口", "记忆窗口", "任务管理器", "资源管理器", "命令提示符", "PowerShell"],  # 要屏蔽的窗口
            )
            print("[自我监控] 线程已创建（未启动，已启用向量记忆系统和记忆检索功能）")

            # 启动自我监控线程
            self.self_monitoring_thread.start_monitoring()
            print("[自我监控] 线程已启动")
        except ImportError as e:
            print(f"[自我监控] 导入失败: {e}")
        except Exception as e:
            print(f"[自我监控] 初始化失败: {e}")

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
        self.add_caption_line(f"[系统] MCP工具加载完成，共 {tool_count} 个工具")

        # 打印工具列表
        # all_tools_mapping 是字典格式: {"1": {...}, "2": {...}, ...}
        for tool_id, tool_info in all_tools_mapping.items():
            tool_name = tool_info.get('name', '无名称')
            tool_description = tool_info.get('description', '无描述')
            tool_server = tool_info.get('server', '未知服务器')
            print(f"- [{tool_id}] {tool_name} (服务器: {tool_server}): {tool_description}")

    def on_tools_loading_failed(self, error):
        """工具加载失败回调"""
        self.tools_loading = False
        print(f"[错误] 工具加载失败: {error}")

        # 添加到字幕显示
        self.add_caption_line(f"[错误] 工具加载失败: {error}")

    def open_tools_window(self):
        """打开工具窗口"""
        if not self.tools_loaded_once:
            # 如果工具还没有加载，先加载工具
            self.async_refresh_tools_list()
            QMessageBox.information(self, "提示", "工具正在加载中，请稍候...")
            return

        # 打开工具窗口
        try:
            dialog = ToolsDialog(self.tools_by_server, self.tools_list, self)
            dialog.exec()
        except Exception as e:
            print(f"[错误] 打开工具窗口失败: {e}")

    def open_memory_window(self):
        """打开记忆窗口"""
        if self.memory_window:
            self.memory_window.show()
            self.memory_window.raise_()
        else:
            QMessageBox.information(self, "提示", "记忆窗口不可用")

    def open_commentary_window(self):
        """打开吐槽窗口"""
        if self.commentary_window:
            self.commentary_window.show()
            self.commentary_window.raise_()
        else:
            QMessageBox.information(self, "提示", "吐槽窗口不可用")

    def add_caption_line(self, text):
        """添加字幕行"""
        print(f"[UI DEBUG] 尝试添加文本: {text}")
        # 检查是否在主线程
        if QThread.currentThread() == self.thread():
            print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
            print("[UI DEBUG] 在主线程，直接添加")
            # 在主线程，直接添加
            # 只显示当前回合的内容，不保留历史
            self.caption_display.setPlainText(text)
            # 滚动到底部
            self.caption_display.verticalScrollBar().setValue(self.caption_display.verticalScrollBar().maximum())
            print("[UI DEBUG] 文本已添加，只显示当前回合内容")
            
            # 设置定时器，10秒后清空字幕
            self.caption_timer = QTimer(self)
            self.caption_timer.setSingleShot(True)
            self.caption_timer.timeout.connect(self.clear_caption)
            self.caption_timer.start(10000)  # 10秒后清空
        else:
            print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
            print("[UI DEBUG] 不在主线程，使用信号转发")
            # 不在主线程，使用信号转发
            self.add_caption_signal.emit(text)

    def on_vlm_result_ready(self, result):
        """VLM结果回调"""
        print(f"[VLM] 分析结果: {result}")
        # 直接设置文本，避免add_caption_line的定时器问题
        self.caption_display.setPlainText(f"[分析] {result}")
        # 设置定时器，10秒后清空字幕
        self.caption_timer = QTimer(self)
        self.caption_timer.setSingleShot(True)
        self.caption_timer.timeout.connect(self.clear_caption)
        self.caption_timer.start(10000)  # 10秒后清空

    def on_vlm_error_ready(self, error):
        """VLM错误回调"""
        print(f"[VLM] 分析错误: {error}")
        self.add_caption_line(f"[错误] VLM分析失败: {error}")
    
    def clear_caption(self):
        """清空字幕显示区域"""
        print("[UI DEBUG] 清空字幕显示")
        if hasattr(self, 'caption_display') and self.caption_display:
            self.caption_display.setPlainText("")
        else:
            print("[UI DEBUG] caption_display不存在，跳过清空")

    def on_input_submitted(self):
        """处理用户输入"""
        input_text = self.input_line.text().strip()
        if not input_text:
            return

        # 清空输入栏
        self.input_line.clear()

        # 处理命令
        if input_text.startswith('/'):
            command = input_text[1:].lower()
            if command == 'h':
                # 打开工具窗口显示工具列表
                self.open_tools_window()
                self.add_caption_line(f"[命令] 打开工具窗口")
            elif command == 'showmemory':
                # 显示数据库中的记忆
                if hasattr(self, 'vector_memory') and self.vector_memory:
                    stats = self.vector_memory.get_stats()
                    memories = self.vector_memory.get_all_memories()
                    memory_info = f"数据库统计:\n"
                    memory_info += f"总记忆数: {stats.get('total_memories', 0)}\n"
                    memory_info += f"使用模型: {stats.get('model', '未知')}\n"
                    memory_info += f"版本: {stats.get('version', '未知')}\n\n"
                    memory_info += "最近记忆:\n"

                    for i, mem in enumerate(memories[:5]):  # 显示最近5组记忆
                        # 检查是否为分组后的记忆
                        if 'main' in mem:
                            # 新格式：分组记忆
                            main_mem = mem['main']
                            memory_info += f"\n[{i+1}] 主记忆 (时间: {main_mem['metadata'].get('datetime', '未知')})\n"
                            memory_info += f"    VLM分析: {main_mem['document'][:100]}...\n"

                            # 显示LLM吐槽
                            commentary_id = main_mem['id'].replace('mem_', 'roast_')
                            related_commentary = [m for m in memories if 'id' in m and m['id'] == commentary_id]
                            if related_commentary:
                                memory_info += f"    LLM吐槽: {related_commentary[0]['document'][:100]}...\n"

                            # 显示用户输入
                            if mem['user_inputs']:
                                memory_info += f"    用户输入 ({len(mem['user_inputs'])}条):\n"
                                for idx, user_input in enumerate(mem['user_inputs'][:3]):  # 最多显示3条
                                    memory_info += f"      - {user_input['document'][:80]}...\n"

                            # 显示VLM分析历史
                            if mem['vlm_analyses']:
                                memory_info += f"    VLM分析历史 ({len(mem['vlm_analyses'])}条):\n"
                                for idx, vlm_analysis in enumerate(mem['vlm_analyses'][:3]):  # 最多显示3条
                                    memory_info += f"      - {vlm_analysis['document'][:80]}...\n"
                        else:
                            # 旧格式：简单记忆
                            memory_info += f"\n[{i+1}] 记忆 (时间: {mem['metadata'].get('datetime', '未知')})\n"
                            memory_info += f"    类型: {mem.get('type', 'unknown')}\n"
                            memory_info += f"    内容: {mem['document'][:100]}...\n"

                    QMessageBox.information(self, "数据库记忆", memory_info)
                else:
                    QMessageBox.information(self, "提示", "向量记忆系统不可用")
                self.add_caption_line(f"[命令] 显示数据库记忆")
            elif command == 'clearmemory':
                # 清空数据库
                if hasattr(self, 'vector_memory') and self.vector_memory:
                    self.vector_memory.clear_all()
                    self.add_caption_line(f"[命令] 数据库已清空")
                else:
                    self.add_caption_line(f"[命令] 向量记忆系统不可用，无法清空数据库")
            elif command.startswith('r '):
                # /r 命令：直接发送给 LLM 执行（支持 function calling）
                query = command[2:].strip()
                if query:
                    self._send_user_input_to_llm_with_tools(query)
                else:
                    self.add_caption_line(f"[命令] 用法: /r <问题描述>")
            else:
                self.add_caption_line(f"[命令] 未知命令: {input_text}")
        else:
            # 处理普通文本输入
            self.add_caption_line(f"[你] {input_text}")

            # 将用户输入添加到用户输入历史中，稍后与VLM信息一起发送给LLM
            self.user_input_history.append(input_text)
            print(f"[用户输入] 已添加到历史记录: {input_text[:30]}...")

            # 将用户输入添加到自我监控线程的历史记录中
            if hasattr(self, 'self_monitoring_thread') and self.self_monitoring_thread:
                self.self_monitoring_thread.add_user_input(input_text)
                print(f"[用户输入] 已添加到自我监控线程历史记录: {input_text[:30]}...")


    def _send_user_input_to_llm(self, input_text):
        """将用户输入发送给LLM处理"""
        print(f"[LLM] 处理用户输入: {input_text[:30]}...")
        
        # 将用户输入添加到对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': input_text
        })
        
        # 构建messages格式，用于LLM服务
        messages = []
        
        # 添加系统消息
        messages.append({
            'role': 'system',
            'content': '你是一个智能助手，根据对话历史回答用户的问题'
        })
        
        # 添加最近的对话历史（最多10条）
        recent_history = self.conversation_history[-10:]
        for item in recent_history:
            messages.append(item)
        
        # 调用LLM服务
        try:
            response = self.llm_service.create(messages)
            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[LLM] 生成回复完成: {content[:30]}...")
                
                # 将LLM回复添加到对话历史
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content
                })
                
                # 显示在主窗口中
                self.add_caption_line(f"[AI] {content}")
        except Exception as e:
            print(f"[错误] 调用LLM服务失败: {e}")
            self.add_caption_line(f"[错误] 处理请求失败: {e}")

    def _send_user_input_to_llm_with_tools(self, input_text):
        """将用户输入发送给LLM处理（支持 function calling）"""
        print(f"[LLM] 处理用户输入（带工具）: {input_text[:30]}...")

        # 构建工具列表格式（OpenAI function calling 格式）
        tools = self._convert_tools_to_openai_format()

        # 将用户输入添加到对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': input_text
        })

        # 构建messages格式，用于LLM服务
        messages = []

        # 添加系统消息
        system_prompt = """你是一个智能助手，可以调用各种工具来帮助用户完成任务。

可用工具包括：
- Blender工具：3D模型处理、导入导出、骨骼绑定等
- Unreal Engine工具：FBX导入、UE启动、MOD构建等
- 计算机控制工具：鼠标点击、键盘输入、滚轮滚动等
- OCR工具：文字识别和坐标定位
- 浏览器工具：打开URL
- 点赞收藏检测工具：检测点赞和收藏按钮

根据用户的需求，选择合适的工具来执行任务。如果需要调用工具，请使用function calling格式。"""
        messages.append({
            'role': 'system',
            'content': system_prompt
        })

        # 添加最近的对话历史（最多10条）
        recent_history = self.conversation_history[-10:]
        for item in recent_history:
            messages.append(item)

        # 调用LLM服务
        try:
            response = self.llm_service.create(messages, tools=tools)
            if response:
                # 提取回复内容
                assistant_message = response['choices'][0]['message']

                # 检查是否有工具调用
                if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
                    print(f"[LLM] 检测到工具调用: {len(assistant_message['tool_calls'])} 个")

                    # 处理工具调用
                    self._execute_tool_calls(assistant_message['tool_calls'])

                    # 将工具调用消息添加到对话历史
                    self.conversation_history.append(assistant_message)

                    # 显示在主窗口中
                    self.add_caption_line(f"[AI] 正在执行 {len(assistant_message['tool_calls'])} 个工具调用...")
                else:
                    # 普通文本回复
                    content = assistant_message.get('content', '')
                    print(f"[LLM] 生成回复完成: {content[:30]}...")

                    # 将LLM回复添加到对话历史
                    self.conversation_history.append({
                        'role': 'assistant',
                        'content': content
                    })

                    # 显示在主窗口中
                    self.add_caption_line(f"[AI] {content}")
        except Exception as e:
            print(f"[错误] 调用LLM服务失败: {e}")
            self.add_caption_line(f"[错误] 处理请求失败: {e}")

    def _convert_tools_to_openai_format(self):
        """将工具列表转换为 OpenAI function calling 格式"""
        tools = []

        for tool_id, tool_info in self.tools_list.items():
            tool_def = {
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
            tools.append(tool_def)

        print(f"[工具转换] 已转换 {len(tools)} 个工具到 OpenAI 格式")
        return tools

    def _execute_tool_calls(self, tool_calls):
        """执行工具调用"""
        import asyncio

        async def execute_single_tool(tool_call):
            """执行单个工具调用"""
            tool_name = tool_call['function']['name']
            try:
                arguments_str = tool_call['function'].get('arguments', '{}')
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except Exception as e:
                print(f"[工具调用] 解析参数失败: {e}")
                arguments = {}

            print(f"[工具调用] 执行工具: {tool_name}")
            print(f"[工具调用] 参数: {arguments}")

            try:
                # 查找工具所属的服务器
                server_name = None
                for tool_id, tool_info in self.tools_list.items():
                    if tool_info['name'] == tool_name:
                        server_name = tool_info['server']
                        break

                if server_name:
                    # 调用 MCP 客户端执行工具
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = await self.mcp_client.call_tool(server_name, tool_name, arguments)
                        print(f"[工具调用] 执行成功: {result}")

                        # 将工具调用结果添加到对话历史
                        self.conversation_history.append({
                            'role': 'tool',
                            'name': tool_name,
                            'content': str(result)
                        })

                        # 获取 LLM 对工具结果的总结
                        self._get_llm_summary_for_tool_result(tool_name, result)

                        return result
                    finally:
                        loop.close()
                else:
                    print(f"[工具调用] 未找到工具 {tool_name} 所属的服务器")
                    self.add_caption_line(f"[错误] 未找到工具: {tool_name}")
                    return None
            except Exception as e:
                print(f"[工具调用] 执行失败: {e}")
                import traceback
                traceback.print_exc()
                self.add_caption_line(f"[错误] 工具执行失败: {e}")
                return None

        # 创建事件循环并执行所有工具调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def execute_all_tools():
            """并行执行所有工具调用"""
            tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        try:
            results = loop.run_until_complete(execute_all_tools())
            print(f"[工具调用] 所有工具执行完成")
        finally:
            loop.close()

    def _get_llm_summary_for_tool_result(self, tool_name, tool_result):
        """获取 LLM 对工具执行结果的总结"""
        try:
            # 构建系统消息
            system_prompt = "你是一个智能助手，请简要总结工具执行的结果。"

            messages = [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': f"工具 {tool_name} 执行完成，结果是：\n{str(tool_result)}\n请简要总结这个结果。"
                }
            ]

            # 调用LLM服务
            response = self.llm_service.create(messages)

            if response:
                summary = response['choices'][0]['message']['content']
                print(f"[LLM] 工具结果总结: {summary[:50]}...")

                # 将总结添加到对话历史
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': summary
                })

                # 显示在主窗口中
                self.add_caption_line(f"[AI] {summary}")

        except Exception as e:
            print(f"[错误] 获取工具结果总结失败: {e}")
            # 如果总结失败，直接显示结果
            self.add_caption_line(f"[工具] {tool_name}: {str(tool_result)[:100]}...")

    def auto_record_game_state(self):
        """自动记录游戏状态"""
        print("[自动记录] 开始记录游戏状态...")

        # 这里可以添加自动记录游戏状态的逻辑
        # 例如，使用VLM分析当前屏幕，记录游戏状态

    def _on_self_monitoring_analysis(self, analysis: str):
        """
        自我监控VLM分析结果回调 - 显示在主窗口字幕区

        Args:
            analysis: VLM分析结果
        """
        # 使用信号在主线程中添加到字幕区
        self.vlm_result_ready.emit(analysis)
        print(f"[VLM] 收到分析结果: {analysis[:30]}...")

    def _on_self_monitoring_commentary(self, commentary: str):
        """
        自我监控吐槽结果回调 - 显示在吐槽窗口

        Args:
            commentary: 吐槽文本
        """
        # 添加到吐槽窗口
        try:
            # 添加文本（add_text 方法会通过信号机制安全地显示窗口）
            self.commentary_window.add_text(f"{commentary}")
            print(f"[回调] 吐槽已添加到吐槽窗口: {commentary[:30]}...")
        except Exception as e:
            print(f"[错误] 添加吐槽到窗口失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_memory_retrieved(self, query_type: str, query_text: str, results: List):
        """
        系统监控回调 - 显示在监控窗口

        Args:
            query_type: 检索类型（用户输入/VLM分析/联合查询）
            query_text: 查询文本
            results: 检索结果
        """
        if hasattr(self, 'memory_window') and self.memory_window:
            # 显示检索到的记忆（log_retrieved_memory 方法会通过信号机制安全地显示窗口）
            self.memory_window.log_retrieved_memory(query_type, query_text, results)

    def _on_memory_saved(self, memory_id: str, vlm_analysis: str, llm_commentary: str):
        """
        系统监控回调 - 显示在监控窗口

        Args:
            memory_id: 记忆ID
            vlm_analysis: VLM分析结果
            llm_commentary: LLM吐槽
        """
        if hasattr(self, 'memory_window') and self.memory_window:
            # 由于我们已经禁用了记忆保存，这里不再记录
            pass

    def _hide_windows(self):
        """
        截图前隐藏窗口
        """
        # 使用 QTimer.singleShot 在主线程中执行UI操作
        def hide_windows_safe():
            # 设置主窗口透明度为0
            self.setWindowOpacity(0)
            # 清空主窗口内容
            self.caption_display.clear()
            # 隐藏输入窗口
            if hasattr(self, 'input_window') and self.input_window:
                self.input_window.hide()
            # 设置吐槽窗口透明度为0
            if hasattr(self, 'commentary_window') and self.commentary_window:
                self.commentary_window.setWindowOpacity(0)
                # 清空吐槽窗口内容
                self.commentary_window.clear_text()
            # 设置记忆窗口透明度为0
            if hasattr(self, 'memory_window') and self.memory_window:
                self.memory_window.setWindowOpacity(0)
                # 清空记忆窗口内容
                self.memory_window.clear_monitoring()

        # 在主线程中执行
        QTimer.singleShot(0, hide_windows_safe)

    def _show_windows(self):
        """
        截图后显示窗口
        """
        # 使用 QTimer.singleShot 在主线程中执行UI操作
        def show_windows_safe():
            # 设置主窗口透明度为1
            self.setWindowOpacity(1)
            # 显示输入窗口
            if hasattr(self, 'input_window') and self.input_window:
                self.input_window.show()
            # 设置吐槽窗口透明度为1
            if hasattr(self, 'commentary_window') and self.commentary_window:
                self.commentary_window.setWindowOpacity(1)
            # 设置记忆窗口透明度为1
            if hasattr(self, 'memory_window') and self.memory_window:
                self.memory_window.setWindowOpacity(1)

        # 在主线程中执行
        QTimer.singleShot(0, show_windows_safe)

    def closeEvent(self, event):
        """关闭事件处理"""
        print("[关闭] MCP AI Caller 正在关闭...")

        # 停止自我监控线程
        if self.self_monitoring_thread:
            self.self_monitoring_thread.stop()
            self.self_monitoring_thread.join(timeout=5)
            print("[关闭] 自我监控线程已停止")

        # 关闭输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.close()
            print("[关闭] 输入窗口已关闭")

        # 关闭记忆窗口
        if self.memory_window:
            self.memory_window.close()
            print("[关闭] 记忆窗口已关闭")

        # 关闭吐槽窗口
        if self.commentary_window:
            self.commentary_window.close()
            print("[关闭] 吐槽窗口已关闭")

        # 关闭工具加载器
        if self.tool_loader:
            self.tool_loader.stop()
            print("[关闭] 工具加载器已停止")

        # 关闭MCP客户端
        if hasattr(self, 'mcp_client') and self.mcp_client:
            self.mcp_client.close()
            print("[关闭] MCP客户端已关闭")

        print("[关闭] MCP AI Caller 已关闭")
        event.accept()


if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建主窗口实例
    window = MCPAICaller()

    # 显示主窗口
    window.show()

    # 进入应用程序主循环
    sys.exit(app.exec())
