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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QMetaObject, pyqtSlot
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

        # 存储VLM历史，用于定期发送给LLM吐槽
        self.vlm_history = []

        # 最近一次记录的游戏状态
        self.last_game_state = ""

        # 存储分割后的内容
        self.split_contents = []

        # 存储工具列表
        self.tools_list = []

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
            from memory_system import VectorMemory
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

        # 设置定时器，每30秒将VLM历史发送给LLM吐槽（已改为基于VLM数量触发）
        # 保留定时器作为备选，但主要逻辑改为每3个VLM消息触发一次
        self.vlm_commentary_timer = QTimer(self)
        self.vlm_commentary_timer.timeout.connect(self._send_vlm_history_to_llm)
        self.vlm_commentary_timer.start(30000)  # 30秒（备用定时器）

    def _setup_window(self):
        """设置窗口属性"""
        print("[初始化] 设置窗口...")

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        # 窗口半透明，底色深色
        self.setWindowOpacity(0.5)  # 0.5是透明度值，范围0-1，更透明一点
        # 使用深色背景
        palette = self.palette()
        # 创建一个深色半透明背景
        dark_semi_transparent = QColor(30, 30, 30, 180)  # 深色半透明背景
        palette.setColor(self.backgroundRole(), dark_semi_transparent)
        self.setPalette(palette)
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
        # 窗口半透明，底色深色
        self.commentary_window.setWindowOpacity(0.5)  # 0.5是透明度值，范围0-1，更透明一点
        # 使用深色背景
        palette = self.commentary_window.palette()
        dark_semi_transparent = QColor(30, 30, 30, 180)  # 深色半透明背景
        palette.setColor(self.commentary_window.backgroundRole(), dark_semi_transparent)
        self.commentary_window.setPalette(palette)
        self.commentary_window.show()

        # 创建记忆窗口 (如果可用)
        if MemoryWindow is not None:
            self.memory_window = MemoryWindow()
            # 记忆窗口放在屏幕中间偏下的位置，变小一点，避免与游戏UI重叠
            self.memory_window.setGeometry(300, 350, 800, 180)
            # 窗口半透明，底色深色
            self.memory_window.setWindowOpacity(0.5)  # 0.5是透明度值，范围0-1，更透明一点
            # 使用深色背景
            palette = self.memory_window.palette()
            dark_semi_transparent = QColor(30, 30, 30, 180)  # 深色半透明背景
            palette.setColor(self.memory_window.backgroundRole(), dark_semi_transparent)
            self.memory_window.setPalette(palette)
            self.memory_window.show()
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

        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建字幕显示区域
        self.caption_display = QTextEdit(self)
        self.caption_display.setReadOnly(True)
        self.caption_display.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setStyleSheet(
            "QTextEdit { background-color: transparent; color: white; font-size: 14px; padding: 10px; }"
        )
        main_layout.addWidget(self.caption_display)

        # 创建输入栏
        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText("输入命令，例如：/h 打开工具窗口")
        self.input_line.setStyleSheet(
            "QLineEdit { background-color: rgba(50, 50, 50, 150); color: white; font-size: 14px; padding: 5px; border: 1px solid rgba(100, 100, 100, 100); }"
        )
        self.input_line.returnPressed.connect(self.on_input_submitted)
        main_layout.addWidget(self.input_line)

        # 创建主窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 初始显示欢迎信息
        self.caption_display.setHtml(
            "<div style='color: white; font-size: 14px;'>"\
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

            # 创建自我监控线程（verbose=False，减少控制台输出）
            self.self_monitoring_thread = SelfMonitoringThread(
                vlm_service=self.vlm_service,
                llm_service=self.llm_service,
                callback_analysis=self._on_self_monitoring_analysis,
                callback_commentary=self._on_self_monitoring_commentary,
                verbose=False,  # 不输出详细日志到控制台
                enable_memory=True,  # 启用向量记忆系统
                enable_memory_retrieval=True,  # 启用记忆检索功能
                callback_memory_retrieved=self._on_memory_retrieved,  # 记忆检索回调
                callback_memory_saved=self._on_memory_saved,  # 记忆保存回调
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
        self.tools_list = tools
        self.tools_loading = False
        self.tools_loaded_once = True
        self.last_tools_load_time = datetime.now().timestamp()

        # 显示工具加载完成信息
        tool_count = len(tools)
        print(f"[工具加载] MCP工具加载完成，共 {tool_count} 个工具")

        # 添加到字幕显示
        self.add_caption_line(f"[系统] MCP工具加载完成，共 {tool_count} 个工具")

        # 打印工具列表
        if isinstance(tools, dict):
            # 如果tools是字典
            for tool_name, tool_info in tools.items():
                print(f"- {tool_name}: {tool_info.get('description', '无描述')}")
        elif isinstance(tools, list) or isinstance(tools, tuple):
            # 如果tools是列表或元组
            for tool in tools:
                if isinstance(tool, dict):
                    # 如果列表中的元素是字典
                    tool_name = tool.get('name', '无名称')
                    tool_description = tool.get('description', '无描述')
                    print(f"- {tool_name}: {tool_description}")
                else:
                    # 如果列表中的元素是其他类型
                    print(f"- {tool}")
        else:
            # 如果tools是其他类型
            print(f"[工具加载] 工具列表格式未知: {type(tools)}")

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
            dialog = ToolsDialog(self.tools_list, self)
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
            current_text = self.caption_display.toPlainText()
            new_text = current_text + "\n" + text if current_text else text
            # 限制显示行数
            lines = new_text.split("\n")
            if len(lines) > 10:
                new_text = "\n".join(lines[-10:])
            self.caption_display.setPlainText(new_text)
            # 滚动到底部
            self.caption_display.verticalScrollBar().setValue(self.caption_display.verticalScrollBar().maximum())
            print(f"[UI DEBUG] 文本已添加，当前行数: {len(lines)}")
        else:
            print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
            print("[UI DEBUG] 不在主线程，使用信号转发")
            # 不在主线程，使用信号转发
            self.add_caption_signal.emit(text)

    def on_vlm_result_ready(self, result):
        """VLM结果回调"""
        print(f"[VLM] 分析结果: {result}")
        self.add_caption_line(f"[分析] {result}")

    def on_vlm_error_ready(self, error):
        """VLM错误回调"""
        print(f"[VLM] 分析错误: {error}")
        self.add_caption_line(f"[错误] VLM分析失败: {error}")

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
                # 打开工具窗口
                self.open_tools_window()
                self.add_caption_line(f"[命令] 已打开工具窗口")
            elif command == 'm':
                # 显示记忆窗口
                self.open_memory_window()
                self.add_caption_line(f"[命令] 已显示记忆窗口")
            elif command == 't':
                # 显示吐槽窗口
                self.open_commentary_window()
                self.add_caption_line(f"[命令] 已显示吐槽窗口")
            else:
                self.add_caption_line(f"[命令] 未知命令: {input_text}")
        else:
            # 处理普通文本输入
            self.add_caption_line(f"[你] {input_text}")
            # 这里可以添加处理普通文本输入的逻辑
            # 例如，将输入发送给LLM，获取回复等

    def auto_record_game_state(self):
        """自动记录游戏状态"""
        print("[自动记录] 开始记录游戏状态...")

        # 这里可以添加自动记录游戏状态的逻辑
        # 例如，使用VLM分析当前屏幕，记录游戏状态

    def _send_vlm_history_to_llm(self):
        """将VLM历史发送给LLM吐槽"""
        if not self.vlm_history:
            return

        print("[LLM] 开始生成吐槽...")

        # 构建提示词
        prompt = "基于以下游戏状态分析，生成一个幽默的吐槽：\n"
        for i, item in enumerate(self.vlm_history):
            prompt += f"{i+1}. {item}\n"

        prompt += "\n要求："
        prompt += "1. 保持幽默风趣"
        prompt += "2. 语言口语化，符合网络用语"
        prompt += "3. 不要太长，控制在50字以内"

        # 调用LLM服务
        try:
            response = self.llm_service.generate_response(prompt)
            if response:
                print(f"[LLM] 吐槽生成完成: {response}")
                self.add_caption_line(f"[吐槽] {response}")

                # 清空VLM历史
                self.vlm_history = []
        except Exception as e:
            print(f"[错误] 生成吐槽失败: {e}")

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
        系统监控回调 - 显示在监控窗口

        Args:
            query_text: 查询文本
            results: 检索结果
        """
        if hasattr(self, 'memory_window') and self.memory_window:
            # 显示窗口（如果还没显示）
            self.memory_window.show()
            # 显示检索到的记忆
            self.memory_window.log_retrieved_memory(query_text, results)

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
        # 在主线程中执行UI操作
        if QApplication.instance() and QApplication.instance().thread() == QThread.currentThread():
            # 设置主窗口透明度为0
            self.setWindowOpacity(0)
            # 设置吐槽窗口透明度为0
            if hasattr(self, 'commentary_window') and self.commentary_window:
                self.commentary_window.setWindowOpacity(0)
            # 设置记忆窗口透明度为0
            if hasattr(self, 'memory_window') and self.memory_window:
                self.memory_window.setWindowOpacity(0)

    def _show_windows(self):
        """
        截图后显示窗口
        """
        # 在主线程中执行UI操作
        if QApplication.instance() and QApplication.instance().thread() == QThread.currentThread():
            # 设置主窗口透明度为1
            self.setWindowOpacity(1)
            # 设置吐槽窗口透明度为1
            if hasattr(self, 'commentary_window') and self.commentary_window:
                self.commentary_window.setWindowOpacity(1)
            # 设置记忆窗口透明度为1
            if hasattr(self, 'memory_window') and self.memory_window:
                self.memory_window.setWindowOpacity(1)

    def closeEvent(self, event):
        """关闭事件处理"""
        print("[关闭] MCP AI Caller 正在关闭...")

        # 停止自我监控线程
        if self.self_monitoring_thread:
            self.self_monitoring_thread.stop()
            self.self_monitoring_thread.join(timeout=5)
            print("[关闭] 自我监控线程已停止")

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
