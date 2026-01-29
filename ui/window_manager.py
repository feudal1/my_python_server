import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout,
    QWidget, QLabel, QDialog, QScrollArea, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QMetaObject, pyqtSlot, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut, QColor


class WindowManager(QMainWindow):
    """窗口管理器类，负责处理所有UI相关的功能"""

    # 定义信号用于线程间通信
    add_caption_signal = pyqtSignal(str)  # 添加字幕信号
    input_submitted_signal = pyqtSignal(str)  # 输入提交信号

    # 工具调用事件流信号
    tool_call_start_signal = pyqtSignal(str, dict)  # 工具调用开始 (tool_name, arguments)
    tool_call_end_signal = pyqtSignal(str, dict, str)  # 工具调用结束 (tool_name, arguments, result)
    tool_result_signal = pyqtSignal(str, str)  # 工具结果 (tool_call_id, result)
    text_delta_signal = pyqtSignal(str)  # 文本增量 (text)
    text_end_signal = pyqtSignal(str)  # 文本结束 (full_text)
    message_end_signal = pyqtSignal(str)  # 消息结束 (role)
    lifecycle_signal = pyqtSignal(str, str)  # 生命周期事件 (event_type, data)

    def __init__(self):
        """初始化窗口管理器"""
        super().__init__()

        # 初始化变量
        self._initialize_variables()

        # 设置定时器
        self._setup_timers()

        # 设置窗口
        self._setup_window()

        # 设置UI
        self._setup_ui()

        # 设置快捷键
        self._setup_shortcuts()

        # 设置信号连接
        self._setup_connections()

        print("[窗口管理器] 初始化完成!")

    def _initialize_variables(self):
        """初始化变量"""
        print("[窗口管理器] 设置初始变量...")

        # 存储字幕显示的定时器
        self.caption_timer = None

    def _setup_timers(self):
        """设置定时器"""
        print("[窗口管理器] 设置定时器...")

        # 设置定时器，每30秒自动记录游戏状态
        self.auto_record_timer = QTimer(self)
        # 注意：这里不直接连接到方法，而是由外部连接

        # 设置定时器，每1秒自动截图并调用VLM
        self.auto_screenshot_timer = QTimer(self)
        # 注意：这里不直接连接到方法，而是由外部连接

    def _setup_window(self):
        """设置窗口属性"""
        print("[窗口管理器] 设置窗口...")

        # 设置无边框窗口和置顶
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        # 设置透明背景属性
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()

        # 计算窗口位置，放在屏幕下方
        window_width = 1200
        window_height = 100
        x = (screen_geometry.width() - window_width) // 2
        y = screen_geometry.height() - window_height - 50  # 距离底部50px

        # 设置窗口位置和大小
        self.setGeometry(x, y, window_width, window_height)
        self.min_height = 180
        self.max_height = 500

        # 设置窗口不透明度为1.0，使其完全可见
        self.setWindowOpacity(1.0)

        # 添加调试信息
        print(f"[窗口调试] 窗口位置: ({x}, {y}), 大小: {window_width}x{window_height}")
        print(f"[窗口调试] 窗口是否可见: {self.isVisible()}")

    def _setup_ui(self):
        """设置UI"""
        print("[窗口管理器] 设置UI...")

        # 创建主布局（仅包含字幕显示）
        main_layout = QVBoxLayout()

        # 创建字幕显示区域
        self.caption_display = QTextEdit(self)
        self.caption_display.setReadOnly(True)
        self.caption_display.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 设置高透明黑色背景的样式（不设置字体样式，在HTML中设置）
        self.caption_display.setStyleSheet(
            """
            QTextEdit {
                background-color: transparent;
                padding: 10px;
            }
            """
        )
        # 不要对caption_display设置WA_TranslucentBackground，否则文本也会透明
        main_layout.addWidget(self.caption_display)

        # 创建主窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        # 设置高透明的黑色背景（更透明）
        central_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.2);")
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
        self.input_line.setPlaceholderText("输入命令，例如：列出所有可用的技能")
        self.input_line.setFixedHeight(30)
        self.input_line.setStyleSheet(
            "QLineEdit { background-color: white; color: black; font-size: 14px; padding: 5px; border: 1px solid #ccc; border-radius: 4px; }"
        )
        self.input_line.returnPressed.connect(self.on_input_submitted)
        input_layout.addWidget(self.input_line)

        self.input_window.setLayout(input_layout)

        # 设置输入窗口在屏幕左下角（x偏移200px，y离底部10px）
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        input_window_width = 300
        input_window_height = 40
        self.input_window.setGeometry(
            screen_geometry.left() + 10 + 200,  # 左边距 10px，再向右偏移 200px
            screen_geometry.bottom() - input_window_height -2,  # 距离底部 10px
            input_window_width,
            input_window_height
        )
        self.input_window.show()

        # 不显示欢迎信息，保持窗口空白

    def _setup_shortcuts(self):
        """设置快捷键（已禁用）"""
        print("[窗口管理器] 设置快捷键...")
        # 禁用所有键盘快捷键
        pass

    def _setup_connections(self):
        """设置信号连接"""
        print("[窗口管理器] 设置信号连接...")

        # 连接添加字幕信号
        self.add_caption_signal.connect(self.add_caption_line)

    def on_input_submitted(self):
        """处理用户输入"""
        input_text = self.input_line.text().strip()
        if not input_text:
            return

        # 清空输入栏
        self.input_line.clear()

        # 发送输入提交信号
        self.input_submitted_signal.emit(input_text)

    def add_caption_line(self, text):
        """添加字幕行"""
        try:
            print(f"[UI DEBUG] 尝试添加文本: {text}")
        except UnicodeEncodeError:
            print(f"[UI DEBUG] 尝试添加文本: {repr(text)}")
        # 检查是否在主线程
        if QThread.currentThread() == self.thread():
            try:
                print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
                print("[UI DEBUG] 在主线程，直接添加")
            except UnicodeEncodeError:
                pass
            # 确保窗口可见
            self.setWindowOpacity(1.0)
            if not self.isVisible():
                self.show()
            # 在主线程，直接添加
            # 只显示当前回合的内容，不保留历史
            # 白色文字，字号更小
            escaped_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            html_content = f"""
            <div style="
                color: #ffffff;
                font-size: 22px;
                font-weight: 900;
                font-family: 'Arial Black', 'Impact', 'Microsoft YaHei', sans-serif;
                text-shadow: 0 0 5px rgba(0,0,0,0.8);
            ">{escaped_text}</div>
            """
            self.caption_display.setHtml(html_content)
            # 滚动到底部
            self.caption_display.verticalScrollBar().setValue(self.caption_display.verticalScrollBar().maximum())

            # 添加调试信息
            try:
                print("[UI DEBUG] 文本已添加，只显示当前回合内容")
                print(f"[UI DEBUG] caption_display文本: {self.caption_display.toPlainText()[:50]}...")
                print(f"[UI DEBUG] caption_display是否可见: {self.caption_display.isVisible()}")
                print(f"[UI DEBUG] 主窗口是否可见: {self.isVisible()}")
            except UnicodeEncodeError:
                pass

            # 取消之前的定时器（如果存在）
            if hasattr(self, 'caption_timer') and self.caption_timer:
                self.caption_timer.stop()

            # 设置定时器，30秒后清空字幕
            self.caption_timer = QTimer(self)
            self.caption_timer.setSingleShot(True)
            self.caption_timer.timeout.connect(self.clear_caption)
            self.caption_timer.start(30000)  # 30秒后清空
        else:
            try:
                print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
                print("[UI DEBUG] 不在主线程，使用信号转发")
            except UnicodeEncodeError:
                pass
            # 不在主线程，使用信号转发
            self.add_caption_signal.emit(text)

    def clear_caption(self):
        """清空字幕显示区域"""
        print("[UI DEBUG] 清空字幕显示")
        if hasattr(self, 'caption_display') and self.caption_display:
            self.caption_display.setPlainText("")
        else:
            print("[UI DEBUG] caption_display不存在，跳过清空")

    def _hide_windows(self):
        """截图前隐藏窗口"""
        # 设置主窗口完全透明（但不隐藏，保持窗口结构）
        self.setWindowOpacity(0.0)
        # 清空主窗口内容
        self.caption_display.clear()
        # 隐藏输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.hide()

    def _show_windows(self):
        """截图后显示窗口"""
        # 设置主窗口完全不透明
        self.setWindowOpacity(1.0)
        # 显示输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.show()

    def closeEvent(self, event):
        """关闭事件处理"""
        print("[窗口管理器] 正在关闭...")

        # 停止定时器
        if hasattr(self, 'auto_record_timer'):
            self.auto_record_timer.stop()
        if hasattr(self, 'auto_screenshot_timer'):
            self.auto_screenshot_timer.stop()

        # 关闭输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.close()
            print("[窗口管理器] 输入窗口已关闭")

        print("[窗口管理器] 已关闭")
        event.accept()


# 测试代码
if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建窗口管理器实例
    window = WindowManager()

    # 显示主窗口
    window.show()

    # 进入应用程序主循环
    sys.exit(app.exec())
