"""
记忆系统可视化窗口 - 只显示检索记录
"""
from typing import List, Dict
from PyQt6.QtWidgets import (
    QMainWindow, QTextEdit, QVBoxLayout,
    QWidget, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer


class MemorySignals(QObject):
    """信号类,用于线程间通信"""
    memory_saved = pyqtSignal(str, str, str)  # id, vlm_analysis, llm_commentary
    memory_retrieved = pyqtSignal(str, list)  # query_text, results


class MemoryWindow(QMainWindow):
    """记忆系统显示窗口 - 只显示检索记录"""

    # 定义信号用于线程间通信
    window_shown = pyqtSignal()
    monitoring_logged = pyqtSignal(str)
    memory_retrieved = pyqtSignal(str, str, list)  # query_type, query_text, memories

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide)
        # 连接信号到槽
        self.window_shown.connect(self._show_safe)
        self.monitoring_logged.connect(self._log_monitoring_safe)
        self.memory_retrieved.connect(self._log_retrieved_memory_safe)
        self.hide()

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("🧠 系统监控")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        # 设置完全透明背景
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 不设置窗口位置和大小，使用默认值，由外部调用者设置

    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 监控记录区域
        self.retrieve_display = QTextEdit()
        self.retrieve_display.setReadOnly(True)
        # 隐藏垂直滚动条
        self.retrieve_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 隐藏水平滚动条
        self.retrieve_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.retrieve_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0);
                color: #ff0000;
                border: none;
                font-family: Consolas, monospace;
                font-size: 18px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.retrieve_display)

    def clear_monitoring(self):
        """清空监控记录"""
        self.retrieve_display.clear()

    def update_stats(self, total_monitors: int):
        """
        更新统计信息

        Args:
            total_monitors: 总监控数
        """
        self.setWindowTitle(f"🧠 系统监控 - {total_monitors} 项")

    def log_retrieved_memory(self, query_type: str, query_text: str, memories: List[Dict]):
        """
        记录检索到的记忆（只显示检索结果，10秒后隐藏）

        Args:
            query_type: 检索类型（用户输入/VLM分析/联合查询）
            query_text: 查询文本
            memories: 检索到的记忆列表
        """
        self.memory_retrieved.emit(query_type, query_text, memories)

    def _log_retrieved_memory_safe(self, query_type: str, query_text: str, memories: List[Dict]):
        """
        在主线程中安全地记录检索到的记忆
        """
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")

        if memories:
            # 有检索到记忆，显示检索内容（每条记忆一行，前10个字符）
            log_lines = [f"[{timestamp}] {query_type}检索到 {len(memories)} 条记忆:"]

            for mem in memories:
                doc = mem.get('document', '')
                # 显示主记忆的前10个字符
                preview = doc[:10] if len(doc) > 10 else doc
                log_lines.append(f"  - {preview}")

            log_text = '\n'.join(log_lines)
        else:
            # 没有检索到记忆
            log_text = f"[{timestamp}] {query_type}未找到相关记忆"

        # 清空旧内容，只显示最新一条
        self.retrieve_display.clear()
        self.retrieve_display.setPlainText(log_text)

        # 显示窗口
        super().show()

        # 重置隐藏定时器（10秒后隐藏）
        self.hide_timer.stop()
        self.hide_timer.start(10000)

    def show(self):
        """
        安全地显示窗口（使用信号在主线程中调用）
        """
        self.window_shown.emit()

    def _show_safe(self):
        """
        在主线程中安全地显示窗口
        """
        super().show()

    def log_monitoring(self, message: str):
        """
        记录监控信息（只显示最新一条，10秒后隐藏）

        Args:
            message: 监控消息文本
        """
        self.monitoring_logged.emit(message)

    def _log_monitoring_safe(self, message: str):
        """
        在主线程中安全地记录监控信息
        """
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"

        # 清空旧内容，只显示最新一条
        self.retrieve_display.clear()
        self.retrieve_display.setPlainText(log_text)

        # 显示窗口
        super().show()

        # 重置隐藏定时器（10秒后隐藏）
        self.hide_timer.stop()
        self.hide_timer.start(10000)


# 测试代码
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    window = MemoryWindow()
    window.show()

    # 模拟一些记录
    window.log_monitoring("系统监控测试：猫在地上")
    window.log_monitoring("系统监控测试：狗在沙发上")

    sys.exit(app.exec())
