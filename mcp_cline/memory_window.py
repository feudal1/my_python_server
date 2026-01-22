"""
记忆系统可视化窗口 - 只显示检索记录
"""
from typing import List, Dict
from PyQt6.QtWidgets import (
    QMainWindow, QTextEdit, QVBoxLayout,
    QWidget, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject


class MemorySignals(QObject):
    """信号类,用于线程间通信"""
    memory_saved = pyqtSignal(str, str, str)  # id, vlm_analysis, llm_commentary
    memory_retrieved = pyqtSignal(str, list)  # query_text, results


class MemoryWindow(QMainWindow):
    """记忆系统显示窗口 - 只显示检索记录"""

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_ui()

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("🧠 系统监控")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        # 不设置完全透明背景，使用半透明背景
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

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
                color: #00aaff;
                border: none;
                font-family: Consolas, monospace;
                font-size: 18px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.retrieve_display)

    def log_monitoring(self, message: str):
        """
        记录监控信息

        Args:
            message: 监控消息文本
        """
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")

        log_text = f"[{timestamp}] {message}\n"
        log_text += "-" * 50 + "\n"

        # 滚动到顶部并插入
        cursor = self.retrieve_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        cursor.insertText(log_text)

        # 自动滚动到顶部
        self.retrieve_display.verticalScrollBar().setValue(0)

        # 保持最多50条记录
        text = self.retrieve_display.toPlainText()
        lines = text.split('\n')
        if len(lines) > 200:
            self.retrieve_display.setPlainText('\n'.join(lines[-200:]))
            # 重新滚动到顶部
            self.retrieve_display.verticalScrollBar().setValue(0)

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

    def log_retrieved_memory(self, query_text: str, memories: List[Dict]):
        """
        记录检索到的记忆

        Args:
            query_text: 查询文本
            memories: 检索到的记忆列表
        """
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")

        log_text = f"[{timestamp}] 检索记忆: {query_text}\n"
        if memories:
            for i, memory in enumerate(memories):
                log_text += f"  记忆 {i+1}: {memory.get('vlm_analysis', '无分析')}\n"
        else:
            log_text += "  无相关记忆\n"
        log_text += "-" * 50 + "\n"

        # 滚动到顶部并插入
        cursor = self.retrieve_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        cursor.insertText(log_text)

        # 自动滚动到顶部
        self.retrieve_display.verticalScrollBar().setValue(0)

        # 保持最多50条记录
        text = self.retrieve_display.toPlainText()
        lines = text.split('\n')
        if len(lines) > 200:
            self.retrieve_display.setPlainText('\n'.join(lines[-200:]))
            # 重新滚动到顶部
            self.retrieve_display.verticalScrollBar().setValue(0)


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
