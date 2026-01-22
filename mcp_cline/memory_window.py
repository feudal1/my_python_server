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
        self.setWindowTitle("🧠 记忆检索 Memory Retrieval")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置窗口位置和大小 (在吐槽窗口下方)
        self.setGeometry(200, 300, 500, 280)

    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 标题栏
        title_label = QLabel("🧠 记忆检索 Memory Retrieval")
        title_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                border: 2px solid #00ff00;
                border-radius: 5px;
                background-color: rgba(0, 0, 0, 180);
            }
        """)
        layout.addWidget(title_label)

        # 检索记录区域
        self.retrieve_display = QTextEdit()
        self.retrieve_display.setReadOnly(True)
        self.retrieve_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 200);
                color: #00aaff;
                border: 2px solid #00aaff;
                border-radius: 5px;
                font-family: Consolas, monospace;
                font-size: 11px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.retrieve_display)

        # 清空按钮
        clear_btn = QPushButton("清空检索记录 Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 170, 255, 150);
                color: white;
                border: 2px solid #00aaff;
                border-radius: 5px;
                padding: 5px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(0, 170, 255, 200);
            }
        """)
        clear_btn.clicked.connect(self.clear_retrieve)
        layout.addWidget(clear_btn)

    def log_retrieval(self, query_text: str, results: List[Dict]):
        """
        记录检索结果

        Args:
            query_text: 查询文本
            results: 检索结果列表
        """
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")

        log_text = f"[{timestamp}] 检索: {query_text[:30]}...\n"

        if results:
            for i, result in enumerate(results[:3], 1):
                similarity = 1 - result['distance']
                memory_text = result['document'][:40]
                memory_type = result['metadata'].get('type', 'unknown')
                log_text += f"  {i}. [{memory_type}] {memory_text}... (相似度: {similarity:.2f})\n"
        else:
            log_text += "  未找到相关记忆\n"

        log_text += "-" * 50 + "\n"

        # 滚动到顶部并插入
        cursor = self.retrieve_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        cursor.insertText(log_text)

        # 保持最多50条记录
        text = self.retrieve_display.toPlainText()
        lines = text.split('\n')
        if len(lines) > 200:
            self.retrieve_display.setPlainText('\n'.join(lines[-200:]))

    def clear_retrieve(self):
        """清空检索记录"""
        self.retrieve_display.clear()

    def update_stats(self, total_memories: int):
        """
        更新统计信息

        Args:
            total_memories: 总记忆数
        """
        self.setWindowTitle(f"🧠 记忆检索 - {total_memories} 条记忆")


# 测试代码
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    window = MemoryWindow()
    window.show()

    # 模拟一些记录
    window.log_retrieval("猫在地上", [
        {'distance': 0.1, 'document': '一只猫在沙发上睡觉', 'metadata': {'type': 'monitoring'}},
        {'distance': 0.2, 'document': '猫从沙发上跳到地板', 'metadata': {'type': 'commentary'}}
    ])

    sys.exit(app.exec())
