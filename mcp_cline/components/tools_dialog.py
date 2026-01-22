from typing import Dict, Optional, Any
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QScrollArea, QWidget, QLabel
from PyQt6.QtCore import Qt


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
