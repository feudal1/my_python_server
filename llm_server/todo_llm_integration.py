# todo_llm_integration.py
import os
import json
import sqlite3
import re
import http.client
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Tuple, Optional

class TodoItem:
    """
    待办事项实体类
    """
    def __init__(self, id: int = None, description: str = "", completed: bool = False, created_at: str = None):
        self.id = id
        self.description = description
        self.completed = completed
        self.created_at = created_at or datetime.now().isoformat()

class TodoManager:
    """
    待办事项管理器
    """
    
    def __init__(self, db_path: str = "todos.db"):
        """
        初始化待办事项管理器
        
        Args:
            db_path (str): SQLite数据库文件路径
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        初始化数据库表结构
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_todo(self, description: str) -> TodoItem:
        """
        添加新的待办事项
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO todos (description, completed) VALUES (?, ?)",
            (description, False)
        )
        
        todo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return TodoItem(id=todo_id, description=description)
    
    def get_pending_todos(self) -> List[TodoItem]:
        """
        获取所有未完成的待办事项
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, description, completed, created_at FROM todos WHERE completed = ? ORDER BY created_at",
            (False,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [TodoItem(id=row[0], description=row[1], completed=bool(row[2]), created_at=row[3]) 
                for row in rows]
    
    def mark_as_completed(self, todo_id: int) -> bool:
        """
        标记待办事项为已完成
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE todos SET completed = ? WHERE id = ?",
            (True, todo_id)
        )
        
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected_rows > 0

class LLMService:
    """
    LLM服务类（简化版，基于您提供的代码）
    """
    
    def __init__(self):
        # 这里使用固定的配置，您可以根据需要修改
        self.api_url = os.getenv('LLM_OPENAI_API_URL', 'https://api.deepseek.com/v1')
        self.model_name = os.getenv('LLM_MODEL_NAME', 'deepseek-chat')
        self.api_key = os.getenv('LLM_OPENAI_API_KEY', '')
        
        if not self.api_key:
            raise ValueError("请设置环境变量 LLM_OPENAI_API_KEY")
    
    def create(self, messages, tools=None):
        """
        调用LLM服务
        """
        # 解析 URL
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path

        # 创建 HTTP 连接
        conn = http.client.HTTPSConnection(host)

        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }

        # 发送 POST 请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        conn.request(
            "POST",
            path,
            body=json.dumps(request_body),
            headers=headers
        )

        # 获取响应
        response = conn.getresponse()
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            conn.close()
            raise Exception(f"LLM服务器错误: {response.status} - {error_msg}")

        # 读取响应内容
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)
        conn.close()
        
        return data

class TodoLLMProcessor:
    """
    待办事项与LLM集成处理器
    """
    
    def __init__(self, todo_manager: TodoManager, llm_service: LLMService):
        self.todo_manager = todo_manager
        self.llm_service = llm_service
    
    def process_user_input(self, user_input: str, conversation_history: List[dict] = None) -> dict:
        """
        处理用户输入，检查待办事项并调用LLM
        
        Args:
            user_input (str): 用户输入
            conversation_history (List[dict]): 对话历史
            
        Returns:
            dict: LLM响应结果
        """
        # 检查用户是否完成了任务
        completion_msg = self._check_task_completion(user_input)
        
        # 准备消息列表
        messages = conversation_history or []
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 如果有任务完成确认，添加系统消息
        if completion_msg:
            messages.append({
                "role": "system",
                "content": completion_msg
            })
        
        # 检查是否有未完成的待办事项
        pending_todos = self.todo_manager.get_pending_todos()
        if pending_todos:
            todo_reminder = self._format_todo_reminder(pending_todos)
            # 将提醒添加到系统消息中
            messages.append({
                "role": "system",
                "content": todo_reminder
            })
        
        # 调用LLM服务
        try:
            result = self.llm_service.create(messages)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "message": "处理请求时发生错误"
            }
    
    def _check_task_completion(self, user_input: str) -> Optional[str]:
        """
        检查用户是否表示完成了任务
        
        Args:
            user_input (str): 用户输入
            
        Returns:
            Optional[str]: 完成确认消息
        """
        # 匹配完成任务的表达
        patterns = [
            r"完成(?:了)?(?:任务)?\s*(?:#?|No\.?)?\s*(\d+)",
            r"(?:任务)?\s*(\d+)\s*完成(?:了)?",
            r"搞定(?:了)?(?:任务)?\s*(?:#?|No\.?)?\s*(\d+)",
            r"(?:任务)?\s*(\d+)\s*搞定(?:了)?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                try:
                    todo_id = int(match.group(1))
                    if self.todo_manager.mark_as_completed(todo_id):
                        return f"好的，我已经将任务 #{todo_id} 标记为完成了！"
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _format_todo_reminder(self, pending_todos: List[TodoItem]) -> str:
        """
        格式化待办事项提醒
        
        Args:
            pending_todos (List[TodoItem]): 未完成的待办事项列表
            
        Returns:
            str: 格式化的提醒消息
        """
        if not pending_todos:
            return ""
        
        todo_lines = [f"{todo.id}. {todo.description}" for todo in pending_todos]
        todo_list = "\n".join(todo_lines)
        
        return (f"提醒：您当前有以下待完成的任务：\n"
                f"{todo_list}\n\n"
                f"如果您已完成其中任何一项，请告诉我类似\"完成任务1\"这样的指令。")

def main():
    """
    主函数 - 演示如何使用待办事项与LLM集成
    """
    print("=== 待办事项与LLM集成系统 ===")
    
    try:
        # 初始化组件
        todo_manager = TodoManager()
        llm_service = LLMService()
        processor = TodoLLMProcessor(todo_manager, llm_service)
        
        # 添加一些示例任务
        print("正在添加示例任务...")
        todo_manager.add_todo("完成项目报告")
        todo_manager.add_todo("购买办公用品")
        todo_manager.add_todo("预约团队会议")
        
        # 对话历史
        conversation_history = [
            {
                "role": "system",
                "content": "你是一个乐于助人的助手，可以帮助用户管理工作任务。"
            }
        ]
        
        print("\n系统已准备就绪！")
        print("您可以：")
        print("1. 与AI助手对话")
        print("2. 输入'完成任务1'来标记任务完成")
        print("3. 输入'quit'退出")
        print("4. 输入'help'查看帮助")
        
        while True:
            user_input = input("\n您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("助手: 再见！")
                break
            elif user_input.lower() == 'help':
                print("\n帮助信息：")
                print("- 直接与AI助手对话")
                print("- 输入'完成任务1'、'任务1完成'等来标记任务完成")
                print("- 输入'quit'退出程序")
                continue
            elif not user_input:
                continue
            
            # 处理用户输入
            print("助手: ", end="")
            result = processor.process_user_input(user_input, conversation_history)
            
            if "error" in result:
                print(f"错误: {result['message']}")
            else:
                # 提取并打印AI回复
                ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "无回复")
                print(ai_response)
                
                # 更新对话历史
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # 限制对话历史长度以避免过长
                if len(conversation_history) > 10:
                    # 保留系统消息和最近的几条对话
                    system_msg = conversation_history[0]
                    conversation_history = [system_msg] + conversation_history[-9:]
                    
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保设置了正确的环境变量：")
        print("  LLM_OPENAI_API_KEY=your_api_key_here")
        print("  LLM_OPENAI_API_URL=https://api.deepseek.com/v1  # 可选")
        print("  LLM_MODEL_NAME=deepseek-chat  # 可选")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
def example_usage():
    """
    使用示例
    """
    try:
        # 初始化组件
        todo_manager = TodoManager()
        llm_service = LLMService()
        processor = TodoLLMProcessor(todo_manager, llm_service)
        
        # 添加示例任务
        todo_manager.add_todo("完成季度报告")
        todo_manager.add_todo("回复客户邮件")
        
        # 模拟对话
        conversation_history = [
            {
                "role": "system",
                "content": "你是一个任务管理助手，帮助用户跟踪和管理他们的待办事项。"
            }
        ]
        
        print("=== 使用示例 ===")
        
        # 示例1: 正常对话
        print("\n1. 用户询问任务情况:")
        result = processor.process_user_input("我今天有什么任务需要完成？", conversation_history)
        if "error" not in result:
            response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"助手: {response}")
        
        # 示例2: 完成任务
        print("\n2. 用户完成任务:")
        result = processor.process_user_input("我已经完成了任务1", conversation_history)
        if "error" not in result:
            response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"助手: {response}")
        
        # 示例3: 再次询问
        print("\n3. 用户再次询问:")
        result = processor.process_user_input("我现在还有什么任务？", conversation_history)
        if "error" not in result:
            response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"助手: {response}")
            
    except Exception as e:
        print(f"示例运行出错: {e}")

if __name__ == "__main__":
    # 首先运行示例
    example_usage()
    
    print("\n" + "="*50)
    
    # 启动交互式程序
    main()