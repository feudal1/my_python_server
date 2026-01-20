# mcp_ai_caller.py
import tkinter as tk
from tkinter import scrolledtext
import asyncio
import sys
import os
import json
import threading
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MCPAICaller:
    def __init__(self, root):
        self.root = root
        # 去除窗口标题栏和边框
        self.root.overrideredirect(True)
        
        # 设置窗口位置和大小
        self.root.geometry("800x600+100+100")
        
        # 动态导入LLM和VLM服务
        import importlib.util
        llm_spec = importlib.util.spec_from_file_location(
            "llm_class", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_server", "llm_class.py")
        )
        llm_module = importlib.util.module_from_spec(llm_spec)
        llm_spec.loader.exec_module(llm_module)
        
        self.LLMService = llm_module.LLMService
        self.VLMService = llm_module.VLMService
        
        # 实例化LLM和VLM服务
        self.llm_service = self.LLMService()
        self.vlm_service = self.VLMService()
        
        # 实例化MCP客户端
        from mcp_cline.mcp_client import MCPClient
        self.mcp_client = MCPClient()

        self.setup_ui()
        
    def setup_ui(self):
        # 主框架 - 半透明背景
        main_frame = tk.Frame(self.root, bg='#f0f0f0', bd=0, highlightthickness=0)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 对话框容器
        dialog_container = tk.Frame(main_frame, bg='#f0f0f0', bd=0, highlightthickness=0)
        dialog_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        dialog_container.columnconfigure(0, weight=1)
        dialog_container.rowconfigure(0, weight=1)  # 输出区域
        dialog_container.rowconfigure(1, weight=0)  # 输入区域
        dialog_container.rowconfigure(2, weight=0)  # 按钮区域
        
        # 输出区域 - 半透明背景，黑色文字
        self.output_text = scrolledtext.ScrolledText(
            dialog_container, 
            height=20,
            bg='#ffffff',         # 白色背景
            fg='#333333',         # 深灰色文字
            bd=0,                 # 无边框
            highlightthickness=0, # 无高亮边框
            wrap=tk.WORD,
            font=('Arial', 11),
            state=tk.NORMAL
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 输入区域 - 明确的输入框，白色背景，黑色文字
        self.input_text = scrolledtext.ScrolledText(
            dialog_container, 
            height=5,
            bg='#ffffff',         # 白色背景，清晰可见
            fg='#000000',         # 黑色文字
            bd=2,                 # 明确边框
            relief=tk.SUNKEN,     # 凹陷效果，明确输入区域
            highlightthickness=1, # 轻微高亮
            highlightbackground='#cccccc',
            wrap=tk.WORD,
            font=('Arial', 12)
        )
        self.input_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 发送按钮
        send_button = tk.Button(
            dialog_container, 
            text="发送", 
            command=self.send_message,
            bg='#4a86e8',
            fg='white',
            relief=tk.FLAT,
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=5
        )
        send_button.grid(row=2, column=0, sticky=tk.E)
        
        # 绑定回车键发送消息
        self.input_text.bind('<Return>', self.on_enter_pressed)
        
        # 添加拖拽窗口的功能
        self.root.bind("<Button-1>", self.start_move)
        self.root.bind("<B1-Motion>", self.do_move)
        
    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def on_enter_pressed(self, event):
        # 如果按下Shift+Enter，则换行；否则发送消息
        if event.state & 0x1:  # Shift键被按下
            return  # 让文本框正常换行
        else:
            self.send_message()
            return "break"  # 阻止文本框换行

    def get_mcp_tool_list(self):
        """
        获取MCP服务器的可用工具列表
        """
        try:
            # 获取MCPClient中定义的所有服务器名称
            all_tools = {}
            
            # 遍历MCPClient中所有的服务器
            for server_name in self.mcp_client.servers.keys():
                try:
                    # 为每个服务器异步获取工具列表
                    tools = asyncio.run(self.mcp_client.list_tools(server_name))
                    # 将Tool对象转换为字典格式以便于使用
                    if tools:
                        tool_dict = {}
                        for tool in tools:
                            tool_dict[tool.name] = {
                                'name': tool.name,
                                'description': tool.description
                            }
                        all_tools[server_name] = tool_dict
                    else:
                        all_tools[server_name] = {}
                except Exception as e:
                    print(f"获取服务器 {server_name} 的工具列表失败: {str(e)}")
                    continue
            
            return all_tools
        except Exception as e:
            print(f"获取MCP工具列表失败: {str(e)}")
            return {}

    def enhance_prompt_with_tools(self, user_input):
        """
        将可用的MCP工具列表添加到用户输入中，增强提示词
        """
        # 获取可用的工具列表
        available_tools = self.get_mcp_tool_list()
        
        if available_tools:
            tool_description = "可用的MCP工具列表:\n"
            for server_name, tools in available_tools.items():
                if tools:
                    tool_names = [f"{name}({details['description']})" for name, details in tools.items()]
                    tool_description += f"- {server_name}: {tool_names}\n"
                else:
                    tool_description += f"- {server_name}: 无可用工具\n"
            
            enhanced_prompt = f"""
{tool_description}

如果您需要调用MCP工具，请严格按照以下格式响应：
[MCPCALL]server_name|tool_name|{{"param1": "value1", "param2": "value2"}}[/MCPCALL]

其中：
- server_name: 服务器名称
- tool_name: 工具名称  
- 参数部分必须是有效的JSON格式

如果不需要调用MCP工具，请正常回答。

用户请求: {user_input}
            """
            return enhanced_prompt, True
        else:
            return user_input, False

    def parse_ai_response_for_mcp(self, ai_response):
        """
        解析AI响应，查找MCP工具调用指令
        """
        import re
        
        # 匹配 [MCPCALL]server_name|tool_name|{"params": "values"}[/MCPCALL] 格式
        pattern = r'\[MCPCALL\](.*?)\|(.*?)\|({.*?})\[/MCPCALL\]'
        matches = re.findall(pattern, ai_response, re.DOTALL)
        
        if matches:
            server_name, tool_name, params_str = matches[0]
            try:
                arguments = json.loads(params_str)
                return True, server_name.strip(), tool_name.strip(), arguments
            except json.JSONDecodeError:
                print(f"参数解析失败: {params_str}")
                return False, None, None, None
        
        return False, None, None, None

    def send_message(self):
        # 获取输入内容
        user_input = self.input_text.get(1.0, tk.END).strip()
        if not user_input:
            return
            
        # 显示用户输入
        self.output_text.insert(tk.END, f">> {user_input}\n")
        self.input_text.delete(1.0, tk.END)
        
        # 在新线程中处理消息
        thread = threading.Thread(target=self.process_message, args=(user_input,))
        thread.start()
        
    def process_message(self, user_input):
        try:
            # 使用增强的提示词获取AI响应
            enhanced_prompt, has_tools = self.enhance_prompt_with_tools(user_input)
            
            if has_tools:
                # 将工具列表信息提供给AI
                messages = [{"role": "user", "content": enhanced_prompt}]
            else:
                messages = [{"role": "user", "content": user_input}]
            
            result = self.llm_service.create(messages)
            
            # 获取AI响应文本
            response_text = json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else str(result)
            
            # 尝试解析AI响应中的MCP调用指令
            is_mcp_call, server_name, tool_name, arguments = self.parse_ai_response_for_mcp(response_text)
            
            if is_mcp_call:
                # 执行MCP调用
                self.root.after(0, lambda: self.output_text.insert(tk.END, f"正在调用 {server_name} 的 {tool_name} 工具...\n"))
                
                try:
                    result = asyncio.run(self.mcp_client.call_tool(server_name, tool_name, arguments))
                    
                    if result:
                        # 处理MCP结果
                        result_content = []
                        for content in result.content:
                            if hasattr(content, 'text'):
                                result_content.append(content.text)
                            elif hasattr(content, 'type'):
                                result_content.append(f"[{content.type}]: {content}")
                        
                        result_text = "\n".join(result_content) if result_content else "无返回内容"
                        
                        # 将MCP结果展示在输出区域
                        self.root.after(0, lambda: self.output_text.insert(tk.END, f"MCP结果: {result_text}\n"))
                        
                        # 将MCP结果反馈给LLM进行进一步处理
                        feedback_prompt = f"用户请求: {user_input}\nMCP调用结果: {result_text}\n请根据这个结果向用户提供适当的回应。"
                        llm_messages = [{"role": "user", "content": feedback_prompt}]
                        
                        llm_result = self.llm_service.create(llm_messages)
                        llm_response = json.dumps(llm_result, indent=2, ensure_ascii=False) if isinstance(llm_result, dict) else str(llm_result)
                        
                        self.root.after(0, lambda: self.output_text.insert(tk.END, f"<< {llm_response}\n\n"))
                    else:
                        self.root.after(0, lambda: self.output_text.insert(tk.END, "MCP调用未返回结果\n\n"))
                        
                except Exception as e:
                    error_msg = f"MCP调用失败: {str(e)}"
                    self.root.after(0, lambda: self.output_text.insert(tk.END, f"{error_msg}\n\n"))
            else:
                # 如果AI响应不包含MCP调用，直接显示结果
                # 检查原响应中是否包含用户输入的原始内容，避免重复显示
                if "[MCPCALL]" not in response_text:
                    self.root.after(0, lambda r=result: self.display_result(r))
                else:
                    # 如果AI响应包含MCP调用格式但解析失败，也显示结果
                    self.root.after(0, lambda r=result: self.display_result(r))
                
        except Exception as e:
            self.root.after(0, lambda: self.output_text.insert(tk.END, f"错误: {str(e)}\n"))

    def display_result(self, result):
        try:
            response_text = json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else str(result)
            self.output_text.insert(tk.END, f"<< {response_text}\n\n")
            self.output_text.see(tk.END)  # 滚动到底部
        except Exception as e:
            self.output_text.insert(tk.END, f"<< 处理结果时出错: {str(e)}\n\n")


def main():
    root = tk.Tk()
    # 设置窗口属性
    root.wm_attributes("-topmost", 1)  # 置顶
    root.attributes('-alpha', 0.95)    # 设置透明度
    app = MCPAICaller(root)
    root.mainloop()


if __name__ == "__main__":
    main()