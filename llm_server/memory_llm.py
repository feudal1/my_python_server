import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
import subprocess
import re
import json
import sys
from pathlib import Path
from llm_class import VLMService  # 假设VLMService基于LLMService
import pyautogui
import base64

from dotenv import load_dotenv

# 定义全局变量
CURRENT_DIR = Path(__file__).parent
dotenv_path = r'my_python_server_private\.env'
load_dotenv(dotenv_path)
workenv=os.getenv('workenv')

CONFIG_PATH = CURRENT_DIR / "config" / (workenv + "_tools_config.json")
KNOWLEDGE_FILE_PATH = CURRENT_DIR / "config" / (workenv + "_knowledge.txt")

WORKFLOW_PATH = CURRENT_DIR / "config" / (workenv + "_workflow.txt")  # 新增：工作流程文件路径
def execute_python_script(script_path, *args):
    """
    执行指定路径的Python脚本
    """
    # 获取项目根目录（从当前脚本位置向上一级）
    current_dir = Path(__file__).parent.parent  # 回到项目根目录
    script_full_path = current_dir / script_path
    
    if not script_full_path.exists():
        return f"错误: 脚本 '{script_path}' 不存在"
    
    if script_full_path.suffix != '.py':
        return f"错误: 文件必须是Python脚本 (.py文件)"
    
    try:
        # 构建命令：工具名称作为脚本的第一个参数
        cmd = [sys.executable, str(script_full_path)] + list(args)
       
        # 执行Python脚本，指定编码为UTF-8
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(current_dir),
            encoding='utf-8',      # 明确指定UTF-8编码
            errors='replace'       # 遇到编码错误时替换字符
        )
     
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"脚本执行失败: {result.stderr.strip()}"
     
    except subprocess.TimeoutExpired:
        return f"脚本执行超时: {script_path}"
    except Exception as e:
        return f"执行脚本时出错: {str(e)}"


def list_available_tools():
    """
    从配置文件中列出所有可用工具
    """
    if not CONFIG_PATH.exists():
        return []
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('tools', [])
    except Exception as e:
        return []


def get_tool_by_name(tool_name):
    """
    根据工具名称获取工具信息
    """
    tools = list_available_tools()
    if not isinstance(tools, list):  # 确保是列表
        return None
    for tool in tools:
        if isinstance(tool, dict) and tool.get('name') == tool_name:  # 确保 tool 是字典
            return tool
    return None


def execute_tool(tool_name, *args):
    """
    执行指定名称的工具
    """
    tool_info = get_tool_by_name(tool_name)
    if not tool_info:
        return f"错误: 未找到工具 '{tool_name}'"
    
    script_path = tool_info['path']
    result = execute_python_script(script_path, tool_name, *args)
   
    return result


def get_available_tools_info():
    """
    获取所有可用工具的信息
    """
    if not CONFIG_PATH.exists():
        return []  # 确保返回列表而不是字符串
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # 确保返回的是工具列表
            tools = config.get('tools', [])
            return tools if isinstance(tools, list) else []
    except Exception as e:
        print(f"加载工具配置失败: {e}")
        return []


def get_tools_description():
    """获取工具描述，用于提供给VLM"""
    tools = get_available_tools_info()
    if not tools or not isinstance(tools, list):  # 检查是否返回错误或不是列表
        return "当前没有可用工具"
    
    tools_desc = "可用工具列表:\n"
    for tool in tools:
        if not isinstance(tool, dict):  # 确保工具是字典格式
            continue
        name = tool.get('name', '未知工具')
        desc = tool.get('description', '无描述')
    
        # 检查parameters是否是OpenAI格式的字典
        params = tool.get('parameters', {})
        if isinstance(params, dict) and 'properties' in params:
            # OpenAI工具格式
            properties = params.get('properties', {})
            required = params.get('required', [])
            
            if properties:
                param_list = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', '')
                    is_required = " (必填)" if param_name in required else ""
                    param_list.append(f"{param_name}({param_type}){is_required}")
                
                if param_list:
                    param_desc = ", ".join(param_list)
                    tools_desc += f"- {name}: {desc} (参数: {param_desc})\n"
                else:
                    tools_desc += f"- {name}: {desc} (无参数)\n"
            else:
                tools_desc += f"- {name}: {desc} (无参数)\n"
        else:
            # 传统格式
            if isinstance(params, list):  # 检查params是否为列表
                param_desc = ", ".join([f"{p.get('name', 'unknown')}({p.get('type', 'unknown')})" for p in params if isinstance(p, dict)])
                tools_desc += f"- {name}: {desc} (参数: {param_desc})\n"
            else:
                tools_desc += f"- {name}: {desc} (无参数)\n"
    
    tools_desc += "\n使用格式: [TOOL:工具名称,arg1,arg2,...] 或 [TOOL:工具名称(param_name=value)]\n"
    return tools_desc

def image_to_base64(image_path):
    """将图像文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def is_task_completed(ai_response, should_check=True):
    """判断任务是否完成 - 使用特定结束标记，但可以控制是否检查"""
    if not should_check or not ai_response:
        return False
    
    # 使用特定的结束标记，而不是通用关键词
    completion_marker = "[TASK_COMPLETED]"
    
    if completion_marker in ai_response:
        return True
    
    return False

def is_workflow_completed(ai_response):
    """判断工作流程是否完成 - 现在总是返回False，因为我们不再使用此标记"""
    # 移除 TOTAL_TASK_COMPLETED 检测
    return False

def send_task_confirmation_to_ai(vlm_service, task_description, system_prompt, memory_file):
    """
    向AI发送任务完成确认消息
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 截取当前屏幕
    screenshot = pyautogui.screenshot()
    screenshot_path = os.path.join(current_dir, "current_screen.png")
    screenshot.save(screenshot_path)
    
    # 准备消息列表
    messages = []
    
    # 添加系统提示（只包含工具信息）
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # 读取 memory 内容并加入到用户消息中 - 只包含工具执行结果和AI响应
    memory_content = ""
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 只保留工具执行结果和AI响应，过滤掉工作流程定义
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            # 保留工具执行结果和AI响应，过滤掉工作流程定义和任务状态
            if not (line.startswith("工作流程定义:") or 
                    line.startswith("执行历史:") or
                    (line.startswith("任务") and " - " in line and ("待完成" in line or "已完成" in line or "待确定" in line))):
                filtered_lines.append(line)
        
        if filtered_lines:
            memory_content = "历史执行记录:\n" + "\n".join(filtered_lines) + "\n"
    
    # 构建确认消息 - 移除 TOTAL_TASK_COMPLETED 提示
    user_message = f"{memory_content}任务 '{task_description}' 已确认完成，请继续下一个任务\n"
    
    # 构建包含图像的消息内容
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_to_base64(screenshot_path)}"
        }
    }
    
    text_content = {
        "type": "text",
        "text": user_message
    }
    
    # 构建用户消息，包含文本和图像
    messages.append({
        "role": "user",
        "content": [text_content, image_content]
    })
    
    try:
        # 调用LLM服务（模拟VLM功能）
        result = vlm_service.create_with_image(messages)
        ai_response = result['choices'][0]['message']['content']
       
        print(f"AI确认响应: {ai_response}")
        
        # 将确认响应写入记忆文件
        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(f"AI确认响应: {ai_response}\n")
        
        return ai_response
    except Exception as e:
        error_msg = f"发送确认消息时出错: {str(e)}"
        print(error_msg)
        return None

def vision_task_loop(task_description, knowledge_file=None, memory_file=None, workflow_state=None, reset_first_iteration=True):
    """
    基于视觉的循环任务执行器 - 支持多截图上下文
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if knowledge_file is None:
        knowledge_file = KNOWLEDGE_FILE_PATH
    if memory_file is None:
        memory_file = os.path.join(current_dir, "memory.txt")

    # 创建LLM服务实例（模拟VLM）
    vlm_service = VLMService()

    # 读取固定知识
    system_prompt_parts = []
    tools_description = get_tools_description()
    system_prompt_parts.append(f"可用工具信息:\n{tools_description}")

    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_content = f.read()
            if knowledge_content.strip():
                system_prompt_parts.append(f"重要知识:\n{knowledge_content}")

    iteration_count = 0
    max_iterations = 5
    screenshot_history = []  # 新增：保存截图历史
    
    while iteration_count < max_iterations:
        iteration_count += 1
        
        # 截取当前屏幕
        screenshot = pyautogui.screenshot()
        screenshot_path = os.path.join(current_dir, f"screenshot_{iteration_count}.png")
        screenshot.save(screenshot_path)
        
        # 添加到截图历史
        screenshot_history.append(screenshot_path)
        
        system_prompt = "\n".join(system_prompt_parts)
        
        # 准备消息列表
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 读取 memory 内容
        memory_content = ""
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 过滤内容
            lines = content.split('\n')
            filtered_lines = []
            for line in lines:
                if not (line.startswith("工作流程定义:") or 
                        line.startswith("执行历史:") or
                        (line.startswith("任务") and " - " in line and 
                         ("待完成" in line or "已完成" in line or "待确定" in line))):
                    filtered_lines.append(line)
            
            if filtered_lines:
                memory_content = "历史执行记录:\n" + "\n".join(filtered_lines) + "\n"
        
        # 添加任务描述
        user_message = f"当前任务: {task_description}\n"
        
        if memory_content:
            user_message += f"{memory_content}\n"
        
        # 添加迭代信息
        if iteration_count > 1:
            user_message += "请确认任务是否已完成，若完成请输出 [TASK_COMPLETED] 标记当前步骤完成\n"
            is_inquiry_phase = True
        else:
            user_message += "请分析当前屏幕截图，并开始执行任务。"
            is_inquiry_phase = False
        
        print(f"user_message: 【{user_message}】")
        
        # 构建消息内容 - 包含所有截图
        content_list = []
        
        # 添加文本内容
        content_list.append({
            "type": "text",
            "text": user_message
        })
        
        # 添加所有历史截图
        for idx, screenshot_path in enumerate(screenshot_history):
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(screenshot_path)}"
                }
            })
        
        # 构建用户消息
        messages.append({
            "role": "user",
            "content": content_list
        })
        
        try:
            # 调用LLM服务
            result = vlm_service.create_with_image(messages)
            ai_response = result['choices'][0]['message']['content']
            
            print(f"LLM服务返回: 【{ai_response}】")
            
            yield f" {ai_response}"
            
            # 处理工具调用
            tool_execution_result = process_tool_calls(ai_response, memory_file, workflow_state, is_inquiry_phase=False)
            
            # 检查任务完成状态
            if is_inquiry_phase and is_task_completed(ai_response) and not has_tool_calls(ai_response):
                task_completed_in_state = False
                for step, completed in workflow_state:
                    if task_description in step and completed == True:
                        task_completed_in_state = True
                        break
                
                if task_completed_in_state:
                    confirmation_response = send_task_confirmation_to_ai(
                        vlm_service, 
                        task_description, 
                        system_prompt, 
                        memory_file
                    )
                    if confirmation_response:
                        yield f"向AI发送任务完成确认: {confirmation_response}"
                    
                    yield "当前任务已完成，退出循环"
                    break

        except Exception as e:
            error_msg = f"执行任务时出错: {str(e)}"
            yield error_msg
            break
    
    # 清理截图文件
    for screenshot_path in screenshot_history:
        try:
            os.remove(screenshot_path)
        except:
            pass
    
    if iteration_count >= max_iterations:
        yield "达到最大迭代次数，停止任务执行"
def has_tool_calls(response_text):
    """检查响应中是否包含工具调用"""
    tool_pattern = r'\[TOOL:([^\],\]]+)(?:,([^\]]*))?\]'
    matches = re.findall(tool_pattern, response_text)
    return len(matches) > 0

def get_workflow_state_from_memory_in_app(memory_file_path):
    """从记忆文件中提取工作流程状态"""
    saved_state = []
    if os.path.exists(memory_file_path):
        try:
            with open(memory_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            in_workflow_section = False
            
            for line in lines:
                line = line.strip()
                
                if line == "工作流程定义:":
                    in_workflow_section = True
                    continue
                elif line == "执行历史:":
                    in_workflow_section = False
                    continue
                elif in_workflow_section and line.startswith("任务") and " - " in line:
                    # 解析格式如 "任务1: 打开模之屋 - 已完成"
                    try:
                        parts = line.split(" - ")
                        if len(parts) >= 2:
                            step_info = parts[0]  # "任务1: 打开模之屋"
                            status = parts[1]     # "已完成" 或 "待完成" 或 "待确定"
                            
                            # 提取任务号
                            step_match = re.search(r'任务(\d+):', step_info)
                            if step_match:
                                step_num = int(step_match.group(1))
                                if status == "已完成":
                                    completed = True
                                elif status == "待确定":
                                    completed = "pending_verification"
                                else:
                                    completed = False
                                
                                # 确保列表长度足够
                                while len(saved_state) < step_num:
                                    saved_state.append((None, False))
                                
                                # 更新对应位置的完成状态
                                step_desc = step_info.split(":", 1)[1].strip()
                                saved_state[step_num - 1] = (step_desc, completed)
                    except Exception as e:
                        print(f"解析记忆中的任务行出错: {line}, 错误: {str(e)}")
                        continue
        except Exception as e:
            print(f"从记忆文件读取工作流程状态失败: {str(e)}")
    
    return saved_state
def parse_named_arguments(params_str):
    """
    解析命名参数格式如 "param1=value1,param2=value2" 或 "url='example.com'"
    """
    args = []
    current_arg = ""
    inside_quotes = False
    quote_char = None
    
    i = 0
    while i < len(params_str):
        char = params_str[i]
        
        if char in ['"', "'"] and not inside_quotes:
            # 开始引号
            inside_quotes = True
            quote_char = char
        elif char == quote_char and inside_quotes:
            # 结束引号
            inside_quotes = False
            quote_char = None
        elif char == ',' and not inside_quotes:
            # 参数分隔符，不在引号内
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char
        i += 1
    
    # 添加最后一个参数
    if current_arg:
        args.append(current_arg.strip())
    
    # 提取实际参数值（跳过参数名）
    values_only = []
    for arg in args:
        if '=' in arg:
            # 分离参数名和值
            _, value = arg.split('=', 1)
            # 移除可能的引号
            value = value.strip().strip('"\'')
            values_only.append(value)
        else:
            # 如果没有等号，直接添加
            values_only.append(arg.strip())
    
    return values_only
def process_tool_calls(response_text, memory_file_path=None, workflow_state_ref=None, is_inquiry_phase=False):
    """
    解析AI响应中的工具调用指令
    支持格式: [TOOL:工具名称,arg1,arg2,arg3...] 或 [TOOL:工具名称(param=value)]
    """
    # 只在质询阶段检测任务完成标记，执行阶段不检测
    task_completed = is_task_completed(response_text) if is_inquiry_phase else False
    workflow_completed = False  # 总是设置为False
    
    if task_completed and memory_file_path and is_inquiry_phase:
        with open(memory_file_path, 'a', encoding='utf-8') as f:
            f.write(f"[TASK_COMPLETED]\n")
    
    if task_completed and is_inquiry_phase:
        print("检测到任务完成标记: [TASK_COMPLETED]")
    
    # 使用更复杂的正则表达式来处理嵌套参数
    # 这个正则可以正确处理带引号的参数和括号
    pattern = r'\[TOOL:([^\[\]]*(?:\([^)]*\)[^\[\]]*)*)\]'
    matches = re.findall(pattern, response_text)
    
    all_results = []
    for match in matches:
        full_match = match.strip()
        
        # 使用更智能的参数分离方法
        tool_name, tool_args = parse_tool_call(full_match)
        
        # 在解析参数后添加调试输出
        print(f"解析到的工具名: {tool_name}")
        print(f"解析到的参数: {tool_args}")
        
        # 验证工具是否存在
        tools = get_available_tools_info()
        if not tools or not isinstance(tools, list):
            all_results.append(f"工具 '{tool_name}' 执行失败: 无法获取工具列表")
            continue
            
        tool_exists = any(isinstance(tool, dict) and tool.get('name') == tool_name for tool in tools)
        if not tool_exists:
            all_results.append(f"工具 '{tool_name}' 执行失败: 工具不存在")
            continue
        
        result = execute_tool(tool_name, *tool_args) if tool_args else execute_tool(tool_name)
        
        if result is None:
            result = "工具执行结果为空"
        
        if memory_file_path:
            try:
                clean_result = result.replace("[TASK_COMPLETED]", "").replace("[TOTAL_TASK_COMPLETED]", "").strip()
                with open(memory_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"工具 '{tool_name}' 执行结果: {clean_result}\n")
            except Exception as e:
                print(f"写入记忆文件失败: {e}")
        
        print(f"工具 '{tool_name}' 执行结果: {result}")
        
        all_results.append(f"工具 '{tool_name}' 执行结果: {result}")
    
    if task_completed and workflow_state_ref and is_inquiry_phase:
        for i, (step, completed) in enumerate(workflow_state_ref):
            if step in response_text or response_text.strip().startswith(step[:min(len(step), 20)]):
                workflow_state_ref[i] = (step, True)
                break
    
    return {
        "results": "\n".join(all_results) if all_results else None,
        "task_completed": task_completed,
        "workflow_completed": workflow_completed
    }

def parse_tool_call(full_match):
    """
    智能解析工具调用字符串
    """
    if '(' in full_match and full_match.count('(') == full_match.count(')'):
        # 命名参数格式: tool_name(param=value)
        paren_start = full_match.rfind('(')  # 使用rfind找到最后一个左括号
        if paren_start != -1:
            tool_name = full_match[:paren_start].strip()
            params_str = full_match[paren_start+1:-1]  # 去掉最外层括号
            
            # 解析命名参数
            tool_args = parse_named_arguments(params_str)
            return tool_name, tool_args
    
    elif ',' in full_match:
        # 位置参数格式: tool_name,arg1,arg2
        parts = full_match.split(',', 1)
        tool_name = parts[0].strip()
        if len(parts) > 1:
            tool_args = [arg.strip() for arg in parts[1].split(',')]
        else:
            tool_args = []
        return tool_name, tool_args
    else:
        # 无参数格式: tool_name
        tool_name = full_match.strip()
        tool_args = []
        return tool_name, tool_args
def parse_history_content(content):
    """
    解析历史对话内容，转换为messages格式
    """
    messages = []
    lines = content.strip().split('\n')
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('用户:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_role = "user"
            current_content = [line[3:].strip()]
        elif line.startswith('AI:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            
            current_role = "assistant"
            current_content = [line[3:].strip()]
        elif line == "" and current_content:
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            current_role = None
            current_content = []
        elif current_role:
            current_content.append(line)
    
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": '\n'.join(current_content).strip()
        })
    
    return messages



import tkinter as tk
from tkinter import messagebox, ttk
import threading
import os
from pathlib import Path
from llm_class import VLMService
import pyautogui
import base64


class VLMTaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VLM牛马")
        self.root.geometry("400x350")
        self.root.resizable(False, False)
        self.root.attributes('-toolwindow', True)
        # 简化颜色主题 - 使用统一的蓝色系
        self.colors = {
            'primary': '#1E88E5',        # 主蓝色
            'primary_dark': '#0D47A1',   # 主蓝色深色
            'success': '#4CAF50',        # 成功绿色
            'success_dark': '#2E7D32',   # 成功绿色深色
            'warning': '#FFC107',        # 警告黄色
            'warning_dark': '#F57F17',   # 警告黄色深色
            'danger': '#F44336',         # 危险红色
            'danger_dark': '#D32F2F',    # 危险红色深色
            'light': '#FFFFFF',          # 白色
            'dark': '#212121',           # 深灰色
            'gray': '#F5F5F5',           # 浅灰色
            'medium_gray': '#E0E0E0',    # 中灰色
            'dark_gray': '#9E9E9E',      # 深灰色
            'text_primary': '#212121',   # 主要文字颜色
            'text_secondary': '#757575', # 次要文字颜色
            'border': '#BDBDBD'          # 边框颜色
        }
        
        # 任务执行标志
        self.is_executing = False 
        self.workflow_state = []  # 工作流程状态
        self.current_page_index = 0  # 当前显示的页面索引
        
        # 创建界面
        self.setup_modern_ui()
        
        # 文件路径 - 使用全局参数
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_file = KNOWLEDGE_FILE_PATH  # 使用全局变量
        self.workflow_path = WORKFLOW_PATH  # 使用全局变量
        self.memory_file = os.path.join(current_dir, "memory.txt")
        
        # 加载工作流程
        self.load_workflow_content()
        self.ensure_memory_file_exists()
    
    def ensure_memory_file_exists(self):
        """确保memory文件存在"""
        if not os.path.exists(self.memory_file):
            try:
                # 创建空的memory文件
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    f.write("工作流程定义:\n")
                    f.write("执行历史:\n")
                print(f"已创建memory文件: {self.memory_file}")
            except Exception as e:
                print(f"创建memory文件失败: {e}")
   
    def setup_modern_ui(self):
        # 主容器
        self.main_frame = tk.Frame(
            self.root,
            bg=self.colors['gray'],
            padx=12,
            pady=12
        )
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 任务信息框架
        task_frame = tk.LabelFrame(
            self.main_frame,
            text="当前任务",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['light'],
            padx=10,
            pady=8,
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['border'],
            highlightthickness=1
        )
        task_frame.pack(fill=tk.X, pady=(0, 12))
        
        # 任务内容框架
        task_content_frame = tk.Frame(task_frame, bg=self.colors['light'])
        task_content_frame.pack(fill=tk.X, padx=6, pady=6)
        
        # 任务标题
        self.task_title_label = tk.Label(
            task_content_frame,
            text="",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['light'],
            anchor="w"
        )
        self.task_title_label.pack(fill=tk.X, pady=(4, 4))
        
        # 任务描述
        self.task_desc_label = tk.Label(
            task_content_frame,
            text="",
            font=('Segoe UI', 9),
            fg=self.colors['text_secondary'],
            bg=self.colors['light'],
            wraplength=350,
            justify=tk.LEFT,
            anchor="w"
        )
        self.task_desc_label.pack(fill=tk.X, pady=(0, 4))
        
        # 进度条框架
        progress_frame = tk.Frame(self.main_frame, bg=self.colors['gray'])
        progress_frame.pack(fill=tk.X, pady=(0, 16))
        
        progress_label = tk.Label(
            progress_frame,
            text="进度:",
            font=('Segoe UI', 9),
            fg=self.colors['text_primary'],
            bg=self.colors['gray']
        )
        progress_label.pack(side=tk.LEFT)
        
        # 进度条容器
        self.progress_container = tk.Frame(
            progress_frame,
            bg=self.colors['medium_gray'],
            height=12,
            relief=tk.FLAT,
            bd=0
        )
        self.progress_container.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.progress_container.pack_propagate(False)
        
        # 进度条
        self.progress_bar = tk.Frame(
            self.progress_container,
            bg=self.colors['success'],
            relief=tk.FLAT,
            bd=0
        )
        self.progress_bar.pack(fill=tk.BOTH, expand=True)
        
        # 按钮框架 - 使用统一设计
        button_frame = tk.Frame(self.main_frame, bg=self.colors['gray'])
        button_frame.pack(fill=tk.X, pady=(8, 12))
        
        # 统一按钮样式
        button_style = {
            'font': ('Segoe UI', 9, 'bold'),
            'width': 12,
            'height': 1,
            'relief': tk.FLAT,
            'border': 0,
            'cursor': 'hand2',
            'activebackground': self.colors['primary_dark'],
            'activeforeground': 'white'
        }
        
        # 上排按钮
        top_button_frame = tk.Frame(button_frame, bg=self.colors['gray'])
        top_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 选择任务按钮 - 使用主色调
        self.select_page_button = tk.Button(
            top_button_frame,
            text="选择任务",
            command=self.open_page_selection_dialog,
            bg=self.colors['primary'],
            fg='white',
            **button_style
        )
        self.select_page_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 执行当前按钮 - 使用警告色
        self.run_current_button = tk.Button(
            top_button_frame,
            text="执行当前",
            command=self.run_current_task,
            bg=self.colors['warning'],
            fg=self.colors['dark'],
            **button_style
        )
        self.run_current_button.pack(side=tk.LEFT, padx=(0, 0))
        
        # 下排按钮
        bottom_button_frame = tk.Frame(button_frame, bg=self.colors['gray'])
        bottom_button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 执行所有按钮 - 使用成功色
        self.run_all_button = tk.Button(
            bottom_button_frame,
            text="执行所有",
            command=self.run_all_tasks,
            bg=self.colors['success'],
            fg='white',
            **button_style
        )
        self.run_all_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按钮 - 使用危险色
        self.stop_button = tk.Button(
            bottom_button_frame,
            text="停止",
            command=self.stop_all_tasks,
            state=tk.DISABLED,
            bg=self.colors['danger'],
            fg='white',
            **button_style
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清除记忆按钮 - 使用灰色
        self.clear_memory_button = tk.Button(
            bottom_button_frame,
            text="清除记忆",
            command=self.clear_short_term_memory,
            bg=self.colors['dark_gray'],
            fg='white',
            **button_style
        )
        self.clear_memory_button.pack(side=tk.LEFT, padx=(0, 0))
        
        # 状态栏
        self.status_frame = tk.Frame(
            self.main_frame,
            bg=self.colors['dark'],
            height=26
        )
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(18, 0))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="状态: 等待任务开始",
            font=('Segoe UI', 9),
            fg='white',
            bg=self.colors['dark'],
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=3)

    def open_page_selection_dialog(self):
        """打开页面选择弹窗"""
        if not self.workflow_state:
            messagebox.showinfo("提示", "没有可选择的页面")
            return

        # 创建顶层弹窗
        dialog = tk.Toplevel(self.root)
        dialog.title("选择任务页面")
        dialog.geometry("360x320")
        dialog.configure(bg=self.colors['gray'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 标题
        title_label = tk.Label(
            dialog,
            text="请选择要跳转的任务页面:",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['gray']
        )
        title_label.pack(pady=(12, 12))

        # 列表框架
        list_frame = tk.Frame(dialog, bg=self.colors['gray'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # 滚动条
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 列表框
        self.page_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=('Segoe UI', 9),
            bg='white',
            fg=self.colors['text_primary'],
            selectbackground=self.colors['primary'],
            selectforeground='white',
            borderwidth=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=self.colors['primary'],
            height=10
        )
        
        # 添加所有页面到列表框
        for i, (step, completed) in enumerate(self.workflow_state):
            status_text = "已完成" if completed == True else "待确定" if completed == "pending_verification" else "待完成"
            status_symbol = "✓" if completed == True else "?" if completed == "pending_verification" else "○"
            display_text = f"{i+1}. {status_symbol} {step} ({status_text})"
            self.page_listbox.insert(tk.END, display_text)
        
        if 0 <= self.current_page_index < len(self.workflow_state):
            self.page_listbox.selection_set(self.current_page_index)
            self.page_listbox.see(self.current_page_index)

        self.page_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        scrollbar.config(command=self.page_listbox.yview)

        # 按钮框架
        button_frame = tk.Frame(dialog, bg=self.colors['gray'])
        button_frame.pack(fill=tk.X, padx=12, pady=18)

        # 统一对话框按钮样式
        dialog_button_style = {
            'font': ('Segoe UI', 9, 'bold'),
            'width': 10,
            'height': 1,
            'relief': tk.FLAT,
            'border': 0,
            'cursor': 'hand2'
        }
        
        ok_button = tk.Button(
            button_frame,
            text="确定",
            command=lambda: self.select_page_from_dialog(dialog),
            bg=self.colors['primary'],
            fg='white',
            activebackground=self.colors['primary_dark'],
            activeforeground='white',
            **dialog_button_style
        )
        ok_button.pack(side=tk.LEFT, padx=(0, 12))

        cancel_button = tk.Button(
            button_frame,
            text="取消",
            command=dialog.destroy,
            bg=self.colors['dark_gray'],
            fg='white',
            activebackground=self.colors['medium_gray'],
            activeforeground='white',
            **dialog_button_style
        )
        cancel_button.pack(side=tk.LEFT)

        # 双击事件
        self.page_listbox.bind("<Double-1>", lambda event: self.select_page_from_dialog(dialog))

    def select_page_from_dialog(self, dialog):
        """从弹窗中选择页面"""
        selection = self.page_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择一个页面")
            return

        selected_index = selection[0]
        if 0 <= selected_index < len(self.workflow_state):
            # 跳转到选中的页面
            self.current_page_index = selected_index
            self.update_task_display()
        
        dialog.destroy()
        
    def load_workflow_content(self):
        """加载并显示工作流程内容"""
        # 首先尝试从记忆文件中获取步骤完成状态
        saved_state = self.get_workflow_state_from_memory()
        
        # 使用全局定义的工作流程路径
        workflow_path = self.workflow_path
        
        if workflow_path.exists():
            try:
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 分割工作流程为独立步骤
                steps = [step.strip() for step in content.split('\n') if step.strip()]
                
                # 初始化工作流程状态，优先使用记忆文件中的状态
                self.workflow_state = []
                for i, step in enumerate(steps):
                    completed = False
                    # 检查记忆文件中是否有该步骤的完成状态
                    if i < len(saved_state):
                        _, saved_completed = saved_state[i]
                        completed = saved_completed
                    self.workflow_state.append((step, completed))
                
                # 显示第一页
                if self.workflow_state:
                    self.update_task_display()
                
            except Exception as e:
                print(f"加载工作流程失败: {str(e)}")
        else:
            print(f"工作流程文件不存在: {workflow_path}")
    def update_task_display(self):
        """更新当前任务显示"""
        if not self.workflow_state or self.current_page_index >= len(self.workflow_state):
            return
            
        step, completed = self.workflow_state[self.current_page_index]
        
        # 更新标题
        if completed == True:
            status_text = "已完成"
            status_color = self.colors['success']
        elif completed == "pending_verification":
            status_text = "待确定"
            status_color = self.colors['warning']
        else:
            status_text = "待完成"
            status_color = self.colors['danger']
            
        self.task_title_label.config(
            text=f"任务 {self.current_page_index + 1}/{len(self.workflow_state)} - {status_text}",
            fg=status_color
        )
        
        # 更新描述
        self.task_desc_label.config(text=step)
        
        # 更新进度条
        completed_count = sum(1 for _, completed in self.workflow_state if completed == True)
        total_count = len(self.workflow_state)
        progress = (completed_count / total_count) * 100 if total_count > 0 else 0
        self.update_progress_bar(progress)

    def update_progress_bar(self, value):
        """更新进度条"""
        # 计算进度条宽度
        container_width = self.progress_container.winfo_width()
        if container_width <= 0:
            # 如果容器宽度未初始化，稍后重试
            self.root.after(100, lambda: self.update_progress_bar(value))
            return
            
        width = int((value / 100) * container_width)
        self.progress_bar.config(width=width)
        self.progress_bar.update()

    def set_current_page(self, page_index):
        """设置当前页面索引并更新显示"""
        self.current_page_index = page_index
        self.update_task_display()

    def run_current_task(self):
        """执行当前任务"""
        if self.is_executing:
            messagebox.showwarning("警告", "任务正在执行中，请等待完成")
            return
        
        if not self.workflow_state:
            messagebox.showwarning("警告", "没有可执行的任务")
            return
        
        current_task_index = self.current_page_index
        if current_task_index >= len(self.workflow_state):
            messagebox.showwarning("警告", "当前页码超出任务范围")
            return
            
        task_step, completed = self.workflow_state[current_task_index]
        
        if completed == True:
            messagebox.showinfo("提示", f"任务 {current_task_index + 1} 已完成，无需再次执行")
            return
        
        self.is_executing = True
        self.run_current_button.config(state=tk.DISABLED)
        self.run_all_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status(f"状态: 正在执行任务 {current_task_index + 1}: {task_step}")
        
        # 在新线程中执行任务以避免界面冻结
        task_thread = threading.Thread(
            target=self.execute_single_task,
            args=(current_task_index,)
        )
        task_thread.daemon = True
        task_thread.start()

    def run_all_tasks(self):
        """执行所有任务"""
        if self.is_executing:
            messagebox.showwarning("警告", "任务正在执行中，请等待完成")
            return
        
        if not self.workflow_state:
            messagebox.showwarning("警告", "没有可执行的任务")
            return
        
        self.is_executing = True
        self.run_current_button.config(state=tk.DISABLED)
        self.run_all_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("状态: 正在执行所有任务...")
        
        # 在新线程中执行所有任务以避免界面冻结
        task_thread = threading.Thread(
            target=self.execute_all_tasks
        )
        task_thread.daemon = True
        task_thread.start()

    def execute_all_tasks(self):
        """执行所有任务，按顺序自动执行"""
        try:
            for task_index in range(len(self.workflow_state)):
                # 检查是否停止了执行
                if not self.is_executing:
                    print("系统: 任务已手动停止")
                    return
                
                task_step, completed = self.workflow_state[task_index]
                
                # 如果任务已完成，则跳过
                if completed == True:
                    print(f"跳过已完成任务 {task_index + 1}: {task_step}")
                    continue
                
                print(f"开始执行任务 {task_index + 1}: {task_step}")
                
                # 切换到当前任务页面
                self.root.after(0, lambda idx=task_index: self.set_current_page(idx))
                
                # 执行任务
                task_output = ""
                completed_flag_found = False
                
                # 从vision_task_loop函数获取生成器
                import memory_llm  # 使用绝对导入
                
                # 使用for循环遍历vision_task_loop的输出
                for output in memory_llm.vision_task_loop(
                    task_step, 
                    self.knowledge_file, 
                    self.memory_file, 
                    self.workflow_state, 
                    reset_first_iteration=True
                ):
                    if not self.is_executing:
                        print("系统: 任务已手动停止")
                        return
                    
                    if "[TASK_COMPLETED]" in output and "[TOOL:" not in output:  # 确保不是工具执行结果中的标记
                        # 如果是子任务完成标记，直接标记为已完成
                        self.root.after(0, lambda idx=task_index: self.mark_step_as_completed(idx))
                        completed_flag_found = True
                        # 跳出当前任务的内部循环，准备执行下一个任务
                        break  
                    else:
                        print(f"任务{task_index + 1}执行结果: {output}")
                        
                        # 将输出追加到记忆文件
                        with open(self.memory_file, 'a', encoding='utf-8') as f:
                            f.write(f"AI响应: {output}\n")
                        
                        task_output += output + "\n"
                
                # 检查是否所有任务都已完成
                all_completed = all(completed == True for _, completed in self.workflow_state)
                if all_completed:
                    print("所有任务已完成")
                    self.root.after(0, lambda: self.update_status("状态: 所有任务已完成"))
                    break

        except Exception as e:
            error_msg = str(e)  # 提前保存错误信息
            print(f"系统: 执行任务时出错: {error_msg}")
            self.root.after(0, lambda msg=error_msg: self.update_status(f"状态: 执行所有任务时出错: {msg}"))
        finally:
            self.is_executing = False
            self.root.after(0, lambda: self.run_current_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.run_all_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.update_status("状态: 所有任务执行完成"))

    def execute_single_task(self, task_index):
        """执行单个任务"""
        try:
            task_step, completed = self.workflow_state[task_index]
            
            if completed == True:
                print(f"任务 {task_index + 1} 已完成")
                self.root.after(0, lambda: self.update_status(f"状态: 任务 {task_index + 1} 已完成"))
                return
            
            # 从vision_task_loop函数获取生成器
            import memory_llm  # 使用绝对导入
            
            # 执行任务
            task_output = ""
            
            # 使用for循环遍历vision_task_loop的输出
            for output in memory_llm.vision_task_loop(
                task_step, 
                self.knowledge_file, 
                self.memory_file, 
                self.workflow_state, 
                reset_first_iteration=True
            ):
                if not self.is_executing:
                    print("系统: 任务已手动停止")
                    return
                
                if "[TASK_COMPLETED]" in output and "[TOOL:" not in output:  # 确保不是工具执行结果中的标记
                    print(f"任务{task_index + 1}执行结果: {output}")
                    # 标记为已完成
                    self.root.after(0, lambda idx=task_index: self.mark_step_as_completed(idx))
                    
                    # 单个任务执行完成后立即退出（关键修改）
                    print(f"任务 {task_index + 1} 已完成，退出执行")
                    break
                else:
                    print(f"任务{task_index + 1}执行结果: {output}")
                    
                    # 将输出追加到记忆文件
                    with open(self.memory_file, 'a', encoding='utf-8') as f:
                        f.write(f"AI响应: {output}\n")
                    
                    task_output += output + "\n"
            
            # 单个任务执行完成后的状态更新
            self.root.after(0, lambda: self.update_status(f"状态: 任务 {task_index + 1} 执行完成"))

        except Exception as e:
            error_msg = str(e)  # 提前保存错误信息
            print(f"系统: 执行任务时出错: {error_msg}")
            self.root.after(0, lambda msg=error_msg: self.update_status(f"状态: 执行任务 {task_index + 1} 时出错: {msg}"))
        finally:
            self.is_executing = False
            self.root.after(0, lambda: self.run_current_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.run_all_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            
    def mark_step_as_completed(self, index):
        """标记步骤为已完成"""
        if 0 <= index < len(self.workflow_state):
            # 将任务状态改为已完成
            self.workflow_state[index] = (self.workflow_state[index][0], True)
            
            # 保存状态
            self.save_workflow_state()
            
            # 更新当前显示（如果当前页是完成的页）
            if index == self.current_page_index:
                self.update_task_display()
            
            # 在记忆文件中记录步骤执行完成
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"任务{index+1}执行完成\n")
            
            # 更新进度条
            completed_count = sum(1 for _, completed in self.workflow_state if completed == True)
            total_count = len(self.workflow_state)
            progress = (completed_count / total_count) * 100 if total_count > 0 else 0
            self.update_progress_bar(progress)

    def mark_step_as_pending_verification(self, index):
        """标记步骤为待确定状态"""
        if 0 <= index < len(self.workflow_state):
            # 将任务状态改为待确定而不是已完成
            self.workflow_state[index] = (self.workflow_state[index][0], "pending_verification")
            
            # 保存状态
            self.save_workflow_state()
            
            # 更新当前显示（如果当前页是完成的页）
            if index == self.current_page_index:
                self.update_task_display()
            
            # 在记忆文件中记录步骤执行完成，等待确认
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                f.write(f"任务{index+1} 执行完成，等待确认: {self.workflow_state[index][0]}\n")

    def confirm_task_completed(self, task_index):
        """确认任务真正完成"""
        if 0 <= task_index < len(self.workflow_state):
            task_desc, status = self.workflow_state[task_index]
            if status == "pending_verification":
                self.workflow_state[task_index] = (task_desc, True)
                self.save_workflow_state()
                
                # 更新显示（如果当前页是确认的页）
                if task_index == self.current_page_index:
                    self.update_task_display()
                
                # 在记忆文件中记录步骤确认完成
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"任务{task_index+1} 确认完成: {task_desc}\n")

    def stop_all_tasks(self):
        """停止所有任务执行"""
        self.is_executing = False
        self.run_current_button.config(state=tk.NORMAL)
        self.run_all_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("状态: 任务已停止")

    def get_workflow_state_from_memory(self):
        """从记忆文件中提取工作流程状态"""
        saved_state = []
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                in_workflow_section = False
                
                for line in lines:
                    line = line.strip()
                    
                    if line == "工作流程定义:":
                        in_workflow_section = True
                        continue
                    elif line == "执行历史:":
                        in_workflow_section = False
                        continue
                    elif in_workflow_section and line.startswith("任务") and " - " in line:
                        # 解析格式如 "任务1: 打开模之屋 - 已完成"
                        try:
                            parts = line.split(" - ")
                            if len(parts) >= 2:
                                step_info = parts[0]  # "任务1: 打开模之屋"
                                status = parts[1]     # "已完成" 或 "待完成" 或 "待确定"
                                
                                # 提取任务号
                                step_match = re.search(r'任务(\d+):', step_info)
                                if step_match:
                                    step_num = int(step_match.group(1))
                                    if status == "已完成":
                                        completed = True
                                    elif status == "待确定":
                                        completed = "pending_verification"
                                    else:
                                        completed = False
                                    
                                    # 确保列表长度足够
                                    while len(saved_state) < step_num:
                                        saved_state.append((None, False))
                                    
                                    # 更新对应位置的完成状态
                                    step_desc = step_info.split(":", 1)[1].strip()
                                    saved_state[step_num - 1] = (step_desc, completed)
                        except Exception as e:
                            print(f"解析记忆中的任务行出错: {line}, 错误: {str(e)}")
                            continue
            except Exception as e:
                print(f"从记忆文件读取工作流程状态失败: {str(e)}")
        
        return saved_state

    def update_status(self, status_text):
        """更新状态栏"""
        self.status_label.config(text=status_text)

    def clear_short_term_memory(self):
        """手动清除短期记忆"""
        if os.path.exists(self.memory_file):
            try:
                # 使用全局工作流程路径
                workflow_path = self.workflow_path
                
                # 保留工作流程定义，只清除执行历史
                if workflow_path.exists():
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        workflow_content = f.read()
                    
                    # 重写memory文件，只保留工作流程定义
                    with open(self.memory_file, 'w', encoding='utf-8') as f:
                        f.write("工作流程定义:\n")
                        steps = [step.strip() for step in workflow_content.split('\n') if step.strip()]
                        for i, step in enumerate(steps):
                            f.write(f"任务{i+1}: {step} - 待完成\n")
                        f.write("\n执行历史:\n")
                else:
                    # 如果工作流文件不存在，清空整个记忆文件
                    with open(self.memory_file, 'w', encoding='utf-8') as f:
                        f.write("")
                
                # 重置所有步骤为未完成
                for i in range(len(self.workflow_state)):
                    self.workflow_state[i] = (self.workflow_state[i][0], False)
                
                # 重置到第一页并更新显示
                self.current_page_index = 0
                self.update_task_display()
                
                print("短期记忆已手动清除，所有任务重置为未完成")
                self.update_status("状态: 短期记忆已清除")
            except Exception as e:
                messagebox.showerror("错误", f"清除短期记忆失败: {str(e)}")
        else:
            print("短期记忆文件不存在")
            self.update_status("状态: 短期记忆文件不存在")
    def save_workflow_state(self):
        """保存工作流程状态到记忆文件"""
        try:
            # 读取当前memory文件内容
            current_content = ""
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            
            # 分离工作流程定义和执行历史
            lines = current_content.split('\n')
            workflow_lines = []
            history_lines = []
            in_history = False
            
            for line in lines:
                if line == "执行历史:":
                    in_history = True
                    history_lines.append(line)
                    continue
                elif line.startswith("工作流程定义:"):
                    workflow_lines.append(line)
                    continue
                elif line.startswith("任务") and " - " in line and not in_history:
                    workflow_lines.append(line)
                    continue
                elif in_history:
                    history_lines.append(line)
                else:
                    workflow_lines.append(line)
            
            # 更新工作流程状态
            updated_workflow_lines = ["工作流程定义:"]
            for i, (step, completed) in enumerate(self.workflow_state):
                if completed == True:
                    status = "已完成"
                elif completed == "pending_verification":
                    status = "待确定"
                else:
                    status = "待完成"
                updated_workflow_lines.append(f"任务{i+1}: {step} - {status}")
            
            # 合并内容并写回文件
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(updated_workflow_lines))
                f.write('\n\n')
                f.write('\n'.join(history_lines))
        except Exception as e:
            print(f"保存工作流程状态失败: {str(e)}")

def main():
    root = tk.Tk()
    try:
        root.iconbitmap('favicon.ico')  # 如果有图标文件
    except:
        pass
    
    
    root.attributes('-topmost', True)
    
    app = VLMTaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()