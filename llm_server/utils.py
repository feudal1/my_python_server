import os
import subprocess
import re
import json
import sys
from pathlib import Path
import pyautogui
import base64
from datetime import datetime
from llm_class import VLMService

def update_paths():
    """更新路径配置"""
    global CONFIG_PATH, KNOWLEDGE_FILE_PATH, WORKFLOW_PATH
    CONFIG_PATH = CURRENT_DIR / "config" / (workenv + "_tools_config.json")
    KNOWLEDGE_FILE_PATH = CURRENT_DIR / "config" / (workenv + "_knowledge.txt")
    WORKFLOW_PATH = CURRENT_DIR / "config" / (workenv + "_workflow.txt")


def switch_environment(new_env):
    """切换工作环境"""
    global workenv
    if new_env in workenvs:
        workenv = new_env
        update_paths()  # 更新路径
        print(f"已切换到 {workenv} 环境")
        
        # 保存当前环境到配置文件
        save_current_environment(new_env)
        
        # 重启程序以应用新环境
        print("正在重启以应用新环境...")
        python = sys.executable
        os.execl(python, python, *sys.argv)
        
        return True
    return False

def save_current_environment(env):
    """保存当前环境到配置文件"""
    config_file = CURRENT_DIR / "config" / "default_env.json"
    
    try:
        config_data = {
            "default_environment": env,
            "last_updated": str(datetime.now())
        }
        
        # 确保config目录存在
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
            
        print(f"环境配置已保存到: {config_file}")
    except Exception as e:
        print(f"保存环境配置失败: {e}")


def load_default_environment():
    """从配置文件加载默认环境"""
    config_file = CURRENT_DIR / "config" / "default_env.json"
    
    try:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                env = config_data.get("default_environment", "web")
                print(f"从配置文件加载默认环境: {env}")
                return env
    except Exception as e:
        print(f"加载环境配置失败: {e}")
    
    print("使用默认环境: web")
    return "web"


def load_workenvs():
    """从配置文件加载工作环境列表"""
    config_file = CURRENT_DIR / "config" / "default_env.json"
    
    try:
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                envs = config_data.get("workenvs", ["web", "blender", "ue"])
                print(f"从配置文件加载工作环境列表: {envs}")
                return envs
    except Exception as e:
        print(f"加载工作环境列表配置失败: {e}")
    
    print("使用默认工作环境列表: ['web', 'blender', 'ue']")
    return ["web", "blender", "ue"]




def execute_python_script(script_path, *args):
    """
    执行指定路径的Python脚本
    """
    # 获取项目根目录（从当前脚本位置向上一级）
    current_dir = Path(__file__).parent.parent  # 回到项目根目录
    script_full_path = current_dir / script_path
    
    print ("脚本路径:",script_full_path)
    if not script_full_path.exists():
        return f"错误: 脚本 '{script_full_path}' 不存在，current_dir：{current_dir}"
    
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


def parse_tool_call(full_match):
    """
    智能解析工具调用字符串
    """
    # 检查是否为命名参数格式: tool_name(param1=value1,param2=value2)
    if '(' in full_match and full_match.count('(') == full_match.count(')'):
        paren_start = full_match.rfind('(')
        if paren_start != -1:
            tool_name = full_match[:paren_start].strip()
            params_str = full_match[paren_start+1:-1]  # 去掉最外层括号
            
            # 解析命名参数
            tool_args = parse_named_arguments(params_str)
            return tool_name, tool_args
    
    # 检查是否为带等号的参数格式: tool_name,param1=value1,param2=value2
    elif '=' in full_match and ',' in full_match:
        # 分割工具名和参数
        parts = full_match.split(',', 1)
        tool_name = parts[0].strip()
        
        # 解析参数部分，可能有多个参数，如: param1=value1,param2=value2
        params_part = parts[1]
        param_list = []
        
        # 处理可能包含引号的参数值
        current_param = ""
        inside_quotes = False
        quote_char = None
        
        i = 0
        while i < len(params_part):
            char = params_part[i]
            
            if char in ['"', "'"] and not inside_quotes:
                inside_quotes = True
                quote_char = char
            elif char == quote_char and inside_quotes:
                inside_quotes = False
                quote_char = None
            elif char == ',' and not inside_quotes:
                param_list.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
            i += 1
        
        # 添加最后一个参数
        if current_param:
            param_list.append(current_param.strip())
        
        # 提取参数值（跳过参数名）
        tool_args = []
        for param in param_list:
            if '=' in param:
                # 分离参数名和值
                _, value = param.split('=', 1)
                # 移除可能的引号
                value = value.strip().strip('"\'')
                tool_args.append(value)
            else:
                tool_args.append(param.strip())
        
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


# 在 vision_task_loop 函数中，修改返回的LLM响应格式
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
    max_iterations = 50
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
            
            # 优化输出格式，提供更清晰的状态信息
            response_preview = ai_response[:60] + "..." if len(ai_response) > 60 else ai_response
            yield f"LLM响应: {response_preview}"
            
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

# 定义全局变量
CURRENT_DIR = Path(__file__).parent
workenvs =load_workenvs()
workenv = load_default_environment()  # 从配置文件加载默认环境

# 初始化路径
update_paths()

