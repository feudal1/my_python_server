import tkinter as tk
from tkinter import messagebox
import threading
import os
from pathlib import Path

# 从utils导入所需的功能
from utils import (
    switch_environment, load_default_environment, 
    get_workflow_state_from_memory_in_app, vision_task_loop, 
    process_tool_calls, has_tool_calls, is_task_completed,
    send_task_confirmation_to_ai, get_tools_description,
    get_available_tools_info
)
from llm_class import VLMService
import pyautogui


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
        # 从utils获取路径
        from utils import KNOWLEDGE_FILE_PATH, WORKFLOW_PATH
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
        self.run_current_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清除记忆按钮 - 移到上排按钮，位于执行当前按钮的右边
        self.clear_memory_button = tk.Button(
            top_button_frame,
            text="清除记忆",
            command=self.clear_short_term_memory,
            bg=self.colors['dark_gray'],
            fg='white',
            **button_style
        )
        self.clear_memory_button.pack(side=tk.LEFT, padx=(0, 0))
        
        # 下排按钮
        bottom_button_frame = tk.Frame(button_frame, bg=self.colors['gray'])
        bottom_button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 环境切换按钮 - 直接放在底部按钮行，不使用标签
        self.env_select_button = tk.Button(
            bottom_button_frame,
            text="选择环境",
            command=self.open_environment_selection_dialog,
            font=('Segoe UI', 9, 'bold'),
            width=12,
            height=1,
            relief=tk.FLAT,
            border=0,
            cursor='hand2',
            bg=self.colors['primary'],
            fg='white',
            activebackground=self.colors['primary_dark'],
            activeforeground='white'
        )
        self.env_select_button.pack(side=tk.LEFT, padx=(0, 10))
        
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
        self.stop_button.pack(side=tk.LEFT, padx=(0, 0))
        
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
    
   
    def open_environment_selection_dialog(self):
        """打开环境选择弹窗"""
        # 创建顶层弹窗
        dialog = tk.Toplevel(self.root)
        dialog.title("选择环境")
        dialog.geometry("200x150")
        dialog.configure(bg=self.colors['gray'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 标题
        title_label = tk.Label(
            dialog,
            text="请选择环境:",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['gray']
        )
        title_label.pack(pady=(12, 12))

        # 列表框
        list_frame = tk.Frame(dialog, bg=self.colors['gray'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # 滚动条
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 列表框
        env_listbox = tk.Listbox(
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
            height=5
        )
        
        # 从utils获取环境列表
        from utils import workenvs
        # 添加所有环境到列表框
        for i, env in enumerate(workenvs):
            # 从utils获取当前环境
            from utils import workenv
            status_symbol = "●" if env == workenv else "○"
            display_text = f"{status_symbol} {env}"
            env_listbox.insert(tk.END, display_text)
            
            # 选中当前环境
            if env == workenv:
                env_listbox.selection_set(i)
                env_listbox.see(i)

        env_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        scrollbar.config(command=env_listbox.yview)

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
            command=lambda: self.select_environment_from_dialog(dialog, env_listbox),
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
        env_listbox.bind("<Double-1>", lambda event: self.select_environment_from_dialog(dialog, env_listbox))

    def select_environment_from_dialog(self, dialog, env_listbox):
        """从弹窗中选择环境"""
        selection = env_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择一个环境")
            return

        selected_index = selection[0]
        from utils import workenvs
        if 0 <= selected_index < len(workenvs):
            selected_env = workenvs[selected_index]
            # 切换环境
            if switch_environment(selected_env):
                # 重新加载工作流程
                self.load_workflow_content()
                self.update_status(f"状态: 已切换到 {selected_env} 环境")
                print(f"已切换到 {selected_env} 环境，重新加载配置")
        
        dialog.destroy()

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
        saved_state = get_workflow_state_from_memory_in_app(self.memory_file)
        
        # 使用全局定义的工作流程路径
        if self.workflow_path.exists():
            try:
                with open(self.workflow_path, 'r', encoding='utf-8') as f:
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
            print(f"工作流程文件不存在: {self.workflow_path}")
            
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
                
                # 使用从utils导入的vision_task_loop函数
                # 使用for循环遍历vision_task_loop的输出
                for output in vision_task_loop(
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
            
            # 执行任务
            task_output = ""
            
            # 使用for循环遍历vision_task_loop的输出
            for output in vision_task_loop(
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
        saved_state = get_workflow_state_from_memory_in_app(self.memory_file)
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