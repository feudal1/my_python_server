import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
from PIL import Image, ImageTk
from collections import deque, defaultdict
import logging
from pathlib import Path
from ultralytics import YOLO
import pyautogui
import json
import queue
from threading import Thread, Lock
import glob
import re
import tkinter as tk
from tkinter import ttk


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config 

CONFIG = load_config()


# 配置日志
def setup_logging():
    """设置日志配置"""
    # 创建logger实例
    logger = logging.getLogger('ppo_agent_logger')
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_ppo_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器，避免重复
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class ResidualBlock(nn.Module):
    """
    ResNet基本块
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet特征提取器
    """
    def __init__(self, input_channels=3, block_channels=[32, 64, 128]):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 使用配置文件中的图像尺寸
        height = CONFIG.get('IMAGE_HEIGHT', 480)
        width = CONFIG.get('IMAGE_WIDTH', 640)
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, block_channels[0], 
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        layers = []
        # 第一层
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        
        # 第二层 - 下采样
        layers.append(ResidualBlock(block_channels[0], block_channels[1], stride=2))
        layers.append(ResidualBlock(block_channels[1], block_channels[1]))
        
        # 第三层 - 下采样
        layers.append(ResidualBlock(block_channels[1], block_channels[2], stride=2))
        layers.append(ResidualBlock(block_channels[2], block_channels[2]))
        
        self.layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_features = nn.Linear(block_channels[2], 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_features(x)
        return x


class PolicyNetwork(nn.Module):
    """
    策略网络
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(3)
        feature_size = 256  # 根据上面的修改调整
        
        # Actor heads
        self.move_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, move_action_dim)
        )
        
        self.turn_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, turn_action_dim)
        )
        
        # Action parameter head
        self.action_param_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Value network
        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化logger
        self.logger = setup_logging()

    def forward(self, state, return_debug_info=False):
        features = self.feature_extractor(state)
        
        # Actor outputs
        move_logits = self.move_actor(features)
        turn_logits = self.turn_actor(features)
        
        # Action parameters
        action_params = torch.tanh(self.action_param_head(features))
        
        # Critic output
        value = self.critic(features)
        
        if return_debug_info:
            debug_info = {
                'move_logits': move_logits.detach().cpu().numpy(),
                'turn_logits': turn_logits.detach().cpu().numpy(),
                'action_params': action_params.detach().cpu().numpy(),
                'value': value.detach().cpu().numpy(),
                'features_shape': features.shape
            }
            
            # 打印模型输出信息
            self.logger.info(f"Move: {[round(float(x), 2) for x in debug_info['move_logits'][0]]}")
            self.logger.info(f"Turn: {[round(float(x), 2) for x in debug_info['turn_logits'][0]]}")
            self.logger.info(f"参数: {[round(float(x), 2) for x in debug_info['action_params'][0]]}")
            self.logger.info(f"价值: {round(float(debug_info['value'][0][0]), 2)}")
            
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value,
                debug_info
            )
        else:
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value
            )


class Memory:
    """
    智能体的记忆存储类
    """
    def __init__(self):
        self.states = []
        self.move_actions = []
        self.turn_actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_params = []

    def clear_memory(self):
        del self.states[:]
        del self.move_actions[:]
        del self.turn_actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_params[:]

    def append(self, state, move_action, turn_action, logprob, reward, is_terminal, action_param=None):
        self.states.append(state)
        self.move_actions.append(move_action)
        self.turn_actions.append(turn_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        if action_param is not None:
            self.action_params.append(action_param)


class RealTimeVisualizer:
    """
    实时可视化器，使用Tkinter显示截图、YOLO检测框和智能体状态
    """
    def __init__(self, window_name="PPO Agent Visualizer"):
        self.window_name = window_name
        self.current_image = None
        self.detections = []
        self.agent_info = {}
        self.image_lock = Lock()
        self.info_queue = queue.Queue(maxsize=10)  # 限制队列大小
        
        # 标记是否在主线程中初始化
        self.root = None
        self.image_frame = None
        self.canvas = None
        self.display_image = None
        self.image_tk = None
        self.timer_id = None
        
        # 使用队列在线程间传递数据
        self.gui_queue = queue.Queue()

    def init_gui(self):
        """在主线程中初始化GUI"""
        if self.root is None:
            self.root = tk.Tk()
            self.root.title(self.window_name)
            self.root.geometry("800x600")  # 减小窗口尺寸
            
            # 获取屏幕尺寸
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # 设置窗口位置到屏幕左侧
            window_width = 800
            window_height = 600
            x_position = 0  # 左侧位置
            y_position = (screen_height - window_height) // 2  # 垂直居中
            
            # 设置窗口几何属性（宽度x高度+x偏移+y偏移）
            self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
            
            # 创建图像显示框架
            self.image_frame = ttk.Frame(self.root)
            self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # 创建Canvas用于显示图像
            self.canvas = tk.Canvas(self.image_frame, bg="black")
            self.canvas.pack(fill=tk.BOTH, expand=True)
    def update_image_and_detections(self, image, detections):
        """
        更新图像和检测结果，并立即触发界面更新
        """
        with self.image_lock:
            self.current_image = image.copy() if image is not None else None
            self.detections = detections.copy() if detections is not None else []

        # 将更新请求放入队列
        try:
            self.gui_queue.put(('update_image', self.current_image, self.detections), block=False)
        except queue.Full:
            pass

    def _update_image_display(self, image, detections):
        """
        更新图像显示
        """
        if image is not None:
            # 绘制检测框
            display_img = self._draw_detections(image.copy())
            
            # 如果有智能体信息，也绘制在图像上
            if hasattr(self, '_last_agent_info') and self._last_agent_info:
                display_img = self._add_agent_info_to_image(display_img, self._last_agent_info)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            
            # 调整图像大小以适应canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage对象
            self.image_tk = ImageTk.PhotoImage(pil_image)
            
            # 更新canvas上的图像
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.image_tk)

    def _add_agent_info_to_image(self, image, info):
        """
        将智能体信息添加到图像上
        """
        # 设置字体和颜色
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (255, 255, 255)  # 白色文字
        text_thickness = 2
        background_color = (0, 0, 0)  # 黑色背景
        
        # 计算起始位置
        start_y = 30
        line_height = 30
        
        # 准备要显示的信息
        lines = []
        
        # Move Probabilities
        move_probs = info.get('move_probs', 'N/A')
        if isinstance(move_probs, list):
            move_probs_str = [round(x, 2) for x in move_probs]
            lines.append(f"Move Probabilities: {move_probs_str}")
        else:
            lines.append(f"Move Probabilities: {move_probs}")
        
        # Turn Probabilities  
        turn_probs = info.get('turn_probs', 'N/A')
        if isinstance(turn_probs, list):
            turn_probs_str = [round(x, 2) for x in turn_probs]
            lines.append(f"Turn Probabilities: {turn_probs_str}")
        else:
            lines.append(f"Turn Probabilities: {turn_probs}")
        
        # Action Params
        action_params = info.get('action_params', 'N/A')
        if isinstance(action_params, list):
            action_params_str = [round(x, 2) for x in action_params]
            lines.append(f"Action Params: {action_params_str}")
        else:
            lines.append(f"Action Params: {action_params}")
        
        # Value Estimation - 安全处理数值格式化
        value = info.get('value', 'N/A')
        if isinstance(value, (int, float)):
            lines.append(f"Value Estimation: {value:.2f}")
        else:
            lines.append("Value Estimation: N/A")
        
        # Reward - 现在能正确处理了
        reward = info.get('reward', 'N/A')
        if isinstance(reward, (int, float)):
            lines.append(f"Reward: {reward:.2f}")
        else:
            lines.append(f"Reward: {reward}")
        
        # Step - 现在能正确处理了
        step = info.get('step', 'N/A')
        if isinstance(step, (int, float)):
            lines.append(f"Step: {step}")
        else:
            lines.append(f"Step: {step}")
        
        # Episode - 现在能正确处理了
        episode = info.get('episode', 'N/A')
        if isinstance(episode, (int, float)):
            lines.append(f"Episode: {episode}")
        else:
            lines.append(f"Episode: {episode}")
            
        # 其他文本
        lines.append(f"Move Action: {info.get('move_action', 'N/A')}")
        lines.append(f"Turn Action: {info.get('turn_action', 'N/A')}")
        lines.append(f"Move Step: {info.get('move_step', 'N/A')}")
        lines.append(f"Turn Angle: {info.get('turn_angle', 'N/A')}")

        # 为文本绘制背景矩形
        overlay = image.copy()
        text_start_x = 10
        
        # 计算文本区域的高度
        text_region_height = len(lines) * line_height + 20
        cv2.rectangle(overlay, (5, 10), (635, 10 + text_region_height), background_color, -1)
        
        # 添加半透明效果
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # 在图像上绘制每一行文本
        for i, line in enumerate(lines):
            y_pos = start_y + i * line_height
            cv2.putText(image, line, (text_start_x, y_pos), font, font_scale, text_color, text_thickness)
        
        return image
 
    def update_agent_info(self, info):
        """
        更新智能体信息
        """
        # 保存最新信息
        self._last_agent_info = info
        
        # 将更新请求放入队列
        try:
            self.gui_queue.put(('update_info', info), block=False)
        except queue.Full:
            pass

    def process_gui_updates(self):
        """
        处理GUI更新，应在主线程中定期调用
        """
        try:
            while True:
                # 非阻塞地从队列获取GUI更新请求
                msg_type, *args = self.gui_queue.get_nowait()
                
                if msg_type == 'update_image':
                    image, detections = args
                    self._update_image_display(image, detections)
                elif msg_type == 'update_info':
                    info = args[0]
                    # 更新当前图像的信息显示
                    if self.current_image is not None:
                        # 更新当前图像的信息显示
                        img_with_info = self._add_agent_info_to_image(self.current_image.copy(), info)
                        self.current_image = img_with_info
                        # 重新显示图像以包含最新信息
                        self.update_image_and_detections(self.current_image, self.detections)
        except queue.Empty:
            pass  # 队列为空，正常情况
        
        # 增加更新频率，改为每50ms更新一次（与原来相同，但如果之前较慢可调整）
        if self.root:
            self.root.after(50, self.process_gui_updates)  # 每50ms检查一次更新
    def _update_info_on_image(self, info):
        """
        删除:在图像上更新信息 - 此功能已集成到_update_image_display中
        """
        pass

    def _draw_detections(self, image):
        """
        在图像上绘制检测框
        """
        if self.detections:
            for detection in self.detections:
                bbox = detection['bbox']
                label = detection['label']
                score = detection['score']
                
                # 转换边界框坐标为整数
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制矩形框
                color = (0, 255, 0)  # 绿色框
                thickness = 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签和置信度文本
                text = f"{label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_color = (255, 255, 255)  # 白色文字
                text_thickness = 1
                
                # 获取文本框的尺寸
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # 绘制文本背景
                cv2.rectangle(image, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # 在图像上绘制文本
                cv2.putText(image, text, (x1, y1 - 5), font, 
                           font_scale, text_color, text_thickness)
        
        return image

    def run(self):
        """
        运行Tkinter主循环
        """
        self.init_gui()
        # 设置定期处理GUI更新
        self.root.after(50, self.process_gui_updates)
        self.root.mainloop()

def add_visualization_to_environment(TargetSearchEnvironment, visualizer):
    """
    为环境类添加可视化功能
    """
    original_step = TargetSearchEnvironment.step
    original_reset = TargetSearchEnvironment.reset
    
    def step_with_visualization(self, move_action, turn_action, move_forward_step=2, turn_angle=30):
        # 获取当前状态
        current_state = self.capture_screen()
        
        # 检测目标
        detections = self.detect_target(current_state)
        
        # 更新可视化
        visualizer.update_image_and_detections(current_state, detections)
        
        # 执行原始step
        result = original_step(self, move_action, turn_action, move_forward_step, turn_angle)
        
        # 新状态
        new_state, reward, done, new_detections = result
        
        # 更新可视化信息 - 确保传递step和episode信息
        visualizer.update_agent_info({
            'step': self.step_count,
            'episode': getattr(self, '_current_episode', 0),  # 确保使用_current_episode
            'reward': reward,
            'move_action': move_action,
            'turn_action': turn_action,
            'move_forward_step': move_forward_step,
            'turn_angle': turn_angle,
            'detections': len(new_detections),
            'done': done
        })
        
        return result
    
    def reset_with_visualization(self):
        result = original_reset(self)
        
        # 更新可视化
        visualizer.update_image_and_detections(result, [])
        visualizer.update_agent_info({
            'step': 0,
            'episode': getattr(self, '_current_episode', 0),  # 确保使用_current_episode
            'reset': True
        })
        
        return result
    
    # 替换环境的方法
    TargetSearchEnvironment.step = step_with_visualization
    TargetSearchEnvironment.reset = reset_with_visualization

def add_visualization_to_agent(PPOAgent, visualizer):
    """
    为智能体添加可视化功能 - 修复版本
    """
    original_act = PPOAgent.act
    
    def act_with_visualization(self, state, memory, return_debug_info=False):
        # 使用旧策略获取动作概率
        result = original_act(self, state, memory, return_debug_info)
        
        if return_debug_info:
            move_action, turn_action, move_forward_step, turn_angle, debug_info = result
            
            # 提取概率信息
            import numpy as np
            move_probs = debug_info.get('move_logits', [0, 0, 0, 0])
            turn_probs = debug_info.get('turn_logits', [0, 0])
            action_params = debug_info.get('action_params', [0, 0])
            value = debug_info.get('value', [[0]])[0][0]
            
            # 使用tolist()方法安全转换numpy数组为Python列表
            move_probs_list = np.asarray(move_probs).flatten().tolist()
            turn_probs_list = np.asarray(turn_probs).flatten().tolist()
            action_params_list = np.asarray(action_params).flatten().tolist()
            
            # 四舍五入到2位小数
            move_probs_list = [round(x, 2) for x in move_probs_list]
            turn_probs_list = [round(x, 2) for x in turn_probs_list]
            action_params_list = [round(x, 2) for x in action_params_list]

            # 获取环境信息
            current_env = None
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, TargetSearchEnvironment):
                    current_env = obj
                    break

            # 获取最新的奖励信息 - 修复：使用当前环境的last_reward
            current_reward = getattr(current_env, 'last_reward', 0) if current_env else 0

            # 更新可视化信息
            visualizer.update_agent_info({
                'move_probs': move_probs_list,
                'turn_probs': turn_probs_list,
                'action_params': action_params_list,
                'value': float(value),
                'move_action': move_action,
                'turn_action': turn_action,
                'move_step': round(move_forward_step, 2),
                'turn_angle': round(turn_angle, 2),
                'step': getattr(current_env, 'step_count', 'N/A'),
                'episode': getattr(current_env, '_current_episode', 'N/A'),
                'reward': current_reward  # 使用环境的last_reward作为当前奖励
            })
        else:
            move_action, turn_action, move_forward_step, turn_angle = result
            
            # 获取环境信息
            current_env = None
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, TargetSearchEnvironment):
                    current_env = obj
                    break

            # 获取最新的奖励信息 - 修复：使用环境的last_reward
            current_reward = getattr(current_env, 'last_reward', 0) if current_env else 0

            # 更新可视化信息
            visualizer.update_agent_info({
                'move_action': move_action,
                'turn_action': turn_action,
                'move_step': round(move_forward_step, 2),
                'turn_angle': round(turn_angle, 2),
                'step': getattr(current_env, 'step_count', 'N/A'),
                'episode': getattr(current_env, '_current_episode', 'N/A'),
                'reward': current_reward  # 使用环境的last_reward作为当前奖励
            })
        
        return result
    
    PPOAgent.act = act_with_visualization


class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description=None):
        # 如果没有传入target_description，使用默认配置
        if target_description is None:
            target_description = CONFIG['TARGET_DESCRIPTION']

        self.last_reward = 0
        
        from control_api_tool import ImprovedMovementController
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = CONFIG['ENV_MAX_STEPS']
        self.last_detection_result = None
        self.last_area = 0
        self.logger = setup_logging()
        
        # 添加存储最近检测图像的队列 - 移到这里，确保在调用检测相关方法前已初始化
        self.recent_detection_images = deque(maxlen=5)
        
        self.yolo_model = self._load_yolo_model()
        self._warm_up_detection_model()
        
        # 成功条件阈值
        self.MIN_GATE_AREA = CONFIG['MIN_GATE_AREA']
        self.CENTER_THRESHOLD = CONFIG['CENTER_THRESHOLD']
        
        # 动作历史记录
        self.action_history = deque(maxlen=10)     
    def _load_yolo_model(self):
        """
        加载YOLO模型
        """
        current_dir = Path(__file__).parent
        model_path = current_dir.parent / "models" / "find_gate.pt"
        
        if not model_path.exists():
            self.logger.error(f"YOLO模型文件不存在: {model_path}")
            return None
        
        try:
            model = YOLO(str(model_path))
            self.logger.info(f"成功加载YOLO模型: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载YOLO模型失败: {e}")
            return None

    def _warm_up_detection_model(self):
        """
        预热检测模型
        """
        self.logger.info("正在预热检测模型，确保模型已加载...")
        try:
            dummy_image = self.capture_screen()
            if dummy_image is not None and dummy_image.size > 0:
                dummy_result = self.detect_target(dummy_image)
                self.logger.info("检测模型已预热完成")
            else:
                self.logger.warning("无法获取初始截图进行模型预热")
        except Exception as e:
            self.logger.warning(f"模型预热过程中出现错误: {e}")

    def capture_screen(self):
        """
        截取当前屏幕画面
        """
        try:
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            from computer_server.prtsc import capture_window_by_title
            result = capture_window_by_title("Sifu", "sifu_window_capture.png")
            if result:
                screenshot = Image.open("sifu_window_capture.png")
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            self.logger.warning("截图功能不可用，使用模拟图片")
            # 使用配置文件中的图像尺寸
            return np.zeros((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 3), dtype=np.uint8)

    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []
        
        try:
            results = self.yolo_model.predict(
                source=image,
                conf=CONFIG['DETECTION_CONFIDENCE'],
                save=False,
                verbose=False
            )
            
            detections = []
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'label': class_name,
                        'score': conf,
                        'width': width,
                        'height': height
                    })
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []
    def _check_climb_conditions(self, detection_results):
        """
        检查climb检测结果是否满足条件
        """
        confidence_threshold = CONFIG.get('CLIMB_CONFIDENCE_THRESHOLD', 0.85)
        if not detection_results:
            return False
        
        # 筛选出所有climb检测结果
        climb_detections = [
            detection for detection in detection_results
            if (
                detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            )
        ]
        
        if not climb_detections:
            return False
        
        # 如果只有一个climb检测，要求其置信度超过阈值
        if len(climb_detections) == 1:
            return climb_detections[0]['score'] > confidence_threshold
        # 如果有多个climb检测，要求至少有一个置信度超过阈值
        else:
            return any(detection['score'] > confidence_threshold for detection in climb_detections)

    def _calculate_target_presence_reward(self, detection_results):
        """
        基于目标检测的奖励
        """
        if not detection_results:
            return -0.05  # 未检测到任何目标的惩罚
        
        # 检测到gate给予较高奖励
        gate_detected = any(
            'gate' in detection['label'].lower() or detection['label'].lower() == 'gate'
            for detection in detection_results
        )
        
        if gate_detected:
            return 0.1
        
        # 检测到其他目标的奖励
        return 0.02

    def _calculate_progress_reward(self, detection_results):
        """
        基于目标检测进展的奖励
        """
        if not detection_results:
            return -0.02
        else:
            return 0.02

    def _calculate_repetition_penalty(self, action_taken):
        """
        重复动作惩罚
        """
        if not action_taken or len(self.action_history) < 3:
            return 0.0
        
        recent_actions = list(self.action_history)[-3:]
        same_action_count = sum(1 for act in recent_actions if act == action_taken)
        
        if same_action_count >= 2:
            return -0.1
        elif same_action_count >= 1:
            return -0.05
            
        return 0.0

    def _get_max_detection_area(self, detection_results):
        """
        获取检测结果中的最大面积
        """
        if not detection_results:
            return 0
        return max([det['width'] * det['height'] for det in detection_results])

    def reset_to_origin(self):
        """
        重置到原点操作
        """
        self.logger.info("执行重置到原点操作...")
        
        # 按键操作序列
        pyautogui.press('esc')
        time.sleep(0.2)
        pyautogui.press('q')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        
        # 检测门是否存在
        gate_detected = False
        while not gate_detected:
            new_state = self.capture_screen()
            detection_results = self.detect_target(new_state)
            
            for detection in detection_results:
                if detection['label'].lower() == 'gate' or 'gate' in detection['label'].lower():
                    gate_detected = True
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(0.3)
                    pyautogui.press('enter')
                    break
            
            if not gate_detected:
                self.logger.info(f"未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.2)
               
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.2)
        pyautogui.press('enter')
        time.sleep(0.2)
        
        # 重置环境内部状态
        self.step_count = 0
        self.last_area = 0
        self.last_detection_result = None
        self.action_history.clear()
        self.logger.info("重置到原点操作完成")

    def step(self, move_action, turn_action, move_forward_step=2, turn_angle=30):
        """
        执行动作并返回新的状态、奖励和是否结束
        """
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        self.logger.debug(f"执行动作: 移动-{move_action_names[move_action]}, 转头-{turn_action_names[turn_action]}, 步长: {move_forward_step}, 角度: {turn_angle}")
        
        # 执行动作前先检查当前状态
        pre_action_state = self.capture_screen()
        pre_action_detections = self.detect_target(pre_action_state)
        
        # 检查当前状态是否有符合条件的climb类别
        pre_climb_detected = self._check_climb_conditions(pre_action_detections)
        
        if pre_climb_detected:
            self.logger.info(f"动作执行前已检测到符合条件的climb类别，立即终止")
            
            # 保存带识别框的图片
           
            # 完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            speed_bonus = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8) * (self.max_steps - self.step_count)
            reward = base_completion_reward + speed_bonus
            
            new_area = self._get_max_detection_area(pre_action_detections)
            self.last_area = new_area
            self.last_detection_result = pre_action_detections
            self.step_count += 1
            
            # 关键：执行游戏重置操作
            self.reset_to_origin()
            
            return pre_action_state, reward, True, pre_action_detections
        
        # 执行移动动作
        if move_action == 0 and move_forward_step > 0:  # forward
            self.controller.move_forward(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 1 and move_forward_step > 0:  # backward
            self.controller.move_backward(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 2 and move_forward_step > 0:  # strafe_left
            self.controller.strafe_left(duration=move_forward_step * CONFIG['forward_coe'])
        elif move_action == 3 and move_forward_step > 0:  # strafe_right
            self.controller.strafe_right(duration=move_forward_step * CONFIG['forward_coe'])
        
        # 执行转头动作
        if turn_action == 0 and turn_angle > 0:  # turn_left
            self.controller.turn_left(turn_angle * CONFIG['turn_coe'], duration=1)
        elif turn_action == 1 and turn_angle > 0:  # turn_right
            self.controller.turn_right(turn_angle * CONFIG['turn_coe'], duration=1)

        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        

        # 计算奖励
        reward, new_area = self.calculate_reward(
            detection_results, 
            self.last_area,
            (move_action, turn_action)
        )
        self.last_reward = reward
        # 更新状态
        self.last_area = new_area
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否检测到符合条件的climb类别
        climb_detected = self._check_climb_conditions(detection_results)

        done = climb_detected or self.step_count >= self.max_steps
        
        # 如果检测到符合条件的climb，给予额外奖励
        if climb_detected:
            # 保存带识别框的图片
            
            
            # 基础完成奖励
            base_completion_reward = CONFIG.get('BASE_COMPLETION_REWARD', 250)
            
            # 快速完成奖励
            quick_completion_factor = CONFIG.get('QUICK_COMPLETION_BONUS_FACTOR', 8)
            quick_completion_bonus = quick_completion_factor * (self.max_steps - self.step_count) 
            
            # 总奖励
            total_completion_bonus = base_completion_reward + quick_completion_bonus
            reward += total_completion_bonus
            self.logger.info(f"检测到符合条件的climb类别！步骤: {self.step_count}, 基础奖励: {base_completion_reward:.2f}, "
                            f"快速完成奖励: {quick_completion_bonus:.2f}, 总奖励: {total_completion_bonus:.2f}")
            
            # 关键：执行游戏重置操作
            self.reset_to_origin()

        # 修改:现在在计算完奖励后才输出日志
        self.logger.info(f"S {self.step_count}, A: {new_area:.2f}, R: {reward:.2f}")
        
        # 更新动作历史
        self.action_history.append((move_action, turn_action))
        
        # 修改：只有在达到最大步数时才重置，避免重复重置
        if done and not climb_detected:  # 如果不是因为climb而结束，也要重置
            self.logger.info(f"达到最大步数 {self.max_steps}")
            # 在episode结束时重置游戏环境
            self.reset_to_origin()
        
        return new_state, reward, done, detection_results 

    def calculate_reward(self, detection_results, last_area, action_taken):
        """
        计算综合奖励
        """
        # 目标检测奖励
        target_reward = self._calculate_target_presence_reward(detection_results)
        
        # 进展奖励
        progress_reward = self._calculate_progress_reward(detection_results)
        
        # 重复动作惩罚
        repetition_penalty = self._calculate_repetition_penalty(action_taken)
        
        # 总奖励
        total_reward = target_reward + progress_reward + repetition_penalty
        
        # 计算新区域面积
        new_area = self._get_max_detection_area(detection_results) if detection_results else 0
        
        return total_reward, new_area

    def _save_detection_image_with_bounding_boxes(self, image, detection_results, prefix="detection"):
        """
        保存带检测框的图像，并保留最近5个
        """
        try:
            # 复制图像以避免修改原始图像
            img_with_boxes = image.copy()
            
            # 绘制检测框
            for detection in detection_results:
                bbox = detection['bbox']
                label = detection['label']
                score = detection['score']
                
                # 转换边界框坐标为整数
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制矩形框
                color = (0, 255, 0)  # 绿色框
                thickness = 2
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签和置信度文本
                text = f"{label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_color = (255, 255, 255)  # 白色文字
                text_thickness = 1
                
                # 获取文本框的尺寸
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # 绘制文本背景
                cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # 在图像上绘制文本
                cv2.putText(img_with_boxes, text, (x1, y1 - 5), font, 
                        font_scale, text_color, text_thickness)
            
            # 生成文件名
            timestamp = int(time.time() * 1000)  # 使用毫秒时间戳
            filename = f"{prefix}_detection_{timestamp}.png"
            
            # 确定保存路径
            save_dir = Path(__file__).parent / "detection_images"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = save_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), img_with_boxes)
            
            if success:
                
                
                # 将文件路径添加到最近检测图像队列
                self.recent_detection_images.append(str(filepath))
            else:
                self.logger.error(f"保存检测结果图像失败: {filepath}")
                
        except Exception as e:
            self.logger.error(f"保存带检测框图像时出错: {e}")

    def get_recent_detection_images(self):
        """
        获取最近5个检测图像的路径
        """
        return list(self.recent_detection_images)

    def reset(self):
        """
        重置环境
        """
        self.logger.debug("重置环境")
        self.step_count = 0
        self.last_area = 0
        self.last_detection_result = None
        self.action_history.clear()
        initial_state = self.capture_screen()
        return initial_state

class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        
        # 添加logger
        self.logger = setup_logging()
        
        # Create policy networks
        self.policy = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            betas=self.betas
        )
        self.policy_old = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def act(self, state, memory, return_debug_info=False):
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        
        # 使用旧策略获取动作概率
        with torch.no_grad():
            if return_debug_info:
                move_probs, turn_probs, action_params, state_val, debug_info = self.policy_old(state_tensor.unsqueeze(0), return_debug_info=True)
            else:
                move_probs, turn_probs, action_params, state_val = self.policy_old(state_tensor.unsqueeze(0))
                
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            # 采样动作
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            logprob = move_logprob + turn_logprob
        
        # 处理动作参数
        move_forward_step_raw = action_params[0][0].item()  # [-1,1]范围
        turn_angle_raw = action_params[0][1].item()         # [-1,1]范围
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 将 [-1, 1] 映射到实际范围
        move_forward_step = ((move_forward_step_raw + 1.0) / 2.0) * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = ((turn_angle_raw + 1.0) / 2.0) * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN
        
        # 存储到记忆中
        memory.append(
            state_tensor,
            move_action.item(),
            turn_action.item(),
            logprob.item(),
            0,  # 奖励稍后更新
            False,  # 是否结束稍后更新
            [move_forward_step, turn_angle]  # 存储参数
        )
        
        if return_debug_info:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle, debug_info
        else:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def update(self, memory):
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 转换为张量 - 注意logprobs需要特殊处理
        old_states = torch.stack(memory.states).detach()
        old_actions_move = torch.LongTensor(memory.move_actions).detach()
        old_actions_turn = torch.LongTensor(memory.turn_actions).detach()
        
        # 修改这一行：将logprobs转换为tensor
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).detach()
        
        # 修改这一行：正确处理action_params
        if memory.action_params:
            # 将列表转换为tensor
            action_params_tensor = torch.tensor(memory.action_params, dtype=torch.float32)
            old_values = action_params_tensor.detach()
        else:
            old_values = None

        # PPO更新
        for _ in range(self.K_epochs):
            # 前向传播
            move_probs, turn_probs, action_params, state_vals = self.policy(old_states)

            # 计算对数概率
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            logprobs = move_dist.log_prob(old_actions_move) + turn_dist.log_prob(old_actions_turn)
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算优势
            advantages = rewards - state_vals.squeeze().detach()

            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = self.MseLoss(state_vals.squeeze(), rewards)

            # 熵损失
            move_entropy = move_dist.entropy().mean()
            turn_entropy = turn_dist.entropy().mean()
            entropy_loss = move_entropy + turn_entropy

            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        memory.clear_memory()  
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        expected_height = CONFIG.get('IMAGE_HEIGHT', 480)
        expected_width = CONFIG.get('IMAGE_WIDTH', 640)
        
        if state_rgb.shape[0] != expected_height or state_rgb.shape[1] != expected_width:
            state_rgb = cv2.resize(state_rgb, (expected_width, expected_height))
        
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        
        return state_tensor

    def save_checkpoint(self, filepath, episode, optimizer_state_dict=None):
        """
        保存模型检查点
        """
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': optimizer_state_dict or self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型检查点已保存: {filepath}")

    def load_checkpoint(self, filepath):
        """
        加载模型检查点
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint.get('episode', 0)
            self.logger.info(f"模型检查点已加载，从第 {start_episode} 轮开始继续训练")
            return start_episode + 1
        else:
            self.logger.info(f"检查点文件不存在: {filepath}")
            return 0


def find_latest_checkpoint(model_path):
    """
    查找最新的检查点文件
    """
    model_dir = os.path.dirname(model_path)
    model_base_name = os.path.basename(model_path).replace('.pth', '')
    
    # 查找所有相关的检查点文件
    checkpoint_pattern = re.compile(rf'{re.escape(model_base_name)}_checkpoint_ep_(\d+)\.pth$')
    found_checkpoints = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            match = checkpoint_pattern.match(file)
            if match:
                full_path = os.path.join(model_dir, file)
                epoch_num = int(match.group(1))
                found_checkpoints.append((full_path, epoch_num))
    
    if found_checkpoints:
        # 返回具有最高epoch编号的检查点
        latest_checkpoint = max(found_checkpoints, key=lambda x: x[1])
        return latest_checkpoint[0]
    
    return None


def create_environment_and_agent():
    """
    创建环境和智能体
    """
    # 创建可视化器
    visualizer = RealTimeVisualizer()
    
    # 创建环境
    env = TargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    
    # 为环境添加可视化功能
    add_visualization_to_environment(TargetSearchEnvironment, visualizer)
    
    move_action_dim = 4  # forward, backward, strafe_left, strafe_right
    turn_action_dim = 2  # turn_left, turn_right
    
    ppo_agent = PPOAgent(
        (3, CONFIG.get('IMAGE_HEIGHT', 480), CONFIG.get('IMAGE_WIDTH', 640)), 
        move_action_dim,
        turn_action_dim
    )
    
    # 为智能体添加可视化功能
    add_visualization_to_agent(PPOAgent, visualizer)
    
    return env, ppo_agent, visualizer


def load_model(ppo_agent, model_path):
    """
    加载模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def initialize_model(model_path, load_existing=True):
    """
    初始化模型，包括加载已存在的模型或创建新模型
    """
    env, ppo_agent, visualizer = create_environment_and_agent()
    
    start_episode = 0
    
    if load_existing:
        # 首先查找检查点文件
        latest_checkpoint = find_latest_checkpoint(model_path)
        
        if latest_checkpoint:
            # 如果找到检查点，优先加载检查点
            start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        elif os.path.exists(model_path):
            # 没有检查点但主模型存在，加载主模型
            if not load_model(ppo_agent, model_path):
                raise Exception(f"加载模型失败: {model_path}")
            start_episode = 0
    
    return env, ppo_agent, visualizer, start_episode


def get_deterministic_action(ppo_agent, state):
    """
    在评估模式下获取确定性动作
    """
    state_tensor = ppo_agent._preprocess_state(state)
    
    # 使用旧策略获取动作概率（但不采样，而是选择最高概率的动作）
    with torch.no_grad():
        move_probs, turn_probs, action_params, state_val = ppo_agent.policy_old(state_tensor.unsqueeze(0))
        
        # 确保处理正确的张量维度
        move_probs_squeezed = move_probs.squeeze() if move_probs.dim() > 1 else move_probs
        turn_probs_squeezed = turn_probs.squeeze() if turn_probs.dim() > 1 else turn_probs
        
        # 选择最高概率的动作
        move_action = torch.argmax(move_probs_squeezed, dim=-1).item()
        turn_action = torch.argmax(turn_probs_squeezed, dim=-1).item()
        
        # 获取动作参数
        move_forward_step_raw = action_params[0][0].item()
        turn_angle_raw = action_params[0][1].item()
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 映射到实际范围
        move_forward_step = (torch.tanh(torch.tensor(move_forward_step_raw)) + 1) / 2 * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = (torch.tanh(torch.tensor(turn_angle_raw)) + 1) / 2 * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN

    return move_action, turn_action, move_forward_step, turn_angle


def run_episode(env, ppo_agent, visualizer, episode_num, total_episodes, training_mode=True, print_debug=False):
    """
    运行单个episode，确保在结束时重置环境
    """
    # 设置当前episode号
    env._current_episode = episode_num
    
    state = env.reset()  # 重置环境
    total_reward = 0
    step_count = 0
    done = False
    final_area = 0
    success_flag = False
    
    # 为每个episode创建独立的记忆
    episode_memory = Memory()
    
    # 定义动作名称映射
    move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
    turn_action_names = ["turn_left", "turn_right"]
    
    # 添加最大步数限制，防止无限循环
    max_steps = CONFIG.get('MAX_STEPS_PER_EPISODE', 1000)
    
    while not done and step_count < max_steps:
        if training_mode:
            # 训练模式：使用act方法
            if print_debug:
                move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(
                    state, episode_memory, return_debug_info=True)
                
                # 打印调试信息
                if 'move_probs' in debug_info:
                    print(f"move_probs shape: {debug_info['move_probs'].shape}")
                    print(f"turn_probs shape: {debug_info['turn_probs'].shape}")
                    print(f"move_action: {move_action}, turn_action: {turn_action}")
            else:
                move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(
                    state, episode_memory)
        else:
            # 评估模式：使用确定性动作
            move_action, turn_action, move_forward_step, turn_angle = get_deterministic_action(ppo_agent, state)
        
        # 执行环境步骤
        next_state, reward, done, detection_results = env.step(
            move_action, turn_action, move_forward_step, turn_angle)
        
        # 更新episode_memory中的奖励和终端状态
        if len(episode_memory.rewards) > 0:
            episode_memory.rewards[-1] = reward
            episode_memory.is_terminals[-1] = done
        
        # 更新状态
        state = next_state
        total_reward += reward
        step_count += 1
        
        # 更新可视化信息
        visualizer.update_agent_info({
            'episode': episode_num,  # 明确传递episode号
            'step': step_count,      # 传递当前step数
            'total_reward': total_reward,
            'reward': reward,        # 传递当前reward
            'done': done,
            'success': done and any(
                detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                for detection in detection_results
            ),
            'move_action': move_action,
            'turn_action': turn_action,
            'move_step': round(move_forward_step, 2),
            'turn_angle': round(turn_angle, 2)
        })
        
        # 记录最终检测面积
        if detection_results:
            final_area = max(d['width'] * d['height'] for d in detection_results 
                           if 'width' in d and 'height' in d)
    
    # 检查是否成功找到目标
    if detection_results:
        climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in detection_results
        )
        success_flag = climb_detected
    
    # 在训练模式下，需要更新智能体
    if training_mode and len(episode_memory.rewards) > 0:
        # 更新智能体
        ppo_agent.update(episode_memory)
    
    # 获取最近的检测图像（如果需要的话）
    recent_detection_images = getattr(env, 'get_recent_detection_images', lambda: [])()
    
    # 确保在episode结束时重置环境
    # 注意：env.reset()已经在env.step()中被调用了，所以这里不需要再次调用
    # env.reset()
    
    return {
        'total_reward': total_reward,
        'step_count': step_count,
        'final_area': final_area,
        'success_flag': success_flag,
        'detection_results': detection_results,
        'recent_detection_images': recent_detection_images
    }

def perform_training_loop(env, ppo_agent, visualizer, start_episode, total_episodes):
    """
    执行训练循环
    """
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    # 训练统计变量
    training_stats = {
        'episode_count': 0,
        'successful_episodes': 0,
        'average_reward_history': []
    }
    
    print(f"开始训练循环，从第 {start_episode} 轮到第 {total_episodes} 轮")
    loop_start_time = time.time()
    
    for episode in range(start_episode, total_episodes):
        print_debug =   True  # 每50轮打印一次详细信息
        result = run_episode(env, ppo_agent, visualizer, episode, total_episodes, training_mode=True, print_debug=print_debug)
        
        # 更新统计数据
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        final_areas.append(result['final_area'])
        training_stats['episode_count'] += 1
        
        if result['success_flag']:
            training_stats['successful_episodes'] += 1
        
        # 每10轮保存一次检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = f"{CONFIG['MODEL_PATH'].rsplit('.', 1)[0]}_checkpoint_ep_{episode + 1}.pth"
            ppo_agent.save_checkpoint(checkpoint_path, episode + 1)
            print(f"检查点已保存: {checkpoint_path}")
        
        print(f"训练进度: {episode+1}/{total_episodes}")
        
      
        if episode % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - loop_start_time
            
            avg_reward = np.mean(total_rewards) if total_rewards else 0
            success_rate = training_stats['successful_episodes'] / training_stats['episode_count'] if training_stats['episode_count'] > 0 else 0
            
            print(f"Episode数 {episode}: 平均奖励: {avg_reward:.3f}, "
                  f"成功率: {success_rate:.3f}, "
                  f"总奖励: {result['total_reward']:.3f}, "
                  f"步数: {result['step_count']}, "
                  f"成功: {result['success_flag']}")
            print(f"当前训练耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    return training_stats


def continue_training_ppo_agent(model_path=None):
    """
    基于现有模型继续训练
    """
    config = CONFIG
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    print(f"基于现有模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    # 开始计时
    start_time = time.time()
    
    # 初始化模型
    env, ppo_agent, visualizer, start_episode = initialize_model(model_path, load_existing=True)
    
    total_training_episodes = start_episode + config['EPISODES']
    print(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 启动可视化线程
    def run_visualizer():
        visualizer.run()
    
    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, visualizer, start_episode, total_training_episodes)
    
    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"继续训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"更新后的PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮，耗时 {training_duration:.2f} 秒", 
        "final_episode": total_training_episodes,
        "training_stats": {
            "successful_episodes": training_stats['successful_episodes'],
            "total_episodes": training_stats['episode_count']
        },
        "training_duration": training_duration
    }


def train_new_ppo_agent(model_path=None):
    """
    从头开始训练PPO智能体
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    print(f"开始从头训练PPO智能体: {model_path}")
    
    # 开始计时
    start_time = time.time()
    
    # 初始化模型（不加载现有模型）
    env, ppo_agent, visualizer, start_episode = initialize_model(model_path, load_existing=False)
    
    total_training_episodes = CONFIG['EPISODES']
    print(f"从第 {start_episode} 轮开始，训练 {CONFIG['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 启动可视化线程
    def run_visualizer():
        visualizer.run()
    
    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, visualizer, start_episode, total_training_episodes)
    
    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"从头训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"新训练的PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"从头训练完成，共训练了 {CONFIG['EPISODES']} 轮，耗时 {training_duration:.2f} 秒", 
        "final_episode": total_training_episodes,
        "training_stats": {
            "successful_episodes": training_stats['successful_episodes'],
            "total_episodes": training_stats['episode_count']
        },
        "training_duration": training_duration
    }


def evaluate_trained_ppo_agent(model_path=None):
    """
    评估已训练的PPO模型
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    print(f"评估已训练的PPO模型: {model_path}")
    
    # 开始计时
    start_time = time.time()
    
    # 创建环境和智能体
    env, ppo_agent, visualizer = create_environment_and_agent()
    
    # 启动可视化线程
    def run_visualizer():
        visualizer.run()
    
    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()
    
    # 加载训练好的模型
    if not load_model(ppo_agent, model_path):
        return {"status": "error", "message": f"模型加载失败"}
    
    # 设置为评估模式
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    
    evaluation_episodes = CONFIG.get('EVALUATION_EPISODES', 10)
    scores = []
    total_rewards = []
    success_count = 0
    all_recent_detection_images = []  # 收集所有episode的检测图像
    
    print(f"开始评估，共 {evaluation_episodes} 个episode")
    
    for episode in range(evaluation_episodes):
        # 每隔几轮打印一次调试信息
        print_debug =   True # 每2轮打印一次调试信息
        result = run_episode(env, ppo_agent, visualizer, episode, evaluation_episodes, training_mode=False, print_debug=print_debug)
        
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        
        if result['success_flag']:
            success_count += 1
        
        # 收集最近检测图像
        all_recent_detection_images.extend(result['recent_detection_images'])
        
        print(f"Eval Ep: {episode+1}/{evaluation_episodes}, Steps: {result['step_count']}, "
              f"Reward: {result['total_reward']:.3f}, Success: {result['success_flag']}")
    
    # 计算评估指标
    avg_score = np.mean(scores) if scores else 0
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    success_rate = success_count / evaluation_episodes if evaluation_episodes > 0 else 0
    
    evaluation_result = {
        'avg_score': avg_score,
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'total_evaluated_episodes': evaluation_episodes,
        'success_count': success_count,
        'recent_detection_images': all_recent_detection_images[-5:]  # 只保留最后5个检测图像
    }
    
    # 结束计时
    end_time = time.time()
    evaluation_duration = end_time - start_time
    
    print(f"\n=== 评估结果 ===")
    print(f"平均步数: {avg_score:.3f}")
    print(f"平均奖励: {avg_reward:.3f}")
    print(f"成功率: {success_rate:.3f}")
    print(f"成功次数: {success_count}/{evaluation_episodes}")
    print(f"评估耗时: {evaluation_duration:.2f} 秒 ({evaluation_duration/60:.2f} 分钟)")
    
    print("模型评估完成！")
    
    return {
        "status": "success",
        "message": f"模型评估完成，耗时 {evaluation_duration:.2f} 秒",
        "evaluation_result": evaluation_result,
        "evaluation_duration": evaluation_duration
    }

def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    """
    
    tool_functions = {
        "continue_train_ppo_agent": continue_training_ppo_agent,
        "train_new_ppo_agent": train_new_ppo_agent,
        "evaluate_trained_ppo_agent": evaluate_trained_ppo_agent
    }
    
    if tool_name not in tool_functions:
        error_msg = f"错误: 未知的PPO工具 '{tool_name}'"
        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        result = tool_functions[tool_name](*args)
        print(f"PPO工具执行成功: {tool_name}")
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"执行PPO工具时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}


def train_with_visualization():
    """
    带可视化的训练函数
    """
    # 创建可视化器
    visualizer = RealTimeVisualizer()
    
    # 为环境和智能体添加可视化功能
    add_visualization_to_environment(TargetSearchEnvironment, visualizer)
    
    # 初始化环境和智能体
    env = TargetSearchEnvironment()
    
    state_dim = (3, CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'])
    move_action_dim = 4
    turn_action_dim = 2
    agent = PPOAgent(state_dim, move_action_dim, turn_action_dim)
    
    # 为智能体添加可视化功能
    add_visualization_to_agent(PPOAgent, visualizer)
    
    # **关键修改：在单独线程中启动可视化**
    def run_visualizer():
        visualizer.run()
    
    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()
    
    # 训练循环
    for episode in range(0, 2001):
        print(f"\n=== Episode {episode} Started ===")
        
        # 设置当前episode号
        env._current_episode = episode
        
        state = env.reset()
        memory = Memory()
        total_reward = 0
        
        for t in range(env.max_steps):
            # 智能体执行动作
            move_action, turn_action, move_step, turn_angle, debug_info = agent.act(
                state, memory, return_debug_info=True
            )
            
            # 执行环境步骤
            next_state, reward, done, detections = env.step(
                move_action, turn_action, move_step, turn_angle
            )
            
            # 更新记忆
            memory.rewards[-1] = reward
            memory.is_terminals[-1] = done
            
            total_reward += reward
            state = next_state
            
            # 更新可视化信息
            visualizer.update_agent_info({
                'episode': episode,     # 明确传递episode号
                'step': t,             # 传递当前step数
                'total_reward': total_reward,
                'reward': reward,      # 传递当前reward
                'done': done,
                'move_action': move_action,
                'turn_action': turn_action,
                'move_step': round(move_step, 2),
                'turn_angle': round(turn_angle, 2)
            })
            
            if done:
                break
        
        # 更新策略
        agent.update(memory)
        
        # 记录训练信息
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t+1}")
        
        # 保存检查点
        if episode % 10 == 0:
            agent.save_checkpoint(f'ppo_model_checkpoint_ep{episode}.pth')
def main():
    """
    主函数，用于直接运行此脚本
    """
    import sys
    if len(sys.argv) < 2:
        print("默认运行继续训练...")
        continue_training_ppo_agent()
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: python ppo_agents.py train_new_ppo_agent [model_path]")
        print("2. 继续训练门搜索智能体: python ppo_agents.py continue_train_ppo_agent [model_path]")
        print("3. 评估已训练模型: python ppo_agents.py evaluate_trained_ppo_agent [model_path]")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(str(response))


if __name__ == "__main__":
    main()