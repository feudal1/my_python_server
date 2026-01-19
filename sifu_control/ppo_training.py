"""
PPO强化学习智能体 - 集成Ground-SAM特征提取

本模块实现了基于近端策略优化(PPO)的强化学习智能体,用于Sifu游戏的自动寻路任务。
集成了Ground-SAM视觉特征提取器,支持在训练过程中进行SAM掩码可视化。

主要功能:
- PPO算法实现: 包含Actor-Critic网络、经验回放、策略更新
- Ground-SAM集成: 使用冻结的SAM和Grounding DINO模型进行特征提取
- SAM掩码可视化: 支持在训练过程中自动保存SAM分割结果
- 多环境支持: 支持Sifu游戏环境和测试环境

SAM掩码可视化使用方法:
    注意: 为提升训练速度,默认已禁用掩码绘制(visualize_mask=False)。
    如需调试,可在代码中手动启用可视化:

    sam_embedding = self._extract_sam_embedding(
        image_np,
        visualize_mask=True,                      # 启用掩码可视化(调试用)
        save_path='./output/sam_masks.png'       # 保存路径
    )

    独立演示SAM掩码可视化:
    python demo_show_sam_masks.py --image your_image.jpg --output result.png

详细文档:
- Ground-SAM: ../grounding_dino/GROUND_SAM_README.md
- SAM掩码: ../grounding_dino/SAM_MASKS_README.md
"""

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
import h5py
import pickle
from datetime import datetime

# 导入Ground-SAM特征提取器
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'grounding_dino'))
try:
    from ground_sam_feature_extractor import get_groundsam_extractor, GroundSAMFeatureExtractor
    GROUND_SAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Ground-SAM不可用: {e}")
    GROUND_SAM_AVAILABLE = False


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config 

CONFIG = load_config()

# 全局Ground-SAM特征提取器实例
_ground_sam_extractor = None

def get_global_ground_sam_extractor(force_reload=False):
    """
    获取全局Ground-SAM特征提取器实例(单例模式)
    Args:
        force_reload: 是否强制重新加载
    Returns:
        GroundSAMFeatureExtractor实例,如果不可用则返回None
    """
    global _ground_sam_extractor
    
    if not GROUND_SAM_AVAILABLE:
        return None
    
    if _ground_sam_extractor is None or force_reload:
        try:
            logger = setup_logging()
            logger.info("正在初始化Ground-SAM特征提取器...")
            
            # 从配置中读取模型设置
            sam_model_type = CONFIG.get('SAM_MODEL_TYPE', 'vit_b')
            dino_model_name = CONFIG.get('DINO_MODEL_NAME', 'IDEA-Research/grounding-dino-tiny')
            
            _ground_sam_extractor = get_groundsam_extractor(
                sam_model_type=sam_model_type,
                dino_model_name=dino_model_name
            )
            logger.info(f"Ground-SAM特征提取器初始化完成 (SAM: {sam_model_type}, DINO: {dino_model_name})")
        except Exception as e:
            logger.error(f"初始化Ground-SAM失败: {e}")
            _ground_sam_extractor = None
    
    return _ground_sam_extractor


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


class SLAMMapBuilder:
    """
    基于Ground-SAM特征的轻量级SLAM地图构建器
    构建语义占用栅格地图
    """
    def __init__(self, map_size=100, resolution=0.1):
        """
        Args:
            map_size: 地图尺寸（栅格数）
            resolution: 每个栅格对应的实际距离（米）
        """
        self.map_size = map_size
        self.resolution = resolution
        self.center = map_size // 2  # 当前位置始终在地图中心

        # 占用栅格：0=未知, 1=空闲, 2=障碍物, 3=门
        self.occupancy_grid = np.zeros((map_size, map_size), dtype=np.int8)

        # 语义地图：存储每个栅格的语义类别
        self.semantic_grid = np.zeros((map_size, map_size), dtype=np.int8)

        # 访问频率：用于衡量探索程度
        self.visit_count = np.zeros((map_size, map_size), dtype=np.int32)

        # 历史路径：存储访问过的位置
        self.path_history = []
        self.max_path_length = 500

        # 统计信息
        self.total_cells = map_size * map_size
        self.explored_cells = 0
        self.obstacle_cells = 0
        self.door_cells = 0

        # 添加：追踪当前位置和朝向
        self.current_map_x = self.center
        self.current_map_y = self.center
        self.current_heading = 0  # 朝向角度（弧度）
        self.current_step_count = 0

    def update(self, detections, move_action, move_step, turn_action, turn_angle, current_image=None):
        """
        根据检测结果和动作更新地图
        使用视觉里程计（Visual Odometry）方法：通过追踪物体相对位置变化来推断实际移动
        Args:
            detections: YOLO检测结果
            move_action: 移动动作索引
            move_step: 移动步长
            turn_action: 转向动作索引
            turn_angle: 转向角度
            current_image: 当前图像（用于检测障碍物）
        Returns:
            地图特征向量（用于PPO状态）
        """
        # 保存上一帧的检测数据用于视觉里程计
        if not hasattr(self, 'prev_detections'):
            self.prev_detections = []
            self.prev_heading = 0

        # 打印检测到的物体
        logger = setup_logging()
        if len(detections) == 0:
            logger.debug(f"[SLAM] YOLO检测到 0 个物体")
        else:
            logger.info(f"[SLAM] YOLO检测到 {len(detections)} 个物体:")
            for i, det in enumerate(detections):
                label = det.get('label', 'unknown')
                bbox = det.get('bbox', [])
                cx = (bbox[0] + bbox[2]) / 2 if len(bbox) >= 4 else 0
                cy = (bbox[1] + bbox[3]) / 2 if len(bbox) >= 4 else 0
                score = det.get('score', 0)
                logger.info(f"  物体{i}: {label}, 中心点=({cx:.1f}, {cy:.1f}), 置信度={score:.3f}")

        # 检测环境中的障碍物（基于图像分析）
        obstacle_points = self._detect_obstacles_from_image(current_image) if current_image is not None else []
        logger.info(f"[SLAM] 从图像中检测到 {len(obstacle_points)} 个障碍物特征点")

        # 使用视觉里程计估计实际移动
        if len(self.prev_detections) > 0 and len(detections) > 0:
            # 计算物体相对位置的变化来估计位移
            dx, dy, dheading = self._estimate_visual_odometry(self.prev_detections, detections)

            logger.info(f"[SLAM] 视觉里程计: dx={dx:.3f}, dy={dy:.3f}, 当前位置=({self.current_map_x:.1f}, {self.current_map_y:.1f})")

            # 如果检测到显著变化，更新位置
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                self.current_map_x += dx
                self.current_map_y += dy

            # 估计转向（基于动作命令，因为转向通常是可靠的）
            if turn_action == 0:  # turn_left
                dheading = np.radians(turn_angle)
            elif turn_action == 1:  # turn_right
                dheading = -np.radians(turn_angle)
            else:
                dheading = 0

            self.current_heading += dheading
            self.current_heading = self.current_heading % (2 * np.pi)

        # 限制位置在地图范围内
        self.current_map_x = np.clip(self.current_map_x, 2, self.map_size - 3)
        self.current_map_y = np.clip(self.current_map_y, 2, self.map_size - 3)

        current_pos = (int(self.current_map_x), int(self.current_map_y))

        # 检测结果映射到地图（相对于当前位置和朝向）
        for detection in detections:
            # 将屏幕坐标映射到地图坐标
            screen_center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2
            screen_center_y = (detection['bbox'][1] + detection['bbox'][3]) / 2

            # 转换为相对于屏幕中心的偏移
            offset_x = (screen_center_x - 320) / 320.0  # 归一化 [-1, 1]
            offset_y = (screen_center_y - 240) / 240.0  # 归一化 [-1, 1]

            # 根据朝向旋转偏移（反向向量）
            rotated_offset_x = offset_x * np.cos(self.current_heading) - offset_y * np.sin(self.current_heading)
            rotated_offset_y = offset_x * np.sin(self.current_heading) + offset_y * np.cos(self.current_heading)

            # 映射到地图坐标（前方为正）
            map_x = int(self.current_map_x + rotated_offset_x * 20)
            map_y = int(self.current_map_y - rotated_offset_y * 20)  # Y轴反转

            logger.info(f"[SLAM] 反向向量计算: 屏幕中心=({screen_center_x:.1f}, {screen_center_y:.1f}), "
                       f"偏移=({offset_x:.2f}, {offset_y:.2f}), 朝向={self.current_heading:.2f}, "
                       f"旋转后=({rotated_offset_x:.2f}, {rotated_offset_y:.2f}), "
                       f"地图坐标=({map_x}, {map_y})")

            # 确保在地图范围内
            if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                # 根据检测类型更新栅格
                label_lower = detection['label'].lower()

                if 'gate' in label_lower or 'door' in label_lower:
                    # 门：类型3，绿色
                    self.occupancy_grid[map_y, map_x] = 3
                    self.semantic_grid[map_y, map_x] = 1
                    self.door_cells += 1
                    logger.info(f"[SLAM] 标记门在地图坐标 ({map_x}, {map_y})")
                else:
                    # 其他YOLO检测的物体作为障碍物（类型2，橙色）
                    self.occupancy_grid[map_y, map_x] = 2
                    self.semantic_grid[map_y, map_x] = 0
                    self.obstacle_cells += 1
                    logger.info(f"[SLAM] 标记物体 {detection.get('label', 'unknown')} 在地图坐标 ({map_x}, {map_y})")

                self.visit_count[map_y, map_x] += 1
            else:
                logger.warning(f"[SLAM] 物体 {detection.get('label', 'unknown')} 坐标 ({map_x}, {map_y}) 超出地图范围")

        # 标记障碍物特征点到地图上
        for ox, oy in obstacle_points:
            # 转换为相对于屏幕中心的偏移
            offset_x = (ox - 320) / 320.0  # 归一化 [-1, 1]
            offset_y = (oy - 240) / 240.0  # 归一化 [-1, 1]

            # 根据朝向旋转偏移
            rotated_offset_x = offset_x * np.cos(self.current_heading) - offset_y * np.sin(self.current_heading)
            rotated_offset_y = offset_x * np.sin(self.current_heading) + offset_y * np.cos(self.current_heading)

            # 映射到地图坐标
            map_x = int(self.current_map_x + rotated_offset_x * 20)
            map_y = int(self.current_map_y - rotated_offset_y * 20)

            # 确保在地图范围内
            if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                # 标记为障碍物（类型2）
                if self.occupancy_grid[map_y, map_x] == 0:  # 如果该位置为空，才标记
                    self.occupancy_grid[map_y, map_x] = 2
                    self.semantic_grid[map_y, map_x] = 0
                    self.obstacle_cells += 1

        # 更新路径历史
        self.path_history.append(current_pos)
        if len(self.path_history) > self.max_path_length:
            self.path_history.pop(0)

        # 标记当前位置为已访问
        self.visit_count[current_pos[1], current_pos[0]] += 1

        # 更新统计
        self.explored_cells = np.sum(self.visit_count > 0)

        self.current_step_count += 1

        # 保存当前检测作为上一帧
        self.prev_detections = detections.copy()

        # 返回地图特征向量
        return self._get_map_features()

    def _estimate_visual_odometry(self, prev_detections, current_detections):
        """
        通过追踪物体相对位置变化来估计位移
        Args:
            prev_detections: 上一帧的检测结果
            current_detections: 当前帧的检测结果
        Returns:
            (dx, dy, dheading): 估计的位移和转向变化
        """
        # 找到两帧之间匹配的物体（通过中心点距离）
        matched_pairs = []
        for prev in prev_detections:
            prev_center_x = (prev['bbox'][0] + prev['bbox'][2]) / 2
            prev_center_y = (prev['bbox'][1] + prev['bbox'][3]) / 2

            for curr in current_detections:
                curr_center_x = (curr['bbox'][0] + curr['bbox'][2]) / 2
                curr_center_y = (curr['bbox'][1] + curr['bbox'][3]) / 2

                # 计算距离
                dist = np.sqrt((prev_center_x - curr_center_x) ** 2 +
                              (prev_center_y - curr_center_y) ** 2)

                # 如果距离小于阈值，认为是同一个物体
                if dist < 50:
                    matched_pairs.append((prev, curr))
                    break

        if len(matched_pairs) < 2:
            # 匹配不够，无法估计位移，返回0
            return 0, 0, 0

        # 计算所有匹配对的位置变化
        dx_list = []
        dy_list = []

        for prev, curr in matched_pairs:
            prev_center_x = (prev['bbox'][0] + prev['bbox'][2]) / 2
            prev_center_y = (prev['bbox'][1] + prev['bbox'][3]) / 2
            curr_center_x = (curr['bbox'][0] + curr['bbox'][2]) / 2
            curr_center_y = (curr['bbox'][1] + curr['bbox'][3]) / 2

            # 屏幕上的位移
            screen_dx = curr_center_x - prev_center_x
            screen_dy = curr_center_y - prev_center_y

            # 将屏幕位移映射到地图坐标系（考虑朝向）
            map_dx = screen_dx * 0.02 * np.cos(self.current_heading) - screen_dy * 0.02 * np.sin(self.current_heading)
            map_dy = screen_dx * 0.02 * np.sin(self.current_heading) + screen_dy * 0.02 * np.cos(self.current_heading)

            dx_list.append(map_dx)
            dy_list.append(map_dy)

        # 使用中位数来过滤异常值
        dx = np.median(dx_list) if dx_list else 0
        dy = np.median(dy_list) if dy_list else 0

        return dx, dy, 0

    def _detect_obstacles_from_image(self, image):
        """
        从图像中检测障碍物特征点
        使用简单的图像处理：边缘检测 + 关键点检测
        Args:
            image: 当前图像（numpy数组，BGR格式）
        Returns:
            list of (x, y) 屏幕坐标系中的障碍物点
        """
        if image is None:
            return []

        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 使用角点检测（Good Features to Track）
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=20,  # 最多检测20个关键点
                qualityLevel=0.01,  # 质量阈值
                minDistance=30,  # 最小距离
                blockSize=3
            )

            obstacle_points = []
            if corners is not None:
                corners = corners.reshape(-1, 2)

                # 筛选有效的角点（避免中心区域）
                screen_center_x, screen_center_y = 320, 240
                for corner in corners:
                    x, y = corner

                    # 计算到中心的距离
                    dist = np.sqrt((x - screen_center_x) ** 2 + (y - screen_center_y) ** 2)

                    # 只选择视野中段的角点（太近或太远都不要）
                    if 50 < dist < 200:
                        obstacle_points.append((x, y))

            logger = setup_logging()
            logger.debug(f"[SLAM] 检测到 {len(obstacle_points)} 个障碍物特征点")

            return obstacle_points

        except Exception as e:
            logger = setup_logging()
            logger.warning(f"[SLAM] 障碍物检测失败: {e}")
            return []

    def _get_map_features(self):
        """
        获取地图特征向量（用于PPO状态）
        Returns:
            map_features: (map_size, map_size, 4) 张量
            [占用状态, 语义类别, 访问频率, 距离中心的距离]
        """
        # 归一化访问频率
        normalized_visit = self.visit_count / (np.max(self.visit_count) + 1)

        # 计算距离中心的距离
        y_indices, x_indices = np.indices((self.map_size, self.map_size))
        distance_from_center = np.sqrt((x_indices - self.center) ** 2 + (y_indices - self.center) ** 2)
        normalized_distance = distance_from_center / (self.map_size / 2)

        # 组合四个通道
        map_features = np.stack([
            self.occupancy_grid / 4.0,  # 归一化到 [0, 1]
            self.semantic_grid / 3.0,    # 归一化到 [0, 1]
            normalized_visit,              # 访问频率
            1.0 - normalized_distance       # 距离中心的远近（反转）
        ], axis=-1)

        return torch.FloatTensor(map_features).permute(2, 0, 1)  # (4, H, W)

    def get_map_image(self, size=(300, 300)):
        """
        获取用于可视化的地图图像
        Args:
            size: 输出图像尺寸
        Returns:
            RGB图像
        """
        # 创建彩色地图
        colormap = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # 颜色映射：0=黑色(未知), 1=灰色(已访问), 2=橙色(障碍物), 3=绿色(门)
        colors = {
            0: [0, 0, 0],      # 未知 - 黑色
            1: [100, 100, 100],  # 已访问/空闲 - 深灰色
            2: [255, 165, 0],  # 障碍物 - 橙色
            3: [0, 255, 0]      # 门 - 绿色（唯一）
        }

        for value, color in colors.items():
            mask = self.occupancy_grid == value
            colormap[mask] = color

        # 绘制已访问区域（淡黄色）
        visited_mask = self.visit_count > 0
        # 创建半透明的黄色覆盖层
        visited_overlay = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        visited_overlay[visited_mask] = [60, 60, 20]
        colormap = cv2.addWeighted(colormap, 1.0, visited_overlay, 0.5, 0)

        # 绘制路径历史（亮黄色）
        if len(self.path_history) > 1:
            for i in range(len(self.path_history) - 1):
                y1, x1 = self.path_history[i]
                y2, x2 = self.path_history[i + 1]
                cv2.line(colormap, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 绘制当前位置（紫色十字）
        cx, cy = int(self.current_map_x), int(self.current_map_y)
        cv2.line(colormap, (cx - 5, cy), (cx + 5, cy), (255, 0, 255), 3)
        cv2.line(colormap, (cx, cy - 5), (cx, cy + 5), (255, 0, 255), 3)

        # 绘制朝向指示（粉红色箭头）
        arrow_length = 10
        end_x = cx + int(arrow_length * np.sin(self.current_heading))
        end_y = cy - int(arrow_length * np.cos(self.current_heading))
        cv2.arrowedLine(colormap, (cx, cy), (end_x, end_y), (255, 105, 180), 3)

        # 绘制检测到的物体（用小点标注，按类型区分颜色）
        for y in range(self.map_size):
            for x in range(self.map_size):
                obj_type = self.occupancy_grid[y, x]
                if obj_type > 1:  # 有物体
                    # 根据物体类型使用不同颜色的小点
                    if obj_type == 2:  # 障碍物 - 橙色
                        cv2.circle(colormap, (x, y), 1, (255, 165, 0), -1)
                    elif obj_type == 3:  # 门 - 绿色（唯一，稍微大一点）
                        cv2.circle(colormap, (x, y), 2, (0, 255, 0), -1)

        # 调整大小
        map_image = cv2.resize(colormap, size, interpolation=cv2.INTER_NEAREST)

        return map_image

    def get_stats(self):
        """获取地图统计信息"""
        return {
            'explored_ratio': self.explored_cells / self.total_cells,
            'obstacle_count': self.obstacle_cells,
            'door_count': self.door_cells,
            'total_cells': self.total_cells
        }

    def reset(self):
        """重置地图"""
        self.occupancy_grid.fill(0)
        self.semantic_grid.fill(0)
        self.visit_count.fill(0)
        self.path_history.clear()
        self.explored_cells = 0
        self.obstacle_cells = 0
        self.door_cells = 0
        self.current_map_x = self.center
        self.current_map_y = self.center
        self.current_heading = 0
        self.current_step_count = 0
        self.prev_detections = []


class GroundSAMFeatureEncoder(nn.Module):
    """
    Ground-SAM特征编码器
    将冻结的SAM特征编码为可训练的特征向量
    """
    def __init__(self, input_channels, feature_dim=256):
        super(GroundSAMFeatureEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, sam_features):
        """
        编码SAM特征
        Args:
            sam_features: SAM图像嵌入 (H, W, C) 或 (B, H, W, C)
        Returns:
            编码后的特征向量 (B, feature_dim)
        """
        if sam_features.dim() == 3:
            # 单张图像: (H, W, C) -> (C, H, W)
            x = sam_features.permute(2, 0, 1).unsqueeze(0)
            x = self.encoder(x)
            return x
        elif sam_features.dim() == 4:
            # 批量: (B, H, W, C) -> (B, C, H, W)
            x = sam_features.permute(0, 3, 1, 2)
            x = self.encoder(x)
            return x
        else:
            raise ValueError(f"不支持的输入维度: {sam_features.dim()}")


class PolicyNetwork(nn.Module):
    """
    策略网络(使用Ground-SAM特征提取 + SLAM地图融合)
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=256, use_ground_sam=False, use_slam_map=False):
        super(PolicyNetwork, self).__init__()

        self.sam_mask_counter = 0  # SAM掩码保存计数器

        self.use_ground_sam = use_ground_sam and GROUND_SAM_AVAILABLE
        self.use_slam_map = use_slam_map

        # 特征维度
        feature_size = 256

        # 如果使用Ground-SAM,初始化SAM特征编码器
        if self.use_ground_sam:
            try:
                extractor = get_global_ground_sam_extractor()
                if extractor is not None:
                    # 初始化SAM特征编码器(可训练部分)
                    sam_channels, _, _ = extractor.get_feature_dim()
                    self.sam_feature_encoder = GroundSAMFeatureEncoder(sam_channels, feature_dim=256)
                    self.has_ground_sam = True
                    logger = setup_logging()
                    logger.info(f"Ground-SAM编码器已初始化, SAM通道数: {sam_channels}, 特征维度: {feature_size}")
                else:
                    self.has_ground_sam = False
            except Exception as e:
                logger = setup_logging()
                logger.warning(f"无法初始化Ground-SAM: {e}")
                self.has_ground_sam = False
        else:
            self.has_ground_sam = False

        # SLAM地图编码器（4通道：占用、语义、访问频率、距离）
        if self.use_slam_map:
            self.slam_map_encoder = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            logger = setup_logging()
            logger.info("SLAM地图编码器已初始化")

        # 融合后的特征维度
        combined_feature_size = feature_size
        if self.use_slam_map:
            combined_feature_size += 128

        # Actor heads
        self.move_actor = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, move_action_dim)
        )

        self.turn_actor = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, turn_action_dim)
        )

        # Action parameter head
        self.action_param_head = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        # Value network
        self.critic = nn.Sequential(
            nn.Linear(combined_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 初始化logger
        self.logger = setup_logging()

    def forward(self, state, return_debug_info=False, slam_map_features=None):
        """
        前向传播
        Args:
            state: 输入状态(图像)
            return_debug_info: 是否返回调试信息
            slam_map_features: SLAM地图特征 (4, H, W) 可选
        Returns:
            策略输出
        """
        # 使用Ground-SAM提取特征
        if self.has_ground_sam and self.training:
            try:
                # 将tensor转换回numpy图像
                if state.dim() == 4:
                    # 批量处理
                    batch_size = state.shape[0]
                    sam_features_list = []
                    for i in range(batch_size):
                        # 将归一化的[0,1]范围转换回[0,255]的uint8格式
                        image_np = (state[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                        sam_embedding = self._extract_sam_embedding(image_np)
                        sam_encoded = self.sam_feature_encoder(sam_embedding)
                        sam_features_list.append(sam_encoded)
                    sam_features = torch.stack(sam_features_list)
                else:
                    # 单张图像
                    # 将归一化的[0,1]范围转换回[0,255]的uint8格式
                    image_np = (state.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    sam_embedding = self._extract_sam_embedding(image_np)
                    logger = setup_logging()
                    logger.info(f"sam_embedding shape: {sam_embedding.shape} (before encoder)")
                    sam_features = self.sam_feature_encoder(sam_embedding)
                    logger.info(f"sam_features shape: {sam_features.shape} (after encoder)")
            except Exception as e:
                # 如果SAM特征提取失败，返回零特征
                logger = setup_logging()
                logger.error(f"Ground-SAM特征提取失败: {e}")
                # 返回零张量作为备选
                if state.dim() == 4:
                    batch_size = state.shape[0]
                    sam_features = torch.zeros(batch_size, 256).to(state.device)
                else:
                    sam_features = torch.zeros(1, 256).to(state.device)
        else:
            # 如果不使用Ground-SAM，返回零特征
            if state.dim() == 4:
                batch_size = state.shape[0]
                sam_features = torch.zeros(batch_size, 256).to(state.device)
            else:
                sam_features = torch.zeros(1, 256).to(state.device)

        # 融合SLAM地图特征
        if self.use_slam_map and slam_map_features is not None:
            # 将SLAM地图特征移到相同设备
            slam_map_features = slam_map_features.to(sam_features.device)

            # 调试信息
            logger = setup_logging()
            logger.info(f"Before encoding - sam_features shape: {sam_features.shape}, slam_map_features shape: {slam_map_features.shape}")

            # 确保sam_features是2D的
            if sam_features.dim() == 3 and sam_features.shape[1] == 1:
                # 如果是 [B, 1, feature_dim]，去掉多余的维度
                sam_features = sam_features.squeeze(1)
                logger.info(f"sam_features squeezed to: {sam_features.shape}")

            # 编码SLAM地图
            if slam_map_features.dim() == 3:
                # 单张地图
                slam_encoded = self.slam_map_encoder(slam_map_features.unsqueeze(0))
            else:
                # 批量地图
                slam_encoded = self.slam_map_encoder(slam_map_features)

            logger.info(f"After encoding - sam_features shape: {sam_features.shape}, slam_encoded shape: {slam_encoded.shape}")

            # 融合SAM特征和SLAM地图特征
            features = torch.cat([sam_features, slam_encoded], dim=-1)
        else:
            # 只使用SAM特征
            features = sam_features

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
                'features_shape': features.shape,
                'using_ground_sam': self.has_ground_sam
            }

            # 安全地打印模型输出信息(处理不同维度的情况)
            try:
                move_logits_flat = debug_info['move_logits'].flatten()
                turn_logits_flat = debug_info['turn_logits'].flatten()
                action_params_flat = debug_info['action_params'].flatten()
                value_flat = debug_info['value'].flatten()

                # 只打印前5个值以避免过多输出
                self.logger.info(f"Move: {[round(float(x), 2) for x in move_logits_flat[:5]]}")
                self.logger.info(f"Turn: {[round(float(x), 2) for x in turn_logits_flat[:5]]}")
                self.logger.info(f"参数: {[round(float(x), 2) for x in action_params_flat[:5]]}")
                self.logger.info(f"价值: {round(float(value_flat[0]), 2) if len(value_flat) > 0 else 'N/A'}")
            except Exception as e:
                self.logger.warning(f"无法打印调试信息: {e}")

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
    
    def _extract_sam_embedding(self, image_np, visualize_mask=False, save_path=None, visualization_interval=1000):
        """
        提取SAM图像嵌入(冻结的预训练模型)
        Args:
            image_np: numpy数组图像 (H, W, 3)
            visualize_mask: 是否可视化SAM掩码 (默认False,大幅提升性能)
            save_path: 掩码图像保存路径(如果为None则自动生成路径)
            visualization_interval: 每N步才可视化一次 (默认1000,几乎不绘制)
        Returns:
            SAM嵌入张量
        """
        extractor = get_global_ground_sam_extractor()
        if extractor is None:
            raise ValueError("Ground-SAM提取器未初始化")

        self.sam_mask_counter += 1  # 每次调用都递增计数器

        # 完全禁用SAM掩码可视化,只提取特征
        # 自动分割非常耗时(2-5秒),严重影响训练速度
        # 如果需要调试,可以手动设置visualize_mask=True

        with torch.no_grad():  # SAM是冻结的,不计算梯度
            # 默认不进行自动分割和可视化,大幅提升性能
            # 只在explicitly要求且间隔满足时才可视化
            if visualize_mask and (self.sam_mask_counter % visualization_interval == 0):
                try:
                    logger = setup_logging()
                    logger.info(f"步骤 {self.sam_mask_counter}: 开始SAM自动分割(仅用于调试)...")

                    start_time = time.time()
                    masks = extractor.sam_backbone.automatic_segmentation(image_np)
                    seg_time = time.time() - start_time

                    logger.info(f"SAM自动分割完成,检测到 {len(masks)} 个对象,耗时: {seg_time:.3f}秒")

                    # 自动生成保存路径
                    if save_path is None and hasattr(self, 'sam_mask_counter'):
                        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam_masks_output')
                        os.makedirs(output_dir, exist_ok=True)
                        save_path = os.path.join(output_dir, f'sam_mask_{self.sam_mask_counter:05d}.png')

                    self._visualize_sam_masks_fast(image_np, masks, save_path)
                except Exception as mask_error:
                    logger = setup_logging()
                    logger.error(f"SAM掩码生成失败: {type(mask_error).__name__}: {mask_error}")

            # 直接提取特征,不进行自动分割
            image_features = extractor.sam_backbone.extract_image_features(image_np)
            sam_embedding = image_features['image_embedding']

            logger = setup_logging()
            logger.debug(f"SAM特征提取成功, embedding形状: {sam_embedding.shape}")

            # 确保sam_embedding在正确的设备上(与sam_feature_encoder一致)
            if hasattr(self, 'sam_feature_encoder'):
                device = next(self.sam_feature_encoder.parameters()).device
                sam_embedding = sam_embedding.to(device)

            return sam_embedding

    def _visualize_sam_masks(self, image, masks, save_path=None):
        """
        可视化SAM自动分割的掩码 (使用OpenCV优化,比matplotlib快3-5倍)
        Args:
            image: 原始图像 (H, W, 3) numpy数组
            masks: SAM生成的掩码列表
            save_path: 保存路径,如果为None则不保存
        """
        logger = setup_logging()
        logger.info(f"开始可视化SAM掩码(OpenCV优化版), 掩码数量: {len(masks)}, 保存路径: {save_path}")

        try:
            # 创建图像副本用于绘制
            vis_image = image.copy()

            # 按面积排序掩码
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

            # 只显示前N个掩码以避免图像过于拥挤 (从20降到10)
            max_masks = min(len(sorted_masks), 10)

            for idx in range(max_masks):
                mask_data = sorted_masks[idx]
                mask = mask_data['segmentation']

                # 转换为numpy数组(如果是torch tensor)
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()

                # 生成随机颜色(高饱和度HSV色彩空间)
                hue = (idx * 137) % 180  # 黄金角度分布,颜色区分度高
                color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

                # 应用掩码(半透明叠加)
                mask_bool = mask.astype(bool)
                vis_image[mask_bool] = (vis_image[mask_bool] * 0.4 + color * 0.6).astype(np.uint8)

                # 绘制边界框
                x, y, w, h = mask_data['bbox']
                cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)),
                            tuple(map(int, color)), 2)

                # 添加文字信息
                info_text = f"#{idx} A:{mask_data['area']:.0f} IoU:{mask_data['predicted_iou']:.2f}"
                cv2.putText(vis_image, info_text, (int(x), max(5, int(y - 5))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(map(int, color)), 1, cv2.LINE_AA)

            # 添加标题信息
            title_text = f"SAM Auto-Seg: {len(masks)} objects (showing {max_masks})"
            cv2.putText(vis_image, title_text, (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # 保存图像(使用OpenCV,比matplotlib快3-5倍)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_image)
                logger.info(f"SAM掩码可视化已保存到: {save_path}")
            else:
                logger.info("未指定保存路径,跳过保存")

        except Exception as e:
            logger.error(f"可视化SAM掩码失败: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _visualize_sam_masks_fast(self, image, masks, save_path=None):
        """
        极速可视化SAM自动分割的掩码 (优化到0.1秒以内)

        优化策略:
        1. 只显示最大的3个掩码
        2. 只绘制边界框,不进行像素级颜色叠加
        3. 简化文字信息

        Args:
            image: 原始图像 (H, W, 3) numpy数组
            masks: SAM生成的掩码列表
            save_path: 保存路径,如果为None则不保存
        """
        logger = setup_logging()

        try:
            start_time = time.time()

            # 创建图像副本用于绘制
            vis_image = image.copy()

            # 按面积排序掩码,只保留前3个最大的
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            max_masks = min(len(sorted_masks), 3)  # 只显示3个,大幅减少绘制时间

            # 预定义颜色(避免重复计算HSV)
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色(OpenCV中是BGR)
                (0, 0, 255)     # 红色
            ]

            for idx in range(max_masks):
                mask_data = sorted_masks[idx]
                mask = mask_data.get('segmentation') or mask_data.get('segmentation', None)

                # 转换为numpy数组(如果是torch tensor)
                if mask is not None and hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()

                # 使用预定义颜色
                color = colors[idx % len(colors)]

                # 不进行像素级颜色叠加(最耗时的部分),只画边界框
                x, y, w, h = mask_data['bbox']
                cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)),
                            color, 2)

                # 简化文字信息,只显示序号
                cv2.putText(vis_image, f"#{idx+1}", (int(x), max(5, int(y - 5))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # 保存图像
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_image)

            elapsed = time.time() - start_time
            logger.info(f"SAM掩码快速可视化完成,耗时: {elapsed:.3f}秒, 保存路径: {save_path}")

        except Exception as e:
            logger.error(f"可视化SAM掩码失败: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())


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
    实时可视化器,使用Tkinter显示截图、YOLO检测框、Ground-SAM特征和智能体状态
    """
    def __init__(self, window_name="PPO Agent Visualizer"):
        self.window_name = window_name
        self.current_image = None
        self.detections = []
        self.agent_info = {}
        self.ground_sam_features = None  # 存储Ground-SAM特征
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
    def update_image_and_detections(self, image, detections, ground_sam_features=None, slam_map_image=None):
        """
        更新图像和检测结果，并立即触发界面更新
        Args:
            image: 当前图像
            detections: YOLO检测结果
            ground_sam_features: Ground-SAM特征(可选)
            slam_map_image: SLAM小地图图像(可选)
        """
        with self.image_lock:
            self.current_image = image.copy() if image is not None else None
            self.detections = detections.copy() if detections is not None else []
            self.ground_sam_features = ground_sam_features
            self.slam_map_image = slam_map_image

        # 将更新请求放入队列
        try:
            self.gui_queue.put(('update_image', self.current_image, self.detections, self.ground_sam_features, self.slam_map_image), block=False)
        except queue.Full:
            pass

    def _update_image_display(self, image, detections, ground_sam_features=None, slam_map_image=None):
        """
        更新图像显示
        Args:
            image: 当前图像
            detections: 检测结果
            ground_sam_features: Ground-SAM特征(可选)
            slam_map_image: SLAM小地图图像(可选)
        """
        if image is not None:
            # 先绘制SAM掩码(重新叠加原始图像)
            if ground_sam_features is not None:
                display_img = self._draw_ground_sam_features(image.copy(), ground_sam_features, reblend_original=True)
            else:
                display_img = image.copy()

            # 再绘制YOLO检测框(在掩码之上)
            display_img = self._draw_detections(display_img)

            # 如果有智能体信息，也绘制在图像上
            if hasattr(self, '_last_agent_info') and self._last_agent_info:
                display_img = self._add_agent_info_to_image(display_img, self._last_agent_info)

            # 如果有SLAM小地图，将地图叠加到图像右上角
            if slam_map_image is not None:
                map_size = 200  # 小地图尺寸
                map_resized = cv2.resize(slam_map_image, (map_size, map_size), interpolation=cv2.INTER_NEAREST)

                # 叠加到显示图像的右上角
                h, w = display_img.shape[:2]
                if w >= map_size + 10 and h >= map_size + 10:
                    # 叠加地图到右上角
                    display_img[5:map_size + 5, w - map_size - 5:w - 5] = map_resized

                    # 添加边框
                    cv2.rectangle(display_img, (w - map_size - 5, 5), (w - 5, map_size + 5), (0, 0, 255), 2)

                    # 添加标题
                    cv2.putText(display_img, "SLAM Map", (w - map_size, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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

    def _update_image_display_with_existing_features(self, display_img):
        """
        更新图像显示(使用已有的特征信息,不重新绘制Ground-SAM特征)
        Args:
            display_img: 已经绘制好agent信息的图像
        """
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

        # Ground-SAM状态
        ground_sam_active = info.get('ground_sam_active', False)
        lines.append(f"Ground-SAM: {'Active' if ground_sam_active else 'Inactive'}")

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
                    if len(args) >= 4:
                        image, detections, ground_sam_features, slam_map_image = args
                    else:
                        image, detections, ground_sam_features = args
                        slam_map_image = None
                    self._update_image_display(image, detections, ground_sam_features, slam_map_image)
                elif msg_type == 'update_info':
                    info = args[0]
                    # 更新当前图像的信息显示
                    if self.current_image is not None:
                        # 保存最新的agent info
                        self._last_agent_info = info
                        # 直接更新显示,避免重复调用update_image_and_detections
                        img_with_info = self._add_agent_info_to_image(self.current_image.copy(), info)
                        # 只更新canvas显示,不触发新的queue消息
                        self._update_image_display_with_existing_features(img_with_info)
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
        在图像上绘制YOLO检测框(使用绿色)
        """
        if self.detections:
            for detection in self.detections:
                bbox = detection['bbox']
                label = detection['label']
                score = detection['score']

                # 转换边界框坐标为整数
                x1, y1, x2, y2 = map(int, bbox)

                # 绘制矩形框(绿色 - YOLO检测结果)
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # 添加标签和置信度文本
                text = f"YOLO {label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_color = (255, 255, 255)
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

    def _draw_ground_sam_features(self, image, features, reblend_original=False):
        """
        在图像上绘制Ground-SAM特征
        Args:
            image: 输入图像
            features: Ground-SAM特征字典
            reblend_original: 是否重新叠加原始图像(如果为False,则在已有图像上叠加掩码)
        Returns:
            绘制后的图像
        """
        # 已禁用掩码绘制以提升性能，直接返回原始图像
        return image

        # 以下代码已禁用
        if features is None:
            return image

        try:
            # 创建半透明叠加层
            overlay = image.copy()

            # 不绘制Grounding DINO检测框，只绘制SAM掩码
            # 绘制SAM分割掩码(使用半透明叠加)
            if 'mask_features' in features and features['mask_features']:
                mask_features = features['mask_features']

                import logging
                logger = logging.getLogger('ppo_agent_logger')
                logger.info(f"准备绘制 {len(mask_features)} 个SAM分割掩码")

                import random
                mask_drawn = 0
                for i, mask_info in enumerate(mask_features):
                    if isinstance(mask_info, dict) and 'mask' in mask_info:
                        mask = mask_info['mask']

                        # 转换为numpy数组
                        if torch.is_tensor(mask):
                            mask = mask.cpu().numpy()

                        # 确保mask是二维的
                        if mask.ndim == 3:
                            mask = mask.squeeze()

                        logger.debug(f"掩码 {i}: 形状={mask.shape}, dtype={mask.dtype}, min={mask.min():.3f}, max={mask.max():.3f}")

                        # 为每个掩码生成随机颜色（使用H色彩空间，保持高亮度和高饱和度）
                        hue = random.randint(0, 179)
                        color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

                        # 应用掩码到overlay(增加掩码颜色的权重)
                        mask_bool = mask.astype(bool)
                        true_count = mask_bool.sum()
                        if true_count > 0:  # 确保掩码有效且有像素
                            logger.debug(f"掩码 {i} 有 {true_count} 个有效像素")
                            # 使用numpy向量化操作进行颜色混合(让掩码颜色更明显)
                            overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.3 + color[0] * 0.7).astype(np.uint8)
                            overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.3 + color[1] * 0.7).astype(np.uint8)
                            overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.3 + color[2] * 0.7).astype(np.uint8)

                            mask_drawn += 1
                        else:
                            logger.warning(f"掩码 {i} 没有有效像素!")

                logger.info(f"实际绘制了 {mask_drawn}/{len(mask_features)} 个掩码")
            else:
                import logging
                logger = logging.getLogger('ppo_agent_logger')
                logger.debug("features字典中没有mask_features键")

            # 根据参数决定是否重新叠加原始图像
            if reblend_original:
                # 叠加原始图像(增加掩码透明度,让掩码更明显)
                alpha = 0.6  # 增加掩码的透明度(0.6 + 0.4)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            else:
                # 不重新叠加,直接返回overlay(这样可以保留YOLO检测结果)
                image = overlay

        except Exception as e:
            # 如果绘制失败,不影响主流程
            import logging
            logger = logging.getLogger('ppo_agent_logger')
            logger.error(f"绘制Ground-SAM特征时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
    为环境类添加可视化功能(包括Ground-SAM特征提取)
    """
    original_step = TargetSearchEnvironment.step
    original_reset = TargetSearchEnvironment.reset

    def step_with_visualization(self, move_action, turn_action, move_forward_step=2, turn_angle=30):
        # 获取当前状态
        current_state = self.capture_screen()

        # 检测目标
        detections = self.detect_target(current_state)

        # Ground-SAM特征提取已禁用以提升性能
        # 自动分割非常耗时，严重影响训练速度
        ground_sam_features = None

        # 获取SLAM小地图图像
        slam_map_image = None
        if hasattr(self, 'slam_map_builder'):
            # 更新SLAM地图（传递当前图像用于障碍物检测）
            slam_features = self.slam_map_builder.update(
                detections,
                move_action,
                move_forward_step,
                turn_action,
                turn_angle,
                current_image=current_state  # 传递当前图像用于障碍物检测
            )
            # 获取可视化地图
            slam_map_image = self.slam_map_builder.get_map_image(size=(200, 200))

        # 更新可视化(包含SLAM地图)
        visualizer.update_image_and_detections(current_state, detections, ground_sam_features, slam_map_image)

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
            'ground_sam_active': ground_sam_features is not None,
            'done': done
        })

        return result

    def reset_with_visualization(self):
        result = original_reset(self)

        # 重置SLAM地图
        if hasattr(self, 'slam_map_builder'):
            self.slam_map_builder.reset()

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

        # 添加用于检查图像变化的变量
        self.previous_image_hash = None
        self.image_change_counter = 0
        self.total_image_checks = 0

        # 存储最近的Ground-SAM特征（用于保存图片）
        self.last_ground_sam_features = None
        
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
                
                # 检查图像是否与上一帧相同
                current_hash = hash(screenshot.tobytes())
                if self.previous_image_hash is not None:
                    self.total_image_checks += 1
                    if current_hash != self.previous_image_hash:
                        self.image_change_counter += 1
                
                self.previous_image_hash = current_hash
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                
                # 检查图像是否与上一帧相同
                current_hash = hash(screenshot.tobytes())
                if self.previous_image_hash is not None:
                    self.total_image_checks += 1
                    if current_hash != self.previous_image_hash:
                        self.image_change_counter += 1
                        
                self.previous_image_hash = current_hash
                return screenshot
        except ImportError:
            self.logger.warning("截图功能不可用，使用模拟图片")
            # 使用配置文件中的图像尺寸
            sim_img = np.zeros((CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], 3), dtype=np.uint8)
            # 检查图像是否与上一帧相同
            current_hash = hash(sim_img.tobytes())
            if self.previous_image_hash is not None:
                self.total_image_checks += 1
                if current_hash != self.previous_image_hash:
                    self.image_change_counter += 1
                    
            self.previous_image_hash = current_hash
            return sim_img

    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []

        try:
            conf_threshold = CONFIG.get('DETECTION_CONFIDENCE', 0.4)
            self.logger.debug(f"YOLO检测开始，置信度阈值: {conf_threshold}")

            results = self.yolo_model.predict(
                source=image,
                conf=conf_threshold,
                save=False,
                verbose=False
            )

            detections = []
            result = results[0]

            # 调试：打印原始检测结果
            if result.boxes is not None:
                self.logger.info(f"YOLO原始检测框数量: {len(result.boxes)}")

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()

                names = result.names if hasattr(result, 'names') else {}

                self.logger.debug(f"置信度值: {confs}")

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")

                    width = x2 - x1
                    height = y2 - y1

                    self.logger.debug(f"检测框{i}: {class_name}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], conf={conf:.3f}")

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
        基于目标检测的奖励 - 优化版本
        """
        if not detection_results:
            return 0.0  # 未检测到目标不给惩罚，避免过于消极
        
        # 检测到gate给予较高奖励
        gate_detected = any(
            'gate' in detection['label'].lower() or detection['label'].lower() == 'gate'
            for detection in detection_results
        )
        
        if gate_detected:
            return 0.15  # 增加gate检测奖励
        
        # 检测到其他目标的奖励
        return 0.05  # 增加其他目标检测奖励

    def _calculate_progress_reward(self, detection_results):
        """
        基于目标检测进展的奖励 - 优化版本
        """
        if not detection_results:
            return 0.0  # 无目标不给负奖励
        else:
            return 0.03  # 检测到目标给予小奖励

    def _calculate_repetition_penalty(self, action_taken):
        """
        重复动作惩罚 - 优化版本
        """
        if not action_taken or len(self.action_history) < 4:
            return 0.0
        
        recent_actions = list(self.action_history)[-4:]
        same_action_count = sum(1 for act in recent_actions if act == action_taken)
        
        # 只在连续重复超过3次才惩罚
        if same_action_count >= 3:
            return -0.05  # 减轻惩罚
        elif same_action_count >= 2:
            return -0.02  # 轻微惩罚
        else:
            return 0.0  # 不惩罚

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
            # 保存带Ground-SAM掩码的图片
            if CONFIG.get('SAVE_DETECTION_IMAGES', False) and self.last_ground_sam_features is not None:
                self._save_detection_image_with_bounding_boxes(
                    new_state,
                    detection_results,
                    self.last_ground_sam_features,
                    prefix=f"episode_{getattr(self, '_current_episode', 0)}_step_{self.step_count}_climb"
                )

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
        # 基础奖励 - 鼓励探索
        base_reward = 0.01  # 小的正奖励，避免所有情况都是负奖励

        # 目标检测奖励
        target_reward = self._calculate_target_presence_reward(detection_results)

        # 进展奖励
        progress_reward = self._calculate_progress_reward(detection_results)

        # 重复动作惩罚
        repetition_penalty = self._calculate_repetition_penalty(action_taken)

        # 总奖励
        total_reward = base_reward + target_reward + progress_reward + repetition_penalty

        # 计算新区域面积
        new_area = self._get_max_detection_area(detection_results) if detection_results else 0

        return total_reward, new_area

    def _save_detection_image_with_bounding_boxes(self, image, detection_results, ground_sam_features=None, prefix="detection"):
        """
        保存带Ground-SAM掩码的图像（不包含检测框），并保留最近5个
        """
        try:
            # 复制图像以避免修改原始图像
            img_with_masks = image.copy()

            # 如果有Ground-SAM特征，绘制掩码
            if ground_sam_features is not None and 'mask_features' in ground_sam_features:
                mask_features = ground_sam_features['mask_features']

                # 创建半透明叠加层
                overlay = img_with_masks.copy()

                for i, mask_info in enumerate(mask_features):
                    if isinstance(mask_info, dict) and 'mask' in mask_info:
                        mask = mask_info['mask']

                        # 转换为numpy数组
                        if torch.is_tensor(mask):
                            mask = mask.cpu().numpy()

                        # 确保mask是二维的
                        if mask.ndim == 3:
                            mask = mask.squeeze()

                        # 为每个掩码生成随机颜色（使用H色彩空间，保持高亮度）
                        import random
                        hue = random.randint(0, 179)
                        color = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]

                        # 应用掩码到overlay(半透明)
                        mask_bool = mask.astype(bool)
                        true_count = mask_bool.sum()
                        if true_count > 0:
                            # 使用numpy向量化操作进行颜色混合
                            overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.5 + color[0] * 0.5).astype(np.uint8)
                            overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.5 + color[1] * 0.5).astype(np.uint8)
                            overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.5 + color[2] * 0.5).astype(np.uint8)

                # 叠加原始图像
                img_with_masks = cv2.addWeighted(overlay, 0.7, img_with_masks, 0.3, 0)

            # 生成文件名
            timestamp = int(time.time() * 1000)  # 使用毫秒时间戳
            filename = f"{prefix}_sam_masks_{timestamp}.png"

            # 确定保存路径
            save_dir = Path(__file__).parent / "detection_images"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)

            filepath = save_dir / filename

            # 保存图像
            success = cv2.imwrite(str(filepath), img_with_masks)

            if success:
                # 将文件路径添加到最近检测图像队列
                self.recent_detection_images.append(str(filepath))
            else:
                self.logger.error(f"保存检测结果图像失败: {filepath}")

        except Exception as e:
            self.logger.error(f"保存带Ground-SAM掩码图像时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

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
        
        # 输出图像变化统计信息
        if self.total_image_checks > 0:
            change_rate = self.image_change_counter / self.total_image_checks
            self.logger.info(f"图像变化统计: {self.image_change_counter}/{self.total_image_checks} ({change_rate:.2%})")
        
        initial_state = self.capture_screen()
        return initial_state


class ExperienceCollector:
    """HDF5经验收集器"""
    def __init__(self):
        self.h5_file = None
        self.state_shape = None
        self.state_dtype = None
        self.config = CONFIG
        self.save_dir = self.config.get('EXPERIENCE_SAVE_DIR', './experience_data')
        self.max_experiences = self.config.get('MAX_EXPERIENCES', 100000)
        self.flush_interval = self.config.get('EXPERIENCE_FLUSH_INTERVAL', 100)
        self.current_count = 0
        self.buffer_count = 0
        self.logger = setup_logging()

    def _init_file(self, state_tensor):
        """初始化HDF5文件"""
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'experience_{timestamp}.h5'
        filepath = os.path.join(self.save_dir, filename)

        self.h5_file = h5py.File(filepath, 'w')
        self.state_shape = state_tensor.shape

        # 将PyTorch dtype转换为numpy dtype
        if torch.is_tensor(state_tensor):
            state_np = state_tensor.cpu().numpy()
            self.state_dtype = state_np.dtype
        else:
            state_np = state_tensor
            self.state_dtype = state_np.dtype

        self.h5_file.create_dataset('states', (0,) + self.state_shape, maxshape=(self.max_experiences,) + self.state_shape, dtype='float32', chunks=True, compression='gzip')
        self.h5_file.create_dataset('move_actions', (0,), maxshape=(self.max_experiences,), dtype='int64', chunks=True, compression='gzip')
        self.h5_file.create_dataset('turn_actions', (0,), maxshape=(self.max_experiences,), dtype='int64', chunks=True, compression='gzip')
        self.h5_file.create_dataset('move_steps', (0,), maxshape=(self.max_experiences,), dtype='float32', chunks=True, compression='gzip')
        self.h5_file.create_dataset('turn_angles', (0,), maxshape=(self.max_experiences,), dtype='float32', chunks=True, compression='gzip')
        self.h5_file.create_dataset('rewards', (0,), maxshape=(self.max_experiences,), dtype='float32', chunks=True, compression='gzip')
        self.h5_file.create_dataset('dones', (0,), maxshape=(self.max_experiences,), dtype='bool', chunks=True, compression='gzip')

        metadata = self.h5_file.create_group('metadata')
        metadata.attrs['total_experiences'] = 0
        metadata.attrs['last_update'] = datetime.now().isoformat()

        self.logger.info(f"ExperienceCollector初始化完成: {filepath}")

    def add_experience(self, state, move_action, turn_action, move_step, turn_angle, reward, done):
        """添加一条经验"""
        if self.h5_file is None:
            self._init_file(state)

        if self.current_count >= self.max_experiences:
            self.logger.warning("已达到最大经验数，停止收集")
            return

        self.h5_file['states'].resize((self.current_count + 1,) + self.state_shape)
        self.h5_file['states'][self.current_count] = state.cpu().numpy() if torch.is_tensor(state) else state

        self.h5_file['move_actions'].resize((self.current_count + 1,))
        self.h5_file['move_actions'][self.current_count] = move_action

        self.h5_file['turn_actions'].resize((self.current_count + 1,))
        self.h5_file['turn_actions'][self.current_count] = turn_action

        self.h5_file['move_steps'].resize((self.current_count + 1,))
        self.h5_file['move_steps'][self.current_count] = float(move_step)

        self.h5_file['turn_angles'].resize((self.current_count + 1,))
        self.h5_file['turn_angles'][self.current_count] = float(turn_angle)

        self.h5_file['rewards'].resize((self.current_count + 1,))
        self.h5_file['rewards'][self.current_count] = float(reward)

        self.h5_file['dones'].resize((self.current_count + 1,))
        self.h5_file['dones'][self.current_count] = bool(done)

        self.current_count += 1
        self.buffer_count += 1

        if self.buffer_count >= self.flush_interval:
            self.flush()

    def update_last_experience(self, reward, done):
        """更新最后一条经验的奖励和done标志"""
        if self.current_count > 0:
            idx = self.current_count - 1
            self.h5_file['rewards'][idx] = float(reward)
            self.h5_file['dones'][idx] = bool(done)

    def flush(self):
        """刷新到磁盘"""
        if self.h5_file is not None:
            self.h5_file['metadata'].attrs['total_experiences'] = self.current_count
            self.h5_file['metadata'].attrs['last_update'] = datetime.now().isoformat()
            self.h5_file.flush()
            self.logger.info(f"已刷新{self.buffer_count}条经验到磁盘, 总数:{self.current_count}")
            self.buffer_count = 0

    def close(self):
        """关闭文件"""
        if self.h5_file is not None:
            self.flush()
            self.h5_file.close()
            self.h5_file = None
            self.logger.info("ExperienceCollector已关闭")

    def get_stats(self):
        """获取统计信息"""
        if self.h5_file is None:
            return {'total_experiences': 0, 'file_size_mb': 0}
        import os
        filepath = self.h5_file.filename
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0
        return {'total_experiences': self.current_count, 'file_size_mb': file_size_mb}


# 全局ExperienceCollector实例
_experience_collector = None


def get_experience_collector():
    """获取全局ExperienceCollector实例"""
    global _experience_collector
    if _experience_collector is None:
        config = CONFIG
        if config.get('SAVE_EXPERIENCE', False):
            _experience_collector = ExperienceCollector()
    return _experience_collector


class PPOAgent:
    """
    PPO智能体(支持Ground-SAM + SLAM地图)
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, use_ground_sam=None, use_slam_map=None):
        self.sam_mask_counter = 0  # SAM掩码保存计数器
        config = CONFIG

        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']

        # 保存维度信息用于策略重置
        self.state_dim = state_dim
        self.move_action_dim = move_action_dim
        self.turn_action_dim = turn_action_dim

        # 确定是否使用Ground-SAM
        if use_ground_sam is None:
            use_ground_sam = config.get('USE_GROUND_SAM', False)

        self.use_ground_sam = use_ground_sam and GROUND_SAM_AVAILABLE

        # 确定是否使用SLAM地图
        if use_slam_map is None:
            use_slam_map = config.get('USE_SLAM_MAP', False)

        self.use_slam_map = use_slam_map

        # 添加logger
        self.logger = setup_logging()

        # 如果使用Ground-SAM,先初始化全局提取器
        if self.use_ground_sam:
            get_global_ground_sam_extractor()
            self.logger.info("PPO智能体已启用Ground-SAM特征提取")

        # 初始化SLAM地图构建器
        if self.use_slam_map:
            self.slam_map_builder = SLAMMapBuilder(
                map_size=config.get('SLAM_MAP_SIZE', 100),
                resolution=config.get('SLAM_MAP_RESOLUTION', 0.1)
            )
            self.logger.info("PPO智能体已启用SLAM地图构建")

        # Create policy networks
        self.policy = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim,
            use_ground_sam=self.use_ground_sam,
            use_slam_map=self.use_slam_map
        )

        # 优化器:只优化可训练参数,不优化冻结的SAM
        if self.use_ground_sam and hasattr(self.policy, 'sam_feature_encoder'):
            # 如果使用Ground-SAM,优化编码器和策略网络
            trainable_params = []
            trainable_params.extend(self.policy.sam_feature_encoder.parameters())
            if self.use_slam_map and hasattr(self.policy, 'slam_map_encoder'):
                trainable_params.extend(self.policy.slam_map_encoder.parameters())
            trainable_params.extend(self.policy.move_actor.parameters())
            trainable_params.extend(self.policy.turn_actor.parameters())
            trainable_params.extend(self.policy.action_param_head.parameters())
            trainable_params.extend(self.policy.critic.parameters())

            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
                betas=self.betas
            )
        else:
            # 不使用Ground-SAM,优化所有参数
            trainable_params = list(self.policy.parameters())
            if self.use_slam_map and hasattr(self.policy, 'slam_map_encoder'):
                trainable_params.extend(self.policy.slam_map_encoder.parameters())
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
                betas=self.betas
            )

        self.policy_old = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim,
            use_ground_sam=self.use_ground_sam,
            use_slam_map=self.use_slam_map
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # 添加梯度检查相关变量
        self.gradient_norms = []
        self.parameter_norms = []

    def reset_policy(self):
        """重置策略网络，用于检测到策略崩溃时"""
        self.policy = PolicyNetwork(
            self.state_dim, self.move_action_dim, self.turn_action_dim,
            use_ground_sam=self.use_ground_sam
        )

        # 重新设置优化器
        if self.use_ground_sam and hasattr(self.policy, 'sam_feature_encoder'):
            trainable_params = []
            trainable_params.extend(self.policy.sam_feature_encoder.parameters())
            trainable_params.extend(self.policy.move_actor.parameters())
            trainable_params.extend(self.policy.turn_actor.parameters())
            trainable_params.extend(self.policy.action_param_head.parameters())
            trainable_params.extend(self.policy.critic.parameters())

            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
                betas=self.betas
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.logger.warning("策略网络已重置")

    def act(self, state, memory, return_debug_info=False):
        # 预处理状态
        state_tensor = self._preprocess_state(state)

        # 获取SLAM地图特征
        slam_map_features = None
        if self.use_slam_map and hasattr(self, 'slam_map_builder'):
            slam_map_features = self.slam_map_builder._get_map_features()

        # 使用旧策略获取动作概率
        with torch.no_grad():
            if return_debug_info:
                move_probs, turn_probs, action_params, state_val, debug_info = self.policy_old(
                    state_tensor.unsqueeze(0),
                    return_debug_info=True,
                    slam_map_features=slam_map_features
                )
            else:
                move_probs, turn_probs, action_params, state_val = self.policy_old(
                    state_tensor.unsqueeze(0),
                    slam_map_features=slam_map_features
                )

            # 检测策略崩溃：如果所有动作概率都接近0或1
            move_probs_squeezed = move_probs.squeeze()
            turn_probs_squeezed = turn_probs.squeeze()

            # 检查是否策略崩溃（概率过于极端）
            move_max_prob = move_probs_squeezed.max().item()
            turn_max_prob = turn_probs_squeezed.max().item()

            # 如果策略崩溃，重置策略
            if move_max_prob > 0.99 or turn_max_prob > 0.99:
                self.logger.warning(f"检测到策略崩溃！Move最大概率: {move_max_prob:.4f}, Turn最大概率: {turn_max_prob:.4f}，重置策略")
                self.reset_policy()

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
        # 确保 action_params 的形状正确
        action_params_flat = action_params.flatten()

        if action_params_flat.shape[0] >= 2:
            move_forward_step_raw = action_params_flat[0].item()  # [-1,1]范围
            turn_angle_raw = action_params_flat[1].item()         # [-1,1]范围
        else:
            # 如果参数不够，使用默认值
            logger.warning(f"action_params 维度异常: {action_params.shape}, 使用默认值")
            move_forward_step_raw = 0.0
            turn_angle_raw = 0.0

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

        # 检查h5py是否可用
        if h5py is not None:
            try:
                collector = get_experience_collector()
                # 此时还不知道reward和done,先用0和False占位,后续会更新
                collector.add_experience(
                    state_tensor,
                    move_action.item(),
                    turn_action.item(),
                    move_forward_step,
                    turn_angle,
                    0.0,  # 临时奖励,在update时更新
                    False  # 临时done标志
                )
            except Exception as e:
                self.logger.error(f"保存经验失败: {e}")
        else:
            self.logger.warning("h5py模块不可用，跳过经验保存")

        # 准备返回值
        if return_debug_info:
            return move_action, turn_action, move_forward_step, turn_angle, debug_info
        else:
            return move_action, turn_action, move_forward_step, turn_angle

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

            # 计算优势 - 使用广义优势估计(GAE)以获得更好的性能
            advantages = rewards - state_vals.squeeze().detach()
            
            # 简化损失函数，避免策略崩溃
            # 标准的PPO clip损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = self.MseLoss(state_vals.squeeze(), rewards)

            # 熵损失 - 提高探索性
            move_entropy = move_dist.entropy().mean()
            turn_entropy = turn_dist.entropy().mean()
            entropy_loss = move_entropy + turn_entropy

            # 总损失 - 增加熵权重以提高探索性
            loss = actor_loss + 0.5 * critic_loss - 0.02 * entropy_loss

            # 反向传播


            self.optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度
            total_norm = 0
            param_norm = 0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm += p.norm(2).item() ** 2
                    total_norm += p.grad.norm(2).item() ** 2
            
            param_norm = param_norm ** 0.5
            total_norm = total_norm ** 0.5
            
            self.gradient_norms.append(total_norm)
            self.parameter_norms.append(param_norm)
            
            # 梯度裁剪以稳定训练
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()

        # 更新旧策略 - 使用更保守的软更新
        with torch.no_grad():
            for old_param, new_param in zip(self.policy_old.parameters(), self.policy.parameters()):
                old_param.data.copy_(0.998 * old_param.data + 0.002 * new_param.data)

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
            
            # 输出加载检查点时的参数统计信息
            param_norm = 0
            for p in self.policy.parameters():
                param_norm += p.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            self.logger.info(f"加载检查点时参数范数: {param_norm:.4f}")
            
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

    # 如果智能体启用了SLAM地图，将slam_map_builder挂载到环境上，供可视化使用
    if hasattr(ppo_agent, 'slam_map_builder'):
        env.slam_map_builder = ppo_agent.slam_map_builder
        print("SLAM地图构建器已挂载到环境，可视化将显示SLAM小地图")

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


# 重复的initialize_model函数已删除,使用上面第2126行的正确版本
# 以下函数已弃用,保留用于向后兼容
# def initialize_model(model_path, load_existing=False):
#     """
#     初始化模型
#     """
#     config = CONFIG
#
#     # 初始化环境
#     env = config['ENV']()  # 这个键不存在,导致KeyError
#
#     # 初始化PPO智能体
#     ppo_agent = PPOAgent(
#         state_dim=env.observation_space.shape[0],
#         action_dim=env.action_space.shape[0],
#         lr=config['LEARNING_RATE'],
#         gamma=config['GAMMA'],
#         K_epochs=config['K_EPOCHS'],
#         eps_clip=config['EPS_CLIP'],
#         logger=config['LOGGER']
#     )
#
#     # 初始化可视化器
#     visualizer = Visualizer(env, ppo_agent, config['VISUALIZER_PORT'])
#
#     start_episode = 0
#     if load_existing:
#         if os.path.exists(model_path):
#             ppo_agent.load_checkpoint(model_path)
#             print(f"模型已加载: {model_path}")
#             start_episode = int(model_path.split('_')[-1].split('.')[0])
#         else:
#             print(f"模型文件不存在: {model_path}")
#
#     return env, ppo_agent, visualizer, start_episode


# 旧式经验缓冲区(用于保持向后兼容)
EXPERIENCE_BUFFER = deque(maxlen=10000)


def train_with_experience_buffer(ppo_agent, experience_buffer):
    """
    使用经验池训练PPO智能体
    """
    for _ in range(ppo_agent.K_epochs):
        for state, action, reward, next_state, done in experience_buffer:
            ppo_agent.update(state, action, reward, next_state, done)


def save_experience_buffer(buffer, path):
    """
    保存经验池到本地文件
    """
    with open(path, 'wb') as f:
        pickle.dump(list(buffer), f)

def load_experience_buffer(path):
    """
    从本地文件加载经验池
    """
    if os.path.exists(path):
        with open(path, 'rb') as f:
            experiences = pickle.load(f)
        return deque(experiences, maxlen=10000)
    return deque(maxlen=10000)

def perform_training_loop(env, ppo_agent, visualizer, start_episode, total_episodes):
    """
    执行训练循环
    """
    global EXPERIENCE_BUFFER
    
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
        print_debug = episode % 50 == 0  # 每50轮打印一次详细信息
        result = run_episode(env, ppo_agent, visualizer, episode, total_episodes, training_mode=True, print_debug=print_debug)
        
        # 更新统计数据
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        final_areas.append(result['final_area'])
        training_stats['episode_count'] += 1
        
        if result['success_flag']:
            training_stats['successful_episodes'] += 1
        
        # 每5轮保存经验并更新策略
        if (episode + 1) % 5 == 0:
            # 保存当前经验池
            experience_path = f"experience_buffer_ep{episode+1}.pkl"
            save_experience_buffer(EXPERIENCE_BUFFER, experience_path)
            
            # 使用经验池训练多次
            if len(EXPERIENCE_BUFFER) > 0:
                print(f"Episode {episode+1}: 使用经验池训练，经验数量: {len(EXPERIENCE_BUFFER)}")
                train_with_experience_buffer(ppo_agent, EXPERIENCE_BUFFER)
        
        # 每10轮保存一次检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = f'ppo_model_checkpoint_ep{episode+1}.pth'
            ppo_agent.save_checkpoint(checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")

    # 训练结束后保存最终的经验池
    final_experience_path = "final_experience_buffer.pkl"
    save_experience_buffer(EXPERIENCE_BUFFER, final_experience_path)
    print(f"最终经验池已保存: {final_experience_path}")
    
    return training_stats


def run_episode(env, ppo_agent, visualizer, episode_num, total_episodes, training_mode=True, print_debug=False):
    """
    运行一个episode
    """
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    final_area = 0
    success_flag = False
    detection_results = []
    
    while not done:
        action = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if training_mode:
            EXPERIENCE_BUFFER.append((state, action, reward, next_state, done))
        
        state = next_state
        
        if print_debug:
            print(f"Episode {episode_num}/{total_episodes}: Step {step_count}, Reward {reward}, Total Reward {total_reward}")
        
        if 'detection_results' in info:
            detection_results.append(info['detection_results'])
        
        if 'final_area' in info:
            final_area = info['final_area']
        
        if 'climb_detected' in info:
            climb_detected = info['climb_detected']
            success_flag = climb_detected
    
    # 在训练模式下，需要更新智能体
    if training_mode and len(episode_memory.rewards) > 0:
        # 更新智能体
        ppo_agent.update(episode_memory)
        
        # 输出梯度和参数范数信息
        if ppo_agent.gradient_norms:
            avg_grad_norm = sum(ppo_agent.gradient_norms[-10:]) / len(ppo_agent.gradient_norms[-10:])
            avg_param_norm = sum(ppo_agent.parameter_norms[-10:]) / len(ppo_agent.parameter_norms[-10:])
            
            ppo_agent.logger.info(f"梯度范数: {avg_grad_norm:.4f}, 参数范数: {avg_param_norm:.4f}")
    
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

    # 初始化模型(使用正确的initialize_model版本)
    env, ppo_agent, visualizer, start_episode = initialize_model(model_path, load_existing=True)

    total_training_episodes = start_episode + config['EPISODES']
    print(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")

    # 启动可视化线程
    def run_visualizer():
        visualizer.run()

    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()

    # 使用train_with_visualization进行训练(已集成ExperienceCollector)
    # 训练循环
    for episode in range(start_episode, total_training_episodes):
        print(f"\n=== Episode {episode} Started ===")

        # 设置当前episode号
        env._current_episode = episode

        state = env.reset()
        memory = Memory()
        total_reward = 0

        for t in range(env.max_steps):
            # 先提取Ground-SAM特征（如果启用）
            ground_sam_features = None
            if CONFIG.get('USE_GROUND_SAM', False) and GROUND_SAM_AVAILABLE:
                try:
                    extractor = get_global_ground_sam_extractor()
                    if extractor is not None:
                        # 将state转换为numpy图像
                        if isinstance(state, torch.Tensor):
                            state_np = state.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        else:
                            state_np = state

                        # 提取Ground-SAM特征(不可视化,保持快速)
                        ground_sam_features = extractor.extract_features(
                            state_np,
                            use_auto_segmentation=False  # 不进行自动分割,保持快速
                        )

                        logger = logging.getLogger('ppo_agent_logger')
                except Exception as e:
                    logger = logging.getLogger('ppo_agent_logger')
                    logger.error(f"Ground-SAM特征提取失败: {e}")

            # 智能体执行动作
            move_action, turn_action, move_step, turn_angle, debug_info = ppo_agent.act(
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
                'episode': episode,
                'step': t,
                'total_reward': total_reward,
                'reward': reward,
                'done': done,
                'move_action': move_action,
                'turn_action': turn_action,
                'move_step': round(move_step, 2),
                'turn_angle': round(turn_angle, 2)
            })

            if done:
                break

        # 更新策略
        ppo_agent.update(memory)

        # 记录训练信息
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t+1}")

        # 保存检查点
        if episode % 10 == 0:
            ppo_agent.save_checkpoint(f'ppo_model_checkpoint_ep{episode}.pth')

        # 定期输出经验收集统计
        if episode % 50 == 0:
            collector = get_experience_collector()
            stats = collector.get_stats()
            print(f"经验收集统计: 总数={stats['total_experiences']}, 文件大小={stats['file_size_mb']:.2f}MB")

    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time

    print(f"继续训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")

    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"更新后的PPO模型已保存为 {model_path}")

    # 保存经验数据
    collector = get_experience_collector()
    collector.close()

    return {
        "status": "success",
        "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮，耗时 {training_duration:.2f} 秒",
        "final_episode": total_training_episodes,
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

    # 初始化模型(不加载现有模型)
    env, ppo_agent, visualizer, start_episode = initialize_model(model_path, load_existing=False)

    total_training_episodes = CONFIG['EPISODES']
    print(f"从第 {start_episode} 轮开始，训练 {CONFIG['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")

    # 启动可视化线程
    def run_visualizer():
        visualizer.run()

    visualizer_thread = Thread(target=run_visualizer, daemon=True)
    visualizer_thread.start()

    # 使用train_with_visualization进行训练
    for episode in range(start_episode, total_training_episodes):
        print(f"\n=== Episode {episode} Started ===")

        # 设置当前episode号
        env._current_episode = episode

        state = env.reset()
        memory = Memory()
        total_reward = 0

        for t in range(env.max_steps):
            # 先提取Ground-SAM特征（如果启用）
            ground_sam_features = None
            if CONFIG.get('USE_GROUND_SAM', False) and GROUND_SAM_AVAILABLE:
                try:
                    extractor = get_global_ground_sam_extractor()
                    if extractor is not None:
                        # 将state转换为numpy图像
                        if isinstance(state, torch.Tensor):
                            state_np = state.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        else:
                            state_np = state

                        # 提取Ground-SAM特征(不可视化,保持快速)
                        ground_sam_features = extractor.extract_features(
                            state_np,
                            use_auto_segmentation=False
                        )

                        logger = logging.getLogger('ppo_agent_logger')
                except Exception as e:
                    logger = logging.getLogger('ppo_agent_logger')
                    logger.error(f"Ground-SAM特征提取失败: {e}")

            # 智能体执行动作
            move_action, turn_action, move_step, turn_angle, debug_info = ppo_agent.act(
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
                'episode': episode,
                'step': t,
                'total_reward': total_reward,
                'reward': reward,
                'done': done,
                'move_action': move_action,
                'turn_action': turn_action,
                'move_step': round(move_step, 2),
                'turn_angle': round(turn_angle, 2)
            })

            if done:
                break

        # 更新策略
        ppo_agent.update(memory)

        # 记录训练信息
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t+1}")

        # 保存检查点
        if episode % 10 == 0:
            ppo_agent.save_checkpoint(f'ppo_model_checkpoint_ep{episode}.pth')

        # 定期输出经验收集统计
        if episode % 50 == 0:
            collector = get_experience_collector()
            stats = collector.get_stats()
            print(f"经验收集统计: 总数={stats['total_experiences']}, 文件大小={stats['file_size_mb']:.2f}MB")

    # 结束计时
    end_time = time.time()
    training_duration = end_time - start_time

    print(f"从头训练完成！")
    print(f"训练耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")

    torch.save(ppo_agent.policy.state_dict(), model_path)
    print(f"新训练的PPO模型已保存为 {model_path}")

    # 保存经验数据
    collector = get_experience_collector()
    collector.close()

    return {
        "status": "success",
        "message": f"从头训练完成，共训练了 {CONFIG['EPISODES']} 轮，耗时 {training_duration:.2f} 秒",
        "final_episode": total_training_episodes,
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

    # 如果智能体启用了SLAM地图，将slam_map_builder挂载到环境上，供可视化使用
    if hasattr(agent, 'slam_map_builder'):
        env.slam_map_builder = agent.slam_map_builder
        print("SLAM地图构建器已挂载到环境，可视化将显示SLAM小地图")

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
            # 先提取Ground-SAM特征（如果启用）
            ground_sam_features = None
            if CONFIG.get('USE_GROUND_SAM', False) and GROUND_SAM_AVAILABLE:
                try:
                    extractor = get_global_ground_sam_extractor()
                    if extractor is not None:
                        # 将state转换为numpy图像
                        if isinstance(state, torch.Tensor):
                            state_np = state.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        else:
                            state_np = state

                        # 提取Ground-SAM特征
                        ground_sam_features = extractor.extract_features(
                            state_np,
                            use_auto_segmentation=True
                        )

                        # 存储到环境中，用于后续保存图片
                        env.last_ground_sam_features = ground_sam_features

                        logger = logging.getLogger('ppo_agent_logger')
                        logger.info(f"步骤 {t}: 提取Ground-SAM特征完成")
                        if ground_sam_features:
                            logger.info(f"Ground-SAM特征键: {list(ground_sam_features.keys())}")
                            if 'auto_masks' in ground_sam_features and ground_sam_features['auto_masks']:
                                logger.info(f"自动分割检测到 {len(ground_sam_features['auto_masks'])} 个对象")
                            if 'mask_features' in ground_sam_features and ground_sam_features['mask_features']:
                                logger.info(f"提取到 {len(ground_sam_features['mask_features'])} 个掩码")
                except Exception as e:
                    logger = logging.getLogger('ppo_agent_logger')
                    logger.error(f"Ground-SAM特征提取失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # 智能体执行动作（使用Ground-SAM特征）
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