"""
经验数据重放训练模块

此模块提供从保存的经验数据中加载和训练PPO智能体的功能。

使用方法:
1. 从HDF5文件加载经验数据
2. 使用加载的数据进行PPO训练
3. 可以在离线模式下训练,无需与游戏环境交互
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
import argparse


def load_experience_data(experience_path: str) -> Optional[Dict]:
    """
    从HDF5文件加载经验数据

    Args:
        experience_path: 经验数据文件路径

    Returns:
        包含经验数据的字典,如果加载失败则返回None
    """
    if not os.path.exists(experience_path):
        print(f"经验文件不存在: {experience_path}")
        return None

    try:
        print(f"正在加载经验数据: {experience_path}")
        with h5py.File(experience_path, 'r') as h5_file:
            total = h5_file['metadata'].attrs.get('total_experiences', h5_file['states'].shape[0])

            data = {
                'states': h5_file['states'][:],  # (N, 3, H, W)
                'move_actions': h5_file['move_actions'][:],  # (N,)
                'turn_actions': h5_file['turn_actions'][:],  # (N,)
                'move_steps': h5_file['move_steps'][:],  # (N,)
                'turn_angles': h5_file['turn_angles'][:],  # (N,)
                'rewards': h5_file['rewards'][:],  # (N,)
                'dones': h5_file['dones'][:],  # (N,)
                'total': total
            }

            print(f"成功加载 {total} 条经验")
            print(f"状态形状: {data['states'].shape}")
            print(f"动作形状: move={data['move_actions'].shape}, turn={data['turn_actions'].shape}")
            print(f"奖励统计: mean={data['rewards'].mean():.3f}, std={data['rewards'].std():.3f}")

            return data

    except Exception as e:
        print(f"加载经验数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_experience_data(data: Dict, batch_size: int = 32, shuffle: bool = True):
    """
    将经验数据分批

    Args:
        data: 经验数据字典
        batch_size: 批次大小
        shuffle: 是否打乱数据

    Yields:
        批次数据
    """
    n_samples = data['states'].shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_data = {
            'states': data['states'][batch_indices],
            'move_actions': data['move_actions'][batch_indices],
            'turn_actions': data['turn_actions'][batch_indices],
            'move_steps': data['move_steps'][batch_indices],
            'turn_angles': data['turn_angles'][batch_indices],
            'rewards': data['rewards'][batch_indices],
            'dones': data['dones'][batch_indices]
        }
        yield batch_data


class OfflinePPOTrainer:
    """
    离线PPO训练器

    使用保存的经验数据进行PPO训练,无需与环境交互
    """

    def __init__(self, experience_path: str, config_path: str = None):
        """
        初始化训练器

        Args:
            experience_path: 经验数据文件路径
            config_path: 配置文件路径
        """
        self.experience_path = experience_path
        self.experience_data = None
        self.config = self._load_config(config_path)

        # 初始化日志
        self.logger = self._setup_logging()

        # 加载经验数据
        self.experience_data = load_experience_data(experience_path)
        if self.experience_data is None:
            raise ValueError(f"无法加载经验数据: {experience_path}")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import json

        if config_path is None:
            config_path = Path(__file__).parent / "ppo_config.json"

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 返回默认配置
            return {
                'LEARNING_RATE': 0.0003,
                'GAMMA': 0.99,
                'K_EPOCHS': 4,
                'EPS_CLIP': 0.2,
                'IMAGE_HEIGHT': 480,
                'IMAGE_WIDTH': 640
            }

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('offline_ppo_trainer')
        logger.setLevel(logging.INFO)

        # 清除已有处理器
        logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def train(self, n_updates: int = 1000, batch_size: int = 32):
        """
        使用经验数据进行训练

        Args:
            n_updates: 更新次数
            batch_size: 批次大小
        """
        print(f"开始离线训练,共 {n_updates} 次更新")

        for update_idx in range(n_updates):
            total_loss = 0

            for batch_idx, batch in enumerate(batch_experience_data(self.experience_data, batch_size, shuffle=True)):
                # 在这里实现PPO训练逻辑
                # 需要与PPOAgent的训练逻辑一致
                loss = self._train_batch(batch)
                total_loss += loss

            avg_loss = total_loss / (batch_idx + 1) if batch_idx >= 0 else 0

            if (update_idx + 1) % 10 == 0:
                print(f"Update {update_idx + 1}/{n_updates}, Avg Loss: {avg_loss:.4f}")

        print("训练完成!")

    def _train_batch(self, batch: Dict) -> float:
        """
        训练单个批次

        Args:
            batch: 批次数据

        Returns:
            损失值
        """
        # 这里需要实现PPO的损失计算和优化
        # 由于需要访问PPOAgent的网络,这里只是一个占位符
        # 实际使用时需要传入PPOAgent实例

        # 示例: 假设我们计算一个简单的MSE损失
        loss = 0.0
        return loss


def train_with_saved_experience(experience_path: str, model_path: str, n_updates: int = 1000):
    """
    使用保存的经验训练模型

    Args:
        experience_path: 经验数据文件路径
        model_path: 模型保存路径
        n_updates: 训练更新次数
    """
    # 加载经验数据
    data = load_experience_data(experience_path)
    if data is None:
        print("无法加载经验数据")
        return

    # 创建训练器
    trainer = OfflinePPOTrainer(experience_path)

    # 开始训练
    trainer.train(n_updates=n_updates, batch_size=32)

    # 保存模型
    print(f"训练完成,模型将保存到: {model_path}")


def list_experience_files(experience_dir: str = "./experience_data"):
    """
    列出所有经验文件

    Args:
        experience_dir: 经验数据目录

    Returns:
        经验文件列表
    """
    experience_path = Path(experience_dir)

    if not experience_path.exists():
        print(f"经验目录不存在: {experience_dir}")
        return []

    h5_files = list(experience_path.glob("*.h5"))

    if not h5_files:
        print(f"未找到经验文件")
        return []

    print(f"\n找到 {len(h5_files)} 个经验文件:")
    for h5_file in sorted(h5_files):
        file_size = h5_file.stat().st_size / (1024 * 1024)  # MB

        # 读取文件信息
        try:
            with h5py.File(h5_file, 'r') as h5_file_obj:
                total = h5_file_obj['metadata'].attrs.get('total_experiences', h5_file_obj['states'].shape[0])
                print(f"  {h5_file.name}")
                print(f"    经验数: {total}")
                print(f"    文件大小: {file_size:.2f} MB")
        except Exception as e:
            print(f"  {h5_file.name} (无法读取: {e})")

    return h5_files


def analyze_experience(experience_path: str):
    """
    分析经验数据的统计信息

    Args:
        experience_path: 经验数据文件路径
    """
    data = load_experience_data(experience_path)
    if data is None:
        return

    print("\n=== 经验数据分析 ===")
    print(f"总经验数: {data['total']}")
    print(f"\n状态统计:")
    print(f"  形状: {data['states'].shape}")
    print(f"  数值范围: [{data['states'].min()}, {data['states'].max()}]")

    print(f"\n移动动作分布:")
    unique, counts = np.unique(data['move_actions'], return_counts=True)
    for action, count in zip(unique, counts):
        print(f"  Action {int(action)}: {count} ({count/len(data['move_actions'])*100:.1f}%)")

    print(f"\n转向动作分布:")
    unique, counts = np.unique(data['turn_actions'], return_counts=True)
    for action, count in zip(unique, counts):
        print(f"  Action {int(action)}: {count} ({count/len(data['turn_actions'])*100:.1f}%)")

    print(f"\n移动步长统计:")
    print(f"  Mean: {data['move_steps'].mean():.3f}")
    print(f"  Std: {data['move_steps'].std():.3f}")
    print(f"  Min: {data['move_steps'].min():.3f}")
    print(f"  Max: {data['move_steps'].max():.3f}")

    print(f"\n转向角度统计:")
    print(f"  Mean: {data['turn_angles'].mean():.3f}")
    print(f"  Std: {data['turn_angles'].std():.3f}")
    print(f"  Min: {data['turn_angles'].min():.3f}")
    print(f"  Max: {data['turn_angles'].max():.3f}")

    print(f"\n奖励统计:")
    print(f"  Mean: {data['rewards'].mean():.3f}")
    print(f"  Std: {data['rewards'].std():.3f}")
    print(f"  Min: {data['rewards'].min():.3f}")
    print(f"  Max: {data['rewards'].max():.3f}")
    print(f"  正奖励数: {np.sum(data['rewards'] > 0)}")
    print(f"  负奖励数: {np.sum(data['rewards'] < 0)}")
    print(f"  零奖励数: {np.sum(data['rewards'] == 0)}")

    print(f"\nDone标志统计:")
    print(f"  True: {np.sum(data['dones'])} ({np.sum(data['dones'])/len(data['dones'])*100:.1f}%)")
    print(f"  False: {np.sum(~data['dones'])} ({np.sum(~data['dones'])/len(data['dones'])*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="经验数据训练工具")
    parser.add_argument("command", choices=["list", "analyze", "train"],
                       help="命令: list-列出文件, analyze-分析数据, train-训练模型")
    parser.add_argument("--path", type=str, default="./experience_data",
                       help="经验文件路径或目录")
    parser.add_argument("--output", type=str, default="./model/offline_trained_model.pth",
                       help="训练输出模型路径")
    parser.add_argument("--updates", type=int, default=1000,
                       help="训练更新次数")

    args = parser.parse_args()

    if args.command == "list":
        list_experience_files(args.path)
    elif args.command == "analyze":
        analyze_experience(args.path)
    elif args.command == "train":
        train_with_saved_experience(args.path, args.output, args.updates)
