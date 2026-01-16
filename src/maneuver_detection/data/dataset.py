"""
PyTorch 数据集定义
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import h5py
from pathlib import Path


class ManeuverDataset(Dataset):
    """
    机动检测数据集 (第一阶段)
    用于训练点火时刻检测器
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        times: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Args:
            windows: 残差窗口数据 [N, window_size, 3]
            labels: 二分类标签 [N] (0: 无机动, 1: 有机动)
            times: 窗口中心时间 [N]
            transform: 数据增强变换
        """
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)
        self.times = times
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.windows[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class DeltaVDataset(Dataset):
    """
    Δv 回归数据集 (第二阶段)
    """

    def __init__(
        self,
        features: np.ndarray,
        delta_v: np.ndarray,
        transform=None
    ):
        """
        Args:
            features: P/T/R特征 [N, n_features]
            delta_v: 真实Δv [N, 3] (RTN分量) 或 [N, 1] (标量)
            transform: 数据变换
        """
        self.features = torch.FloatTensor(features)
        self.delta_v = torch.FloatTensor(delta_v)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.delta_v)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.delta_v[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class SequenceDataset(Dataset):
    """
    序列到序列数据集
    用于端到端训练 (同时检测时刻和估计Δv)
    """

    def __init__(
        self,
        sequences: np.ndarray,
        maneuver_masks: np.ndarray,
        delta_v_labels: np.ndarray,
        ignition_times: Optional[np.ndarray] = None
    ):
        """
        Args:
            sequences: 残差序列 [N, seq_len, 3]
            maneuver_masks: 机动掩码 [N, seq_len] (每个时间点是否机动)
            delta_v_labels: Δv标签 [N, 3]
            ignition_times: 点火时刻索引 [N]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.maneuver_masks = torch.FloatTensor(maneuver_masks)
        self.delta_v_labels = torch.FloatTensor(delta_v_labels)
        self.ignition_times = ignition_times

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return {
            'sequence': self.sequences[idx],
            'mask': self.maneuver_masks[idx],
            'delta_v': self.delta_v_labels[idx],
            'ignition_idx': self.ignition_times[idx] if self.ignition_times is not None else -1
        }


class DataAugmentation:
    """数据增强"""

    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_shift_range: int = 5
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_range = time_shift_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 随机缩放
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.empty(1).uniform_(*self.scale_range)
            x = x * scale

        return x


def create_data_loaders(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Returns:
        train_loader, val_loader
    """
    transform = DataAugmentation() if augment else None

    train_dataset = ManeuverDataset(
        train_windows, train_labels, transform=transform
    )
    val_dataset = ManeuverDataset(val_windows, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def load_from_hdf5(filepath: str) -> Dict[str, np.ndarray]:
    """从HDF5文件加载数据"""
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


def save_to_hdf5(filepath: str, data: Dict[str, np.ndarray]):
    """保存数据到HDF5文件"""
    with h5py.File(filepath, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value, compression='gzip')
