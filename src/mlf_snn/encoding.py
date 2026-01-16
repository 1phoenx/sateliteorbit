"""
脉冲编码模块
将连续值特征转换为脉冲序列
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class RateEncoder(nn.Module):
    """
    速率编码器
    将特征值转换为发放概率，生成泊松脉冲序列
    """

    def __init__(self, time_steps: int = 16):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, features]，值域 [0, 1]
        Returns:
            spikes: 脉冲序列 [batch, time_steps, features]
        """
        # 确保输入在 [0, 1] 范围
        x = torch.clamp(x, 0, 1)

        # 生成脉冲序列
        batch_size, n_features = x.shape
        spikes = torch.zeros(batch_size, self.time_steps, n_features, device=x.device)

        for t in range(self.time_steps):
            rand = torch.rand_like(x)
            spikes[:, t, :] = (rand < x).float()

        return spikes


class DirectEncoder(nn.Module):
    """
    直接编码器
    将时序特征直接作为输入电流
    """

    def __init__(self, scale: float = 1.0, normalize: bool = True):
        super().__init__()
        self.scale = scale
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入时序 [batch, time, features]
        Returns:
            current: 输入电流 [batch, time, features]
        """
        if self.normalize:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + 1e-8
            x = (x - mean) / std

        return x * self.scale


class TemporalEncoder(nn.Module):
    """
    时间编码器
    基于时间到首次脉冲 (Time-to-First-Spike) 编码
    """

    def __init__(self, time_steps: int = 16, tau: float = 1.0):
        super().__init__()
        self.time_steps = time_steps
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, features]，值域 [0, 1]
        Returns:
            spikes: 脉冲序列 [batch, time_steps, features]
        """
        x = torch.clamp(x, 0.01, 1)
        batch_size, n_features = x.shape

        # 计算首次发放时间：值越大，发放越早
        spike_times = (1 - x) * self.time_steps
        spike_times = spike_times.long().clamp(0, self.time_steps - 1)

        # 生成脉冲序列
        spikes = torch.zeros(batch_size, self.time_steps, n_features, device=x.device)

        for b in range(batch_size):
            for f in range(n_features):
                t = spike_times[b, f].item()
                spikes[b, t, f] = 1.0

        return spikes
