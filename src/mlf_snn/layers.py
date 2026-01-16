"""
SNN 层模块
包含线性层和卷积层的脉冲神经网络实现
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .neurons import LIFNeuron, MLFNeuron


class SNNLinear(nn.Module):
    """
    SNN 全连接层
    线性变换 + 脉冲神经元
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neuron_type: str = 'lif',
        threshold: float = 1.0,
        decay: float = 0.9,
        bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if neuron_type == 'mlf':
            self.neuron = MLFNeuron(decay=decay)
        else:
            self.neuron = LIFNeuron(threshold=threshold, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, in_features]
        Returns:
            spike: 脉冲输出 [batch, out_features]
        """
        current = self.linear(x)
        spike = self.neuron(current)
        return spike

    def reset(self):
        """重置神经元状态"""
        self.neuron.membrane = None


class SNNConv1d(nn.Module):
    """
    SNN 1D卷积层
    卷积 + 脉冲神经元
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        neuron_type: str = 'lif',
        threshold: float = 1.0,
        decay: float = 0.9
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)

        if neuron_type == 'mlf':
            self.neuron = MLFNeuron(decay=decay)
        else:
            self.neuron = LIFNeuron(threshold=threshold, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, channels, length]
        Returns:
            spike: 脉冲输出
        """
        current = self.bn(self.conv(x))
        spike = self.neuron(current)
        return spike

    def reset(self):
        """重置神经元状态"""
        self.neuron.membrane = None


class SNNReadout(nn.Module):
    """
    SNN 读出层
    将脉冲序列转换为分类输出
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        readout_mode: str = 'mean'
    ):
        """
        Args:
            readout_mode: 'mean' (平均发放率), 'last' (最后时刻), 'max' (最大值)
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.readout_mode = readout_mode

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_train: 脉冲序列 [batch, time, features]
        Returns:
            output: 分类输出 [batch, out_features]
        """
        if self.readout_mode == 'mean':
            x = spike_train.mean(dim=1)
        elif self.readout_mode == 'last':
            x = spike_train[:, -1, :]
        elif self.readout_mode == 'max':
            x = spike_train.max(dim=1)[0]
        else:
            x = spike_train.mean(dim=1)

        return self.linear(x)
