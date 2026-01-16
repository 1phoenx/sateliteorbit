"""
脉冲神经元模型
包含 LIF 神经元和 MLF 多阈值神经元
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from .surrogate import spike_function, multi_spike_function


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) 神经元
    标准单阈值脉冲神经元
    """

    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        reset_mode: str = 'soft',
        surrogate: str = 'fast_sigmoid'
    ):
        """
        Args:
            threshold: 发放阈值
            decay: 膜电位衰减系数 (beta)
            reset_mode: 重置模式 ('soft' 或 'hard')
            surrogate: 替代梯度类型
        """
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.reset_mode = reset_mode
        self.surrogate = surrogate
        self.membrane = None

    def reset_state(self, batch_size: int, device: torch.device):
        """重置膜电位状态"""
        self.membrane = torch.zeros(batch_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单时间步前向传播

        Args:
            x: 输入电流 [batch_size] 或 [batch_size, features]

        Returns:
            spike: 脉冲输出
        """
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)

        # 膜电位更新: u(t+1) = beta * u(t) + x(t)
        self.membrane = self.decay * self.membrane + x

        # 脉冲发放
        spike = spike_function(self.membrane, self.threshold, self.surrogate)

        # 膜电位重置
        if self.reset_mode == 'hard':
            self.membrane = self.membrane * (1 - spike)
        else:  # soft reset
            self.membrane = self.membrane - spike * self.threshold

        return spike


class MLFNeuron(nn.Module):
    """
    Multi-Level Firing (MLF) 多阈值神经元
    支持 0.6/1.6/2.6 三阈值发放，输出 0/1/2/3 多级脉冲
    """

    def __init__(
        self,
        thresholds: List[float] = None,
        decay: float = 0.9,
        reset_mode: str = 'soft'
    ):
        """
        Args:
            thresholds: 多阈值列表，默认 [0.6, 1.6, 2.6]
            decay: 膜电位衰减系数
            reset_mode: 重置模式
        """
        super().__init__()
        self.thresholds = thresholds or [0.6, 1.6, 2.6]
        self.decay = decay
        self.reset_mode = reset_mode
        self.membrane = None

    def reset_state(self, batch_size: int, device: torch.device):
        """重置膜电位状态"""
        self.membrane = torch.zeros(batch_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单时间步前向传播

        Args:
            x: 输入电流

        Returns:
            spike: 多级脉冲输出 (0, 1, 2, 3)
        """
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)

        # 膜电位更新
        self.membrane = self.decay * self.membrane + x

        # 多阈值脉冲发放
        spike = multi_spike_function(self.membrane, self.thresholds)

        # 膜电位重置（根据发放级别）
        if self.reset_mode == 'hard':
            self.membrane = self.membrane * (spike == 0).float()
        else:
            # 软重置：减去对应阈值
            reset_value = torch.zeros_like(spike)
            for i, theta in enumerate(self.thresholds):
                reset_value = torch.where(
                    spike >= (i + 1),
                    torch.tensor(theta, device=spike.device),
                    reset_value
                )
            self.membrane = self.membrane - reset_value

        return spike

    @property
    def info_capacity(self) -> float:
        """信息容量：每次发放携带的比特数"""
        import math
        return math.log2(len(self.thresholds) + 1)
