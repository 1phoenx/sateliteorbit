"""
MLF-SNN 网络架构
用于变轨检测的脉冲神经网络
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from .neurons import LIFNeuron, MLFNeuron
from .layers import SNNLinear, SNNConv1d, SNNReadout
from .encoding import DirectEncoder, RateEncoder


class MLFSNN(nn.Module):
    """
    MLF-SNN 多阈值脉冲神经网络
    用于变轨检测任务
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 2,
        time_steps: int = 16,
        neuron_type: str = 'mlf',
        decay: float = 0.9,
        readout_mode: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_steps = time_steps
        self.neuron_type = neuron_type

        # 编码层
        self.encoder = RateEncoder(time_steps=time_steps)

        # 构建SNN层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(SNNLinear(
                prev_dim, hidden_dim,
                neuron_type=neuron_type,
                decay=decay
            ))
            prev_dim = hidden_dim

        self.snn_layers = nn.ModuleList(layers)

        # 读出层
        self.readout = SNNReadout(
            prev_dim, output_dim,
            readout_mode=readout_mode
        )

    def reset_states(self):
        """重置所有神经元状态"""
        for layer in self.snn_layers:
            layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch, features]
        Returns:
            output: 分类输出 [batch, output_dim]
        """
        batch_size = x.shape[0]
        self.reset_states()

        # 归一化到 [0, 1]
        x_norm = torch.sigmoid(x)

        # 编码为脉冲序列
        spikes = self.encoder(x_norm)

        # 收集每个时间步的输出
        spike_outputs = []

        for t in range(self.time_steps):
            h = spikes[:, t, :]

            for layer in self.snn_layers:
                h = layer(h)

            spike_outputs.append(h)

        # 堆叠时间维度
        spike_train = torch.stack(spike_outputs, dim=1)

        # 读出
        output = self.readout(spike_train)

        return output

    def get_spike_statistics(self) -> Dict[str, float]:
        """获取脉冲统计信息"""
        stats = {}
        for i, layer in enumerate(self.snn_layers):
            if hasattr(layer.neuron, 'membrane') and layer.neuron.membrane is not None:
                stats[f'layer_{i}_mean_membrane'] = layer.neuron.membrane.mean().item()
        return stats


class SNNClassifier(nn.Module):
    """
    SNN 分类器
    支持时序输入的脉冲神经网络分类器
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 2,
        neuron_type: str = 'mlf',
        decay: float = 0.9
    ):
        super().__init__()

        self.input_dim = input_dim
        self.neuron_type = neuron_type

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # SNN层
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(SNNLinear(
                hidden_dims[i], hidden_dims[i + 1],
                neuron_type=neuron_type,
                decay=decay
            ))
        self.snn_layers = nn.ModuleList(layers)

        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # 神经元
        if neuron_type == 'mlf':
            self.input_neuron = MLFNeuron(decay=decay)
        else:
            self.input_neuron = LIFNeuron(decay=decay)

    def reset_states(self):
        """重置状态"""
        self.input_neuron.membrane = None
        for layer in self.snn_layers:
            layer.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 时序输入 [batch, time, features]
        Returns:
            output: 分类输出 [batch, output_dim]
        """
        batch_size, time_steps, _ = x.shape
        self.reset_states()

        outputs = []

        for t in range(time_steps):
            h = self.input_proj(x[:, t, :])
            h = self.input_neuron(h)

            for layer in self.snn_layers:
                h = layer(h)

            outputs.append(h)

        # 平均池化
        h_mean = torch.stack(outputs, dim=1).mean(dim=1)
        output = self.output_layer(h_mean)

        return output
