"""
第一阶段：点火时刻检测模型
- 1D-CNN: 捕捉局部突变特征
- BiLSTM: 建模时序依赖
- 混合模型: CNN + LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class IgnitionCNN(nn.Module):
    """
    1D-CNN 点火检测器
    适合捕捉残差序列中的局部突变特征
    """

    def __init__(
        self,
        input_channels: int = 3,      # RTN三通道
        window_size: int = 120,
        hidden_dims: list = [32, 64, 128],
        kernel_sizes: list = [7, 5, 3],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        self.input_channels = input_channels
        self.window_size = window_size

        # 卷积层
        layers = []
        in_channels = input_channels

        for i, (out_channels, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # 计算卷积后的特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, window_size)
            conv_out = self.conv_layers(dummy)
            self.conv_out_dim = conv_out.view(1, -1).size(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # 时间定位头 (回归点火时刻在窗口内的相对位置)
        self.time_head = nn.Sequential(
            nn.Linear(self.conv_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 [0, 1] 表示窗口内相对位置
        )

    def forward(
        self,
        x: torch.Tensor,
        return_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入序列 [batch, window_size, 3]
            return_time: 是否返回时间定位

        Returns:
            logits: 分类logits [batch, 2]
            time_pred: 相对时间预测 [batch, 1] (可选)
        """
        # [batch, window_size, channels] -> [batch, channels, window_size]
        x = x.permute(0, 2, 1)

        # 卷积特征提取
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)

        # 分类
        logits = self.fc(conv_features)

        if return_time:
            time_pred = self.time_head(conv_features)
            return logits, time_pred

        return logits, None


class IgnitionLSTM(nn.Module):
    """
    BiLSTM 点火检测器
    适合建模长程时序依赖
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类头
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 时间定位头
        self.time_head = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入序列 [batch, seq_len, 3]

        Returns:
            logits: 分类logits [batch, 2]
            time_pred: 相对时间预测 [batch, 1]
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # 注意力加权
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]

        # 分类
        logits = self.fc(context)

        if return_time:
            time_pred = self.time_head(context)
            return logits, time_pred

        return logits, None


class IgnitionCNNLSTM(nn.Module):
    """
    CNN + LSTM 混合模型
    CNN提取局部特征，LSTM建模时序关系
    """

    def __init__(
        self,
        input_channels: int = 3,
        window_size: int = 120,
        cnn_channels: list = [32, 64],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        # CNN特征提取
        cnn_layers = []
        in_channels = input_channels
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 时间定位头
        self.time_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_time: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [batch, seq, channels] -> [batch, channels, seq]
        x = x.permute(0, 2, 1)

        # CNN特征
        cnn_out = self.cnn(x)  # [batch, cnn_channels[-1], reduced_seq]

        # [batch, channels, seq] -> [batch, seq, channels]
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(cnn_out)

        # 使用最后时刻的隐藏状态
        h_n = h_n.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        last_hidden = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)

        # 分类
        logits = self.fc(last_hidden)

        if return_time:
            time_pred = self.time_head(last_hidden)
            return logits, time_pred

        return logits, None


class IgnitionDetector:
    """
    点火时刻检测器封装类
    包含模型推理和后处理
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        threshold: float = 0.5,
        nms_window: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.nms_window = nms_window
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        windows: np.ndarray,
        window_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        检测点火时刻

        Args:
            windows: 滑动窗口数据 [N, window_size, 3]
            window_times: 窗口中心时间 [N]

        Returns:
            dict: 包含检测结果
                - probabilities: 机动概率 [N]
                - detections: 检测到的机动索引
                - ignition_times: 精确点火时刻
        """
        # 转换为tensor
        x = torch.FloatTensor(windows).to(self.device)

        # 批量推理
        batch_size = 256
        all_probs = []
        all_times = []

        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            logits, time_pred = self.model(batch, return_time=True)

            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())

            if time_pred is not None:
                all_times.append(time_pred.cpu().numpy())

        probabilities = np.concatenate(all_probs)
        relative_times = np.concatenate(all_times) if all_times else None

        # 阈值过滤
        candidates = np.where(probabilities > self.threshold)[0]

        # 非极大值抑制
        detections = self._nms(candidates, probabilities)

        # 计算精确点火时刻
        ignition_times = []
        for det_idx in detections:
            if relative_times is not None:
                # 使用时间回归头的输出
                rel_t = relative_times[det_idx, 0]
                window_size = windows.shape[1]
                # 相对位置转实际时间偏移
                time_offset = (rel_t - 0.5) * window_size * 60  # 假设60秒采样间隔
                t_ignition = window_times[det_idx] + time_offset
            else:
                t_ignition = window_times[det_idx]
            ignition_times.append(t_ignition)

        return {
            'probabilities': probabilities,
            'detections': detections,
            'ignition_times': np.array(ignition_times) if ignition_times else np.array([])
        }

    def _nms(
        self,
        candidates: np.ndarray,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """非极大值抑制"""
        if len(candidates) == 0:
            return np.array([])

        # 按概率排序
        sorted_indices = np.argsort(probabilities[candidates])[::-1]
        candidates = candidates[sorted_indices]

        keep = []
        suppressed = set()

        for idx in candidates:
            if idx in suppressed:
                continue

            keep.append(idx)

            # 抑制邻近检测
            for j in range(idx - self.nms_window, idx + self.nms_window + 1):
                if j >= 0 and j < len(probabilities):
                    suppressed.add(j)

        return np.array(keep)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'nms_window': self.nms_window
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint.get('threshold', 0.5)
        self.nms_window = checkpoint.get('nms_window', 10)
        self.model.eval()
