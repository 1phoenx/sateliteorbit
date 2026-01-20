"""
神经网络特征提取模块
使用1D UNet从原始时序数据中提取P/T/R特征

相比传统数学方法的优势:
1. 端到端学习 - 自动学习最优特征表示
2. 噪声鲁棒性 - 通过训练学习去噪
3. 非线性特征 - 捕获复杂的时序模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class ConvBlock1D(nn.Module):
    """1D卷积块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock1D(nn.Module):
    """下采样块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        conv_out = self.conv(x)
        return self.pool(conv_out), conv_out


class UpBlock1D(nn.Module):
    """上采样块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = F.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D UNet用于时序特征提取

    输入: (batch, 1, seq_length) - 原始推力时序
    输出: (batch, 3) - P, T, R特征
    """

    def __init__(self, seq_length: int = 1000, base_channels: int = 32):
        super().__init__()

        self.seq_length = seq_length

        # 编码器
        self.down1 = DownBlock1D(1, base_channels)
        self.down2 = DownBlock1D(base_channels, base_channels * 2)
        self.down3 = DownBlock1D(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock1D(base_channels * 4, base_channels * 8)

        # 瓶颈层
        self.bottleneck = ConvBlock1D(base_channels * 8, base_channels * 16)

        # 解码器
        self.up1 = UpBlock1D(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock1D(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock1D(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock1D(base_channels * 2, base_channels)

        # 特征提取头
        self.feature_conv = nn.Conv1d(base_channels, 8, kernel_size=1)

        # 全局池化 + 全连接层输出P/T/R
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # 输出 P, T, R
        )

        # 点火时刻检测头 (逐点分类)
        self.ignition_head = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, seq_length) 原始推力时序

        Returns:
            features: (batch, 3) P, T, R特征
            ignition_prob: (batch, seq_length) 点火概率序列
        """
        # 编码
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        # 瓶颈
        x = self.bottleneck(x4)

        # 解码
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # 特征提取
        feat = self.feature_conv(x)
        feat_pooled = self.global_pool(feat).squeeze(-1)
        features = self.fc(feat_pooled)

        # 点火时刻检测
        ignition_logits = self.ignition_head(x)
        ignition_prob = torch.sigmoid(ignition_logits).squeeze(1)

        return features, ignition_prob


class FeatureExtractorCNN(nn.Module):
    """
    简化版CNN特征提取器

    相比UNet更轻量，适合快速推理
    """

    def __init__(self, seq_length: int = 1000):
        super().__init__()

        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            # 第二层
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            # 第三层
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # 全局池化
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3)  # P, T, R
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, seq_length)

        Returns:
            (batch, 3) - P, T, R
        """
        feat = self.encoder(x)
        feat = feat.squeeze(-1)
        return self.fc(feat)


class NeuralFeatureExtractor:
    """神经网络特征提取器封装类"""

    def __init__(
        self,
        model_type: str = 'cnn',  # 'unet' or 'cnn'
        seq_length: int = 1000,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.model_type = model_type

        if model_type == 'unet':
            self.model = UNet1D(seq_length).to(self.device)
        else:
            self.model = FeatureExtractorCNN(seq_length).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 特征标准化参数
        self.feature_mean = None
        self.feature_std = None

    def train(
        self,
        thrust_data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2
    ) -> Dict:
        """
        训练特征提取器

        Args:
            thrust_data: (N, seq_length) 推力时序数据
            labels: (N, 3) P, T, R真实值
            epochs: 训练轮数
            batch_size: 批大小
            val_split: 验证集比例

        Returns:
            训练历史
        """
        # 数据准备
        n_samples = len(thrust_data)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # 标准化
        self.feature_mean = labels[train_idx].mean(axis=0)
        self.feature_std = labels[train_idx].std(axis=0) + 1e-8
        labels_normalized = (labels - self.feature_mean) / self.feature_std

        # 转换为张量
        X_train = torch.FloatTensor(thrust_data[train_idx]).unsqueeze(1).to(self.device)
        y_train = torch.FloatTensor(labels_normalized[train_idx]).to(self.device)
        X_val = torch.FloatTensor(thrust_data[val_idx]).unsqueeze(1).to(self.device)
        y_val = torch.FloatTensor(labels_normalized[val_idx]).to(self.device)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                self.optimizer.zero_grad()

                if self.model_type == 'unet':
                    pred, _ = self.model(batch_X)
                else:
                    pred = self.model(batch_X)

                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            # 验证
            self.model.eval()
            with torch.no_grad():
                if self.model_type == 'unet':
                    val_pred, _ = self.model(X_val)
                else:
                    val_pred = self.model(X_val)
                val_loss = F.mse_loss(val_pred, y_val).item()

            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss / n_batches)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/n_batches:.4f}, Val Loss: {val_loss:.4f}")

        return history

    def extract_features(self, thrust_data: np.ndarray) -> np.ndarray:
        """
        提取特征

        Args:
            thrust_data: (N, seq_length) 或 (seq_length,)

        Returns:
            (N, 3) 或 (3,) - P, T, R特征
        """
        self.model.eval()

        if thrust_data.ndim == 1:
            thrust_data = thrust_data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        # 调整序列长度
        if thrust_data.shape[1] != self.seq_length:
            # 使用插值调整长度
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, thrust_data.shape[1])
            x_new = np.linspace(0, 1, self.seq_length)
            thrust_data_resized = np.array([
                interp1d(x_old, row, kind='linear')(x_new)
                for row in thrust_data
            ])
        else:
            thrust_data_resized = thrust_data

        X = torch.FloatTensor(thrust_data_resized).unsqueeze(1).to(self.device)

        with torch.no_grad():
            if self.model_type == 'unet':
                features, _ = self.model(X)
            else:
                features = self.model(X)

            features = features.cpu().numpy()

        # 反标准化
        if self.feature_mean is not None:
            features = features * self.feature_std + self.feature_mean

        if squeeze_output:
            features = features.squeeze(0)

        return features

    def detect_ignition_time(self, thrust_data: np.ndarray, sampling_rate: float = 100.0) -> float:
        """
        检测点火时刻 (仅UNet支持)

        Args:
            thrust_data: (seq_length,) 推力时序

        Returns:
            点火时刻 (秒)
        """
        if self.model_type != 'unet':
            raise ValueError("点火时刻检测仅UNet模型支持")

        self.model.eval()

        # 调整序列长度
        if len(thrust_data) != self.seq_length:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(thrust_data))
            x_new = np.linspace(0, 1, self.seq_length)
            thrust_data = interp1d(x_old, thrust_data, kind='linear')(x_new)

        X = torch.FloatTensor(thrust_data).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, ignition_prob = self.model(X)
            ignition_prob = ignition_prob.squeeze().cpu().numpy()

        # 找到概率最大的点
        ignition_idx = np.argmax(ignition_prob)

        # 转换为原始时间
        original_length = len(thrust_data)
        ignition_time = (ignition_idx / self.seq_length) * (original_length / sampling_rate)

        return ignition_time

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'seq_length': self.seq_length,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }, filepath)

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.feature_mean = checkpoint['feature_mean']
        self.feature_std = checkpoint['feature_std']


class HybridFeatureExtractor:
    """
    混合特征提取器

    结合传统数学方法和神经网络方法的优势
    """

    def __init__(self, sampling_rate: float = 100.0, device: str = None):
        self.sampling_rate = sampling_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 传统方法
        from src.feature_extraction_v2 import ThrusterFeatureExtractor
        self.traditional_extractor = ThrusterFeatureExtractor(sampling_rate)

        # 神经网络方法 (可选)
        self.neural_extractor = None

    def enable_neural_extraction(self, model_path: str = None):
        """启用神经网络特征提取"""
        self.neural_extractor = NeuralFeatureExtractor(
            model_type='cnn',
            device=self.device
        )
        if model_path:
            self.neural_extractor.load(model_path)

    def extract_features(
        self,
        thrust: np.ndarray,
        ton: np.ndarray,
        use_neural: bool = False
    ) -> Dict:
        """
        提取特征

        Args:
            thrust: 推力时序
            ton: 推力器开关状态
            use_neural: 是否使用神经网络

        Returns:
            特征字典
        """
        # 传统方法
        B_thrust, sigma = self.traditional_extractor.compute_baseline(thrust, ton)
        P_trad = self.traditional_extractor.extract_P(thrust, ton, B_thrust, sigma)
        T_trad, ignition_time, true_thrust = self.traditional_extractor.extract_T(
            thrust, B_thrust, sigma, ton
        )
        R_trad = self.traditional_extractor.extract_R(thrust, T_trad)

        result = {
            'P': P_trad,
            'T': T_trad,
            'R': R_trad,
            'ignition_time': ignition_time,
            'true_thrust': true_thrust,
            'method': 'traditional'
        }

        # 神经网络方法 (如果启用)
        if use_neural and self.neural_extractor is not None:
            neural_features = self.neural_extractor.extract_features(thrust)
            result['P_neural'] = neural_features[0]
            result['T_neural'] = neural_features[1]
            result['R_neural'] = neural_features[2]

            # 融合: 加权平均
            alpha = 0.7  # 传统方法权重
            result['P_fused'] = alpha * P_trad + (1 - alpha) * neural_features[0]
            result['T_fused'] = alpha * T_trad + (1 - alpha) * neural_features[1]
            result['R_fused'] = alpha * R_trad + (1 - alpha) * neural_features[2]
            result['method'] = 'hybrid'

        return result
