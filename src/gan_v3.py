"""
GAN网络优化版：使用LayerNorm + GELU激活函数
用于小样本数据扩充，支持原始时序数据扩充

优化点:
1. 使用LayerNorm替代BatchNorm - 对小批量更稳定
2. 使用GELU替代LeakyReLU - 更平滑的激活函数
3. 添加残差连接 - 提升训练稳定性
4. 支持条件生成 (cGAN) - 按类别生成样本
5. 支持原始时序数据扩充
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import os


class ResidualBlock(nn.Module):
    """残差块 with LayerNorm + GELU"""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class GeneratorV2(nn.Module):
    """优化版生成器 - LayerNorm + GELU + 残差连接"""

    def __init__(self, latent_dim: int = 100, output_dim: int = 3, condition_dim: int = 0):
        """
        Args:
            latent_dim: 潜在空间维度
            output_dim: 输出特征维度 (P, T, R)
            condition_dim: 条件维度 (用于cGAN)
        """
        super().__init__()

        input_dim = latent_dim + condition_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            ResidualBlock(256),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),

            ResidualBlock(512),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=1)

        x = self.input_layer(z)
        x = self.hidden_layers(x)
        return self.output_layer(x)


class DiscriminatorV2(nn.Module):
    """优化版判别器 - LayerNorm + GELU"""

    def __init__(self, input_dim: int = 3, condition_dim: int = 0):
        super().__init__()

        total_input_dim = input_dim + condition_dim

        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            ResidualBlock(256),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition=None):
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        return self.model(x)


class TimeSeriesGenerator(nn.Module):
    """时序数据生成器 - 用于生成原始CSV格式的时序数据"""

    def __init__(self, latent_dim: int = 100, seq_length: int = 1000, n_features: int = 4):
        """
        Args:
            latent_dim: 潜在空间维度
            seq_length: 序列长度
            n_features: 特征数量 (time, ton, thrust, mfr)
        """
        super().__init__()

        self.seq_length = seq_length
        self.n_features = n_features

        # 使用1D转置卷积生成时序
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * (seq_length // 16)),
            nn.LayerNorm(256 * (seq_length // 16)),
            nn.GELU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, seq_length // 8]),
            nn.GELU(),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, seq_length // 4]),
            nn.GELU(),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([32, seq_length // 2]),
            nn.GELU(),

            nn.ConvTranspose1d(32, n_features, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 256, self.seq_length // 16)
        x = self.conv_layers(x)
        return x.transpose(1, 2)  # (batch, seq_length, n_features)


class FeatureGANV2:
    """优化版特征生成对抗网络"""

    def __init__(
        self,
        latent_dim: int = 100,
        feature_dim: int = 3,
        condition_dim: int = 0,
        device: str = None,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # 初始化网络
        self.generator = GeneratorV2(latent_dim, feature_dim, condition_dim).to(self.device)
        self.discriminator = DiscriminatorV2(feature_dim, condition_dim).to(self.device)

        # 优化器
        self.optimizer_G = optim.AdamW(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2),
            weight_decay=1e-4
        )
        self.optimizer_D = optim.AdamW(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2),
            weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=100)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=100)

        # 损失函数
        self.criterion = nn.BCELoss()

        # 训练历史
        self.history = {'g_loss': [], 'd_loss': [], 'd_real_acc': [], 'd_fake_acc': []}

        # 归一化参数
        self.feature_min = None
        self.feature_max = None

    def train(
        self,
        real_features: np.ndarray,
        labels: np.ndarray = None,
        epochs: int = 300,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """训练GAN"""

        # 数据归一化
        real_features_normalized = self._normalize_features(real_features)

        # 创建数据集
        if labels is not None and self.condition_dim > 0:
            labels_onehot = np.eye(self.condition_dim)[labels.astype(int)]
            dataset = TensorDataset(
                torch.FloatTensor(real_features_normalized),
                torch.FloatTensor(labels_onehot)
            )
        else:
            dataset = TensorDataset(torch.FloatTensor(real_features_normalized))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # 训练循环
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_d_real_acc = 0
            epoch_d_fake_acc = 0
            n_batches = 0

            pbar = tqdm(dataloader, disable=not verbose, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_data in pbar:
                if len(batch_data) == 2:
                    real_batch, condition = batch_data
                    condition = condition.to(self.device)
                else:
                    real_batch = batch_data[0]
                    condition = None

                real_batch = real_batch.to(self.device)
                current_batch_size = real_batch.size(0)

                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)

                # 训练判别器
                self.optimizer_D.zero_grad()

                real_output = self.discriminator(real_batch, condition)
                d_real_loss = self.criterion(real_output, real_labels)

                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z, condition)
                fake_output = self.discriminator(fake_batch.detach(), condition)
                d_fake_loss = self.criterion(fake_output, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # 训练生成器
                self.optimizer_G.zero_grad()

                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z, condition)
                fake_output = self.discriminator(fake_batch, condition)

                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                # 统计
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real_acc += (real_output > 0.5).float().mean().item()
                epoch_d_fake_acc += (fake_output < 0.5).float().mean().item()
                n_batches += 1

                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })

            # 更新学习率
            self.scheduler_G.step()
            self.scheduler_D.step()

            # 记录历史
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            self.history['d_real_acc'].append(epoch_d_real_acc / n_batches)
            self.history['d_fake_acc'].append(epoch_d_fake_acc / n_batches)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}]")
                print(f"  G_loss: {self.history['g_loss'][-1]:.4f}")
                print(f"  D_loss: {self.history['d_loss'][-1]:.4f}")

        return self.history

    def generate_samples(self, n_samples: int, condition: np.ndarray = None) -> np.ndarray:
        """生成合成样本"""
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)

            if condition is not None:
                condition_tensor = torch.FloatTensor(condition).to(self.device)
                fake_features = self.generator(z, condition_tensor)
            else:
                fake_features = self.generator(z)

            fake_features = fake_features.cpu().numpy()

        return self._denormalize_features(fake_features)

    def augment_dataset(
        self,
        real_features: np.ndarray,
        labels: np.ndarray = None,
        expansion_factor: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        扩充数据集

        Returns:
            (augmented_features, augmented_labels, is_synthetic)
        """
        n_real = len(real_features)
        n_synthetic = n_real * (expansion_factor - 1)

        print(f"[GAN V2] 数据扩充")
        print(f"  真实样本: {n_real}")
        print(f"  生成样本: {n_synthetic}")

        synthetic_features = self.generate_samples(n_synthetic)
        augmented_features = np.vstack([real_features, synthetic_features])

        is_synthetic = np.concatenate([
            np.zeros(n_real),
            np.ones(n_synthetic)
        ])

        if labels is not None:
            synthetic_labels = np.random.choice(labels, size=n_synthetic)
            augmented_labels = np.concatenate([labels, synthetic_labels])
        else:
            augmented_labels = None

        return augmented_features, augmented_labels, is_synthetic

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)
        return 2 * (features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8) - 1

    def _denormalize_features(self, normalized: np.ndarray) -> np.ndarray:
        return (normalized + 1) / 2 * (self.feature_max - self.feature_min) + self.feature_min

    def save_model(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_G_state': self.optimizer_G.state_dict(),
            'optimizer_D_state': self.optimizer_D.state_dict(),
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
            'history': self.history
        }, filepath)
        print(f"模型已保存至 {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.feature_min = checkpoint['feature_min']
        self.feature_max = checkpoint['feature_max']
        self.history = checkpoint.get('history', self.history)
        print(f"模型已从 {filepath} 加载")


class TimeSeriesGAN:
    """时序数据GAN - 用于扩充原始CSV文件数据"""

    def __init__(
        self,
        latent_dim: int = 100,
        seq_length: int = 1000,
        n_features: int = 4,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.n_features = n_features

        self.generator = TimeSeriesGenerator(latent_dim, seq_length, n_features).to(self.device)

        # 判别器使用1D卷积
        self.discriminator = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([32, seq_length // 2]),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, seq_length // 4]),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([128, seq_length // 8]),
            nn.GELU(),

            nn.Flatten(),
            nn.Linear(128 * (seq_length // 8), 1),
            nn.Sigmoid()
        ).to(self.device)

        self.optimizer_G = optim.AdamW(self.generator.parameters(), lr=2e-4)
        self.optimizer_D = optim.AdamW(self.discriminator.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()

        self.feature_stats = None

    def train(self, time_series_data: List[np.ndarray], epochs: int = 300, batch_size: int = 16):
        """
        训练时序GAN

        Args:
            time_series_data: 时序数据列表，每个元素为 (seq_length, n_features)
        """
        # 标准化数据
        all_data = np.stack(time_series_data)
        self.feature_stats = {
            'mean': all_data.mean(axis=(0, 1)),
            'std': all_data.std(axis=(0, 1)) + 1e-8
        }

        normalized_data = (all_data - self.feature_stats['mean']) / self.feature_stats['std']

        dataset = TensorDataset(torch.FloatTensor(normalized_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                real_batch = batch_data[0].to(self.device)
                current_batch_size = real_batch.size(0)

                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)

                # 训练判别器
                self.optimizer_D.zero_grad()
                real_output = self.discriminator(real_batch.transpose(1, 2))
                d_real_loss = self.criterion(real_output, real_labels)

                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_output = self.discriminator(fake_batch.transpose(1, 2).detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # 训练生成器
                self.optimizer_G.zero_grad()
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_output = self.discriminator(fake_batch.transpose(1, 2))
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

    def generate_time_series(self, n_samples: int) -> List[np.ndarray]:
        """生成时序数据"""
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            fake_data = self.generator(z).cpu().numpy()

        # 反标准化
        fake_data = fake_data * self.feature_stats['std'] + self.feature_stats['mean']

        return [fake_data[i] for i in range(n_samples)]
