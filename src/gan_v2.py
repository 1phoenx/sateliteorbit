"""
GAN小样本扩充模块 v2
针对推力器特征数据的条件GAN实现
支持按类别生成和质量过滤
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConditionalGenerator(nn.Module):
    """条件生成器 - 支持按类别生成"""

    def __init__(self, latent_dim: int = 64, output_dim: int = 3, n_classes: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # 类别嵌入
        self.label_embedding = nn.Embedding(n_classes, latent_dim)

        # 生成器网络
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, output_dim)
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    """条件判别器"""

    def __init__(self, input_dim: int = 3, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes

        # 类别嵌入
        self.label_embedding = nn.Embedding(n_classes, input_dim)

        # 判别器网络
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        x = torch.cat([x, label_emb], dim=1)
        return self.model(x)


class ThrusterFeatureGAN:
    """推力器特征条件GAN"""

    def __init__(
        self,
        latent_dim: int = 64,
        feature_dim: int = 3,
        n_classes: int = 2,
        device: str = None
    ):
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化网络
        self.generator = ConditionalGenerator(
            latent_dim, feature_dim, n_classes
        ).to(self.device)

        self.discriminator = ConditionalDiscriminator(
            feature_dim, n_classes
        ).to(self.device)

        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 归一化参数
        self.feature_mean = None
        self.feature_std = None

        # 训练历史
        self.history = {'g_loss': [], 'd_loss': []}

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score 归一化"""
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8
        return (features - self.feature_mean) / self.feature_std

    def _denormalize(self, features: np.ndarray) -> np.ndarray:
        """反归一化"""
        return features * self.feature_std + self.feature_mean

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        训练条件GAN

        Args:
            features: 特征数据 (N, 3) - P, T, R
            labels: 标签 (N,) - 0: 正常, 1: 异常
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否显示进度
        """
        logger.info(f"开始训练GAN，设备: {self.device}")
        logger.info(f"样本数: {len(features)}, 特征维度: {features.shape[1]}")

        # 归一化
        features_norm = self._normalize(features)

        # 创建数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(features_norm),
            torch.LongTensor(labels)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            n_batches = 0

            for real_features, real_labels in dataloader:
                real_features = real_features.to(self.device)
                real_labels = real_labels.to(self.device)
                batch_size_curr = real_features.size(0)

                # 标签
                valid = torch.ones(batch_size_curr, 1).to(self.device)
                fake = torch.zeros(batch_size_curr, 1).to(self.device)

                # 训练判别器
                self.optimizer_D.zero_grad()

                real_pred = self.discriminator(real_features, real_labels)
                d_real_loss = self.criterion(real_pred, valid)

                z = torch.randn(batch_size_curr, self.latent_dim).to(self.device)
                gen_labels = torch.randint(0, self.n_classes, (batch_size_curr,)).to(self.device)
                fake_features = self.generator(z, gen_labels)
                fake_pred = self.discriminator(fake_features.detach(), gen_labels)
                d_fake_loss = self.criterion(fake_pred, fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # 训练生成器
                self.optimizer_G.zero_grad()

                z = torch.randn(batch_size_curr, self.latent_dim).to(self.device)
                gen_labels = torch.randint(0, self.n_classes, (batch_size_curr,)).to(self.device)
                fake_features = self.generator(z, gen_labels)
                fake_pred = self.discriminator(fake_features, gen_labels)
                g_loss = self.criterion(fake_pred, valid)

                g_loss.backward()
                self.optimizer_G.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1

            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}"
                )

        logger.info("GAN训练完成")
        return self.history

    def generate(
        self,
        n_samples: int,
        label: int = None,
        quality_filter: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成样本

        Args:
            n_samples: 生成样本数
            label: 指定类别 (None表示随机)
            quality_filter: 是否进行质量过滤
        """
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)

            if label is not None:
                labels = torch.full((n_samples,), label, dtype=torch.long).to(self.device)
            else:
                labels = torch.randint(0, self.n_classes, (n_samples,)).to(self.device)

            fake_features = self.generator(z, labels)
            fake_features = fake_features.cpu().numpy()
            labels = labels.cpu().numpy()

        # 反归一化
        fake_features = self._denormalize(fake_features)

        # 质量过滤
        if quality_filter:
            fake_features, labels = self._quality_filter(fake_features, labels)

        return fake_features, labels

    def _quality_filter(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        质量过滤 - 移除不合理的生成样本

        过滤条件:
        - P (峰值强度) > 0
        - T (持续时间) >= 0.1
        - R (频率比) > 0 或 NaN
        """
        P, T, R = features[:, 0], features[:, 1], features[:, 2]

        valid_mask = (P > 0) & (T >= 0.1) & ((R > 0) | np.isnan(R))

        filtered_features = features[valid_mask]
        filtered_labels = labels[valid_mask]

        filter_rate = 1 - valid_mask.sum() / len(valid_mask)
        if filter_rate > 0.1:
            logger.warning(f"质量过滤移除了 {filter_rate*100:.1f}% 的样本")

        return filtered_features, filtered_labels

    def augment_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        expansion_factor: int = 10,
        balance_classes: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        扩充数据集

        Args:
            features: 原始特征
            labels: 原始标签
            expansion_factor: 扩充倍数
            balance_classes: 是否平衡类别
        """
        logger.info(f"数据扩充: 原始样本 {len(features)}, 扩充倍数 {expansion_factor}")

        if balance_classes:
            # 按类别分别扩充
            unique_labels = np.unique(labels)
            all_features = [features]
            all_labels = [labels]

            for lbl in unique_labels:
                class_count = (labels == lbl).sum()
                n_generate = class_count * (expansion_factor - 1)

                gen_features, gen_labels = self.generate(
                    n_generate, label=int(lbl), quality_filter=True
                )
                all_features.append(gen_features)
                all_labels.append(gen_labels)

            aug_features = np.vstack(all_features)
            aug_labels = np.concatenate(all_labels)
        else:
            n_generate = len(features) * (expansion_factor - 1)
            gen_features, gen_labels = self.generate(n_generate, quality_filter=True)

            aug_features = np.vstack([features, gen_features])
            aug_labels = np.concatenate([labels, gen_labels])

        logger.info(f"扩充后样本数: {len(aug_features)}")
        return aug_features, aug_labels

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'history': self.history
        }, path)
        logger.info(f"模型已保存至: {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.feature_mean = checkpoint['feature_mean']
        self.feature_std = checkpoint['feature_std']
        self.history = checkpoint.get('history', {'g_loss': [], 'd_loss': []})
        logger.info(f"模型已加载: {path}")


def main():
    """主函数 - GAN数据扩充"""
    import argparse

    parser = argparse.ArgumentParser(description='GAN小样本扩充')
    parser.add_argument('--input', type=str, default='data/feature_dataset.csv',
                        help='输入特征文件')
    parser.add_argument('--output', type=str, default='data/augmented_dataset.csv',
                        help='输出扩充文件')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--expansion', type=int, default=10,
                        help='扩充倍数')
    parser.add_argument('--model_path', type=str, default='models/thruster_gan.pth',
                        help='模型保存路径')

    args = parser.parse_args()

    # 加载特征数据
    df = pd.read_csv(args.input)
    features = df[['P', 'T', 'R']].values
    labels = df['is_anomalous'].values.astype(int)

    # 处理NaN
    features = np.nan_to_num(features, nan=0.0)

    # 创建并训练GAN
    gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
    gan.train(features, labels, epochs=args.epochs)

    # 保存模型
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    gan.save(args.model_path)

    # 扩充数据
    aug_features, aug_labels = gan.augment_dataset(
        features, labels, expansion_factor=args.expansion
    )

    # 保存扩充数据
    aug_df = pd.DataFrame({
        'P': aug_features[:, 0],
        'T': aug_features[:, 1],
        'R': aug_features[:, 2],
        'is_anomalous': aug_labels,
        'is_synthetic': [0] * len(features) + [1] * (len(aug_features) - len(features))
    })
    aug_df.to_csv(args.output, index=False)

    print(f"\n扩充完成:")
    print(f"  原始样本: {len(features)}")
    print(f"  扩充后: {len(aug_features)}")
    print(f"  保存至: {args.output}")


if __name__ == '__main__':
    main()
