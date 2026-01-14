"""
GAN网络：用于小样本数据扩充
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
from src.config import Config
from src.utils import timing_decorator, save_checkpoint

class Generator(nn.Module):
    """生成器网络"""

    def __init__(self, latent_dim: int = 100, output_dim: int = 3):
        """
        Args:
            latent_dim: 潜在空间维度
            output_dim: 输出特征维度 (P, T, R)
        """
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: 潜在向量 (batch_size, latent_dim)

        Returns:
            生成的特征 (batch_size, output_dim)
        """
        return self.model(z)

class Discriminator(nn.Module):
    """判别器网络"""

    def __init__(self, input_dim: int = 3):
        """
        Args:
            input_dim: 输入特征维度
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch_size, input_dim)

        Returns:
            真实概率 (batch_size, 1)
        """
        return self.model(x)

class FeatureGAN:
    """特征生成对抗网络"""

    def __init__(self, config: Config = None):
        """
        初始化GAN

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.device = self.config.DEVICE

        gan_config = self.config.GAN_CONFIG
        self.latent_dim = gan_config['latent_dim']
        self.batch_size = gan_config['batch_size']
        self.expansion_factor = gan_config['sample_expansion_factor']

        # 初始化网络
        self.generator = Generator(self.latent_dim, output_dim=3).to(self.device)
        self.discriminator = Discriminator(input_dim=3).to(self.device)

        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=gan_config['generator_lr'],
            betas=(gan_config['beta1'], gan_config['beta2'])
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=gan_config['discriminator_lr'],
            betas=(gan_config['beta1'], gan_config['beta2'])
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 训练历史
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }

    @timing_decorator
    def train(self,
              real_features: np.ndarray,
              epochs: int = None,
              verbose: bool = True) -> dict:
        """
        训练GAN

        Args:
            real_features: 真实特征数据 (N, 3)
            epochs: 训练轮数
            verbose: 是否显示进度

        Returns:
            训练历史
        """
        if epochs is None:
            epochs = self.config.GAN_CONFIG['epochs']

        # 数据归一化到 [-1, 1]
        real_features_normalized = self._normalize_features(real_features)

        # 创建数据加载器
        dataset = TensorDataset(torch.FloatTensor(real_features_normalized))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # 训练循环
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_d_real_acc = 0
            epoch_d_fake_acc = 0
            n_batches = 0

            pbar = tqdm(dataloader, disable=not verbose) if verbose else dataloader

            for batch_idx, (real_batch,) in enumerate(pbar):
                real_batch = real_batch.to(self.device)
                current_batch_size = real_batch.size(0)

                # 真实和虚假标签
                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)

                # ==================
                # 训练判别器
                # ==================
                self.optimizer_D.zero_grad()

                # 真实数据
                real_output = self.discriminator(real_batch)
                d_real_loss = self.criterion(real_output, real_labels)

                # 生成假数据
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_output = self.discriminator(fake_batch.detach())
                d_fake_loss = self.criterion(fake_output, fake_labels)

                # 总判别器损失
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_D.step()

                # ==================
                # 训练生成器
                # ==================
                self.optimizer_G.zero_grad()

                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z)
                fake_output = self.discriminator(fake_batch)

                # 生成器希望判别器认为生成的数据是真的
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

                # 记录统计
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_d_real_acc += (real_output > 0.5).float().mean().item()
                epoch_d_fake_acc += (fake_output < 0.5).float().mean().item()
                n_batches += 1

                if verbose:
                    pbar.set_description(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"D_loss: {d_loss.item():.4f} | "
                        f"G_loss: {g_loss.item():.4f}"
                    )

            # 记录epoch统计
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            self.history['d_real_acc'].append(epoch_d_real_acc / n_batches)
            self.history['d_fake_acc'].append(epoch_d_fake_acc / n_batches)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}]")
                print(f"  G_loss: {self.history['g_loss'][-1]:.4f}")
                print(f"  D_loss: {self.history['d_loss'][-1]:.4f}")
                print(f"  D_real_acc: {self.history['d_real_acc'][-1]:.4f}")
                print(f"  D_fake_acc: {self.history['d_fake_acc'][-1]:.4f}")

        return self.history

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        生成合成样本

        Args:
            n_samples: 要生成的样本数量

        Returns:
            生成的特征 (n_samples, 3)
        """
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            fake_features = self.generator(z)
            fake_features = fake_features.cpu().numpy()

        # 反归一化
        fake_features = self._denormalize_features(fake_features)

        return fake_features

    def augment_dataset(self,
                        real_features: np.ndarray,
                        labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        扩充数据集

        Args:
            real_features: 真实特征 (N, 3)
            labels: 真实标签 (N,)

        Returns:
            (augmented_features, augmented_labels)
        """
        n_real = len(real_features)
        n_synthetic = n_real * (self.expansion_factor - 1)

        print(f"[GAN] 数据扩充")
        print(f"  真实样本: {n_real}")
        print(f"  生成样本: {n_synthetic}")
        print(f"  总样本: {n_real + n_synthetic}")

        # 生成合成样本
        synthetic_features = self.generate_samples(n_synthetic)

        # 合并真实和合成数据
        augmented_features = np.vstack([real_features, synthetic_features])

        if labels is not None:
            # 假设合成样本与真实样本标签相同
            synthetic_labels = np.random.choice(labels, size=n_synthetic)
            augmented_labels = np.concatenate([labels, synthetic_labels])
        else:
            augmented_labels = None

        return augmented_features, augmented_labels

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征到 [-1, 1]"""
        self.feature_min = features.min(axis=0)
        self.feature_max = features.max(axis=0)

        normalized = 2 * (features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8) - 1
        return normalized

    def _denormalize_features(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化特征"""
        features = (normalized + 1) / 2 * (self.feature_max - self.feature_min) + self.feature_min
        return features

    def save_model(self, filepath: str):
        """保存GAN模型"""
        save_checkpoint(
            model=self.generator,
            optimizer=self.optimizer_G,
            epoch=len(self.history['g_loss']),
            loss=self.history['g_loss'][-1] if self.history['g_loss'] else 0,
            filepath=filepath.replace('.pth', '_generator.pth'),
            discriminator_state=self.discriminator.state_dict(),
            feature_min=self.feature_min,
            feature_max=self.feature_max
        )

    def load_model(self, filepath: str):
        """加载GAN模型"""
        from src.utils import load_checkpoint

        checkpoint = load_checkpoint(
            filepath.replace('.pth', '_generator.pth'),
            model=self.generator,
            optimizer=self.optimizer_G
        )

        if 'discriminator_state' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state'])

        self.feature_min = checkpoint.get('feature_min')
        self.feature_max = checkpoint.get('feature_max')
