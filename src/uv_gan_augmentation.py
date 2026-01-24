"""
UV 识别系统 - GAN数据增强训练
===========================================

使用GAN扩充特征数据，增加训练epoch，生成论文级别的结果

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FeatureGAN(nn.Module):
    """特征GAN - 用于生成合成特征数据"""

    def __init__(self, feature_dim=16, latent_dim=64, hidden_dim=128):
        super(FeatureGAN, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()
        )

        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.generator(z)


class GANTrainer:
    """GAN训练器"""

    def __init__(
        self,
        feature_dim=16,
        latent_dim=64,
        hidden_dim=128,
        lr=0.0002,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.gan = FeatureGAN(feature_dim, latent_dim, hidden_dim).to(self.device)

        # 优化器
        self.g_optimizer = optim.Adam(
            self.gan.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.gan.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 训练历史
        self.g_losses = []
        self.d_losses = []

    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 500,
        batch_size: int = 64,
        save_interval: int = 50
    ):
        """
        训练GAN

        参数:
            X_train: 训练数据 (N, D)
            epochs: 训练轮数
            batch_size: 批大小
            save_interval: 保存间隔
        """
        print(f"\n开始训练GAN...")
        print(f"  训练样本: {len(X_train)}")
        print(f"  特征维度: {X_train.shape[1]}")
        print(f"  训练轮数: {epochs}")
        print(f"  批大小: {batch_size}")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 转换为Tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练循环
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            n_batches = 0

            for batch_idx, (real_data,) in enumerate(dataloader):
                batch_size_actual = real_data.size(0)

                # 真实和假标签
                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)

                # ==================== 训练判别器 ====================
                self.d_optimizer.zero_grad()

                # 真实数据
                real_output = self.gan.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)

                # 生成假数据
                z = torch.randn(batch_size_actual, self.gan.latent_dim).to(self.device)
                fake_data = self.gan.generator(z)
                fake_output = self.gan.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)

                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # ==================== 训练生成器 ====================
                self.g_optimizer.zero_grad()

                # 生成假数据
                z = torch.randn(batch_size_actual, self.gan.latent_dim).to(self.device)
                fake_data = self.gan.generator(z)
                fake_output = self.gan.discriminator(fake_data)

                # 生成器损失（希望判别器认为是真的）
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                # 记录损失
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1

            # 平均损失
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            # 打印进度
            if (epoch + 1) % save_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")

        print(f"\n✓ GAN训练完成！")

    def generate(self, n_samples: int) -> np.ndarray:
        """
        生成合成样本

        参数:
            n_samples: 生成样本数

        返回:
            synthetic_data: 合成数据 (n_samples, feature_dim)
        """
        self.gan.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.gan.latent_dim).to(self.device)
            fake_data = self.gan.generator(z)
            fake_data = fake_data.cpu().numpy()

        # 反标准化
        synthetic_data = self.scaler.inverse_transform(fake_data)

        return synthetic_data

    def save(self, filepath: Path):
        """保存模型"""
        torch.save({
            'generator': self.gan.generator.state_dict(),
            'discriminator': self.gan.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'scaler': self.scaler,
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }, filepath)
        print(f"模型已保存: {filepath}")

    def load(self, filepath: Path):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.gan.generator.load_state_dict(checkpoint['generator'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.scaler = checkpoint['scaler']
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        print(f"模型已加载: {filepath}")


def augment_features_with_gan(
    features_file: Path,
    output_file: Path,
    augmentation_factor: int = 5,
    epochs: int = 500,
    batch_size: int = 64
):
    """
    使用GAN扩充特征数据

    参数:
        features_file: 原始特征文件
        output_file: 输出文件
        augmentation_factor: 扩充倍数
        epochs: 训练轮数
        batch_size: 批大小
    """
    print("=" * 70)
    print("GAN特征数据增强")
    print("=" * 70)

    # 1. 加载原始特征
    print("\n[1/5] 加载原始特征...")
    df = pd.read_csv(features_file)

    # 选择数值特征
    feature_cols = [
        'background_mean', 'background_std', 'threshold',
        'peak_intensity', 'mean_intensity', 'total_energy',
        'num_pulses',
        'mean_pulse_duration', 'max_pulse_duration',
        'mean_pulse_peak', 'max_pulse_peak',
        'mean_pulse_energy',
        'max_rise_rate', 'mean_rise_rate',
        'mean_pulse_interval', 'min_pulse_interval'
    ]

    X = df[feature_cols].fillna(0).values
    print(f"  原始样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")

    # 2. 训练GAN
    print("\n[2/5] 训练GAN...")
    trainer = GANTrainer(
        feature_dim=X.shape[1],
        latent_dim=64,
        hidden_dim=128,
        lr=0.0002
    )

    trainer.train(X, epochs=epochs, batch_size=batch_size, save_interval=50)

    # 3. 生成合成数据
    print(f"\n[3/5] 生成合成数据 (扩充{augmentation_factor}倍)...")
    n_synthetic = len(X) * augmentation_factor
    X_synthetic = trainer.generate(n_synthetic)
    print(f"  生成样本数: {len(X_synthetic)}")

    # 4. 合并数据
    print("\n[4/5] 合并原始数据和合成数据...")
    X_augmented = np.vstack([X, X_synthetic])
    print(f"  总样本数: {len(X_augmented)}")

    # 创建标签（原始=0, 合成=1）
    labels = np.concatenate([
        np.zeros(len(X)),
        np.ones(len(X_synthetic))
    ])

    # 创建DataFrame
    df_augmented = pd.DataFrame(X_augmented, columns=feature_cols)
    df_augmented['is_synthetic'] = labels

    # 复制其他列
    for col in df.columns:
        if col not in feature_cols and col not in df_augmented.columns:
            # 对于合成数据，使用原始数据的平均值或众数
            if df[col].dtype in [np.float64, np.int64]:
                default_value = df[col].mean()
            else:
                default_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0

            # 原始数据保持不变
            original_values = df[col].values
            # 合成数据使用默认值
            synthetic_values = np.full(len(X_synthetic), default_value)

            df_augmented[col] = np.concatenate([original_values, synthetic_values])

    # 5. 保存
    print(f"\n[5/5] 保存增强后的数据...")
    df_augmented.to_csv(output_file, index=False)
    print(f"  保存到: {output_file}")

    # 保存GAN模型
    model_dir = output_file.parent / 'gan_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(model_dir / 'feature_gan.pth')

    # 绘制训练曲线
    plot_training_curves(trainer.g_losses, trainer.d_losses, model_dir)

    print("\n" + "=" * 70)
    print("✓ GAN数据增强完成！")
    print("=" * 70)
    print(f"原始样本: {len(X)}")
    print(f"合成样本: {len(X_synthetic)}")
    print(f"总样本: {len(X_augmented)}")
    print(f"扩充倍数: {augmentation_factor}x")
    print("=" * 70)

    return df_augmented


def plot_training_curves(g_losses, d_losses, output_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))

    plt.plot(g_losses, label='Generator Loss', linewidth=2)
    plt.plot(d_losses, label='Discriminator Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GAN Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    save_path = output_dir / 'gan_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练曲线已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("UV 识别系统 - GAN数据增强训练")
    print("=" * 70)

    # 参数设置
    augmentation_factor = 5  # 扩充5倍
    epochs = 500  # 训练500轮
    batch_size = 64

    # 增强训练集
    print("\n增强训练集...")
    df_train_augmented = augment_features_with_gan(
        features_file=Path('data/uv_features_train.csv'),
        output_file=Path('data/uv_features_train_gan_augmented.csv'),
        augmentation_factor=augmentation_factor,
        epochs=epochs,
        batch_size=batch_size
    )

    # 增强测试集（可选，通常不增强测试集）
    # print("\n增强测试集...")
    # df_test_augmented = augment_features_with_gan(
    #     features_file=Path('data/uv_features_test.csv'),
    #     output_file=Path('data/uv_features_test_gan_augmented.csv'),
    #     augmentation_factor=2,  # 测试集扩充较少
    #     epochs=300,
    #     batch_size=batch_size
    # )

    print("\n" + "=" * 70)
    print("✓ 所有数据增强完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
