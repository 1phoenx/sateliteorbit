"""
完整训练脚本 - 支持300 epochs训练
生成训练曲线图和性能报告
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_feature_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载特征数据集"""
    df = pd.read_csv(data_path)

    # 过滤有效样本
    df = df[df['is_valid'] == 1].copy()

    # 填充NaN
    df['R'] = df['R'].fillna(df['R'].median())

    # 提取特征和标签
    X = df[['P', 'T', 'R']].values
    y = df['is_anomalous'].values
    t = df['ignition_time'].values

    return X, y, t


def train_gan_300epochs(X_train: np.ndarray, save_dir: str = 'models'):
    """训练GAN 300 epochs"""
    try:
        from src.gan_v3 import FeatureGANV2
    except ImportError:
        from src.gan import FeatureGAN as FeatureGANV2

    print("\n" + "=" * 60)
    print("GAN训练 (300 epochs)")
    print("=" * 60)

    gan = FeatureGANV2(
        latent_dim=100,
        feature_dim=3,
        lr_g=2e-4,
        lr_d=1e-4
    )

    history = gan.train(X_train, epochs=300, batch_size=32, verbose=True)

    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    gan.save_model(f'{save_dir}/gan_300epochs.pth')

    return gan, history


def train_classifier_300epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = 'models'
) -> Tuple[object, Dict]:
    """训练分类器 300 epochs"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 60)
    print("分类器训练 (300 epochs)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 定义模型
    class Classifier(nn.Module):
        def __init__(self, input_dim=3):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),

                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),

                nn.Linear(128, 2)
            )

        def forward(self, x):
            return self.model(x)

    model = Classifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = nn.CrossEntropyLoss()

    # 数据加载
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = None

    for epoch in range(300):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/300 - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/classifier_300epochs.pth')

    return model, history


def plot_training_curves(
    gan_history: Dict,
    classifier_history: Dict,
    save_dir: str = 'figures'
):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # GAN损失曲线
    ax1 = axes[0, 0]
    epochs = range(1, len(gan_history['g_loss']) + 1)
    ax1.plot(epochs, gan_history['g_loss'], 'b-', label='Generator Loss', linewidth=1.5)
    ax1.plot(epochs, gan_history['d_loss'], 'r-', label='Discriminator Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('GAN Training Loss (300 Epochs)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, 300])

    # GAN判别器准确率
    ax2 = axes[0, 1]
    ax2.plot(epochs, gan_history['d_real_acc'], 'g-', label='Real Accuracy', linewidth=1.5)
    ax2.plot(epochs, gan_history['d_fake_acc'], 'm-', label='Fake Accuracy', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('GAN Discriminator Accuracy (300 Epochs)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 300])
    ax2.set_ylim([0, 1])

    # 分类器损失曲线
    ax3 = axes[1, 0]
    epochs_clf = range(1, len(classifier_history['train_loss']) + 1)
    ax3.plot(epochs_clf, classifier_history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax3.plot(epochs_clf, classifier_history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Classifier Training Loss (300 Epochs)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1, 300])

    # 分类器准确率曲线
    ax4 = axes[1, 1]
    ax4.plot(epochs_clf, classifier_history['train_acc'], 'b-', label='Train Accuracy', linewidth=1.5)
    ax4.plot(epochs_clf, classifier_history['val_acc'], 'r-', label='Val Accuracy', linewidth=1.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Classifier Accuracy (300 Epochs)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([1, 300])
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves_300epochs.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n训练曲线已保存至 {save_dir}/training_curves_300epochs.png")

    # 单独保存GAN曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, gan_history['g_loss'], 'b-', label='Generator Loss', linewidth=2)
    ax.plot(epochs, gan_history['d_loss'], 'r-', label='Discriminator Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('GAN Training Loss Curve (300 Epochs)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 300])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gan_loss_300epochs.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 单独保存分类器曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_clf, classifier_history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs_clf, classifier_history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Classifier Accuracy Curve (300 Epochs)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 300])
    ax.set_ylim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/classifier_accuracy_300epochs.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("完整训练流程 (300 Epochs)")
    print("=" * 60)

    # 加载数据
    feature_path = 'data/feature_dataset.csv'
    if not os.path.exists(feature_path):
        print(f"错误: 找不到特征数据集 {feature_path}")
        return

    print("\n加载数据...")
    X, y, t = load_feature_dataset(feature_path)
    print(f"总样本数: {len(X)}")
    print(f"正常样本: {np.sum(y == 0)}")
    print(f"异常样本: {np.sum(y == 1)}")

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X, y, t, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n训练集: {len(X_train)}")
    print(f"验证集: {len(X_val)}")

    # 训练GAN
    gan, gan_history = train_gan_300epochs(X_train)

    # 使用GAN扩充数据
    print("\n使用GAN扩充数据...")
    augmented_features, augmented_labels, is_synthetic = gan.augment_dataset(
        X_train, y_train, expansion_factor=10
    )
    print(f"扩充后训练集: {len(augmented_features)}")

    # 训练分类器
    _, classifier_history = train_classifier_300epochs(
        augmented_features, augmented_labels, X_val, y_val
    )

    # 绘制训练曲线
    plot_training_curves(gan_history, classifier_history)

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, 301),
        'gan_g_loss': gan_history['g_loss'],
        'gan_d_loss': gan_history['d_loss'],
        'classifier_train_loss': classifier_history['train_loss'],
        'classifier_val_loss': classifier_history['val_loss'],
        'classifier_train_acc': classifier_history['train_acc'],
        'classifier_val_acc': classifier_history['val_acc']
    })
    history_df.to_csv('results/training_history_300epochs.csv', index=False)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最终验证准确率: {classifier_history['val_acc'][-1]:.4f}")
    print(f"最佳验证准确率: {max(classifier_history['val_acc']):.4f}")


if __name__ == '__main__':
    main()
