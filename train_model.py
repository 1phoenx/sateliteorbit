"""
双分支模型训练脚本
==================

训练双分支1D CNN模型进行变轨检测：
- 分类任务：异常检测
- 回归任务：推力估计

日期: 2026-01-23
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_dual_branch import DualBranchModel


class ThrusterDataset(Dataset):
    """推力器数据集"""

    def __init__(self, data_dir, split='train'):
        """
        参数:
            data_dir: 数据目录
            split: 'train' or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # 加载数据
        self.statistical_features = np.load(
            self.data_dir / f'{split}_statistical_features.npy'
        )
        self.thrust_sequences = np.load(
            self.data_dir / f'{split}_thrust_sequences.npy'
        )
        self.labels_anomaly = np.load(
            self.data_dir / f'{split}_labels_anomaly.npy'
        )
        self.labels_thrust = np.load(
            self.data_dir / f'{split}_labels_thrust.npy'
        )

        # 归一化统计特征
        if split == 'train':
            self.stat_mean = self.statistical_features.mean(axis=0)
            self.stat_std = self.statistical_features.std(axis=0) + 1e-8
            # 保存归一化参数
            np.save(self.data_dir / 'stat_mean.npy', self.stat_mean)
            np.save(self.data_dir / 'stat_std.npy', self.stat_std)
        else:
            # 使用训练集的归一化参数
            self.stat_mean = np.load(self.data_dir / 'stat_mean.npy')
            self.stat_std = np.load(self.data_dir / 'stat_std.npy')

        self.statistical_features = (
            self.statistical_features - self.stat_mean
        ) / self.stat_std

        # 归一化时序特征
        if split == 'train':
            self.ts_mean = self.thrust_sequences.mean()
            self.ts_std = self.thrust_sequences.std() + 1e-8
            np.save(self.data_dir / 'ts_mean.npy', self.ts_mean)
            np.save(self.data_dir / 'ts_std.npy', self.ts_std)
        else:
            self.ts_mean = np.load(self.data_dir / 'ts_mean.npy')
            self.ts_std = np.load(self.data_dir / 'ts_std.npy')

        self.thrust_sequences = (
            self.thrust_sequences - self.ts_mean
        ) / self.ts_std

        print(f"{split} dataset loaded:")
        print(f"  Samples: {len(self)}")
        print(f"  Anomaly: {self.labels_anomaly.sum()}/{len(self)}")

    def __len__(self):
        return len(self.statistical_features)

    def __getitem__(self, idx):
        return {
            'statistical': torch.FloatTensor(self.statistical_features[idx]),
            'timeseries': torch.FloatTensor(self.thrust_sequences[idx]),
            'label_anomaly': torch.LongTensor([self.labels_anomaly[idx]])[0],
            'label_thrust': torch.FloatTensor([self.labels_thrust[idx]])[0]
        }


def train_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        # 数据移到设备
        stat_features = batch['statistical'].to(device)
        time_series = batch['timeseries'].to(device)
        label_anomaly = batch['label_anomaly'].to(device)
        label_thrust = batch['label_thrust'].to(device)

        # 前向传播
        optimizer.zero_grad()
        cls_output, reg_output = model(stat_features, time_series)

        # 计算损失
        cls_loss = criterion_cls(cls_output, label_anomaly)
        reg_loss = criterion_reg(reg_output.squeeze(), label_thrust)

        # 加权总损失
        loss = cls_loss + 0.1 * reg_loss  # 回归损失权重较小

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

        _, predicted = cls_output.max(1)
        total += label_anomaly.size(0)
        correct += predicted.eq(label_anomaly).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, avg_cls_loss, avg_reg_loss, accuracy


def validate_epoch(model, dataloader, criterion_cls, criterion_reg, device):
    """验证一个epoch"""
    model.eval()

    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0

    # 用于计算虚警率
    false_positives = 0
    true_negatives = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            stat_features = batch['statistical'].to(device)
            time_series = batch['timeseries'].to(device)
            label_anomaly = batch['label_anomaly'].to(device)
            label_thrust = batch['label_thrust'].to(device)

            cls_output, reg_output = model(stat_features, time_series)

            cls_loss = criterion_cls(cls_output, label_anomaly)
            reg_loss = criterion_reg(reg_output.squeeze(), label_thrust)
            loss = cls_loss + 0.1 * reg_loss

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()

            _, predicted = cls_output.max(1)
            total += label_anomaly.size(0)
            correct += predicted.eq(label_anomaly).sum().item()

            # 计算虚警率
            for pred, label in zip(predicted, label_anomaly):
                if label == 0:  # 真实为正常
                    if pred == 1:  # 预测为异常
                        false_positives += 1
                    else:
                        true_negatives += 1

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    accuracy = 100. * correct / total

    # 虚警率
    if (false_positives + true_negatives) > 0:
        false_alarm_rate = 100. * false_positives / (false_positives + true_negatives)
    else:
        false_alarm_rate = 0.0

    return avg_loss, avg_cls_loss, avg_reg_loss, accuracy, false_alarm_rate


def train_model(
    data_dir='data/features_timeseries',
    output_dir='models',
    epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
):
    """训练模型"""

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    train_dataset = ThrusterDataset(data_dir, split='train')
    test_dataset = ThrusterDataset(data_dir, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 模型
    model = DualBranchModel(
        statistical_input_dim=3,
        sequence_length=300,
        num_classes=2
    ).to(device)

    # 损失函数
    # 分类：交叉熵（带类别权重处理不平衡）
    class_weights = torch.FloatTensor([1.0, 10.0]).to(device)  # 异常类权重更高
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

    # 回归：MSE
    criterion_reg = nn.MSELoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练历史
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_reg_loss': [],
        'val_accuracy': [],
        'val_false_alarm_rate': []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_cls_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )

        # 验证
        val_loss, val_cls_loss, val_reg_loss, val_acc, val_far = validate_epoch(
            model, test_loader, criterion_cls, criterion_reg, device
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_reg_loss'].append(train_reg_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_accuracy'].append(val_acc)
        history['val_false_alarm_rate'].append(val_far)

        # 打印结果
        print(f"Train - Loss: {train_loss:.4f}, Cls: {train_cls_loss:.4f}, "
              f"Reg: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Cls: {val_cls_loss:.4f}, "
              f"Reg: {val_reg_loss:.4f}, Acc: {val_acc:.2f}%, FAR: {val_far:.2f}%")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_false_alarm_rate': val_far
            }, output_path / 'best_model.pth')
            print(f"✅ Saved best model (epoch {epoch+1})")

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 保存训练历史
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 绘制训练曲线
    plot_training_history(history, output_path)

    return model, history


def plot_training_history(history, output_dir):
    """绘制训练历史曲线"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train')
    axes[0, 1].plot(history['val_accuracy'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Classification Loss
    axes[1, 0].plot(history['train_cls_loss'], label='Train')
    axes[1, 0].plot(history['val_cls_loss'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # False Alarm Rate
    axes[1, 1].plot(history['val_false_alarm_rate'], label='Val FAR', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('False Alarm Rate (%)')
    axes[1, 1].set_title('False Alarm Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()

    print(f"Training curves saved to {output_dir / 'training_history.png'}")


if __name__ == '__main__':
    train_model(
        data_dir='data/features_timeseries',
        output_dir='models',
        epochs=50,
        batch_size=32,
        learning_rate=1e-4
    )
