"""
模型训练脚本
支持变轨检测、点火时刻定位、Δv回归等任务的训练
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.maneuver_detection.models.ignition_detector import (
    IgnitionCNN, IgnitionLSTM, IgnitionCNNLSTM
)
from src.maneuver_detection.models.delta_v_regressor import (
    DeltaVMLP, GradientBoostingDeltaV, DeltaVEnsemble
)
from src.mlf_snn.network import MLFSNN, SNNClassifier
from src.gan import FeatureGAN


class Trainer:
    """通用训练器"""

    def __init__(self, config: Config = None, device: str = None):
        self.config = config or Config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)

            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if output.dim() > 1 and output.size(1) > 1:
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, model, val_loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)

            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output, target)
            total_loss += loss.item()

            if output.dim() > 1 and output.size(1) > 1:
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                all_preds.extend(pred.cpu().numpy())
            total += target.size(0)
            all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy, all_preds, all_targets

    def train(
        self,
        model,
        train_loader,
        val_loader,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        save_path: str = None
    ):
        """完整训练流程"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"开始训练，设备: {self.device}")
        print(f"训练集: {len(train_loader.dataset)} 样本")
        print(f"验证集: {len(val_loader.dataset)} 样本")
        print("-" * 50)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            val_loss, val_acc, _, _ = self.evaluate(
                model, val_loader, criterion
            )

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"  模型已保存到 {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch+1}")
                    break

        return self.history


def load_dataset(data_path: str, task: str = 'detection'):
    """
    加载数据集

    Args:
        data_path: 数据文件路径
        task: 任务类型 ('detection', 'ignition', 'delta_v')
    """
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)

    # 特征列
    feature_cols = [col for col in df.columns if col.startswith(('F', 'P_', 'T_', 'R_', 'res_', 'rad_'))]

    if not feature_cols:
        feature_cols = [col for col in df.columns if col not in ['label', 'label_maneuver', 'timestamp', 'target_id']]

    X = df[feature_cols].values

    # 标签
    if task == 'detection':
        if 'label_maneuver' in df.columns:
            y = df['label_maneuver'].values
        elif 'label' in df.columns:
            y = df['label'].values
        else:
            y = np.zeros(len(df))
    elif task == 'delta_v':
        delta_v_cols = ['delta_v_R', 'delta_v_T', 'delta_v_N']
        if all(col in df.columns for col in delta_v_cols):
            y = df[delta_v_cols].values
        else:
            y = np.zeros((len(df), 3))
    else:
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))

    print(f"  特征维度: {X.shape}")
    print(f"  标签维度: {y.shape}")

    return X, y, feature_cols


def prepare_dataloaders(X, y, batch_size=32, test_size=0.2, val_size=0.1):
    """准备数据加载器"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y if y.ndim == 1 else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    # 转换为Tensor
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)

    if y.ndim == 1:
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        y_test = torch.LongTensor(y_test)
    else:
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        y_test = torch.FloatTensor(y_test)

    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader, scaler


def train_detection_model(args):
    """训练变轨检测模型"""
    print("=" * 50)
    print("训练变轨检测模型")
    print("=" * 50)

    # 加载数据
    X, y, feature_cols = load_dataset(args.data_path, task='detection')
    input_dim = X.shape[1]

    # 准备数据加载器
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        X, y, batch_size=args.batch_size
    )

    # 创建模型
    if args.model == 'mlf_snn':
        model = MLFSNN(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            output_dim=2,
            time_steps=16,
            neuron_type='mlf'
        )
    elif args.model == 'cnn':
        model = IgnitionCNN(
            input_channels=input_dim,
            window_size=1,
            num_classes=2
        )
    else:
        model = MLFSNN(input_dim=input_dim, output_dim=2)

    print(f"模型: {args.model}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    # 训练
    trainer = Trainer()
    save_path = f"models/{args.model}_detection.pth"
    os.makedirs("models", exist_ok=True)

    history = trainer.train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=save_path
    )

    # 测试评估
    print("\n测试集评估:")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = trainer.evaluate(
        model, test_loader, criterion
    )

    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    if len(preds) > 0:
        print(f"  Precision: {precision_score(targets, preds):.4f}")
        print(f"  Recall: {recall_score(targets, preds):.4f}")
        print(f"  F1: {f1_score(targets, preds):.4f}")

    return history


def train_gan_augmentation(args):
    """训练GAN进行数据扩充"""
    print("=" * 50)
    print("训练GAN数据扩充")
    print("=" * 50)

    X, y, _ = load_dataset(args.data_path, task='detection')

    # 训练GAN
    gan = FeatureGAN()
    gan.train(X, epochs=args.epochs, verbose=True)

    # 扩充数据
    X_aug, y_aug = gan.augment_dataset(X, y)
    print(f"扩充后样本数: {len(X_aug)}")

    # 保存
    os.makedirs("models", exist_ok=True)
    gan.save_model("models/feature_gan.pth")

    return X_aug, y_aug


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='变轨检测模型训练')

    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'gan', 'all'],
                        help='训练任务')
    parser.add_argument('--model', type=str, default='mlf_snn',
                        choices=['mlf_snn', 'cnn', 'lstm'],
                        help='模型类型')
    parser.add_argument('--data_path', type=str,
                        default='data/geo_optical_dataset.csv',
                        help='数据路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    args = parser.parse_args()

    # 设置随机种子
    Config.set_seed(42)

    if args.task == 'detection':
        train_detection_model(args)
    elif args.task == 'gan':
        train_gan_augmentation(args)
    elif args.task == 'all':
        train_gan_augmentation(args)
        train_detection_model(args)


if __name__ == '__main__':
    main()
