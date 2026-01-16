"""
1D CNN 时序分类模型
用于推力器点火检测和异常识别
"""

import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Conv1DBlock(nn.Module):
    """1D卷积块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ThrusterCNN1D(nn.Module):
    """推力器1D CNN分类器"""

    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 1000,
        num_classes: int = 2,
        base_filters: int = 32
    ):
        """
        Args:
            input_channels: 输入通道数 (thrust信号为1)
            seq_length: 序列长度
            num_classes: 分类数 (2: 正常/异常)
            base_filters: 基础滤波器数量
        """
        super().__init__()

        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes

        # 卷积层
        self.conv1 = Conv1DBlock(input_channels, base_filters, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = Conv1DBlock(base_filters, base_filters * 2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = Conv1DBlock(base_filters * 2, base_filters * 4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = Conv1DBlock(base_filters * 4, base_filters * 4, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(base_filters * 4, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_length) 或 (batch, channels, seq_length)

        Returns:
            (logits, features)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.global_pool(x)

        features = x.squeeze(-1)

        x = self.fc1(features)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, features


class ResBlock1D(nn.Module):
    """1D残差块"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ThrusterResNet1D(nn.Module):
    """推力器1D ResNet分类器"""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 64,
        num_blocks: int = 3
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(base_filters) for _ in range(num_blocks)]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        features = x.squeeze(-1)
        logits = self.fc(features)

        return logits, features


class FeatureClassifier(nn.Module):
    """基于P/T/R特征的分类器"""

    def __init__(self, input_dim: int = 3, num_classes: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DPCRedundancyRemover:
    """DPC去冗余模块"""

    def __init__(self, dc_percent: float = 0.02):
        self.dc_percent = dc_percent

    def remove_redundancy(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        keep_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用DPC去除冗余样本

        Args:
            features: 特征矩阵
            labels: 标签
            keep_ratio: 保留比例

        Returns:
            (filtered_features, filtered_labels, selected_indices)
        """
        from scipy.spatial.distance import pdist, squareform

        n_samples = len(features)
        n_keep = int(n_samples * keep_ratio)

        # 计算距离矩阵
        dist_matrix = squareform(pdist(features, metric='euclidean'))

        # 计算截断距离
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        dc = np.percentile(distances, self.dc_percent * 100)

        # 计算局部密度
        rho = np.sum(np.exp(-(dist_matrix / dc) ** 2), axis=1) - 1

        # 选择密度最高的样本
        selected_indices = np.argsort(-rho)[:n_keep]

        return features[selected_indices], labels[selected_indices], selected_indices


class CNN1DTrainer:
    """1D CNN训练器"""

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        save_path: str = None
    ) -> Dict:
        """训练模型"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"开始训练，设备: {self.device}")

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}"
                )

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break

        return self.history

    @torch.no_grad()
    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return total_loss / len(loader), correct / total


def main():
    """主函数 - DPC-1D CNN训练"""
    import argparse
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description='DPC-1D CNN训练')
    parser.add_argument('--input', type=str, default='data/augmented_dataset.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='models/cnn1d_model.pth')
    parser.add_argument('--use_dpc', action='store_true', help='使用DPC去冗余')

    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.input)
    features = df[['P', 'T', 'R']].values
    labels = df['is_anomalous'].values.astype(int)

    # 处理NaN
    features = np.nan_to_num(features, nan=0.0)

    # DPC去冗余
    if args.use_dpc:
        logger.info("使用DPC去冗余...")
        dpc = DPCRedundancyRemover()
        features, labels, _ = dpc.remove_redundancy(features, labels, keep_ratio=0.8)
        logger.info(f"去冗余后样本数: {len(features)}")

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=args.batch_size
    )

    # 创建模型
    model = FeatureClassifier(input_dim=3, num_classes=2)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 训练
    import os
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    trainer = CNN1DTrainer()
    history = trainer.train(
        model, train_loader, test_loader,
        epochs=args.epochs, save_path=args.model_path
    )

    logger.info(f"训练完成，模型保存至: {args.model_path}")


if __name__ == '__main__':
    main()
