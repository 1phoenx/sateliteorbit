"""
双分支1D CNN模型 - 统计特征 + 时序特征
========================================

模型架构:
- 统计特征分支: [P, T, R] → FC层
- 时序特征分支: [300点推力序列] → 1D CNN
- 特征融合: Concat
- 双输出: 分类（异常检测）+ 回归（推力估计）

作者: Claude Code
日期: 2026-01-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticalBranch(nn.Module):
    """统计特征分支"""

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=32):
        super(StatisticalBranch, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        """
        参数:
            x: (batch_size, 3) - [P, T, R]

        返回:
            (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return x


class TimeSeriesBranch(nn.Module):
    """时序特征分支 - 1D CNN"""

    def __init__(self, input_channels=1, sequence_length=300, output_dim=64):
        super(TimeSeriesBranch, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 300 -> 150

        # 第二层卷积
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 150 -> 75

        # 第三层卷积
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 75 -> 37

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        参数:
            x: (batch_size, sequence_length) - 推力时序

        返回:
            (batch_size, output_dim)
        """
        # 添加通道维度: (batch, seq_len) -> (batch, 1, seq_len)
        x = x.unsqueeze(1)

        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # 全局平均池化
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # FC
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class DualBranchModel(nn.Module):
    """双分支模型 - 统计特征 + 时序特征"""

    def __init__(
        self,
        statistical_input_dim=3,
        sequence_length=300,
        statistical_output_dim=32,
        timeseries_output_dim=64,
        num_classes=2
    ):
        super(DualBranchModel, self).__init__()

        # 统计特征分支
        self.statistical_branch = StatisticalBranch(
            input_dim=statistical_input_dim,
            hidden_dim=64,
            output_dim=statistical_output_dim
        )

        # 时序特征分支
        self.timeseries_branch = TimeSeriesBranch(
            input_channels=1,
            sequence_length=sequence_length,
            output_dim=timeseries_output_dim
        )

        # 融合后的维度
        fusion_dim = statistical_output_dim + timeseries_output_dim

        # 融合层
        self.fusion_fc = nn.Linear(fusion_dim, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_dropout = nn.Dropout(0.3)

        # 分类分支（异常检测）
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # 回归分支（推力估计）
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, statistical_features, time_series):
        """
        参数:
            statistical_features: (batch_size, 3) - [P, T, R]
            time_series: (batch_size, 300) - 推力时序

        返回:
            classification_output: (batch_size, num_classes)
            regression_output: (batch_size, 1)
        """
        # 统计特征分支
        stat_features = self.statistical_branch(statistical_features)

        # 时序特征分支
        ts_features = self.timeseries_branch(time_series)

        # 特征融合
        fused_features = torch.cat([stat_features, ts_features], dim=1)

        # 融合层
        x = self.fusion_fc(fused_features)
        x = self.fusion_bn(x)
        x = F.relu(x)
        x = self.fusion_dropout(x)

        # 分类输出
        classification_output = self.classifier(x)

        # 回归输出
        regression_output = self.regressor(x)

        return classification_output, regression_output


def test_model():
    """测试模型"""
    print("Testing Dual Branch Model...")

    # 创建模型
    model = DualBranchModel(
        statistical_input_dim=3,
        sequence_length=300,
        num_classes=2
    )

    # 模拟输入
    batch_size = 16
    statistical_features = torch.randn(batch_size, 3)
    time_series = torch.randn(batch_size, 300)

    # 前向传播
    classification_output, regression_output = model(statistical_features, time_series)

    print(f"Input shapes:")
    print(f"  Statistical features: {statistical_features.shape}")
    print(f"  Time series: {time_series.shape}")
    print(f"\nOutput shapes:")
    print(f"  Classification: {classification_output.shape}")
    print(f"  Regression: {regression_output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✅ Model test passed!")


if __name__ == '__main__':
    test_model()
