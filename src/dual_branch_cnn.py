"""
双分支1D CNN模型
- 分类分支: 判断是否发生变轨事件
- 回归分支: 估计推力大小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class DualBranchCNN(nn.Module):
    """双分支1D CNN: 分类 + 回归"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list = [64, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.shared_backbone = nn.Sequential(*layers)

        # 分类分支
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

        # 回归分支 (推力估计)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_both: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 [batch, input_dim]
            return_both: 是否返回两个分支输出

        Returns:
            class_logits: 分类logits [batch, num_classes]
            thrust_pred: 推力预测 [batch, 1]
        """
        features = self.shared_backbone(x)
        class_logits = self.classification_head(features)
        thrust_pred = self.regression_head(features)

        return class_logits, thrust_pred


class DualBranchLoss(nn.Module):
    """双分支损失函数"""

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 0.5):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

    def forward(
        self,
        class_logits: torch.Tensor,
        thrust_pred: torch.Tensor,
        class_labels: torch.Tensor,
        thrust_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失

        Returns:
            total_loss: 加权总损失
            loss_dict: 各分支损失详情
        """
        cls_loss = self.cls_criterion(class_logits, class_labels)
        reg_loss = self.reg_criterion(thrust_pred.squeeze(), thrust_labels)

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total_loss, {
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
