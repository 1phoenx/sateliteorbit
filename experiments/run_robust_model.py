"""
小样本不平衡数据的鲁棒变轨识别模型
论文核心创新点：
1. 类别平衡采样策略 (Class-Balanced Sampling)
2. 原型网络 (Prototypical Networks) - 小样本学习
3. 自适应阈值决策 (Adaptive Threshold)
4. 不确定性感知预测 (Uncertainty-Aware Prediction)
5. 课程学习 (Curriculum Learning)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_absolute_error, roc_auc_score,
                             precision_recall_curve)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings('ignore')


def extract_features(df):
    """提取增强特征 (20维)"""
    P = df['P'].values
    T = df['T'].values
    R = df['R'].values

    features = np.column_stack([
        P, T, R,
        P * T, P / (R + 1e-6), T / (R + 1e-6), P * T / (R + 1e-6),
        P ** 2, T ** 2, R ** 2,
        np.sqrt(P + 1e-6), np.sqrt(T + 1e-6),
        np.log1p(P), np.log1p(T), np.log1p(R),
        T * R, P / (T + 1e-6), P * T * R,
        np.exp(-T), T ** 0.5 * P
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ==================== 损失函数 ====================

class ClassBalancedLoss(nn.Module):
    """类别平衡损失 - 基于有效样本数"""
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.FloatTensor(weights)

    def forward(self, pred, target):
        weights = self.weights.to(pred.device)
        return F.cross_entropy(pred, target, weight=weights)


class FocalLoss(nn.Module):
    """Focal Loss - 聚焦难分类样本"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(pred.device)
            alpha_t = alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ==================== 模型组件 ====================

class PrototypicalEncoder(nn.Module):
    """原型网络编码器 - 学习类别原型"""
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class UncertaintyHead(nn.Module):
    """不确定性估计头 - 输出均值和方差"""
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, output_dim)
        self.var_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6  # 确保正值
        return mean, var


class SmallSampleRobustNet(nn.Module):
    """
    小样本鲁棒网络

    创新点：
    1. 原型学习 - 学习类别原型进行分类
    2. 不确定性估计 - 预测置信度
    3. 多任务学习 - 分类+回归联合优化
    """
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # 特征编码器
        self.encoder = PrototypicalEncoder(input_dim, hidden_dim, embed_dim)

        # 分类头 (带不确定性)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 回归头 (带不确定性)
        self.regressor = UncertaintyHead(embed_dim, 1)

        # 类别原型 (可学习)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, x, return_embedding=False):
        # 编码
        z = self.encoder(x)

        # 分类 (结合原型距离)
        cls_logits = self.classifier(z)

        # 原型距离
        proto_dist = -torch.cdist(z, self.prototypes)  # 负距离作为相似度
        cls_logits = cls_logits + 0.5 * proto_dist  # 融合

        # 回归 (带不确定性)
        reg_mean, reg_var = self.regressor(z)

        if return_embedding:
            return cls_logits, reg_mean, reg_var, z
        return cls_logits, reg_mean, reg_var

    def compute_prototypes(self, support_x, support_y):
        """从支持集计算类别原型"""
        z = self.encoder(support_x)
        prototypes = []
        for c in range(self.num_classes):
            mask = support_y == c
            if mask.sum() > 0:
                prototypes.append(z[mask].mean(dim=0))
            else:
                prototypes.append(self.prototypes[c])
        return torch.stack(prototypes)


class AdaptiveThresholdClassifier:
    """自适应阈值分类器 - 基于验证集优化阈值"""
    def __init__(self, metric='f1'):
        self.threshold = 0.5
        self.metric = metric

    def fit(self, probs, labels):
        """在验证集上寻找最优阈值"""
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)

        if self.metric == 'f1':
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            self.threshold = thresholds[min(best_idx, len(thresholds)-1)]
        elif self.metric == 'balanced':
            # 平衡精确率和召回率
            balanced = np.sqrt(precisions * recalls)
            best_idx = np.argmax(balanced)
            self.threshold = thresholds[min(best_idx, len(thresholds)-1)]

        return self.threshold

    def predict(self, probs):
        return (probs >= self.threshold).astype(int)


# ==================== 训练器 ====================

class RobustTrainer:
    """鲁棒训练器 - 集成多种小样本学习技术"""

    def __init__(self, model, device, samples_per_class, lr=1e-3):
        self.model = model.to(device)
        self.device = device

        # 类别平衡损失
        self.cls_criterion = ClassBalancedLoss(samples_per_class, beta=0.9999)

        # 回归损失 (负对数似然，考虑不确定性)
        self.reg_criterion = self._gaussian_nll_loss

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.threshold_classifier = AdaptiveThresholdClassifier(metric='f1')

    def _gaussian_nll_loss(self, mean, var, target):
        """高斯负对数似然损失"""
        return 0.5 * (torch.log(var) + (target - mean) ** 2 / var).mean()

    def _create_balanced_sampler(self, labels):
        """创建类别平衡采样器"""
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        weights = torch.DoubleTensor(weights)
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def train(self, X_train, y_train, t_train, X_val, y_val, t_val,
              epochs=200, batch_size=32, curriculum=True):
        """训练模型"""

        # 类别平衡采样
        sampler = self._create_balanced_sampler(y_train)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
            torch.FloatTensor(t_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

        best_val_f1 = 0
        best_state = None
        patience = 0

        for epoch in range(epochs):
            self.model.train()
            total_cls_loss = 0
            total_reg_loss = 0

            # 课程学习：逐渐增加难样本权重
            if curriculum:
                difficulty_weight = min(1.0, (epoch + 1) / 50)
            else:
                difficulty_weight = 1.0

            for x, y, t in train_loader:
                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

                cls_logits, reg_mean, reg_var = self.model(x)

                # 分类损失
                cls_loss = self.cls_criterion(cls_logits, y)

                # 回归损失 (带不确定性)
                reg_loss = self._gaussian_nll_loss(reg_mean.squeeze(), reg_var.squeeze(), t)

                # 总损失
                loss = cls_loss + 0.3 * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()

            scheduler.step()

            # 验证
            val_metrics = self._evaluate(X_val, y_val, t_val, update_threshold=True)

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = self.model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= 30:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: cls_loss={total_cls_loss/len(train_loader):.4f}, "
                      f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}, "
                      f"val_mae={val_metrics['mae']:.2f}s")

        self.model.load_state_dict(best_state)
        print(f"  Best val F1: {best_val_f1:.4f}, Threshold: {self.threshold_classifier.threshold:.4f}")

    def _evaluate(self, X, y, t, update_threshold=False):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            cls_logits, reg_mean, reg_var = self.model(x_tensor)

            probs = F.softmax(cls_logits, dim=1)[:, 1].cpu().numpy()
            reg_pred = reg_mean.squeeze().cpu().numpy()
            reg_uncertainty = reg_var.squeeze().cpu().numpy()

        # 更新自适应阈值
        if update_threshold:
            self.threshold_classifier.fit(probs, y)

        y_pred = self.threshold_classifier.predict(probs)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average='macro'),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'mae': mean_absolute_error(t, reg_pred),
            'probs': probs,
            'reg_pred': reg_pred,
            'reg_uncertainty': reg_uncertainty
        }

    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            cls_logits, reg_mean, reg_var = self.model(x_tensor)

            probs = F.softmax(cls_logits, dim=1)[:, 1].cpu().numpy()
            y_pred = self.threshold_classifier.predict(probs)
            reg_pred = reg_mean.squeeze().cpu().numpy()
            reg_uncertainty = reg_var.squeeze().cpu().numpy()

        return {
            'y_pred': y_pred,
            'y_prob': probs,
            'ignition_time': reg_pred,
            'uncertainty': reg_uncertainty
        }


# ==================== 主实验 ====================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载数据
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = extract_features(df_valid)
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    # 数据划分
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    # 标准化
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    samples_per_class = [np.sum(y_train == 0), np.sum(y_train == 1)]

    print(f"训练集: {len(X_train)} (正常: {samples_per_class[0]}, 异常: {samples_per_class[1]})")
    print(f"验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print(f"类别不平衡比例: 1:{samples_per_class[0]/samples_per_class[1]:.1f}")
    print("=" * 80)

    # 创建模型
    model = SmallSampleRobustNet(input_dim, hidden_dim=128, embed_dim=64)
    trainer = RobustTrainer(model, device, samples_per_class, lr=1e-3)

    # 训练
    print("\n训练小样本鲁棒模型...")
    trainer.train(X_train, y_train, t_train, X_val, y_val, t_val,
                  epochs=200, batch_size=32, curriculum=True)

    # 测试
    print("\n" + "=" * 80)
    print("测试集评估结果")
    print("=" * 80)

    result = trainer.predict(X_test)
    y_pred = result['y_pred']
    t_pred = result['ignition_time']
    uncertainty = result['uncertainty']

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(t_test, t_pred)

    # 虚警率
    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    # 漏检率
    anomaly_mask = y_test == 1
    miss_rate = ((y_pred == 0) & anomaly_mask).sum() / anomaly_mask.sum() if anomaly_mask.sum() > 0 else 0

    print(f"准确率: {acc:.2%}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"虚警率: {far:.2%}")
    print(f"漏检率: {miss_rate:.2%}")
    print(f"点火时刻MAE: {mae:.2f}s")
    print(f"平均不确定性: {uncertainty.mean():.4f}")
    print(f"自适应阈值: {trainer.threshold_classifier.threshold:.4f}")

    # 按不确定性分析
    print("\n" + "=" * 80)
    print("不确定性分析")
    print("=" * 80)

    # 高置信度样本
    high_conf_mask = uncertainty < np.percentile(uncertainty, 50)
    if high_conf_mask.sum() > 0:
        high_conf_mae = mean_absolute_error(t_test[high_conf_mask], t_pred[high_conf_mask])
        print(f"高置信度样本 (前50%): MAE = {high_conf_mae:.2f}s")

    # 低置信度样本
    low_conf_mask = uncertainty >= np.percentile(uncertainty, 50)
    if low_conf_mask.sum() > 0:
        low_conf_mae = mean_absolute_error(t_test[low_conf_mask], t_pred[low_conf_mask])
        print(f"低置信度样本 (后50%): MAE = {low_conf_mae:.2f}s")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame({
        'metric': ['accuracy', 'f1', 'precision', 'recall', 'far', 'miss_rate', 'mae', 'threshold'],
        'value': [acc, f1, precision, recall, far, miss_rate, mae, trainer.threshold_classifier.threshold]
    })
    results_df.to_csv("results/robust_model_results.csv", index=False)
    print("\n结果已保存到 results/robust_model_results.csv")

    return {
        'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
        'far': far, 'miss_rate': miss_rate, 'mae': mae
    }


if __name__ == '__main__':
    main()
