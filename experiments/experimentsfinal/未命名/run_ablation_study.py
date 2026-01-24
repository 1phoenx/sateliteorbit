"""
小样本鲁棒变轨识别 - 完整消融实验
用于论文写作的系统性实验
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
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))


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


# ==================== 模型组件 ====================

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * confidence + (1 - one_hot) * smooth_value
        log_prob = F.log_softmax(pred, dim=1)
        return -(smooth_target * log_prob).sum(dim=1).mean()


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """对比学习损失 (InfoNCE)"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature
        neg_sim_i = torch.mm(z_i, z_j.t()) / self.temperature
        mask = torch.eye(batch_size, device=z_i.device).bool()
        neg_sim_i = neg_sim_i.masked_fill(mask, -1e4)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim_i], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        return F.cross_entropy(logits, labels)


# ==================== 模型定义 ====================

class BaselineMLP(nn.Module):
    """基线MLP"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.classifier(x), self.regressor(x)


class MLPWithDropout(nn.Module):
    """MLP + Dropout"""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.classifier(x), self.regressor(x)


class MLPWithAttention(nn.Module):
    """MLP + 自注意力"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out).squeeze(1)
        return self.classifier(x), self.regressor(x)


class RobustModel(nn.Module):
    """完整鲁棒模型: Dropout + 注意力 + 对比学习编码器"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        # 投影头 (对比学习)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        # 自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, x):
        return self.encoder(x)

    def project(self, z):
        return self.projector(z)

    def forward(self, x):
        z = self.encode(x)
        z = z.unsqueeze(1)
        attn_out, _ = self.attention(z, z, z)
        z = self.norm(z + attn_out).squeeze(1)
        return self.classifier(z), self.regressor(z)


# ==================== 训练函数 ====================

def train_model(model, data, device, epochs=200, lr=1e-3, weight_decay=1e-4,
                use_focal=False, use_label_smooth=False, use_mixup=False):
    """通用训练函数"""
    model = model.to(device)

    if use_focal:
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif use_label_smooth:
        cls_criterion = LabelSmoothingLoss(smoothing=0.1)
    else:
        cls_criterion = nn.CrossEntropyLoss()

    reg_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(data['X_train']),
            torch.LongTensor(data['y_train']),
            torch.FloatTensor(data['t_train'])
        ),
        batch_size=32, shuffle=True, drop_last=True
    )

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for x, y, t in train_loader:
            x, y, t = x.to(device), y.to(device), t.to(device)

            # Mixup数据增强
            if use_mixup and np.random.random() > 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[idx]
                cls_out, reg_out = model(x)
                cls_loss = lam * cls_criterion(cls_out, y) + (1 - lam) * cls_criterion(cls_out, y[idx])
            else:
                cls_out, reg_out = model(x)
                cls_loss = cls_criterion(cls_out, y)

            reg_loss = reg_criterion(reg_out.squeeze(), t)
            loss = cls_loss + 0.5 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(data['X_val']).to(device)
            y_val = torch.LongTensor(data['y_val']).to(device)
            t_val = torch.FloatTensor(data['t_val']).to(device)
            cls_out, reg_out = model(X_val)
            val_loss = cls_criterion(cls_out, y_val) + 0.5 * reg_criterion(reg_out.squeeze(), t_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break

    model.load_state_dict(best_state)
    return model


def pretrain_contrastive(model, X_train, device, epochs=50, lr=1e-3):
    """对比学习预训练"""
    model = model.to(device)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=32, shuffle=True, drop_last=True
    )

    for epoch in range(epochs):
        model.train()
        for (x,) in loader:
            x = x.to(device)
            # 两个增强视图
            noise1 = torch.randn_like(x) * 0.05
            noise2 = torch.randn_like(x) * 0.05
            x1, x2 = x + noise1, x + noise2

            z1 = model.project(model.encode(x1))
            z2 = model.project(model.encode(x2))

            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, data, device):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(data['X_test']).to(device)
        cls_out, reg_out = model(X_test)
        y_pred = cls_out.argmax(dim=1).cpu().numpy()
        t_pred = reg_out.squeeze().cpu().numpy()

    y_test = data['y_test']
    t_test = data['t_test']

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(t_test, t_pred)

    # 虚警率
    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    return {
        'accuracy': acc, 'f1': f1, 'precision': precision,
        'recall': recall, 'mae': mae, 'far': far
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载数据
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = extract_features(df_valid)
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    data = {
        'X_train': X_train, 'y_train': y_train, 't_train': t_train,
        'X_val': X_val, 'y_val': y_val, 't_val': t_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test
    }

    input_dim = X_train.shape[1]
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}, 特征维度: {input_dim}")
    print("=" * 100)

    results = {}

    # A0: 基线MLP
    print("A0: 基线MLP...")
    model_a0 = BaselineMLP(input_dim)
    model_a0 = train_model(model_a0, data, device)
    results['A0: Baseline MLP'] = evaluate(model_a0, data, device)

    # A1: + Dropout + BatchNorm
    print("A1: + Dropout + BatchNorm...")
    model_a1 = MLPWithDropout(input_dim)
    model_a1 = train_model(model_a1, data, device)
    results['A1: + Dropout/BN'] = evaluate(model_a1, data, device)

    # A2: + Focal Loss
    print("A2: + Focal Loss...")
    model_a2 = MLPWithDropout(input_dim)
    model_a2 = train_model(model_a2, data, device, use_focal=True)
    results['A2: + Focal Loss'] = evaluate(model_a2, data, device)

    # A3: + Label Smoothing
    print("A3: + Label Smoothing...")
    model_a3 = MLPWithDropout(input_dim)
    model_a3 = train_model(model_a3, data, device, use_label_smooth=True)
    results['A3: + Label Smooth'] = evaluate(model_a3, data, device)

    # A4: + Mixup
    print("A4: + Mixup...")
    model_a4 = MLPWithDropout(input_dim)
    model_a4 = train_model(model_a4, data, device, use_label_smooth=True, use_mixup=True)
    results['A4: + Mixup'] = evaluate(model_a4, data, device)

    # A5: + Self-Attention
    print("A5: + Self-Attention...")
    model_a5 = MLPWithAttention(input_dim)
    model_a5 = train_model(model_a5, data, device, use_label_smooth=True, use_mixup=True)
    results['A5: + Attention'] = evaluate(model_a5, data, device)

    # A6: + 对比学习预训练 (完整模型)
    print("A6: + Contrastive Pretrain (Full)...")
    model_a6 = RobustModel(input_dim)
    model_a6 = pretrain_contrastive(model_a6, X_train, device, epochs=50)
    model_a6 = train_model(model_a6, data, device, use_label_smooth=True, use_mixup=True)
    results['A6: Full Model'] = evaluate(model_a6, data, device)

    # 输出结果
    print("\n" + "=" * 120)
    print("消融实验结果 (Ablation Study)")
    print("=" * 120)
    print(f"{'配置':<25} {'准确率':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'虚警率':>10} {'点火MAE':>12}")
    print("-" * 120)

    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.2%} {res['f1']:>10.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>10.4f} "
              f"{res['far']:>10.2%} {res['mae']:>10.2f}s")

    print("=" * 120)

    # 计算各组件贡献
    print("\n组件贡献分析:")
    baseline_acc = results['A0: Baseline MLP']['accuracy']
    for name, res in list(results.items())[1:]:
        delta = (res['accuracy'] - baseline_acc) * 100
        print(f"  {name}: {delta:+.2f}% (相对基线)")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results).T
    df_results.to_csv("results/ablation_study_full.csv")
    print("\n结果已保存到 results/ablation_study_full.csv")

    return results


if __name__ == '__main__':
    main()
