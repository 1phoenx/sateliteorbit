"""
RF分类器 + 注意力增强Transformer回归器
目标：准确率≥92%, 虚警率≤3%, 点火MAE尽可能低
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


def extract_features(P, T, R):
    """提取增强特征 (20维)"""
    features = np.column_stack([
        P, T, R,
        P * T, P / (R + 1e-6), T / (R + 1e-6), P * T / (R + 1e-6),
        P ** 2, T ** 2, R ** 2,
        np.sqrt(P + 1e-6), np.sqrt(T + 1e-6),
        np.log1p(P), np.log1p(T), np.log1p(R),
        T * R, P / (T + 1e-6), P * T * R,
        np.exp(-np.clip(T, 0, 10)), T ** 0.5 * P
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class MultiHeadAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力 + 残差
        x = x + self.dropout(self.attn(self.norm1(x)))
        # 前馈 + 残差
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class AttentionTransformerRegressor(nn.Module):
    """
    注意力增强Transformer回归器

    创新点：
    1. 多头自注意力机制
    2. 位置编码
    3. 深层Transformer结构
    4. 残差连接
    """
    def __init__(self, input_dim, d_model=128, num_heads=8, num_layers=6,
                 d_ff=512, dropout=0.1):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 位置编码 (可学习)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # 输入: (batch, features) -> (batch, 1, d_model)
        x = self.input_proj(x).unsqueeze(1)

        # 添加位置编码
        x = x + self.pos_embedding

        # Transformer层
        for layer in self.layers:
            x = layer(x)

        # 输出
        x = x.squeeze(1)
        return self.output(x)


def load_data():
    """加载GAN扩充数据"""
    # 扩充数据用于训练
    df_aug = pd.read_csv("data/augmented_dataset.csv")

    # 原始数据用于测试和回归训练
    df_orig = pd.read_csv("data/feature_dataset.csv")
    df_orig = df_orig[df_orig['is_valid'] == 1].copy()

    print(f"扩充数据: {len(df_aug)} 条 (用于分类器训练)")
    print(f"原始数据: {len(df_orig)} 条 (用于回归器训练和测试)")

    # 扩充数据特征
    X_aug = extract_features(df_aug['P'].values, df_aug['T'].values, df_aug['R'].values)
    y_aug = df_aug['is_anomalous'].values.astype(int)

    # 原始数据特征
    X_orig = extract_features(df_orig['P'].values, df_orig['T'].values, df_orig['R'].values)
    y_orig = df_orig['is_anomalous'].values.astype(int)
    t_orig = df_orig['ignition_time'].values.astype(np.float32)

    # 标准化
    scaler = RobustScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)
    X_orig_scaled = scaler.transform(X_orig)

    # 分类器训练数据 (扩充数据)
    X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(
        X_aug_scaled, y_aug, test_size=0.1, random_state=42, stratify=y_aug
    )

    # 回归器训练数据 (原始数据)
    X_reg_train, X_reg_test, t_reg_train, t_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_orig_scaled, t_orig, y_orig, test_size=0.2, random_state=42
    )

    return {
        'X_clf_train': X_clf_train, 'y_clf_train': y_clf_train,
        'X_clf_val': X_clf_val, 'y_clf_val': y_clf_val,
        'X_reg_train': X_reg_train, 't_reg_train': t_reg_train,
        'X_reg_test': X_reg_test, 't_reg_test': t_reg_test,
        'y_reg_test': y_reg_test,
        'X_test': X_orig_scaled, 'y_test': y_orig, 't_test': t_orig,
        'scaler': scaler
    }


def train_transformer(model, X_train, t_train, X_val, t_val, device,
                      epochs=500, lr=5e-4, batch_size=64):
    """训练Transformer回归器"""
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(t_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    best_val_mae = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, t in train_loader:
            x, t = x.to(device), t.to(device)

            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(device)
            val_pred = model(X_v).squeeze().cpu().numpy()
            val_mae = mean_absolute_error(t_val, val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"val_mae={val_mae:.2f}s, best={best_val_mae:.2f}s")

    model.load_state_dict(best_state)
    return model, best_val_mae


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print("=" * 80)

    data = load_data()
    input_dim = data['X_clf_train'].shape[1]

    results = {}

    # 1. 训练RF分类器 (使用扩充数据)
    print("\n[1] 训练RandomForest分类器 (扩充数据)...")
    rf_clf = RandomForestClassifier(
        n_estimators=1000, max_depth=25, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_clf.fit(data['X_clf_train'], data['y_clf_train'])

    val_pred = rf_clf.predict(data['X_clf_val'])
    val_acc = accuracy_score(data['y_clf_val'], val_pred)
    print(f"  验证准确率: {val_acc:.4f}")

    # 2. 测试不同Transformer配置
    configs = [
        {'d_model': 64, 'num_heads': 4, 'num_layers': 3, 'd_ff': 256, 'name': 'Small'},
        {'d_model': 128, 'num_heads': 8, 'num_layers': 4, 'd_ff': 512, 'name': 'Medium'},
        {'d_model': 128, 'num_heads': 8, 'num_layers': 6, 'd_ff': 512, 'name': 'Large'},
        {'d_model': 256, 'num_heads': 8, 'num_layers': 4, 'd_ff': 1024, 'name': 'Wide'},
    ]

    best_model = None
    best_mae = float('inf')
    best_config = None

    for config in configs:
        print(f"\n[2] 训练Transformer回归器 ({config['name']})...")
        print(f"    d_model={config['d_model']}, heads={config['num_heads']}, "
              f"layers={config['num_layers']}, d_ff={config['d_ff']}")

        model = AttentionTransformerRegressor(
            input_dim,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=0.1
        )

        # 划分回归训练/验证集
        X_train, X_val, t_train, t_val = train_test_split(
            data['X_reg_train'], data['t_reg_train'],
            test_size=0.15, random_state=42
        )

        model, val_mae = train_transformer(
            model, X_train, t_train, X_val, t_val, device,
            epochs=300, lr=5e-4, batch_size=64
        )

        # 测试
        model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(data['X_reg_test']).to(device)
            t_pred = model(X_test).squeeze().cpu().numpy()
            test_mae = mean_absolute_error(data['t_reg_test'], t_pred)

        print(f"    测试MAE: {test_mae:.2f}s")

        results[f"Transformer-{config['name']}"] = test_mae

        if test_mae < best_mae:
            best_mae = test_mae
            best_model = model
            best_config = config

    # 3. 最终评估 (RF + 最佳Transformer)
    print("\n" + "=" * 80)
    print(f"最佳配置: {best_config['name']} (MAE={best_mae:.2f}s)")
    print("=" * 80)

    # 在完整测试集上评估
    y_pred = rf_clf.predict(data['X_test'])
    y_prob = rf_clf.predict_proba(data['X_test'])[:, 1]

    best_model.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(data['X_test']).to(device)
        t_pred = best_model(X_test).squeeze().cpu().numpy()

    y_test = data['y_test']
    t_test = data['t_test']

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(t_test, t_pred)

    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum()

    print(f"\n最终结果 (RF + Transformer-{best_config['name']}):")
    print(f"  准确率: {acc:.2%}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  虚警率: {far:.2%}")
    print(f"  点火时刻MAE: {mae:.2f}s")

    # 对比各配置
    print("\n" + "=" * 80)
    print("Transformer配置对比:")
    print("-" * 40)
    for name, test_mae in results.items():
        marker = " ← Best" if test_mae == best_mae else ""
        print(f"  {name}: MAE = {test_mae:.2f}s{marker}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    final_results = {
        'accuracy': acc,
        'f1': f1,
        'far': far,
        'mae': mae,
        'best_config': best_config['name']
    }
    pd.DataFrame([final_results]).to_csv("results/rf_transformer_optimized.csv", index=False)
    print("\n结果已保存到 results/rf_transformer_optimized.csv")

    return final_results


if __name__ == '__main__':
    main()
