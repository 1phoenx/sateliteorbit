"""
使用GAN扩充数据训练模型
数据量：24,610条 (原始2,612条 × ~10倍)
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


class DeepClassifier(nn.Module):
    """深度分类器"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        return self.model(x)


class DeepRegressor(nn.Module):
    """深度回归器"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.model(x)


class TransformerRegressor(nn.Module):
    """Transformer回归器"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


def load_augmented_data():
    """加载GAN扩充后的数据"""
    # 扩充数据 (用于训练)
    df_aug = pd.read_csv("data/augmented_dataset.csv")

    # 原始数据 (用于测试，确保测试集是真实数据)
    df_orig = pd.read_csv("data/feature_dataset.csv")
    df_orig = df_orig[df_orig['is_valid'] == 1].copy()

    print(f"扩充数据集: {len(df_aug)} 条")
    print(f"  - 原始样本: {(df_aug['is_synthetic'] == 0).sum()}")
    print(f"  - 生成样本: {(df_aug['is_synthetic'] == 1).sum()}")
    print(f"  - 异常比例: {df_aug['is_anomalous'].mean()*100:.1f}%")

    # 提取特征
    X_aug = extract_features(df_aug['P'].values, df_aug['T'].values, df_aug['R'].values)
    y_aug = df_aug['is_anomalous'].values.astype(int)

    # 原始数据用于测试
    X_orig = extract_features(df_orig['P'].values, df_orig['T'].values, df_orig['R'].values)
    y_orig = df_orig['is_anomalous'].values.astype(int)
    t_orig = df_orig['ignition_time'].values.astype(np.float32)

    # 从扩充数据中分离训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.1, random_state=42, stratify=y_aug
    )

    # 原始数据作为测试集
    X_test, y_test, t_test = X_orig, y_orig, t_orig

    # 标准化
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} (异常: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)} (原始真实数据)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test,
        'scaler': scaler
    }


def train_classifier(model, data, device, epochs=200, lr=1e-3):
    """训练分类器"""
    model = model.to(device)

    # 类别权重
    class_counts = np.bincount(data['y_train'])
    weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']), torch.LongTensor(data['y_train'])),
        batch_size=128, shuffle=True, drop_last=True
    )

    best_val_acc = 0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(data['X_val']).to(device)
            val_pred = model(X_val).argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(data['y_val'], val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: val_acc = {val_acc:.4f}, best = {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    return model


def train_regressor(model, data, device, epochs=200, lr=1e-3):
    """训练回归器 (使用原始数据的ignition_time)"""
    # 需要原始数据的ignition_time来训练回归器
    df_orig = pd.read_csv("data/feature_dataset.csv")
    df_orig = df_orig[df_orig['is_valid'] == 1].copy()

    X_orig = extract_features(df_orig['P'].values, df_orig['T'].values, df_orig['R'].values)
    t_orig = df_orig['ignition_time'].values.astype(np.float32)

    X_orig = data['scaler'].transform(X_orig)

    # 划分
    X_train, X_val, t_train, t_val = train_test_split(
        X_orig, t_orig, test_size=0.2, random_state=42
    )

    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(t_train)),
        batch_size=64, shuffle=True, drop_last=True
    )

    best_val_mae = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze()
            loss = criterion(out, t)
            loss.backward()
            optimizer.step()
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

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: val_mae = {val_mae:.2f}s, best = {best_val_mae:.2f}s")

    model.load_state_dict(best_state)
    return model


def evaluate(clf, reg, data, device):
    """评估模型"""
    # 分类
    if isinstance(clf, nn.Module):
        clf.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(data['X_test']).to(device)
            y_pred = clf(X_test).argmax(dim=1).cpu().numpy()
    else:
        y_pred = clf.predict(data['X_test'])

    # 回归
    if isinstance(reg, nn.Module):
        reg.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(data['X_test']).to(device)
            t_pred = reg(X_test).squeeze().cpu().numpy()
    else:
        t_pred = reg.predict(data['X_test'])

    y_test = data['y_test']
    t_test = data['t_test']

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(t_test, t_pred)

    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    return {'accuracy': acc, 'f1': f1, 'mae': mae, 'far': far}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print("=" * 80)

    data = load_augmented_data()
    input_dim = data['X_train'].shape[1]

    print("\n" + "=" * 80)
    results = {}

    # 方案1: RF分类 + RF回归 (基线)
    print("\n[1] RandomForest基线...")
    rf_clf = RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(data['X_train'], data['y_train'])

    from sklearn.ensemble import RandomForestRegressor
    df_orig = pd.read_csv("data/feature_dataset.csv")
    df_orig = df_orig[df_orig['is_valid'] == 1].copy()
    X_orig = extract_features(df_orig['P'].values, df_orig['T'].values, df_orig['R'].values)
    t_orig = df_orig['ignition_time'].values.astype(np.float32)
    X_orig = data['scaler'].transform(X_orig)

    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf_reg.fit(X_orig, t_orig)

    results['RF + RF'] = evaluate(rf_clf, rf_reg, data, device)

    # 方案2: DNN分类 + DNN回归
    print("\n[2] DNN分类 + DNN回归...")
    dnn_clf = DeepClassifier(input_dim, hidden_dim=256)
    dnn_clf = train_classifier(dnn_clf, data, device, epochs=200)

    dnn_reg = DeepRegressor(input_dim, hidden_dim=256)
    dnn_reg = train_regressor(dnn_reg, data, device, epochs=200)

    results['DNN + DNN'] = evaluate(dnn_clf, dnn_reg, data, device)

    # 方案3: DNN分类 + Transformer回归
    print("\n[3] DNN分类 + Transformer回归...")
    trans_reg = TransformerRegressor(input_dim, d_model=128, nhead=4, num_layers=4)
    trans_reg = train_regressor(trans_reg, data, device, epochs=200)

    results['DNN + Transformer'] = evaluate(dnn_clf, trans_reg, data, device)

    # 方案4: RF分类 (扩充数据) + Transformer回归
    print("\n[4] RF分类 (扩充数据) + Transformer回归...")
    results['RF(aug) + Transformer'] = evaluate(rf_clf, trans_reg, data, device)

    # 输出结果
    print("\n" + "=" * 90)
    print("实验结果 (使用GAN扩充数据训练)")
    print("=" * 90)
    print(f"{'模型组合':<25} {'准确率':>10} {'F1':>10} {'虚警率':>10} {'点火MAE':>12}")
    print("-" * 90)

    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.2%} {res['f1']:>10.4f} "
              f"{res['far']:>10.2%} {res['mae']:>10.2f}s")

    print("=" * 90)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv("results/augmented_data_results.csv")
    print("\n结果已保存到 results/augmented_data_results.csv")

    return results


if __name__ == '__main__':
    main()
