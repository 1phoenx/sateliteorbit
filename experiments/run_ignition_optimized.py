"""
点火时刻估计优化实验
目标: 降低点火时刻MAE误差
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def extract_enhanced_features_v2(df):
    """提取增强特征 - 增加时间相关特征"""
    P = df['P'].values
    T = df['T'].values
    R = df['R'].values

    # 基础特征
    features = [P, T, R]

    # 交互特征
    features.append(P * T)
    features.append(P / (R + 1e-6))
    features.append(T / (R + 1e-6))
    features.append(P * T / (R + 1e-6))

    # 多项式特征
    features.append(P ** 2)
    features.append(T ** 2)
    features.append(R ** 2)
    features.append(np.sqrt(P + 1e-6))
    features.append(np.sqrt(T + 1e-6))
    features.append(np.log1p(P))
    features.append(np.log1p(T))
    features.append(np.log1p(R))

    # 时间相关特征 (T是持续时间，与点火时刻强相关)
    features.append(T * R)  # 持续时间与频域比的交互
    features.append(P / (T + 1e-6))  # 功率密度
    features.append(P * T * R)  # 三特征交互
    features.append(np.exp(-T))  # 指数衰减
    features.append(T ** 0.5 * P)  # 根号时间与峰值

    return np.column_stack(features)


class ImprovedIgnitionTransformer(nn.Module):
    """改进的点火时刻估计Transformer - 更深更宽"""

    def __init__(self, input_dim=20, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class IgnitionMLP(nn.Module):
    """深度MLP回归器"""

    def __init__(self, input_dim=20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


def load_data():
    """加载数据"""
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    # 增强特征
    features = extract_enhanced_features_v2(df_valid)
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    ignition_time = np.nan_to_num(ignition_time, nan=0.0)

    # 划分数据
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    # 特征标准化
    feature_scaler = RobustScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)

    # 目标值标准化 (关键优化!)
    target_scaler = StandardScaler()
    t_train_scaled = target_scaler.fit_transform(t_train.reshape(-1, 1)).flatten()
    t_val_scaled = target_scaler.transform(t_val.reshape(-1, 1)).flatten()
    t_test_scaled = target_scaler.transform(t_test.reshape(-1, 1)).flatten()

    logger.info(f"特征维度: {X_train.shape[1]}")
    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    logger.info(f"点火时刻范围: [{t_train.min():.2f}, {t_train.max():.2f}]")

    return {
        'X_train': X_train, 'y_train': y_train,
        't_train': t_train, 't_train_scaled': t_train_scaled,
        'X_val': X_val, 'y_val': y_val,
        't_val': t_val, 't_val_scaled': t_val_scaled,
        'X_test': X_test, 'y_test': y_test,
        't_test': t_test, 't_test_scaled': t_test_scaled,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }


def train_rf_regressor(data):
    """训练随机森林回归器"""
    logger.info("训练RandomForest回归器...")

    model = RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    model.fit(data['X_train'], data['t_train'])

    val_pred = model.predict(data['X_val'])
    val_mae = mean_absolute_error(data['t_val'], val_pred)
    logger.info(f"  RF验证MAE: {val_mae:.2f}s")

    return model


def train_transformer(data, device, epochs=500, use_scaled=True):
    """训练Transformer回归器"""
    logger.info("训练Transformer回归器...")

    input_dim = data['X_train'].shape[1]
    model = ImprovedIgnitionTransformer(input_dim=input_dim).to(device)

    # 使用MSE损失 (对精确回归更好)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    t_train = data['t_train_scaled'] if use_scaled else data['t_train']
    t_val = data['t_val_scaled'] if use_scaled else data['t_val']

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(t_train)),
        batch_size=32, shuffle=True, drop_last=True
    )

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, t_batch in train_loader:
            X_batch, t_batch = X_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            loss = criterion(out, t_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(data['X_val']).to(device)
            t_v = torch.FloatTensor(t_val).to(device)
            val_pred = model(X_v).squeeze()
            val_loss = criterion(val_pred, t_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 100:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 100 == 0:
            logger.info(f"  Epoch {epoch+1}: val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

    model.load_state_dict(best_state)
    return model


def train_mlp(data, device, epochs=500, use_scaled=True):
    """训练MLP回归器"""
    logger.info("训练MLP回归器...")

    input_dim = data['X_train'].shape[1]
    model = IgnitionMLP(input_dim=input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    t_train = data['t_train_scaled'] if use_scaled else data['t_train']
    t_val = data['t_val_scaled'] if use_scaled else data['t_val']

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(t_train)),
        batch_size=32, shuffle=True, drop_last=True
    )

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, t_batch in train_loader:
            X_batch, t_batch = X_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            loss = criterion(out, t_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(data['X_val']).to(device)
            t_v = torch.FloatTensor(t_val).to(device)
            val_pred = model(X_v).squeeze()
            val_loss = criterion(val_pred, t_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 100:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 100 == 0:
            logger.info(f"  Epoch {epoch+1}: val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

    model.load_state_dict(best_state)
    return model


def evaluate_regressor(model, data, device, model_type='rf', use_scaled=True):
    """评估回归器"""
    X_test = data['X_test']
    t_test = data['t_test']

    if model_type == 'rf':
        t_pred = model.predict(X_test)
    else:
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            t_pred = model(X_t).squeeze().cpu().numpy()

            # 反标准化
            if use_scaled:
                t_pred = data['target_scaler'].inverse_transform(t_pred.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(t_test, t_pred)
    rmse = np.sqrt(np.mean((t_test - t_pred) ** 2))

    # 计算误差分布
    errors = np.abs(t_test - t_pred)
    p50 = np.percentile(errors, 50)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)

    return {
        'mae': mae,
        'rmse': rmse,
        'p50': p50,
        'p90': p90,
        'p95': p95
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device}")

    data = load_data()

    results = {}

    # 方案1: RandomForest回归器
    rf_reg = train_rf_regressor(data)
    res_rf = evaluate_regressor(rf_reg, data, device, 'rf')
    results['RandomForest'] = res_rf

    # 方案2: Transformer (标准化目标)
    trans_reg = train_transformer(data, device, epochs=500, use_scaled=True)
    res_trans = evaluate_regressor(trans_reg, data, device, 'transformer', use_scaled=True)
    results['Transformer (scaled)'] = res_trans

    # 方案3: MLP (标准化目标)
    mlp_reg = train_mlp(data, device, epochs=500, use_scaled=True)
    res_mlp = evaluate_regressor(mlp_reg, data, device, 'mlp', use_scaled=True)
    results['MLP (scaled)'] = res_mlp

    # 方案4: 集成 (RF + Transformer + MLP 平均)
    rf_pred = rf_reg.predict(data['X_test'])

    trans_reg.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(data['X_test']).to(device)
        trans_pred = trans_reg(X_t).squeeze().cpu().numpy()
        trans_pred = data['target_scaler'].inverse_transform(trans_pred.reshape(-1, 1)).flatten()

    mlp_reg.eval()
    with torch.no_grad():
        mlp_pred = mlp_reg(X_t).squeeze().cpu().numpy()
        mlp_pred = data['target_scaler'].inverse_transform(mlp_pred.reshape(-1, 1)).flatten()

    ensemble_pred = (rf_pred + trans_pred + mlp_pred) / 3
    ensemble_mae = mean_absolute_error(data['t_test'], ensemble_pred)
    ensemble_rmse = np.sqrt(np.mean((data['t_test'] - ensemble_pred) ** 2))
    errors = np.abs(data['t_test'] - ensemble_pred)
    results['Ensemble (RF+Trans+MLP)'] = {
        'mae': ensemble_mae,
        'rmse': ensemble_rmse,
        'p50': np.percentile(errors, 50),
        'p90': np.percentile(errors, 90),
        'p95': np.percentile(errors, 95)
    }

    # 输出结果
    print("\n" + "=" * 90)
    print("点火时刻估计优化实验结果")
    print("=" * 90)
    print(f"{'模型':<25} {'MAE':>10} {'RMSE':>10} {'P50':>10} {'P90':>10} {'P95':>10}")
    print("-" * 90)

    for name, res in results.items():
        print(f"{name:<25} {res['mae']:>10.2f}s {res['rmse']:>10.2f}s "
              f"{res['p50']:>10.2f}s {res['p90']:>10.2f}s {res['p95']:>10.2f}s")

    print("\n" + "=" * 90)

    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"最佳模型: {best_model[0]}, MAE: {best_model[1]['mae']:.2f}s")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv("results/ignition_optimized_results.csv")
    logger.info("结果已保存到 results/ignition_optimized_results.csv")

    return results


if __name__ == '__main__':
    main()
