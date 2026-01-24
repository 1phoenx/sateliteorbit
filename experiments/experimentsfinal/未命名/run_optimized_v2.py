"""
优化实验v2 - 保守策略
目标: 准确率≥92%, 虚警率≤3%, 响应时间≤5s
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


class ImprovedDNN(nn.Module):
    """改进的DNN - 更深更宽"""

    def __init__(self, input_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)


class ImprovedTransformer(nn.Module):
    """改进的Transformer"""

    def __init__(self, input_dim=3, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


def load_data():
    """加载数据 - 不使用HMSE，保持原始特征"""
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = df_valid[['P', 'T', 'R']].values
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0)
    ignition_time = np.nan_to_num(ignition_time, nan=0.0)

    # 划分数据
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    # 使用RobustScaler更鲁棒
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    logger.info(f"异常比例: 训练{y_train.mean():.2%}, 测试{y_test.mean():.2%}")

    return {
        'X_train': X_train, 'y_train': y_train, 't_train': t_train,
        'X_val': X_val, 'y_val': y_val, 't_val': t_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test
    }


def train_ensemble_classifier(data):
    """训练集成分类器"""
    logger.info("训练集成分类器...")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
    )

    # 投票集成
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    ensemble.fit(data['X_train'], data['y_train'])

    # 验证
    val_pred = ensemble.predict(data['X_val'])
    val_acc = accuracy_score(data['y_val'], val_pred)
    logger.info(f"  集成验证准确率: {val_acc:.4f}")

    return ensemble


def train_dnn_classifier(data, device, epochs=300):
    """训练DNN分类器"""
    logger.info("训练DNN分类器...")

    model = ImprovedDNN(input_dim=3).to(device)

    # 计算类别权重
    class_counts = np.bincount(data['y_train'])
    weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']), torch.LongTensor(data['y_train'])),
        batch_size=32, shuffle=True, drop_last=True
    )

    best_val_acc = 0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
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
            if patience >= 50:
                break

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1}: val_acc={val_acc:.4f}, best={best_val_acc:.4f}")

    model.load_state_dict(best_state)
    logger.info(f"  DNN最佳验证准确率: {best_val_acc:.4f}")
    return model


def train_transformer_regressor(data, device, epochs=300):
    """训练Transformer回归器"""
    logger.info("训练Transformer回归器...")

    model = ImprovedTransformer(input_dim=3).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(data['X_train']), torch.FloatTensor(data['t_train'])),
        batch_size=32, shuffle=True, drop_last=True
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, t_batch in train_loader:
            X_batch, t_batch = X_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            loss = criterion(out, t_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(data['X_val']).to(device)
            t_val = torch.FloatTensor(data['t_val']).to(device)
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, t_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    return model


def evaluate(clf, reg, data, device, clf_type='ensemble'):
    """评估模型"""
    X_test = data['X_test']
    y_test = data['y_test']
    t_test = data['t_test']

    start_time = time.time()

    # 分类
    if clf_type == 'ensemble':
        y_pred = clf.predict(X_test)
    else:
        clf.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            y_pred = clf(X_t).argmax(dim=1).cpu().numpy()

    # 回归
    reg.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        t_pred = reg(X_t).squeeze().cpu().numpy()

    inference_time = time.time() - start_time

    # 指标
    accuracy = accuracy_score(y_test, y_pred)
    normal_mask = y_test == 0
    false_alarm = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0
    ignition_mae = np.mean(np.abs(t_test - t_pred))

    return {
        'accuracy': accuracy,
        'false_alarm_rate': false_alarm,
        'ignition_mae': ignition_mae,
        'inference_time': inference_time
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device}")

    data = load_data()

    # 训练回归器
    transformer_reg = train_transformer_regressor(data, device, epochs=300)

    results = {}

    # 方案1: 集成分类器 + Transformer
    ensemble_clf = train_ensemble_classifier(data)
    res1 = evaluate(ensemble_clf, transformer_reg, data, device, 'ensemble')
    results['Ensemble+Transformer'] = res1

    # 方案2: DNN + Transformer
    dnn_clf = train_dnn_classifier(data, device, epochs=300)
    res2 = evaluate(dnn_clf, transformer_reg, data, device, 'dnn')
    results['DNN+Transformer'] = res2

    # 输出结果
    print("\n" + "=" * 75)
    print("优化实验结果 v2")
    print("=" * 75)
    print(f"{'模型组合':<25} {'准确率':>10} {'虚警率':>10} {'点火MAE':>12} {'推理时间':>12} {'达标'}")
    print("-" * 75)

    for name, res in results.items():
        acc_ok = res['accuracy'] >= 0.92
        far_ok = res['false_alarm_rate'] <= 0.03
        time_ok = res['inference_time'] <= 5.0
        status = "✓" if (acc_ok and far_ok and time_ok) else "✗"
        print(f"{name:<25} {res['accuracy']:>10.2%} {res['false_alarm_rate']:>10.2%} "
              f"{res['ignition_mae']:>10.2f}s {res['inference_time']:>10.4f}s {status}")

    print("\n目标: 准确率≥92%, 虚警率≤3%, 响应时间≤5s")

    # 保存
    df = pd.DataFrame(results).T
    df.to_csv("results/optimized_v2_results.csv")
    logger.info("结果已保存")

    return results


if __name__ == '__main__':
    main()
