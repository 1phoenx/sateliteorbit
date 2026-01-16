"""
优化实验 - 目标: 准确率≥92%, 虚警率≤3%, 响应时间≤5s
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.optimized_models import OptimizedDNN, OptimizedTransformer, FocalLoss
from src.hmse_processor import apply_hmse_preprocessing


def augment_data(X, y, t, factor=3):
    """数据增强 - 对少数类进行过采样和噪声增强"""
    X_aug, y_aug, t_aug = [X], [y], [t]

    # 对异常类(少数类)进行增强
    anomaly_mask = y == 1
    X_anomaly = X[anomaly_mask]
    y_anomaly = y[anomaly_mask]
    t_anomaly = t[anomaly_mask]

    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.05, X_anomaly.shape)
        X_aug.append(X_anomaly + noise * X_anomaly)
        y_aug.append(y_anomaly)
        t_aug.append(t_anomaly + np.random.normal(0, 0.1, t_anomaly.shape))

    return np.vstack(X_aug), np.concatenate(y_aug), np.concatenate(t_aug)


def load_and_prepare_data():
    """加载并准备数据"""
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = df_valid[['P', 'T', 'R']].values
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0)
    ignition_time = np.nan_to_num(ignition_time, nan=0.0)

    # HMSE预处理
    logger.info("应用HMSE预处理...")
    features, _ = apply_hmse_preprocessing(features, scales=[1, 2, 4])

    # 划分数据
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=42, stratify=y_temp
    )

    # 数据增强
    logger.info("应用数据增强...")
    X_train, y_train, t_train = augment_data(X_train, y_train, t_train, factor=5)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"训练集: {len(X_train)} (增强后), 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train, 't_train': t_train,
        'X_val': X_val, 'y_val': y_val, 't_val': t_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test,
        'scaler': scaler
    }


def train_optimized_classifier(data, device, epochs=200):
    """训练优化的DNN分类器"""
    logger.info("训练优化DNN分类器...")

    model = OptimizedDNN(input_dim=3).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Focal Loss处理不平衡
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(data['X_train']),
            torch.LongTensor(data['y_train'])
        ),
        batch_size=64, shuffle=True, drop_last=True
    )

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(data['X_val']).to(device)
            y_val = data['y_val']
            val_pred = model(X_val).argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1}: val_acc={val_acc:.4f}")

    model.load_state_dict(best_model_state)
    return model


def train_optimized_regressor(data, device, epochs=200):
    """训练优化的Transformer回归器"""
    logger.info("训练优化Transformer回归器...")

    model = OptimizedTransformer(input_dim=3).to(device)
    criterion = nn.SmoothL1Loss()  # Huber Loss更鲁棒
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(data['X_train']),
            torch.FloatTensor(data['t_train'])
        ),
        batch_size=64, shuffle=True, drop_last=True
    )

    best_val_loss = float('inf')
    best_model_state = None

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

        # 验证
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(data['X_val']).to(device)
            t_val = torch.FloatTensor(data['t_val']).to(device)
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, t_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    return model


def train_rf_classifier(data):
    """训练随机森林分类器"""
    logger.info("训练随机森林分类器...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(data['X_train'], data['y_train'])
    return clf


def train_gb_classifier(data):
    """训练梯度提升分类器"""
    logger.info("训练梯度提升分类器...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    clf.fit(data['X_train'], data['y_train'])
    return clf


def evaluate_model(clf, reg, data, device, clf_type='DNN'):
    """评估模型组合"""
    X_test = data['X_test']
    y_test = data['y_test']
    t_test = data['t_test']

    # 分类预测
    start_time = time.time()
    if clf_type in ['RF', 'GB']:
        y_pred = clf.predict(X_test)
    else:
        clf.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            y_pred = clf(X_t).argmax(dim=1).cpu().numpy()

    # 回归预测
    reg.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        t_pred = reg(X_t).squeeze().cpu().numpy()

    inference_time = time.time() - start_time

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)

    normal_mask = y_test == 0
    false_alarm = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    ignition_mae = np.mean(np.abs(t_test - t_pred))

    return {
        'accuracy': accuracy,
        'false_alarm_rate': false_alarm,
        'ignition_mae': ignition_mae,
        'inference_time': inference_time,
        'y_pred': y_pred,
        'y_test': y_test
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device}")

    # 准备数据
    data = load_and_prepare_data()

    # 训练回归器 (共用)
    transformer_reg = train_optimized_regressor(data, device, epochs=200)

    results = {}

    # 方案1: RF + Transformer
    rf_clf = train_rf_classifier(data)
    res1 = evaluate_model(rf_clf, transformer_reg, data, device, 'RF')
    results['RF+Transformer'] = res1
    logger.info(f"RF+Transformer: 准确率={res1['accuracy']:.4f}, 虚警率={res1['false_alarm_rate']:.4f}")

    # 方案2: GB + Transformer
    gb_clf = train_gb_classifier(data)
    res2 = evaluate_model(gb_clf, transformer_reg, data, device, 'GB')
    results['GB+Transformer'] = res2
    logger.info(f"GB+Transformer: 准确率={res2['accuracy']:.4f}, 虚警率={res2['false_alarm_rate']:.4f}")

    # 方案3: DNN + Transformer
    dnn_clf = train_optimized_classifier(data, device, epochs=200)
    res3 = evaluate_model(dnn_clf, transformer_reg, data, device, 'DNN')
    results['DNN+Transformer'] = res3
    logger.info(f"DNN+Transformer: 准确率={res3['accuracy']:.4f}, 虚警率={res3['false_alarm_rate']:.4f}")

    # 输出结果
    print("\n" + "=" * 70)
    print("优化实验结果")
    print("=" * 70)
    print(f"{'模型组合':<20} {'准确率':>10} {'虚警率':>10} {'点火MAE':>12} {'推理时间':>12}")
    print("-" * 70)
    for name, res in results.items():
        status = "✓" if res['accuracy'] >= 0.92 and res['false_alarm_rate'] <= 0.03 else "✗"
        print(f"{name:<20} {res['accuracy']:>10.2%} {res['false_alarm_rate']:>10.2%} "
              f"{res['ignition_mae']:>12.2f}s {res['inference_time']:>10.4f}s {status}")

    print("\n目标: 准确率≥92%, 虚警率≤3%, 响应时间≤5s")

    # 找最佳模型
    best = max(results.items(), key=lambda x: x[1]['accuracy'] - x[1]['false_alarm_rate'])
    print(f"\n推荐模型: {best[0]}")

    # 保存结果
    df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk not in ['y_pred', 'y_test']}
                       for k, v in results.items()}).T
    df.to_csv("results/optimized_results.csv")

    return results


if __name__ == '__main__':
    main()
