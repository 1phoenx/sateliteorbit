"""
模型组合对比实验
比较不同变轨检测模型和点火时刻估计模型的组合效果
- 变轨检测: RandomForest (RF), DNN
- 点火时刻估计: LSTM, Transformer
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.models import DNNClassifier, LSTMRegressor, TransformerRegressor, ModelCombination
from src.hmse_processor import apply_hmse_preprocessing
from sklearn.ensemble import RandomForestClassifier


def load_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载并预处理数据"""
    df = pd.read_csv(f"{data_dir}/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = df_valid[['P', 'T', 'R']].values
    labels = df_valid['is_anomalous'].values.astype(int)
    ignition_time = df_valid['ignition_time'].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0)
    ignition_time = np.nan_to_num(ignition_time, nan=0.0)

    # HMSE预处理
    logger.info("应用HMSE预处理...")
    features, _ = apply_hmse_preprocessing(features, scales=[1, 2, 4])

    return features, labels, ignition_time


def split_data(features, labels, ignition_time, seed=42):
    """划分数据集"""
    X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
        features, labels, ignition_time, test_size=0.2, random_state=seed, stratify=labels
    )
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.125, random_state=seed, stratify=y_temp
    )
    return {
        'X_train': X_train, 'y_train': y_train, 't_train': t_train,
        'X_val': X_val, 'y_val': y_val, 't_val': t_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test
    }


def evaluate_combination(clf_type: str, reg_type: str, data: Dict, device: str) -> Dict:
    """评估一种模型组合"""
    logger.info(f"评估组合: {clf_type} + {reg_type}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_train'])
    X_val = scaler.transform(data['X_val'])
    X_test = scaler.transform(data['X_test'])

    # 训练分类器
    if clf_type == 'RF':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, data['y_train'])
        y_pred = classifier.predict(X_test)
    else:  # DNN
        classifier = DNNClassifier(input_dim=3).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.LongTensor(data['y_train']).to(device)

        for epoch in range(100):
            classifier.train()
            optimizer.zero_grad()
            out = classifier(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            y_pred = classifier(X_test_t).argmax(dim=1).cpu().numpy()

    # 训练回归器
    if reg_type == 'LSTM':
        regressor = LSTMRegressor(input_dim=3).to(device)
    else:  # Transformer
        regressor = TransformerRegressor(input_dim=3).to(device)

    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    t_t = torch.FloatTensor(data['t_train']).to(device)

    for epoch in range(100):
        regressor.train()
        optimizer.zero_grad()
        out = regressor(X_t).squeeze()
        loss = criterion(out, t_t)
        loss.backward()
        optimizer.step()

    # 预测点火时刻
    regressor.eval()
    start_time = time.time()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        t_pred = regressor(X_test_t).squeeze().cpu().numpy()
    inference_time = (time.time() - start_time) / len(X_test)

    # 计算指标
    accuracy = accuracy_score(data['y_test'], y_pred)

    normal_mask = data['y_test'] == 0
    if normal_mask.sum() > 0:
        false_alarm = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum()
    else:
        false_alarm = 0.0

    ignition_mae = np.mean(np.abs(data['t_test'] - t_pred))

    return {
        'accuracy': accuracy,
        'false_alarm_rate': false_alarm,
        'ignition_mae': ignition_mae,
        'inference_time': inference_time
    }


def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    # 加载数据
    features, labels, ignition_time = load_data()
    data = split_data(features, labels, ignition_time)
    logger.info(f"数据划分: 训练{len(data['X_train'])}, 验证{len(data['X_val'])}, 测试{len(data['X_test'])}")

    # 4种组合
    combinations = [
        ('RF', 'LSTM'),
        ('RF', 'Transformer'),
        ('DNN', 'LSTM'),
        ('DNN', 'Transformer')
    ]

    results = {}
    for clf_type, reg_type in combinations:
        name = f"{clf_type}+{reg_type}"
        results[name] = evaluate_combination(clf_type, reg_type, data, device)

    # 输出结果
    df = pd.DataFrame(results).T
    df.index.name = 'Model'

    print("\n" + "=" * 70)
    print("模型组合对比实验结果")
    print("=" * 70)
    print(df.to_string())

    # 保存结果
    df.to_csv("results/model_comparison.csv")
    logger.info("结果已保存至 results/model_comparison.csv")

    # 找出最佳组合
    best_acc = df['accuracy'].idxmax()
    best_ign = df['ignition_mae'].idxmin()
    print(f"\n最佳准确率: {best_acc} ({df.loc[best_acc, 'accuracy']:.4f})")
    print(f"最佳点火时刻估计: {best_ign} (MAE: {df.loc[best_ign, 'ignition_mae']:.4f})")

    return df


if __name__ == '__main__':
    main()
