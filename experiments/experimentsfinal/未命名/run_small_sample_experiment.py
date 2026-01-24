"""
小样本鲁棒模型实验
对比：基线方法 vs 创新方法
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from src.small_sample_model import (
    SmallSampleRobustModel, SmallSampleTrainer, extract_features_v2
)


def load_data():
    """加载数据"""
    df = pd.read_csv("data/feature_dataset.csv")
    df_valid = df[df['is_valid'] == 1].copy()

    features = extract_features_v2(df_valid)
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

    # 标准化
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"数据集统计:")
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {len(X_train)} (异常: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train, 't_train': t_train,
        'X_val': X_val, 'y_val': y_val, 't_val': t_val,
        'X_test': X_test, 'y_test': y_test, 't_test': t_test,
        'scaler': scaler
    }


def train_baseline_mlp(data, device, epochs=300):
    """基线MLP (无任何优化技术)"""
    import torch.nn as nn

    class BaselineMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.classifier(x), self.regressor(x)

    model = BaselineMLP(data['X_train'].shape[1]).to(device)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.FloatTensor(data['X_train']).to(device)
    y_train = torch.LongTensor(data['y_train']).to(device)
    t_train = torch.FloatTensor(data['t_train']).to(device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        cls_out, reg_out = model(X_train)
        loss = cls_criterion(cls_out, y_train) + 0.5 * reg_criterion(reg_out.squeeze(), t_train)
        loss.backward()
        optimizer.step()

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

    model.load_state_dict(best_state)
    return model


def train_baseline_with_dropout(data, device, epochs=300):
    """基线MLP + Dropout"""
    import torch.nn as nn

    class MLPWithDropout(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
            self.regressor = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.classifier(x), self.regressor(x)

    model = MLPWithDropout(data['X_train'].shape[1]).to(device)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_train = torch.FloatTensor(data['X_train']).to(device)
    y_train = torch.LongTensor(data['y_train']).to(device)
    t_train = torch.FloatTensor(data['t_train']).to(device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        cls_out, reg_out = model(X_train)
        loss = cls_criterion(cls_out, y_train) + 0.5 * reg_criterion(reg_out.squeeze(), t_train)
        loss.backward()
        optimizer.step()

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

    model.load_state_dict(best_state)
    return model


def evaluate_baseline(model, data, device):
    """评估基线模型"""
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
    mae = mean_absolute_error(t_test, t_pred)

    # 虚警率
    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    return {'accuracy': acc, 'f1': f1, 'mae': mae, 'far': far}


def evaluate_robust_model(trainer, data):
    """评估鲁棒模型 (带不确定性)"""
    result = trainer.predict(data['X_test'], with_uncertainty=True)

    y_pred = result['cls_pred'].cpu().numpy()
    t_pred = result['reg_pred'].cpu().numpy()
    cls_uncertainty = result['cls_uncertainty'].cpu().numpy()
    reg_uncertainty = result['reg_uncertainty'].cpu().numpy()

    y_test = data['y_test']
    t_test = data['t_test']

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(t_test, t_pred)

    # 虚警率
    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    # 不确定性统计
    mean_cls_unc = cls_uncertainty.mean()
    mean_reg_unc = reg_uncertainty.mean()

    return {
        'accuracy': acc, 'f1': f1, 'mae': mae, 'far': far,
        'cls_uncertainty': mean_cls_unc, 'reg_uncertainty': mean_reg_unc
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print("=" * 80)

    data = load_data()
    print("=" * 80)

    results = {}

    # 1. 基线MLP
    print("\n[1/4] 训练基线MLP...")
    baseline_mlp = train_baseline_mlp(data, device, epochs=300)
    results['Baseline MLP'] = evaluate_baseline(baseline_mlp, data, device)

    # 2. MLP + Dropout
    print("\n[2/4] 训练MLP + Dropout...")
    mlp_dropout = train_baseline_with_dropout(data, device, epochs=300)
    results['MLP + Dropout'] = evaluate_baseline(mlp_dropout, data, device)

    # 3. 小样本鲁棒模型 (无预训练)
    print("\n[3/4] 训练小样本鲁棒模型 (无预训练)...")
    model_no_pretrain = SmallSampleRobustModel(
        input_dim=data['X_train'].shape[1],
        hidden_dim=128, num_heads=4, mc_samples=10
    )
    trainer_no_pretrain = SmallSampleTrainer(model_no_pretrain, device, lr=1e-3)
    trainer_no_pretrain.train(
        data['X_train'], data['y_train'], data['t_train'],
        data['X_val'], data['y_val'], data['t_val'],
        epochs=300, cls_weight=1.0, reg_weight=0.5
    )
    results['Ours (w/o pretrain)'] = evaluate_robust_model(trainer_no_pretrain, data)

    # 4. 小样本鲁棒模型 (完整版: 对比学习预训练)
    print("\n[4/4] 训练小样本鲁棒模型 (完整版)...")
    model_full = SmallSampleRobustModel(
        input_dim=data['X_train'].shape[1],
        hidden_dim=128, num_heads=4, mc_samples=10
    )
    trainer_full = SmallSampleTrainer(model_full, device, lr=1e-3)

    # 对比学习预训练
    trainer_full.pretrain_contrastive(data['X_train'], epochs=100)

    # 多任务微调
    trainer_full.train(
        data['X_train'], data['y_train'], data['t_train'],
        data['X_val'], data['y_val'], data['t_val'],
        epochs=300, cls_weight=1.0, reg_weight=0.5
    )
    results['Ours (Full)'] = evaluate_robust_model(trainer_full, data)

    # 输出结果
    print("\n" + "=" * 100)
    print("实验结果对比")
    print("=" * 100)
    print(f"{'方法':<25} {'准确率':>10} {'F1':>10} {'虚警率':>10} {'点火MAE':>12} {'分类不确定性':>12} {'回归不确定性':>12}")
    print("-" * 100)

    for name, res in results.items():
        cls_unc = res.get('cls_uncertainty', '-')
        reg_unc = res.get('reg_uncertainty', '-')
        cls_unc_str = f"{cls_unc:.4f}" if isinstance(cls_unc, float) else cls_unc
        reg_unc_str = f"{reg_unc:.2f}" if isinstance(reg_unc, float) else reg_unc

        print(f"{name:<25} {res['accuracy']:>10.2%} {res['f1']:>10.4f} {res['far']:>10.2%} "
              f"{res['mae']:>10.2f}s {cls_unc_str:>12} {reg_unc_str:>12}")

    print("=" * 100)

    # 消融实验
    print("\n" + "=" * 100)
    print("消融实验分析")
    print("=" * 100)

    baseline_acc = results['Baseline MLP']['accuracy']
    dropout_acc = results['MLP + Dropout']['accuracy']
    no_pretrain_acc = results['Ours (w/o pretrain)']['accuracy']
    full_acc = results['Ours (Full)']['accuracy']

    print(f"Dropout贡献: {(dropout_acc - baseline_acc)*100:+.2f}%")
    print(f"注意力+Mixup+标签平滑贡献: {(no_pretrain_acc - dropout_acc)*100:+.2f}%")
    print(f"对比学习预训练贡献: {(full_acc - no_pretrain_acc)*100:+.2f}%")
    print(f"总提升: {(full_acc - baseline_acc)*100:+.2f}%")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv("results/small_sample_experiment.csv")
    print("\n结果已保存到 results/small_sample_experiment.csv")

    return results


if __name__ == '__main__':
    main()
