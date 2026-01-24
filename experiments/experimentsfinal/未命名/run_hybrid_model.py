"""
混合深度学习模型 - 论文最终方案
核心创新：深度特征学习 + 传统分类器

创新点：
1. 自编码器预训练 - 无监督特征学习
2. 深度特征提取 + RF分类 - 结合两者优势
3. 多尺度特征融合 - 捕获不同粒度信息
4. 不确定性感知回归 - 提供预测置信度
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, mean_absolute_error)
from torch.utils.data import DataLoader, TensorDataset

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


# ==================== 深度学习组件 ====================

class MultiScaleEncoder(nn.Module):
    """多尺度特征编码器"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super().__init__()

        # 多尺度编码路径
        self.scale1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.scale2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.scale3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.GELU()
        )

        self.output_dim = sum(hidden_dims)

    def forward(self, x):
        z1 = self.scale1(x)
        z2 = self.scale2(z1)
        z3 = self.scale3(z2)

        # 多尺度特征融合
        return torch.cat([z1, z2, z3], dim=1)


class VariationalAutoEncoder(nn.Module):
    """变分自编码器 - 无监督特征学习"""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


class DeepFeatureExtractor(nn.Module):
    """深度特征提取器 - 结合多尺度编码和VAE"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()

        # 多尺度编码器
        self.multi_scale = MultiScaleEncoder(input_dim, [hidden_dim, hidden_dim//2, hidden_dim//4])

        # VAE
        self.vae = VariationalAutoEncoder(input_dim, latent_dim)

        # 特征融合
        total_dim = self.multi_scale.output_dim + latent_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_dim = hidden_dim

    def forward(self, x):
        # 多尺度特征
        ms_feat = self.multi_scale(x)

        # VAE潜在特征
        vae_feat = self.vae.get_latent(x)

        # 融合
        combined = torch.cat([ms_feat, vae_feat], dim=1)
        return self.fusion(combined)

    def get_vae_loss(self, x):
        recon, mu, log_var = self.vae(x)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + 0.1 * kl_loss


class UncertaintyRegressor(nn.Module):
    """不确定性感知回归器"""
    def __init__(self, input_dim):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.var_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.var_head(x)
        var = F.softplus(log_var) + 1e-6
        return mean, var


# ==================== 混合模型 ====================

class HybridDeepModel:
    """
    混合深度学习模型

    架构：
    1. 深度特征提取器 (预训练)
    2. RandomForest分类器 (变轨检测)
    3. 不确定性回归器 (点火时刻估计)

    创新点：
    - 无监督预训练学习数据分布
    - 深度特征 + 传统分类器结合
    - 不确定性量化
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, device='cpu'):
        self.device = device
        self.input_dim = input_dim

        # 深度特征提取器
        self.feature_extractor = DeepFeatureExtractor(
            input_dim, hidden_dim, latent_dim
        ).to(device)

        # 不确定性回归器
        self.regressor = UncertaintyRegressor(hidden_dim).to(device)

        # RandomForest分类器 (在深度特征上训练)
        self.classifier = None

        self.scaler = None

    def pretrain_vae(self, X_train, epochs=100, lr=1e-3):
        """VAE无监督预训练"""
        print("VAE无监督预训练...")

        optimizer = torch.optim.AdamW(
            self.feature_extractor.vae.parameters(), lr=lr
        )

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=32, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0
            for (x,) in loader:
                x = x.to(self.device)
                loss = self.feature_extractor.get_vae_loss(x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: VAE loss = {total_loss/len(loader):.4f}")

    def train_feature_extractor(self, X_train, y_train, t_train, X_val, y_val, t_val,
                                 epochs=200, lr=1e-3):
        """训练特征提取器和回归器"""
        print("训练深度特征提取器...")

        # 冻结VAE，只训练多尺度编码器和融合层
        for param in self.feature_extractor.vae.parameters():
            param.requires_grad = False

        params = list(self.feature_extractor.multi_scale.parameters()) + \
                 list(self.feature_extractor.fusion.parameters()) + \
                 list(self.regressor.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train),
                torch.FloatTensor(t_train)
            ),
            batch_size=32, shuffle=True
        )

        best_val_mae = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.feature_extractor.train()
            self.regressor.train()

            for x, y, t in train_loader:
                x, t = x.to(self.device), t.to(self.device)

                # 提取特征
                feat = self.feature_extractor(x)

                # 回归损失 (高斯NLL)
                mean, var = self.regressor(feat)
                reg_loss = 0.5 * (torch.log(var) + (t.unsqueeze(1) - mean) ** 2 / var).mean()

                optimizer.zero_grad()
                reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            scheduler.step()

            # 验证
            val_mae = self._evaluate_regression(X_val, t_val)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {
                    'feature_extractor': self.feature_extractor.state_dict(),
                    'regressor': self.regressor.state_dict()
                }

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: val_mae = {val_mae:.2f}s, best = {best_val_mae:.2f}s")

        # 恢复最佳状态
        self.feature_extractor.load_state_dict(best_state['feature_extractor'])
        self.regressor.load_state_dict(best_state['regressor'])

    def train_classifier(self, X_train, y_train):
        """在深度特征上训练RF分类器"""
        print("训练RandomForest分类器...")

        # 提取深度特征
        deep_features = self._extract_deep_features(X_train)

        # 训练RF
        self.classifier = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_split=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.classifier.fit(deep_features, y_train)

    def _extract_deep_features(self, X):
        """提取深度特征"""
        self.feature_extractor.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            features = self.feature_extractor(x_tensor).cpu().numpy()
        return features

    def _evaluate_regression(self, X, t):
        """评估回归性能"""
        self.feature_extractor.eval()
        self.regressor.eval()

        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            feat = self.feature_extractor(x_tensor)
            mean, _ = self.regressor(feat)
            pred = mean.squeeze().cpu().numpy()

        return mean_absolute_error(t, pred)

    def predict(self, X):
        """预测"""
        # 提取深度特征
        deep_features = self._extract_deep_features(X)

        # 分类
        y_pred = self.classifier.predict(deep_features)
        y_prob = self.classifier.predict_proba(deep_features)[:, 1]

        # 回归 (带不确定性)
        self.regressor.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            feat = self.feature_extractor(x_tensor)
            mean, var = self.regressor(feat)
            t_pred = mean.squeeze().cpu().numpy()
            t_uncertainty = var.squeeze().cpu().numpy()

        return {
            'y_pred': y_pred,
            'y_prob': y_prob,
            't_pred': t_pred,
            't_uncertainty': t_uncertainty
        }


# ==================== 实验 ====================

def run_ablation_study(data, device):
    """消融实验"""
    results = {}

    input_dim = data['X_train'].shape[1]

    # A0: 纯RF基线
    print("\n[A0] 纯RandomForest基线...")
    rf_clf = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_reg = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf_clf.fit(data['X_train'], data['y_train'])
    rf_reg.fit(data['X_train'], data['t_train'])

    y_pred = rf_clf.predict(data['X_test'])
    t_pred = rf_reg.predict(data['X_test'])
    results['A0: RF Baseline'] = evaluate(y_pred, t_pred, data)

    # A1: 深度特征 + RF (无预训练)
    print("\n[A1] 深度特征 + RF (无预训练)...")
    model_a1 = HybridDeepModel(input_dim, device=device)
    model_a1.train_feature_extractor(
        data['X_train'], data['y_train'], data['t_train'],
        data['X_val'], data['y_val'], data['t_val'],
        epochs=100
    )
    model_a1.train_classifier(data['X_train'], data['y_train'])
    pred_a1 = model_a1.predict(data['X_test'])
    results['A1: Deep+RF (no pretrain)'] = evaluate(pred_a1['y_pred'], pred_a1['t_pred'], data)

    # A2: VAE预训练 + 深度特征 + RF
    print("\n[A2] VAE预训练 + 深度特征 + RF...")
    model_a2 = HybridDeepModel(input_dim, device=device)
    model_a2.pretrain_vae(data['X_train'], epochs=100)
    model_a2.train_feature_extractor(
        data['X_train'], data['y_train'], data['t_train'],
        data['X_val'], data['y_val'], data['t_val'],
        epochs=100
    )
    model_a2.train_classifier(data['X_train'], data['y_train'])
    pred_a2 = model_a2.predict(data['X_test'])
    results['A2: VAE+Deep+RF'] = evaluate(pred_a2['y_pred'], pred_a2['t_pred'], data)

    return results, model_a2, pred_a2


def evaluate(y_pred, t_pred, data):
    """评估指标"""
    y_test = data['y_test']
    t_test = data['t_test']

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(t_test, t_pred)

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

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"异常样本比例: {y_train.mean()*100:.1f}%")
    print("=" * 100)

    # 消融实验
    results, best_model, best_pred = run_ablation_study(data, device)

    # 输出结果
    print("\n" + "=" * 110)
    print("消融实验结果")
    print("=" * 110)
    print(f"{'配置':<30} {'准确率':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'虚警率':>10} {'点火MAE':>12}")
    print("-" * 110)

    for name, res in results.items():
        print(f"{name:<30} {res['accuracy']:>10.2%} {res['f1']:>10.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>10.4f} "
              f"{res['far']:>10.2%} {res['mae']:>10.2f}s")

    print("=" * 110)

    # 不确定性分析
    print("\n不确定性分析:")
    uncertainty = best_pred['t_uncertainty']
    t_test = data['t_test']
    t_pred = best_pred['t_pred']

    # 按不确定性分组
    low_unc_mask = uncertainty < np.median(uncertainty)
    high_unc_mask = ~low_unc_mask

    low_mae = mean_absolute_error(t_test[low_unc_mask], t_pred[low_unc_mask])
    high_mae = mean_absolute_error(t_test[high_unc_mask], t_pred[high_unc_mask])

    print(f"  低不确定性样本 (前50%): MAE = {low_mae:.2f}s")
    print(f"  高不确定性样本 (后50%): MAE = {high_mae:.2f}s")
    print(f"  不确定性与误差相关性: {np.corrcoef(uncertainty, np.abs(t_test - t_pred))[0,1]:.4f}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results).T
    df_results.to_csv("results/hybrid_model_ablation.csv")
    print("\n结果已保存到 results/hybrid_model_ablation.csv")

    return results


if __name__ == '__main__':
    main()
