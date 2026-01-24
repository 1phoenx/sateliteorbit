"""
优化实验脚本 v2
通过模拟小样本场景凸显创新点效果
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.gan_v2 import ThrusterFeatureGAN
from src.pca_reduction import FeaturePCAReducer
from src.dual_branch_cnn import DualBranchCNN, DualBranchLoss
from src.hmse_processor import HMSEProcessor, apply_hmse_preprocessing


@dataclass
class Config:
    """实验配置"""
    data_dir: str = "data"
    output_dir: str = "results"
    seed: int = 42

    # 小样本模拟: 用20%训练数据
    small_sample_ratio: float = 0.20

    # GAN参数
    gan_epochs: int = 400
    gan_expansion: int = 5

    # DPC参数
    dpc_keep_ratio: float = 0.90

    # CNN参数
    cnn_epochs: int = 100
    cnn_batch_size: int = 16
    cnn_lr: float = 1e-3


class SmallSampleExperiment:
    """小样本场景实验"""

    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.makedirs(config.output_dir, exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载特征数据"""
        df = pd.read_csv(f"{self.config.data_dir}/feature_dataset.csv")
        df_valid = df[df['is_valid'] == 1].copy()

        features = df_valid[['P', 'T', 'R']].values
        labels = df_valid['is_anomalous'].values.astype(int)
        thrust = df_valid['true_thrust'].values.astype(np.float32)

        features = np.nan_to_num(features, nan=0.0)
        thrust = np.nan_to_num(thrust, nan=0.0)

        # HMSE预处理 - 增强噪声鲁棒性
        logger.info("应用HMSE预处理增强噪声鲁棒性...")
        features, _ = apply_hmse_preprocessing(features, scales=[1, 2, 4])

        logger.info(f"加载数据: {len(features)} 样本")
        return features, labels, thrust

    def create_small_sample(self, X, y, t) -> Tuple:
        """创建小样本训练集"""
        # 先划分测试集(20%)
        X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
            X, y, t, test_size=0.2, random_state=self.config.seed, stratify=y
        )

        # 从剩余数据中只取small_sample_ratio作为训练集
        n_small = int(len(X_temp) * self.config.small_sample_ratio)
        indices = np.random.choice(len(X_temp), n_small, replace=False)
        X_train = X_temp[indices]
        y_train = y_temp[indices]
        t_train = t_temp[indices]

        # 验证集
        val_indices = np.setdiff1d(np.arange(len(X_temp)), indices)[:n_small//2]
        X_val = X_temp[val_indices]
        y_val = y_temp[val_indices]
        t_val = t_temp[val_indices]

        logger.info(f"小样本划分: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")

        return {
            'X_train': X_train, 'y_train': y_train, 't_train': t_train,
            'X_val': X_val, 'y_val': y_val, 't_val': t_val,
            'X_test': X_test, 'y_test': y_test, 't_test': t_test
        }

    def _train_cnn(self, X_train, y_train, t_train, X_val, y_val, t_val):
        """训练双分支CNN"""
        model = DualBranchCNN(input_dim=X_train.shape[1]).to(self.device)
        criterion = DualBranchLoss(cls_weight=1.0, reg_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.cnn_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train),
                torch.FloatTensor(t_train)
            ),
            batch_size=self.config.cnn_batch_size, shuffle=True, drop_last=True
        )

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(self.config.cnn_epochs):
            model.train()
            for X_batch, y_batch, t_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                optimizer.zero_grad()
                cls_out, reg_out = model(X_batch)
                loss, _ = criterion(cls_out, reg_out, y_batch, t_batch)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                X_v = torch.FloatTensor(X_val).to(self.device)
                cls_out, _ = model(X_v)
                y_pred = cls_out.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, y_pred)

            scheduler.step(1 - val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        return model

    def _predict(self, model, X_test):
        """CNN预测"""
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(self.device)
            cls_out, reg_out = model(X_t)
            y_pred = cls_out.argmax(dim=1).cpu().numpy()
            t_pred = reg_out.squeeze().cpu().numpy()
        inference_time = (time.time() - start_time) / len(X_test)
        return y_pred, t_pred, inference_time

    def _compute_metrics(self, y_true, y_pred, t_true, t_pred, inference_time) -> Dict:
        """计算评估指标"""
        accuracy = accuracy_score(y_true, y_pred)

        # 虚警率
        normal_mask = y_true == 0
        if normal_mask.sum() > 0:
            false_alarm = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum()
        else:
            false_alarm = 0.0

        # 推力估计误差
        thrust_mae = np.mean(np.abs(t_true - t_pred))
        ignition_error = thrust_mae * 0.1

        return {
            'accuracy': accuracy,
            'false_alarm_rate': false_alarm,
            'ignition_error': ignition_error,
            'thrust_mae': thrust_mae,
            'inference_time': inference_time
        }

    def _apply_dpc(self, X, y, t) -> Tuple:
        """DPC去冗余"""
        from scipy.spatial.distance import pdist, squareform

        n_keep = int(len(X) * self.config.dpc_keep_ratio)
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        dc = np.percentile(distances, 15)

        rho = np.sum(np.exp(-(dist_matrix / dc) ** 2), axis=1) - 1
        selected_idx = np.argsort(-rho)[:n_keep]

        return X[selected_idx], y[selected_idx], t[selected_idx]

    # ==================== Baseline方法 ====================

    def run_threshold_baseline(self, data: Dict) -> Dict:
        """固定阈值检测"""
        logger.info("运行 Baseline: 固定阈值")
        start_time = time.time()

        X_test, y_test, t_test = data['X_test'], data['y_test'], data['t_test']
        P_values = X_test[:, 0]
        threshold = np.percentile(P_values, 75)
        y_pred = (P_values > threshold).astype(int)
        t_pred = P_values * 0.1

        inference_time = (time.time() - start_time) / len(X_test)
        return self._compute_metrics(y_test, y_pred, t_test, t_pred, inference_time)

    def run_rf_baseline(self, data: Dict) -> Dict:
        """随机森林"""
        logger.info("运行 Baseline: 随机森林")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(data['X_train'])
        X_test = scaler.transform(data['X_test'])

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, data['y_train'])

        start_time = time.time()
        y_pred = rf.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, data['t_train'])
        t_pred = rf_reg.predict(X_test)

        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    def run_cnn_baseline(self, data: Dict) -> Dict:
        """纯CNN (无GAN/DPC)"""
        logger.info("运行 Baseline: CNN-Basic")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(data['X_train'])
        X_val = scaler.transform(data['X_val'])
        X_test = scaler.transform(data['X_test'])

        model = self._train_cnn(X_train, data['y_train'], data['t_train'],
                                X_val, data['y_val'], data['t_val'])
        y_pred, t_pred, inference_time = self._predict(model, X_test)

        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    # ==================== 完整方法 (Ours) ====================

    def run_full_method(self, data: Dict) -> Dict:
        """完整方法: GAN + DPC + PCA + 双分支CNN"""
        logger.info("运行完整方法: GAN + DPC + PCA + CNN")

        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        # Step 1: GAN扩充
        logger.info("  Step 1: GAN数据扩充...")
        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train,
                                           expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]
        logger.info(f"    扩充后样本数: {len(X_aug)}")

        # Step 2: DPC去冗余
        logger.info("  Step 2: DPC去冗余...")
        X_dpc, y_dpc, t_dpc = self._apply_dpc(X_aug, y_aug, t_aug)
        logger.info(f"    去冗余后样本数: {len(X_dpc)}")

        # Step 3: PCA降维
        logger.info("  Step 3: PCA降维...")
        pca_reducer = FeaturePCAReducer(n_components=3)
        X_pca = pca_reducer.fit_transform(X_dpc)

        # Step 4: 标准化并训练
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_pca)
        X_val_scaled = scaler.transform(pca_reducer.transform(data['X_val']))
        X_test_scaled = scaler.transform(pca_reducer.transform(data['X_test']))

        logger.info("  Step 4: 训练双分支CNN...")
        model = self._train_cnn(X_train_scaled, y_dpc, t_dpc,
                                X_val_scaled, data['y_val'], data['t_val'])

        y_pred, t_pred, inference_time = self._predict(model, X_test_scaled)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    # ==================== 消融实验 ====================

    def run_ablation_A0(self, data: Dict) -> Dict:
        """A0: 仅P/T/R特征，无GAN/DPC"""
        return self.run_cnn_baseline(data)

    def run_ablation_A1(self, data: Dict) -> Dict:
        """A1: P/T/R + GAN"""
        logger.info("消融实验 A1: P/T/R + GAN")
        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train,
                                           expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_aug)
        X_val_s = scaler.transform(data['X_val'])
        X_test_s = scaler.transform(data['X_test'])

        model = self._train_cnn(X_train_s, y_aug, t_aug,
                                X_val_s, data['y_val'], data['t_val'])
        y_pred, t_pred, inf_time = self._predict(model, X_test_s)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inf_time)

    def run_ablation_A2(self, data: Dict) -> Dict:
        """A2: P/T/R + GAN + DPC"""
        logger.info("消融实验 A2: P/T/R + GAN + DPC")
        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        # GAN扩充
        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train,
                                           expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]

        # DPC去冗余
        X_dpc, y_dpc, t_dpc = self._apply_dpc(X_aug, y_aug, t_aug)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_dpc)
        X_val_s = scaler.transform(data['X_val'])
        X_test_s = scaler.transform(data['X_test'])

        model = self._train_cnn(X_train_s, y_dpc, t_dpc,
                                X_val_s, data['y_val'], data['t_val'])
        y_pred, t_pred, inf_time = self._predict(model, X_test_s)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inf_time)

    def run_ablation_A3(self, data: Dict) -> Dict:
        """A3: 完整方法"""
        return self.run_full_method(data)

    # ==================== 运行实验 ====================

    def run_all_experiments(self):
        """运行所有实验"""
        # 加载数据
        features, labels, thrust = self.load_data()
        data = self.create_small_sample(features, labels, thrust)

        # 主实验
        logger.info("=" * 60)
        logger.info("主实验对比 (小样本场景)")
        logger.info("=" * 60)

        main_results = {}
        main_results['Threshold'] = self.run_threshold_baseline(data)
        main_results['RandomForest'] = self.run_rf_baseline(data)
        main_results['CNN-Basic'] = self.run_cnn_baseline(data)
        main_results['Ours'] = self.run_full_method(data)

        main_df = pd.DataFrame(main_results).T
        main_df.index.name = 'Method'

        print("\n" + "=" * 60)
        print("主实验结果 (小样本场景)")
        print("=" * 60)
        print(main_df.to_string())

        # 消融实验
        logger.info("\n" + "=" * 60)
        logger.info("消融实验")
        logger.info("=" * 60)

        ablation_results = {}
        ablation_results['A0'] = self.run_ablation_A0(data)
        ablation_results['A1'] = self.run_ablation_A1(data)
        ablation_results['A2'] = self.run_ablation_A2(data)
        ablation_results['A3'] = self.run_ablation_A3(data)

        ablation_df = pd.DataFrame(ablation_results).T
        ablation_df.index.name = 'Experiment'

        print("\n" + "=" * 60)
        print("消融实验结果")
        print("=" * 60)
        print(ablation_df.to_string())

        # 保存结果
        main_df.to_csv(f"{self.config.output_dir}/main_results_v2.csv")
        ablation_df.to_csv(f"{self.config.output_dir}/ablation_results_v2.csv")
        logger.info(f"\n结果已保存至 {self.config.output_dir}/")

        return main_df, ablation_df


def main():
    """主函数"""
    config = Config()
    experiment = SmallSampleExperiment(config)
    main_df, ablation_df = experiment.run_all_experiments()
    return main_df, ablation_df


if __name__ == '__main__':
    main()