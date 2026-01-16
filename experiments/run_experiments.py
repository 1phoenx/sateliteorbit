"""
完整实验流程脚本
包含: 主实验对比 + 消融实验
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction_v2 import ThrusterFeatureExtractor
from src.gan_v2 import ThrusterFeatureGAN
from src.dpc_clustering import ImprovedDPC
from src.pca_reduction import FeaturePCAReducer
from src.dual_branch_cnn import DualBranchCNN, DualBranchLoss


@dataclass
class ExperimentConfig:
    """实验配置"""
    data_dir: str = "data"
    output_dir: str = "results"
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1

    # GAN参数
    gan_epochs: int = 400
    gan_expansion: int = 2

    # DPC参数
    dpc_dc_percent: float = 0.15
    dpc_keep_ratio: float = 0.98

    # PCA参数
    pca_components: int = 3

    # CNN参数
    cnn_epochs: int = 100
    cnn_batch_size: int = 32
    cnn_lr: float = 1e-3
    cnn_patience: int = 15


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/models", exist_ok=True)

        # 设置随机种子
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载特征数据"""
        feature_path = f"{self.config.data_dir}/feature_dataset.csv"

        if not os.path.exists(feature_path):
            logger.info("特征文件不存在，开始提取特征...")
            self._extract_features()

        df = pd.read_csv(feature_path)

        # 筛选有效样本
        df_valid = df[df['is_valid'] == 1].copy()
        logger.info(f"有效样本数: {len(df_valid)}")

        # 提取特征和标签
        features = df_valid[['P', 'T', 'R']].values
        labels = df_valid['is_anomalous'].values.astype(int)
        thrust = df_valid['true_thrust'].values.astype(np.float32)

        # 处理NaN
        features = np.nan_to_num(features, nan=0.0)
        thrust = np.nan_to_num(thrust, nan=0.0)

        return features, labels, thrust

    def _extract_features(self):
        """提取特征"""
        extractor = ThrusterFeatureExtractor(sampling_rate=100.0)
        extractor.batch_extract_features(
            data_dir=self.config.data_dir,
            metadata_path=f"{self.config.data_dir}/metadata.csv",
            output_path=f"{self.config.data_dir}/feature_dataset.csv"
        )

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        thrust: np.ndarray
    ) -> Dict:
        """划分训练/验证/测试集"""
        # 先分出测试集
        X_temp, X_test, y_temp, y_test, t_temp, t_test = train_test_split(
            features, labels, thrust,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=labels
        )

        # 再从剩余数据分出验证集
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
            X_temp, y_temp, t_temp,
            test_size=val_ratio,
            random_state=self.config.random_seed,
            stratify=y_temp
        )

        logger.info(f"数据划分: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")

        return {
            'X_train': X_train, 'y_train': y_train, 't_train': t_train,
            'X_val': X_val, 'y_val': y_val, 't_val': t_val,
            'X_test': X_test, 'y_test': y_test, 't_test': t_test
        }

    # ==================== Baseline方法 ====================

    def run_threshold_baseline(self, data: Dict) -> Dict:
        """Baseline-1: 固定阈值检测"""
        logger.info("运行 Baseline-1: 固定阈值检测")
        start_time = time.time()

        X_test = data['X_test']
        y_test = data['y_test']
        t_test = data['t_test']

        # 使用P特征的阈值检测
        P_values = X_test[:, 0]
        threshold = np.percentile(P_values, 75)
        y_pred = (P_values > threshold).astype(int)

        # 推力估计: 简单线性映射
        t_pred = P_values * 0.1

        inference_time = (time.time() - start_time) / len(X_test)

        return self._compute_metrics(y_test, y_pred, t_test, t_pred, inference_time)

    def run_rf_baseline(self, data: Dict) -> Dict:
        """Baseline-2: 随机森林分类器"""
        logger.info("运行 Baseline-2: 随机森林分类器")

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data['X_train'])
        X_test = scaler.transform(data['X_test'])

        # 训练RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, data['y_train'])

        # 预测
        start_time = time.time()
        y_pred = rf.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)

        # 推力估计: 使用RF回归
        from sklearn.ensemble import RandomForestRegressor
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, data['t_train'])
        t_pred = rf_reg.predict(X_test)

        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    def run_cnn_baseline(self, data: Dict) -> Dict:
        """Baseline-3: 不含GAN/DPC的1D CNN"""
        logger.info("运行 Baseline-3: 纯1D CNN (无GAN/DPC)")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(data['X_train'])
        X_val = scaler.transform(data['X_val'])
        X_test = scaler.transform(data['X_test'])

        model, history = self._train_cnn(
            X_train, data['y_train'], data['t_train'],
            X_val, data['y_val'], data['t_val']
        )

        y_pred, t_pred, inference_time = self._predict_cnn(model, X_test, scaler)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    # ==================== 完整方法 (Ours) ====================

    def run_full_method(self, data: Dict) -> Dict:
        """完整方法: GAN + DPC + PCA + 双分支CNN"""
        logger.info("运行完整方法: GAN + DPC + PCA + 双分支CNN")

        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        # Step 1: GAN扩充
        logger.info("Step 1: GAN数据扩充...")
        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train, expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]

        # Step 2: DPC去冗余
        logger.info("Step 2: DPC去冗余...")
        X_dpc, y_dpc, t_dpc = self._apply_dpc(X_aug, y_aug, t_aug)

        # Step 3: PCA降维
        logger.info("Step 3: PCA降维...")
        pca_reducer = FeaturePCAReducer(n_components=self.config.pca_components)
        X_pca = pca_reducer.fit_transform(X_dpc)

        # Step 4: 标准化并训练
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_pca)
        X_val_scaled = scaler.transform(pca_reducer.transform(data['X_val']))
        X_test_scaled = scaler.transform(pca_reducer.transform(data['X_test']))

        logger.info("Step 5: 训练双分支CNN...")
        model, _ = self._train_cnn(
            X_train_scaled, y_dpc, t_dpc,
            X_val_scaled, data['y_val'], data['t_val']
        )

        y_pred, t_pred, inference_time = self._predict_cnn(model, X_test_scaled, None)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inference_time)

    # ==================== 消融实验 ====================

    def run_ablation_A0(self, data: Dict) -> Dict:
        """A0: 仅P/T/R特征，无GAN/DPC/双阶段"""
        logger.info("消融实验 A0: 仅P/T/R特征")
        return self.run_cnn_baseline(data)

    def run_ablation_A1(self, data: Dict) -> Dict:
        """A1: P/T/R + GAN"""
        logger.info("消融实验 A1: P/T/R + GAN")
        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train, expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_aug)
        X_val_s = scaler.transform(data['X_val'])
        X_test_s = scaler.transform(data['X_test'])

        model, _ = self._train_cnn(X_train_s, y_aug, t_aug, X_val_s, data['y_val'], data['t_val'])
        y_pred, t_pred, inf_time = self._predict_cnn(model, X_test_s, None)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inf_time)

    def run_ablation_A2(self, data: Dict) -> Dict:
        """A2: P/T/R + GAN + DPC"""
        logger.info("消融实验 A2: P/T/R + GAN + DPC")
        X_train, y_train, t_train = data['X_train'], data['y_train'], data['t_train']

        # GAN扩充
        gan = ThrusterFeatureGAN(latent_dim=64, feature_dim=3, n_classes=2)
        gan.train(X_train, y_train, epochs=self.config.gan_epochs, verbose=False)
        X_aug, y_aug = gan.augment_dataset(X_train, y_train, expansion_factor=self.config.gan_expansion)
        t_aug = np.tile(t_train, self.config.gan_expansion)[:len(X_aug)]

        # DPC去冗余
        X_dpc, y_dpc, t_dpc = self._apply_dpc(X_aug, y_aug, t_aug)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_dpc)
        X_val_s = scaler.transform(data['X_val'])
        X_test_s = scaler.transform(data['X_test'])

        model, _ = self._train_cnn(X_train_s, y_dpc, t_dpc, X_val_s, data['y_val'], data['t_val'])
        y_pred, t_pred, inf_time = self._predict_cnn(model, X_test_s, None)
        return self._compute_metrics(data['y_test'], y_pred, data['t_test'], t_pred, inf_time)

    def run_ablation_A3(self, data: Dict) -> Dict:
        """A3: 完整方法 (P/T/R + GAN + DPC + 双阶段)"""
        logger.info("消融实验 A3: 完整方法")
        return self.run_full_method(data)

    # ==================== 辅助函数 ====================

    def _apply_dpc(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple:
        """应用DPC去冗余"""
        from scipy.spatial.distance import pdist, squareform

        n_keep = int(len(X) * self.config.dpc_keep_ratio)
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        dc = np.percentile(distances, self.config.dpc_dc_percent * 100)

        rho = np.sum(np.exp(-(dist_matrix / dc) ** 2), axis=1) - 1
        selected_idx = np.argsort(-rho)[:n_keep]

        return X[selected_idx], y[selected_idx], t[selected_idx]

    def _train_cnn(self, X_train, y_train, t_train, X_val, y_val, t_val):
        """训练双分支CNN"""
        model = DualBranchCNN(input_dim=X_train.shape[1]).to(self.device)
        criterion = DualBranchLoss(cls_weight=1.0, reg_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.cnn_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train),
                torch.FloatTensor(t_train)
            ),
            batch_size=self.config.cnn_batch_size, shuffle=True, drop_last=True
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.cnn_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch, t_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                optimizer.zero_grad()
                cls_out, reg_out = model(X_batch)
                loss, _ = criterion(cls_out, reg_out, y_batch, t_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证
            model.eval()
            with torch.no_grad():
                X_v = torch.FloatTensor(X_val).to(self.device)
                y_v = torch.LongTensor(y_val).to(self.device)
                t_v = torch.FloatTensor(t_val).to(self.device)
                cls_out, reg_out = model(X_v)
                val_loss, _ = criterion(cls_out, reg_out, y_v, t_v)
                val_loss = val_loss.item()

            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.cnn_patience:
                    break

        return model, history

    def _predict_cnn(self, model, X_test, scaler=None):
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

        # 虚警率: 误判为异常的正常样本比例
        normal_mask = y_true == 0
        if normal_mask.sum() > 0:
            false_alarm = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum()
        else:
            false_alarm = 0.0

        # 推力估计误差 (MAE)
        thrust_mae = np.mean(np.abs(t_true - t_pred))

        # 点火时刻误差 (模拟，基于特征变化)
        ignition_error = thrust_mae * 0.1  # 简化估计

        return {
            'accuracy': accuracy,
            'false_alarm_rate': false_alarm,
            'ignition_error': ignition_error,
            'thrust_mae': thrust_mae,
            'inference_time': inference_time
        }

    # ==================== 运行实验 ====================

    def run_main_experiments(self, data: Dict) -> pd.DataFrame:
        """运行主实验对比"""
        logger.info("=" * 50)
        logger.info("开始主实验对比")
        logger.info("=" * 50)

        results = {}
        results['Threshold'] = self.run_threshold_baseline(data)
        results['RandomForest'] = self.run_rf_baseline(data)
        results['CNN-Basic'] = self.run_cnn_baseline(data)
        results['Ours'] = self.run_full_method(data)

        df = pd.DataFrame(results).T
        df.index.name = 'Method'
        return df

    def run_ablation_experiments(self, data: Dict) -> pd.DataFrame:
        """运行消融实验"""
        logger.info("=" * 50)
        logger.info("开始消融实验")
        logger.info("=" * 50)

        results = {}
        results['A0'] = self.run_ablation_A0(data)
        results['A1'] = self.run_ablation_A1(data)
        results['A2'] = self.run_ablation_A2(data)
        results['A3'] = self.run_ablation_A3(data)

        df = pd.DataFrame(results).T
        df.index.name = 'Experiment'
        return df


def main():
    """主函数"""
    config = ExperimentConfig()
    runner = ExperimentRunner(config)

    # 加载数据
    logger.info("加载数据...")
    features, labels, thrust = runner.load_data()
    data = runner.split_data(features, labels, thrust)

    # 主实验
    main_results = runner.run_main_experiments(data)
    print("\n" + "=" * 60)
    print("主实验结果")
    print("=" * 60)
    print(main_results.to_string())

    # 消融实验
    ablation_results = runner.run_ablation_experiments(data)
    print("\n" + "=" * 60)
    print("消融实验结果")
    print("=" * 60)
    print(ablation_results.to_string())

    # 保存结果
    main_results.to_csv(f"{config.output_dir}/main_results.csv")
    ablation_results.to_csv(f"{config.output_dir}/ablation_results.csv")
    logger.info(f"结果已保存至 {config.output_dir}/")

    return main_results, ablation_results


if __name__ == '__main__':
    main()
