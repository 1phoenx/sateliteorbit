"""
第三阶段：识别模型构建
===========================================

基于 UV 特征构建四个识别模型：
1. 点火时刻识别
2. 是否变轨（二分类）
3. 推力大小回归
4. 变轨类型分类

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
import json

# 机器学习模型
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

# 深度学习模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 点火时刻识别模型
# ============================================================================

class IgnitionDetector:
    """
    点火时刻识别器

    方法：基于 UV 强度上升沿的阈值检测 + dI/dt 判据
    """

    def __init__(
        self,
        threshold_factor: float = 3.0,
        min_rise_rate: float = 10.0,
        sampling_rate: float = 100.0
    ):
        self.threshold_factor = threshold_factor
        self.min_rise_rate = min_rise_rate
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

    def detect_ignition(
        self,
        uv_series: np.ndarray
    ) -> Tuple[float, float]:
        """
        检测点火时刻

        参数:
            uv_series: UV 强度时间序列

        返回:
            (ignition_time, confidence): 点火时刻（秒）和置信度
        """
        # 计算背景和阈值
        n_background = max(10, int(len(uv_series) * 0.1))
        background_mean = np.mean(uv_series[:n_background])
        background_std = np.std(uv_series[:n_background])
        threshold = background_mean + self.threshold_factor * background_std

        # 计算 dI/dt
        dI_dt = np.diff(uv_series) / self.dt

        # 查找第一个满足条件的点
        for i in range(len(uv_series) - 1):
            # 条件1: UV 强度超过阈值
            # 条件2: dI/dt 超过最小上升率
            if uv_series[i] > threshold and dI_dt[i] > self.min_rise_rate:
                ignition_time = i * self.dt
                confidence = min(1.0, dI_dt[i] / (self.min_rise_rate * 10))
                return ignition_time, confidence

        # 未检测到点火
        return -1.0, 0.0

    def batch_detect(
        self,
        uv_series_list: List[np.ndarray]
    ) -> List[Tuple[float, float]]:
        """批量检测点火时刻"""
        results = []
        for uv_series in uv_series_list:
            result = self.detect_ignition(uv_series)
            results.append(result)
        return results


# ============================================================================
# 2. 是否变轨（二分类）模型
# ============================================================================

class ManeuverClassifier:
    """
    变轨二分类器

    方法：随机森林分类器
    输入：UV 特征
    输出：是否变轨（0=否，1=是）
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None
    ):
        """训练模型"""
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = feature_names

        # 训练集性能
        y_pred = self.model.predict(X_train_scaled)
        acc = accuracy_score(y_train, y_pred)
        print(f"训练集准确率: {acc:.4f}")

    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        return y_pred, y_proba

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """评估模型"""
        y_pred, y_proba = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        return metrics

    def save(self, filepath: Path):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

    def load(self, filepath: Path):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']


# ============================================================================
# 3. 推力大小回归模型
# ============================================================================

class ThrustRegressor:
    """
    推力大小回归器

    方法：随机森林回归器
    输入：UV 特征
    输出：推力估计值（N）
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None
    ):
        """训练模型"""
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = feature_names

        # 训练集性能
        y_pred = self.model.predict(X_train_scaled)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        print(f"训练集 MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """预测"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """评估模型"""
        y_pred = self.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        return metrics

    def save(self, filepath: Path):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)

    def load(self, filepath: Path):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']


# ============================================================================
# 4. 变轨类型分类模型
# ============================================================================

class ManeuverTypeClassifier:
    """
    变轨类型分类器

    方法：随机森林多分类器
    输入：UV 特征
    输出：变轨类型
        - 0: 短脉冲姿态修正
        - 1: 长时低推变轨
        - 2: 多脉冲调整
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = [
            '短脉冲姿态修正',
            '长时低推变轨',
            '多脉冲调整'
        ]

    def _infer_maneuver_type(
        self,
        features: Dict
    ) -> int:
        """
        根据特征推断变轨类型

        规则：
        - 短脉冲姿态修正: num_pulses <= 2, mean_pulse_duration < 1.0
        - 长时低推变轨: num_pulses <= 2, mean_pulse_duration >= 1.0
        - 多脉冲调整: num_pulses > 2
        """
        num_pulses = features.get('num_pulses', 0)
        mean_duration = features.get('mean_pulse_duration', 0.0)

        if num_pulses > 2:
            return 2  # 多脉冲调整
        elif mean_duration >= 1.0:
            return 1  # 长时低推变轨
        else:
            return 0  # 短脉冲姿态修正

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str] = None
    ):
        """训练模型"""
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 训练
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = feature_names

        # 训练集性能
        y_pred = self.model.predict(X_train_scaled)
        acc = accuracy_score(y_train, y_pred)
        print(f"训练集准确率: {acc:.4f}")

    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        return y_pred, y_proba

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """评估模型"""
        y_pred, y_proba = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        return metrics

    def save(self, filepath: Path):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'class_names': self.class_names
            }, f)

    def load(self, filepath: Path):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.class_names = data['class_names']


# ============================================================================
# 数据准备
# ============================================================================

def prepare_data(
    features_df: pd.DataFrame
) -> Tuple[np.ndarray, Dict]:
    """
    准备训练数据

    参数:
        features_df: 特征 DataFrame

    返回:
        (X, labels): 特征矩阵和标签字典
    """
    # 选择数值特征
    feature_cols = [
        'background_mean', 'background_std', 'threshold',
        'peak_intensity', 'mean_intensity', 'total_energy',
        'num_pulses',
        'mean_pulse_duration', 'max_pulse_duration',
        'mean_pulse_peak', 'max_pulse_peak',
        'mean_pulse_energy',
        'max_rise_rate', 'mean_rise_rate',
        'mean_pulse_interval', 'min_pulse_interval'
    ]

    # 提取特征矩阵
    X = features_df[feature_cols].fillna(0).values

    # 准备标签
    labels = {}

    # 1. 变轨标签（基于 num_pulses > 0）
    labels['is_maneuver'] = (features_df['num_pulses'] > 0).astype(int).values

    # 2. 推力标签
    if 'true_thrust' in features_df.columns:
        labels['thrust'] = features_df['true_thrust'].fillna(0).values
    else:
        # 使用 peak_intensity 作为代理
        labels['thrust'] = features_df['peak_intensity'].values * 0.01

    # 3. 变轨类型标签
    maneuver_types = []
    for _, row in features_df.iterrows():
        num_pulses = row['num_pulses']
        mean_duration = row['mean_pulse_duration']

        if num_pulses > 2:
            maneuver_type = 2  # 多脉冲调整
        elif mean_duration >= 1.0:
            maneuver_type = 1  # 长时低推变轨
        else:
            maneuver_type = 0  # 短脉冲姿态修正

        maneuver_types.append(maneuver_type)

    labels['maneuver_type'] = np.array(maneuver_types)

    return X, labels, feature_cols


# ============================================================================
# 主训练流程
# ============================================================================

def train_all_models(
    train_features_file: Path,
    test_features_file: Path,
    output_dir: Path
):
    """
    训练所有识别模型

    参数:
        train_features_file: 训练集特征文件
        test_features_file: 测试集特征文件
        output_dir: 输出目录
    """
    print("=" * 70)
    print("第三阶段：识别模型构建")
    print("=" * 70)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv(train_features_file)
    test_df = pd.read_csv(test_features_file)

    print(f"训练集: {len(train_df)} 个样本")
    print(f"测试集: {len(test_df)} 个样本")

    # 准备数据
    X_train, y_train, feature_names = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)

    print(f"特征维度: {X_train.shape[1]}")

    # ========================================================================
    # 1. 点火时刻识别
    # ========================================================================
    print("\n" + "=" * 70)
    print("1️⃣ 点火时刻识别模型")
    print("=" * 70)

    ignition_detector = IgnitionDetector(
        threshold_factor=3.0,
        min_rise_rate=10.0,
        sampling_rate=100.0
    )

    print("点火检测器已创建（基于阈值 + dI/dt 判据）")
    print("注意：此模型需要完整的 UV 时间序列，在推理时使用")

    # 保存
    with open(output_dir / 'ignition_detector.pkl', 'wb') as f:
        pickle.dump(ignition_detector, f)

    # ========================================================================
    # 2. 是否变轨（二分类）
    # ========================================================================
    print("\n" + "=" * 70)
    print("2️⃣ 是否变轨（二分类）模型")
    print("=" * 70)

    maneuver_clf = ManeuverClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    print("\n训练变轨分类器...")
    maneuver_clf.train(X_train, y_train['is_maneuver'], feature_names)

    print("\n测试集评估:")
    metrics = maneuver_clf.evaluate(X_test, y_test['is_maneuver'])
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存
    maneuver_clf.save(output_dir / 'maneuver_classifier.pkl')

    # ========================================================================
    # 3. 推力大小回归
    # ========================================================================
    print("\n" + "=" * 70)
    print("3️⃣ 推力大小回归模型")
    print("=" * 70)

    thrust_reg = ThrustRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    print("\n训练推力回归器...")
    thrust_reg.train(X_train, y_train['thrust'], feature_names)

    print("\n测试集评估:")
    metrics = thrust_reg.evaluate(X_test, y_test['thrust'])
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存
    thrust_reg.save(output_dir / 'thrust_regressor.pkl')

    # ========================================================================
    # 4. 变轨类型分类
    # ========================================================================
    print("\n" + "=" * 70)
    print("4️⃣ 变轨类型分类模型")
    print("=" * 70)

    type_clf = ManeuverTypeClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    print("\n训练变轨类型分类器...")
    type_clf.train(X_train, y_train['maneuver_type'], feature_names)

    print("\n测试集评估:")
    metrics = type_clf.evaluate(X_test, y_test['maneuver_type'])
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 保存
    type_clf.save(output_dir / 'maneuver_type_classifier.pkl')

    # ========================================================================
    # 保存元信息
    # ========================================================================
    metadata = {
        'feature_names': feature_names,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'models': {
            'ignition_detector': 'ignition_detector.pkl',
            'maneuver_classifier': 'maneuver_classifier.pkl',
            'thrust_regressor': 'thrust_regressor.pkl',
            'maneuver_type_classifier': 'maneuver_type_classifier.pkl'
        }
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("第三阶段完成！")
    print("=" * 70)
    print(f"所有模型已保存到: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    train_all_models(
        train_features_file='data/uv_features_train.csv',
        test_features_file='data/uv_features_test.csv',
        output_dir='models/uv_recognition'
    )
