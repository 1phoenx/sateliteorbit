"""
PCA降维模块
用于DPC聚类后、1D CNN训练前的特征降维
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePCAReducer:
    """特征PCA降维器"""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        """
        初始化PCA降维器

        Args:
            n_components: 降维后维度 (2-3, 默认3)
            random_state: 随机种子
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> 'FeaturePCAReducer':
        """
        拟合PCA模型

        Args:
            features: 特征矩阵 (N, D)
        """
        # 标准化
        features_scaled = self.scaler.fit_transform(features)
        # PCA拟合
        self.pca.fit(features_scaled)
        self.is_fitted = True

        logger.info(f"PCA拟合完成:")
        logger.info(f"  原始维度: {features.shape[1]}")
        logger.info(f"  降维后维度: {self.n_components}")
        logger.info(f"  解释方差比: {self.pca.explained_variance_ratio_}")
        logger.info(f"  累计解释方差: {sum(self.pca.explained_variance_ratio_):.4f}")

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        应用PCA降维

        Args:
            features: 特征矩阵 (N, D)

        Returns:
            降维后特征 (N, n_components)
        """
        if not self.is_fitted:
            raise ValueError("请先调用fit()方法")

        features_scaled = self.scaler.transform(features)
        return self.pca.transform(features_scaled)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, features_reduced: np.ndarray) -> np.ndarray:
        """反向转换"""
        if not self.is_fitted:
            raise ValueError("请先调用fit()方法")

        features_scaled = self.pca.inverse_transform(features_reduced)
        return self.scaler.inverse_transform(features_scaled)

    def get_explained_variance(self) -> Dict:
        """获取解释方差信息"""
        if not self.is_fitted:
            raise ValueError("请先调用fit()方法")

        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': float(sum(self.pca.explained_variance_ratio_)),
            'n_components': self.n_components
        }

    def save(self, path: str):
        """保存模型"""
        joblib.dump({
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }, path)
        logger.info(f"PCA模型已保存至: {path}")

    def load(self, path: str):
        """加载模型"""
        data = joblib.load(path)
        self.pca = data['pca']
        self.scaler = data['scaler']
        self.n_components = data['n_components']
        self.is_fitted = data['is_fitted']
        logger.info(f"PCA模型已加载: {path}")


def apply_pca_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 3,
    return_reducer: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[FeaturePCAReducer]]:
    """
    PCA降维流水线

    Args:
        features: 特征矩阵
        labels: 标签
        n_components: 目标维度
        return_reducer: 是否返回降维器

    Returns:
        (降维后特征, 标签, 降维器)
    """
    reducer = FeaturePCAReducer(n_components=n_components)
    features_reduced = reducer.fit_transform(features)

    if return_reducer:
        return features_reduced, labels, reducer
    return features_reduced, labels, None
