"""
HMSE (Hierarchical Multi-Scale Entropy) 数据处理模块
用于增强特征数据的噪声鲁棒性
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import uniform_filter1d


class HMSEProcessor:
    """分层多尺度熵处理器"""

    def __init__(self, scales: list = [1, 2, 4], m: int = 2, r_ratio: float = 0.15):
        """
        初始化HMSE处理器

        Args:
            scales: 多尺度因子列表
            m: 嵌入维度
            r_ratio: 容差比例 (相对于标准差)
        """
        self.scales = scales
        self.m = m
        self.r_ratio = r_ratio

    def coarse_grain(self, signal: np.ndarray, scale: int) -> np.ndarray:
        """
        粗粒化处理 - 多尺度分解

        Args:
            signal: 输入信号
            scale: 尺度因子

        Returns:
            粗粒化后的信号
        """
        if scale == 1:
            return signal

        n = len(signal)
        n_coarse = n // scale
        coarse_signal = np.zeros(n_coarse)

        for i in range(n_coarse):
            coarse_signal[i] = np.mean(signal[i * scale:(i + 1) * scale])

        return coarse_signal

    def sample_entropy(self, signal: np.ndarray, m: int = None, r: float = None) -> float:
        """
        计算样本熵 (Sample Entropy)

        Args:
            signal: 输入信号
            m: 嵌入维度
            r: 容差阈值

        Returns:
            样本熵值
        """
        if m is None:
            m = self.m
        if r is None:
            r = self.r_ratio * np.std(signal)

        n = len(signal)
        if n < m + 2:
            return 0.0

        # 构建嵌入向量
        def count_matches(templates, r_val):
            count = 0
            n_templates = len(templates)
            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    if np.max(np.abs(templates[i] - templates[j])) < r_val:
                        count += 1
            return count

        # m维嵌入
        templates_m = np.array([signal[i:i + m] for i in range(n - m)])
        B = count_matches(templates_m, r)

        # m+1维嵌入
        templates_m1 = np.array([signal[i:i + m + 1] for i in range(n - m - 1)])
        A = count_matches(templates_m1, r)

        if B == 0 or A == 0:
            return 0.0

        return -np.log(A / B)

    def compute_mse(self, signal: np.ndarray) -> np.ndarray:
        """
        计算多尺度熵 (Multi-Scale Entropy)

        Args:
            signal: 输入信号

        Returns:
            各尺度的熵值数组
        """
        mse_values = []
        for scale in self.scales:
            coarse_signal = self.coarse_grain(signal, scale)
            se = self.sample_entropy(coarse_signal)
            mse_values.append(se)
        return np.array(mse_values)

    def denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        基于HMSE的自适应去噪

        通过多尺度分析识别噪声成分并滤除

        Args:
            signal: 输入信号

        Returns:
            去噪后的信号
        """
        if len(signal) < 10:
            return signal

        # 计算各尺度的熵值
        mse_values = self.compute_mse(signal)

        # 根据熵值自适应选择滤波强度
        # 高熵表示高复杂度/噪声，需要更强的滤波
        avg_entropy = np.mean(mse_values)
        entropy_ratio = avg_entropy / (np.max(mse_values) + 1e-10)

        # 自适应滤波窗口大小
        base_window = 3
        adaptive_window = int(base_window + entropy_ratio * 4)
        adaptive_window = min(adaptive_window, len(signal) // 4)
        adaptive_window = max(adaptive_window, 3)

        # 应用均值滤波
        denoised = uniform_filter1d(signal, size=adaptive_window, mode='nearest')

        return denoised

    def extract_entropy_features(self, signal: np.ndarray) -> dict:
        """
        提取基于熵的特征

        Args:
            signal: 输入信号

        Returns:
            熵特征字典
        """
        mse_values = self.compute_mse(signal)

        return {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'mse_slope': (mse_values[-1] - mse_values[0]) / len(mse_values) if len(mse_values) > 1 else 0,
            'complexity_index': np.sum(mse_values)
        }

    def process_features(self, features: np.ndarray) -> np.ndarray:
        """
        对特征矩阵进行HMSE增强处理

        Args:
            features: 特征矩阵 (N, D)

        Returns:
            增强后的特征矩阵
        """
        n_samples, n_features = features.shape
        processed = np.zeros_like(features)

        for i in range(n_features):
            feature_col = features[:, i]

            # 去除异常值
            q1, q3 = np.percentile(feature_col, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # 裁剪异常值
            clipped = np.clip(feature_col, lower, upper)

            # 自适应平滑
            if len(clipped) > 10:
                smoothed = self.denoise_signal(clipped)
            else:
                smoothed = clipped

            processed[:, i] = smoothed

        return processed


def apply_hmse_preprocessing(
    features: np.ndarray,
    scales: list = [1, 2, 4],
    return_entropy_features: bool = False
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    应用HMSE预处理的便捷函数

    Args:
        features: 特征矩阵 (N, D)
        scales: 多尺度因子
        return_entropy_features: 是否返回熵特征

    Returns:
        (处理后的特征, 熵特征字典或None)
    """
    processor = HMSEProcessor(scales=scales)
    processed = processor.process_features(features)

    entropy_features = None
    if return_entropy_features:
        entropy_features = {}
        for i in range(features.shape[1]):
            col_entropy = processor.extract_entropy_features(features[:, i])
            for key, val in col_entropy.items():
                entropy_features[f'feature_{i}_{key}'] = val

    return processed, entropy_features
