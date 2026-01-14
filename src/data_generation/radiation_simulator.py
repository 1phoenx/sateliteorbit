"""
辐射强度数据模拟生成模块（Monte-Carlo方法）
模拟226nm/306nm波段辐射强度数据
"""
import numpy as np
import pandas as pd
from typing import Tuple


class RadiationIntensitySimulator:
    """辐射强度模拟器"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

        # 物理参数
        self.k_coefficient = 10000  # P = k * delta_v 的比例系数
        self.r_maneuver = 1.2       # 变轨时226nm/306nm强度比
        self.r_maneuver_std = 0.1
        self.r_background = 0.9    # 非变轨时强度比
        self.r_background_std = 0.1

    def generate_background_radiation(
        self,
        n_samples: int,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成背景辐射强度"""
        noise_level = 1.0 / (10 ** (snr_db / 20))

        # 背景强度（低水平随机波动）
        base_226 = 100 + np.random.normal(0, 10 * noise_level, n_samples)

        # 306nm基于强度比计算
        r_values = np.random.normal(self.r_background, self.r_background_std, n_samples)
        base_306 = base_226 / np.clip(r_values, 0.5, 1.5)

        return base_226, base_306

    def inject_maneuver_radiation(
        self,
        intensity_226: np.ndarray,
        intensity_306: np.ndarray,
        start_idx: int,
        delta_v: float,
        duration: int,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """注入变轨辐射信号"""
        i226 = intensity_226.copy()
        i306 = intensity_306.copy()

        noise_level = 1.0 / (10 ** (snr_db / 20))
        end_idx = min(start_idx + duration, len(i226))

        # 峰值强度 P = k * delta_v
        peak = self.k_coefficient * delta_v * np.random.uniform(0.9, 1.1)

        for i in range(start_idx, end_idx):
            t = i - start_idx
            # 脉冲信号：快速上升，缓慢衰减
            signal = peak * np.exp(-0.5 * ((t - duration/2) / (duration/4))**2)
            signal += np.random.normal(0, peak * 0.05 * noise_level)

            i226[i] += signal
            r = np.random.normal(self.r_maneuver, self.r_maneuver_std)
            i306[i] += signal / max(r, 0.5)

        return i226, i306

    def extract_features(
        self,
        intensity_226: np.ndarray,
        intensity_306: np.ndarray
    ) -> dict:
        """提取三维特征向量(P, T, R)"""
        # 峰值强度P
        P = np.max(intensity_226)

        # 持续时间T（超过阈值的采样点数）
        threshold = np.mean(intensity_226) + 2 * np.std(intensity_226)
        T = np.sum(intensity_226 > threshold)

        # 强度比R
        R = np.mean(intensity_226) / (np.mean(intensity_306) + 1e-8)

        return {'P': P, 'T': T, 'R': R}
