"""
GEO光学残差数据模拟生成模块
模拟赤经(RA)、赤纬(DEC)残差数据
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional


class GEOOpticalResidualGenerator:
    """GEO卫星光学残差数据生成器"""

    def __init__(self, seed: int = 42):
        """
        初始化生成器

        Args:
            seed: 随机种子，确保可复现性
        """
        np.random.seed(seed)
        self.seed = seed

        # 默认参数
        self.sampling_interval = 30  # 采样间隔(秒)
        self.noise_std_ra = 0.5      # RA残差噪声标准差(arcsec)
        self.noise_std_dec = 0.5     # DEC残差噪声标准差(arcsec)

    def generate_baseline_residuals(
        self,
        duration_hours: float,
        sampling_interval: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成基线残差数据（无变轨）

        Args:
            duration_hours: 数据持续时间(小时)
            sampling_interval: 采样间隔(秒)

        Returns:
            timestamps, ra_residuals, dec_residuals
        """
        n_samples = int(duration_hours * 3600 / sampling_interval)

        # 生成时间戳
        timestamps = np.arange(n_samples) * sampling_interval

        # 生成高斯噪声残差
        ra_residuals = np.random.normal(0, self.noise_std_ra, n_samples)
        dec_residuals = np.random.normal(0, self.noise_std_dec, n_samples)

        return timestamps, ra_residuals, dec_residuals

    def inject_maneuver(
        self,
        ra_residuals: np.ndarray,
        dec_residuals: np.ndarray,
        maneuver_start_idx: int,
        delta_v: float,
        duration_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在残差数据中注入变轨信号

        Args:
            ra_residuals: RA残差数组
            dec_residuals: DEC残差数组
            maneuver_start_idx: 变轨起始索引
            delta_v: 速度增量(m/s)
            duration_samples: 变轨持续采样点数

        Returns:
            注入变轨后的ra_residuals, dec_residuals
        """
        ra_modified = ra_residuals.copy()
        dec_modified = dec_residuals.copy()

        # 变轨导致的残差突变幅度与delta_v成正比
        # 系数基于物理模型估计
        amplitude_factor = 10.0  # arcsec/(m/s)

        end_idx = min(maneuver_start_idx + duration_samples, len(ra_residuals))

        # 生成变轨信号（阶跃+衰减）
        for i in range(maneuver_start_idx, end_idx):
            t = i - maneuver_start_idx
            # 阶跃响应 + 指数衰减
            signal = delta_v * amplitude_factor * (1 - np.exp(-t / 5))
            ra_modified[i] += signal * np.random.uniform(0.8, 1.2)
            dec_modified[i] += signal * np.random.uniform(0.6, 1.0)

        return ra_modified, dec_modified

    def generate_dataset(
        self,
        n_events: int = 100,
        duration_hours: float = 24,
        maneuver_ratio: float = 0.3,
        snr_range: Tuple[float, float] = (5, 10)
    ) -> pd.DataFrame:
        """
        生成完整的GEO光学残差数据集

        Args:
            n_events: 事件数量
            duration_hours: 每个事件持续时间
            maneuver_ratio: 变轨事件比例
            snr_range: 信噪比范围(dB)

        Returns:
            包含所有数据的DataFrame
        """
        all_data = []
        n_maneuvers = int(n_events * maneuver_ratio)

        for event_id in range(n_events):
            # 设置SNR
            snr = np.random.uniform(*snr_range)
            self.noise_std_ra = 1.0 / (10 ** (snr / 20))
            self.noise_std_dec = 1.0 / (10 ** (snr / 20))

            # 生成基线数据
            ts, ra, dec = self.generate_baseline_residuals(duration_hours)

            # 是否为变轨事件
            is_maneuver = event_id < n_maneuvers
            maneuver_time = -1
            delta_v = 0.0

            if is_maneuver:
                # 随机变轨参数
                delta_v = np.random.uniform(0.01, 0.5)
                duration = np.random.randint(5, 30)
                start_idx = np.random.randint(100, len(ts) - 200)
                maneuver_time = ts[start_idx]

                ra, dec = self.inject_maneuver(ra, dec, start_idx, delta_v, duration)

            # 构建DataFrame
            event_df = pd.DataFrame({
                'event_id': event_id,
                'timestamp': ts,
                'ra_residual': ra,
                'dec_residual': dec,
                'ra_residual_norm': (ra - ra.mean()) / (ra.std() + 1e-8),
                'dec_residual_norm': (dec - dec.mean()) / (dec.std() + 1e-8),
                'maneuver_label': int(is_maneuver),
                'ignition_time': maneuver_time,
                'delta_v': delta_v,
                'snr_db': snr
            })
            all_data.append(event_df)

        return pd.concat(all_data, ignore_index=True)
