"""
LEO雷达残差数据模拟生成模块
模拟方位角(AZ)、仰角(EL)、斜距(Ra)、距离率(RR)残差数据
"""
import numpy as np
import pandas as pd
from typing import Tuple


class LEORadarResidualGenerator:
    """LEO卫星雷达残差数据生成器"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

    def set_snr(self, snr_db: float):
        """根据SNR设置噪声水平"""
        noise_level = 1.0 / (10 ** (snr_db / 20))
        self.noise_std = {
            'az': noise_level * 0.1,      # 方位角噪声(度)
            'el': noise_level * 0.1,      # 仰角噪声(度)
            'range': noise_level * 10,    # 斜距噪声(米)
            'range_rate': noise_level * 0.1  # 距离率噪声(m/s)
        }

    def generate_baseline(self, n_samples: int) -> dict:
        """生成基线残差数据"""
        return {
            'az': np.random.normal(0, self.noise_std['az'], n_samples),
            'el': np.random.normal(0, self.noise_std['el'], n_samples),
            'range': np.random.normal(0, self.noise_std['range'], n_samples),
            'range_rate': np.random.normal(0, self.noise_std['range_rate'], n_samples)
        }

    def inject_maneuver(
        self,
        residuals: dict,
        start_idx: int,
        delta_v: float,
        duration: int
    ) -> dict:
        """注入变轨信号"""
        modified = {k: v.copy() for k, v in residuals.items()}
        end_idx = min(start_idx + duration, len(residuals['az']))

        for i in range(start_idx, end_idx):
            t = i - start_idx
            signal = delta_v * (1 - np.exp(-t / 3))

            modified['az'][i] += signal * 0.5
            modified['el'][i] += signal * 0.3
            modified['range'][i] += signal * 50
            modified['range_rate'][i] += signal * 0.8

        return modified

    def generate_dataset(
        self,
        n_events: int = 100,
        n_samples_per_event: int = 2880,
        maneuver_ratio: float = 0.3,
        snr_range: Tuple[float, float] = (3, 10)
    ) -> pd.DataFrame:
        """生成完整LEO雷达残差数据集"""
        all_data = []
        n_maneuvers = int(n_events * maneuver_ratio)

        for event_id in range(n_events):
            snr = np.random.uniform(*snr_range)
            self.set_snr(snr)

            residuals = self.generate_baseline(n_samples_per_event)
            timestamps = np.arange(n_samples_per_event) * 30

            is_maneuver = event_id < n_maneuvers
            maneuver_time = -1
            delta_v = 0.0

            if is_maneuver:
                delta_v = np.random.uniform(0.01, 0.5)
                duration = np.random.randint(5, 30)
                start_idx = np.random.randint(100, n_samples_per_event - 200)
                maneuver_time = timestamps[start_idx]
                residuals = self.inject_maneuver(residuals, start_idx, delta_v, duration)

            event_df = pd.DataFrame({
                'event_id': event_id,
                'timestamp': timestamps,
                'az_residual': residuals['az'],
                'el_residual': residuals['el'],
                'range_residual': residuals['range'],
                'range_rate_residual': residuals['range_rate'],
                'maneuver_label': int(is_maneuver),
                'ignition_time': maneuver_time,
                'delta_v': delta_v,
                'snr_db': snr
            })
            all_data.append(event_df)

        return pd.concat(all_data, ignore_index=True)
