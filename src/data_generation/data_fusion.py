"""
数据融合模块
将光学残差、雷达残差与辐射强度数据融合
"""
import numpy as np
import pandas as pd
from typing import Optional

from .optical_residual import GEOOpticalResidualGenerator
from .radar_residual import LEORadarResidualGenerator
from .radiation_simulator import RadiationIntensitySimulator


class DataFusion:
    """数据融合器"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.optical_gen = GEOOpticalResidualGenerator(seed)
        self.radar_gen = LEORadarResidualGenerator(seed)
        self.radiation_sim = RadiationIntensitySimulator(seed)

    def generate_fused_geo_dataset(
        self,
        n_events: int = 100,
        duration_hours: float = 24,
        maneuver_ratio: float = 0.3,
        snr_range: tuple = (5, 10)
    ) -> pd.DataFrame:
        """生成融合的GEO数据集"""
        all_data = []
        n_maneuvers = int(n_events * maneuver_ratio)

        for event_id in range(n_events):
            snr = np.random.uniform(*snr_range)
            self.optical_gen.noise_std_ra = 1.0 / (10 ** (snr / 20))
            self.optical_gen.noise_std_dec = 1.0 / (10 ** (snr / 20))

            # 生成光学残差
            ts, ra, dec = self.optical_gen.generate_baseline_residuals(duration_hours)
            n_samples = len(ts)

            # 生成辐射强度
            i226, i306 = self.radiation_sim.generate_background_radiation(n_samples, snr)

            is_maneuver = event_id < n_maneuvers
            maneuver_time = -1
            delta_v = 0.0

            if is_maneuver:
                delta_v = np.random.uniform(0.01, 0.5)
                duration = np.random.randint(5, 30)
                start_idx = np.random.randint(100, n_samples - 200)
                maneuver_time = ts[start_idx]

                ra, dec = self.optical_gen.inject_maneuver(
                    ra, dec, start_idx, delta_v, duration
                )
                i226, i306 = self.radiation_sim.inject_maneuver_radiation(
                    i226, i306, start_idx, delta_v, duration, snr
                )

            all_data.append(self._build_dataframe(
                event_id, ts, ra, dec, i226, i306,
                is_maneuver, maneuver_time, delta_v, snr
            ))

        return pd.concat(all_data, ignore_index=True)

    def _build_dataframe(
        self, event_id, ts, ra, dec, i226, i306,
        is_maneuver, maneuver_time, delta_v, snr
    ) -> pd.DataFrame:
        """构建单个事件的DataFrame"""
        return pd.DataFrame({
            'event_id': event_id,
            'timestamp': ts,
            'ra_residual': ra,
            'dec_residual': dec,
            'intensity_226nm': i226,
            'intensity_306nm': i306,
            'maneuver_label': int(is_maneuver),
            'ignition_time': maneuver_time,
            'delta_v': delta_v,
            'snr_db': snr
        })

    def generate_fused_leo_dataset(
        self,
        n_events: int = 100,
        n_samples: int = 2880,
        maneuver_ratio: float = 0.3,
        snr_range: tuple = (3, 10)
    ) -> pd.DataFrame:
        """生成融合的LEO雷达数据集"""
        all_data = []
        n_maneuvers = int(n_events * maneuver_ratio)

        for event_id in range(n_events):
            snr = np.random.uniform(*snr_range)
            self.radar_gen.set_snr(snr)

            residuals = self.radar_gen.generate_baseline(n_samples)
            ts = np.arange(n_samples) * 30
            i226, i306 = self.radiation_sim.generate_background_radiation(n_samples, snr)

            is_maneuver = event_id < n_maneuvers
            maneuver_time = -1
            delta_v = 0.0

            if is_maneuver:
                delta_v = np.random.uniform(0.01, 0.5)
                duration = np.random.randint(5, 30)
                start_idx = np.random.randint(100, n_samples - 200)
                maneuver_time = ts[start_idx]

                residuals = self.radar_gen.inject_maneuver(
                    residuals, start_idx, delta_v, duration
                )
                i226, i306 = self.radiation_sim.inject_maneuver_radiation(
                    i226, i306, start_idx, delta_v, duration, snr
                )

            df = pd.DataFrame({
                'event_id': event_id,
                'timestamp': ts,
                'az_residual': residuals['az'],
                'el_residual': residuals['el'],
                'range_residual': residuals['range'],
                'range_rate_residual': residuals['range_rate'],
                'intensity_226nm': i226,
                'intensity_306nm': i306,
                'maneuver_label': int(is_maneuver),
                'ignition_time': maneuver_time,
                'delta_v': delta_v,
                'snr_db': snr
            })
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)
