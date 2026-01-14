"""
数据一致性验证模块
验证时序一致性、特征一致性、噪声兼容性
"""
import numpy as np
import pandas as pd
from typing import Dict, List


class DataValidator:
    """数据验证器"""

    def __init__(self, sampling_interval: int = 30):
        self.sampling_interval = sampling_interval

    def validate_all(self, df: pd.DataFrame) -> Dict:
        """执行所有验证"""
        results = {
            'temporal': self.check_temporal_consistency(df),
            'feature': self.check_feature_consistency(df),
            'noise': self.check_noise_compatibility(df)
        }
        results['passed'] = all(r['passed'] for r in results.values())
        return results

    def check_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """检查时序一致性：变轨信号在标注的点火时刻附近"""
        results = {'passed': True, 'errors': []}

        for event_id in df['event_id'].unique():
            event = df[df['event_id'] == event_id]

            if event['maneuver_label'].iloc[0] == 0:
                continue

            ignition_time = event['ignition_time'].iloc[0]
            if ignition_time < 0:
                continue

            # 找辐射峰值时刻
            rad_peak_idx = event['intensity_226nm'].idxmax()
            rad_peak_time = event.loc[rad_peak_idx, 'timestamp']

            # 检查辐射峰值是否在点火时刻附近（允许较大容差）
            time_diff = abs(rad_peak_time - ignition_time)
            max_tolerance = self.sampling_interval * 50  # 允许50个采样间隔

            if time_diff > max_tolerance:
                results['errors'].append(f"Event {event_id}: 时间差{time_diff}s")
                results['passed'] = False

        return results

    def check_feature_consistency(self, df: pd.DataFrame) -> Dict:
        """检查特征一致性：P、T与delta_v正相关"""
        results = {'passed': True, 'errors': []}

        maneuver_events = df[df['maneuver_label'] == 1]
        if len(maneuver_events) == 0:
            return results

        # 按事件聚合
        stats = []
        for eid in maneuver_events['event_id'].unique():
            event = maneuver_events[maneuver_events['event_id'] == eid]
            P = event['intensity_226nm'].max()
            dv = event['delta_v'].iloc[0]
            stats.append({'P': P, 'delta_v': dv})

        stats_df = pd.DataFrame(stats)
        corr = stats_df['P'].corr(stats_df['delta_v'])

        if corr < 0.5:
            results['passed'] = False
            results['errors'].append(f"P与delta_v相关性过低: {corr:.3f}")

        return results

    def check_noise_compatibility(self, df: pd.DataFrame) -> Dict:
        """检查噪声兼容性：变轨事件的信号应明显高于背景"""
        results = {'passed': True, 'errors': []}

        # 只检查变轨事件
        maneuver_events = df[df['maneuver_label'] == 1]
        if len(maneuver_events) == 0:
            return results

        for eid in maneuver_events['event_id'].unique():
            event = df[df['event_id'] == eid]
            signal = event['intensity_226nm'].values

            # 计算峰值与背景的比值
            peak = np.max(signal)
            background = np.median(signal)
            ratio = peak / (background + 1e-8)

            # 变轨事件的峰值应至少是背景的2倍
            if ratio < 2:
                results['errors'].append(f"Event {eid}: 峰值/背景={ratio:.2f}")
                results['passed'] = False

        return results
