"""
批量特征提取模块 v2
基于 Spacecraft Thruster Firing Test Dataset 提取 P/T/R 特征

物理背景与公式支撑:
====================

1. 辐射强度峰值 P (Radiation Intensity Peak)
   物理意义: 推力器点火时产生的尾焰辐射强度峰值，与推力大小正相关
   物理公式: P = η × F^β + P₀
   其中:
   - η: 辐射效率系数 (取决于推进剂类型)
   - F: 推力大小 (N)
   - β: 幂指数 (通常 0.8~1.2)
   - P₀: 背景辐射

   在本数据集中: P = max(thrust_filtered) - B_thrust
   有效性判据: P > 3σ (3倍背景噪声标准差)

2. 持续时间 T (Duration)
   物理意义: 推力器点火持续时间，与速度增量Δv相关
   物理公式: Δv = (F × T) / m = Isp × g₀ × ln(m₀/m₁)
   其中:
   - F: 推力 (N)
   - T: 持续时间 (s)
   - m: 航天器质量 (kg)
   - Isp: 比冲 (s)
   - g₀: 标准重力加速度 (9.80665 m/s²)

   在本数据集中: T = (last_idx - first_idx) × dt
   其中 first_idx/last_idx 为推力超过阈值的首末时刻

3. 频域强度比 R (Frequency Domain Intensity Ratio)
   物理意义: 基频与二次谐波的幅值比，反映推进剂燃烧状态和推力器类型
   物理公式: R = |FFT(signal)|_f₀ / |FFT(signal)|_2f₀
   其中:
   - f₀: 基频 (主频率分量)
   - 2f₀: 二次谐波

   物理解释:
   - 稳定燃烧: R较大 (基频占主导)
   - 不稳定燃烧: R较小 (谐波分量增加)
   - 不同推进剂类型有不同的R特征值

参考文献:
- Sutton, G.P., Biblarz, O. "Rocket Propulsion Elements" (推进剂燃烧特性)
- Humble, R.W. et al. "Space Propulsion Analysis and Design" (推力器性能分析)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThrusterFeatureExtractor:
    """推力器特征提取器"""

    def __init__(self, sampling_rate: float = 100.0, filter_window: int = 5):
        """
        初始化特征提取器

        Args:
            sampling_rate: 采样率 (Hz)
            filter_window: 移动平均滤波窗口大小
        """
        self.sampling_rate = sampling_rate
        self.filter_window = filter_window
        self.dt = 1.0 / sampling_rate

    def moving_average_filter(self, data: np.ndarray) -> np.ndarray:
        """移动平均滤波去噪"""
        if len(data) < self.filter_window:
            return data
        kernel = np.ones(self.filter_window) / self.filter_window
        return np.convolve(data, kernel, mode='same')

    def compute_baseline(self, thrust: np.ndarray, ton: np.ndarray) -> Tuple[float, float]:
        """
        计算背景基线值和标准差

        Args:
            thrust: 推力数据
            ton: 推力器开关状态

        Returns:
            (背景均值, 背景标准差)
        """
        background_mask = ton == 0
        if np.sum(background_mask) > 0:
            background_thrust = thrust[background_mask]
            B_thrust = np.mean(background_thrust)
            sigma_background = np.std(background_thrust)
        else:
            B_thrust = 0.0
            sigma_background = np.std(thrust) * 0.1
        return B_thrust, sigma_background

    def extract_P(self, thrust: np.ndarray, ton: np.ndarray,
                  B_thrust: float, sigma: float) -> float:
        """
        提取辐射强度峰值 P

        P = 点火时段去噪后thrust最大值 - 背景值
        有效阈值: P > 3×σ
        """
        thrust_filtered = self.moving_average_filter(thrust)
        firing_mask = ton == 1

        if np.sum(firing_mask) > 0:
            max_thrust = np.max(thrust_filtered[firing_mask])
        else:
            max_thrust = np.max(thrust_filtered)

        P = max_thrust - B_thrust
        if P < 3 * sigma:
            P = 0.0
        return P

    def extract_T(self, thrust: np.ndarray, B_thrust: float,
                  sigma: float, ton: np.ndarray = None,
                  use_precise_detection: bool = True) -> Tuple[float, float, float]:
        """
        提取持续时间 T 和点火时刻

        T = thrust超过阈值的首次/末次时刻差值
        阈值 = 背景值 + 3×σ

        Args:
            thrust: 推力数据
            B_thrust: 背景均值
            sigma: 背景标准差
            ton: 推力器开关状态 (用于精确检测)
            use_precise_detection: 是否使用精确点火检测算法
        """
        threshold = B_thrust + 3 * sigma
        thrust_filtered = self.moving_average_filter(thrust)
        above_threshold = thrust_filtered > threshold

        if not np.any(above_threshold):
            return 0.0, np.nan, 0.0

        indices = np.where(above_threshold)[0]
        first_idx = indices[0]
        last_idx = indices[-1]

        T = (last_idx - first_idx) * self.dt
        true_thrust = np.mean(thrust_filtered[above_threshold])

        # 使用精确点火检测算法
        if use_precise_detection:
            try:
                from src.ignition_detector import detect_ignition_ensemble
                result = detect_ignition_ensemble(
                    thrust, ton, sampling_rate=self.sampling_rate
                )
                if not np.isnan(result['ignition_time']):
                    ignition_time = result['ignition_time']
                    # 更新持续时间
                    if result['duration'] > 0:
                        T = result['duration']
                else:
                    ignition_time = first_idx * self.dt
            except ImportError:
                ignition_time = first_idx * self.dt
        else:
            ignition_time = first_idx * self.dt

        return T, ignition_time, true_thrust

    def extract_R(self, thrust: np.ndarray, T: float) -> float:
        """
        提取频域强度比 R

        R = 基频幅值 / 二次谐波幅值
        T < 0.1秒时返回 NaN
        """
        if T < 0.1:
            return np.nan

        thrust_filtered = self.moving_average_filter(thrust)
        n = len(thrust_filtered)

        fft_vals = fft(thrust_filtered)
        freqs = fftfreq(n, self.dt)

        positive_mask = freqs > 0
        fft_magnitude = np.abs(fft_vals[positive_mask])
        positive_freqs = freqs[positive_mask]

        if len(fft_magnitude) < 2:
            return np.nan

        base_idx = np.argmax(fft_magnitude)
        base_freq = positive_freqs[base_idx]
        base_amp = fft_magnitude[base_idx]

        harmonic_freq = 2 * base_freq
        harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
        harmonic_amp = fft_magnitude[harmonic_idx]

        if harmonic_amp < 1e-10:
            return np.nan

        R = base_amp / harmonic_amp
        return R

    def extract_features(self, file_path: str) -> Optional[Dict]:
        """
        从单个CSV文件提取特征

        Args:
            file_path: CSV文件路径

        Returns:
            特征字典，失败返回None
        """
        try:
            df = pd.read_csv(file_path)

            required_cols = ['time', 'ton', 'thrust']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"缺少必要列: {file_path}")
                return None

            thrust = df['thrust'].values
            ton = df['ton'].values

            B_thrust, sigma = self.compute_baseline(thrust, ton)
            P = self.extract_P(thrust, ton, B_thrust, sigma)
            T, ignition_time, true_thrust = self.extract_T(thrust, B_thrust, sigma, ton)
            R = self.extract_R(thrust, T)

            is_valid = 1 if (P > 0 and T >= 0.1) else 0

            return {
                'P': P,
                'T': T,
                'R': R,
                'ignition_time': ignition_time,
                'true_thrust': true_thrust,
                'is_valid': is_valid
            }

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return None

    def batch_extract_features(
        self,
        data_dir: str,
        metadata_path: str,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        批量提取特征

        Args:
            data_dir: 数据目录 (包含 train/ 和 test/ 子目录)
            metadata_path: metadata.csv 路径
            output_path: 输出文件路径

        Returns:
            特征数据集 DataFrame
        """
        logger.info(f"开始批量特征提取，数据目录: {data_dir}")

        # 读取元数据
        metadata = pd.read_csv(metadata_path)
        logger.info(f"元数据记录数: {len(metadata)}")

        results = []
        processed = 0
        failed = 0

        for _, row in metadata.iterrows():
            uid = row['uid']
            filename = row['filename']
            sn = row['sn']
            test_id = row['test_id']
            is_anomalous = row.get('anomalous', 0)

            # 确定文件路径 (train 或 test)
            train_path = Path(data_dir) / 'train' / filename
            test_path = Path(data_dir) / 'test' / filename

            if train_path.exists():
                file_path = train_path
                split = 'train'
            elif test_path.exists():
                file_path = test_path
                split = 'test'
            else:
                logger.warning(f"文件不存在: {filename}")
                failed += 1
                continue

            # 提取特征
            features = self.extract_features(str(file_path))

            if features is not None:
                results.append({
                    'uid': uid,
                    'sn': sn,
                    'test_id': test_id,
                    'split': split,
                    'P': features['P'],
                    'T': features['T'],
                    'R': features['R'],
                    'ignition_time': features['ignition_time'],
                    'true_thrust': features['true_thrust'],
                    'is_anomalous': is_anomalous,
                    'is_valid': features['is_valid']
                })
                processed += 1
            else:
                failed += 1

            if (processed + failed) % 100 == 0:
                logger.info(f"进度: {processed + failed}/{len(metadata)}")

        logger.info(f"批量提取完成: 成功 {processed}, 失败 {failed}")

        # 创建 DataFrame
        df = pd.DataFrame(results)

        # 保存结果
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"特征数据已保存至: {output_path}")

        return df


def main():
    """主函数 - 批量特征提取"""
    import argparse

    parser = argparse.ArgumentParser(description='推力器特征批量提取')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--metadata', type=str, default='data/metadata.csv',
                        help='元数据文件路径')
    parser.add_argument('--output', type=str, default='data/feature_dataset.csv',
                        help='输出文件路径')
    parser.add_argument('--sampling_rate', type=float, default=100.0,
                        help='采样率 (Hz)')

    args = parser.parse_args()

    # 创建特征提取器
    extractor = ThrusterFeatureExtractor(sampling_rate=args.sampling_rate)

    # 批量提取
    df = extractor.batch_extract_features(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        output_path=args.output
    )

    # 输出统计信息
    print("\n" + "=" * 50)
    print("特征提取统计")
    print("=" * 50)
    print(f"总样本数: {len(df)}")
    print(f"有效样本: {df['is_valid'].sum()}")
    print(f"异常样本: {df['is_anomalous'].sum()}")
    print(f"\n特征统计:")
    print(df[['P', 'T', 'R']].describe())


if __name__ == '__main__':
    main()
