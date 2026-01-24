"""
第二阶段：UV 特征工程
===========================================

基于生成的 uv_360nm 时间序列，提取以下特征：
1. dI/dt（上升沿斜率）
2. 峰值强度
3. 峰值持续时间
4. 脉冲面积（积分）
5. 脉冲数量与间隔

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, integrate
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class UVFeatureExtractor:
    """
    UV 特征提取器

    参数:
        threshold_factor: 阈值因子（相对于背景标准差的倍数）
        min_pulse_duration: 最小脉冲持续时间（秒）
        sampling_rate: 采样率（Hz）
    """

    def __init__(
        self,
        threshold_factor: float = 3.0,
        min_pulse_duration: float = 0.1,
        sampling_rate: float = 100.0
    ):
        self.threshold_factor = threshold_factor
        self.min_pulse_duration = min_pulse_duration
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate

    def extract_features(
        self,
        uv_series: np.ndarray,
        time_series: np.ndarray = None
    ) -> Dict:
        """
        从 UV 时间序列提取完整特征

        参数:
            uv_series: UV 强度时间序列
            time_series: 时间序列（可选）

        返回:
            features: 特征字典
        """
        # 如果没有提供时间序列，生成默认时间序列
        if time_series is None:
            time_series = np.arange(len(uv_series)) * self.dt

        # 计算背景值和阈值
        background_mean, background_std, threshold = self._compute_threshold(uv_series)

        # 检测脉冲
        pulses = self._detect_pulses(uv_series, threshold)

        # 提取特征
        features = {
            # 背景特征
            'background_mean': float(background_mean),
            'background_std': float(background_std),
            'threshold': float(threshold),

            # 全局特征
            'peak_intensity': float(np.max(uv_series)),
            'mean_intensity': float(np.mean(uv_series)),
            'total_energy': float(integrate.trapz(uv_series, dx=self.dt)),

            # 脉冲特征
            'num_pulses': len(pulses),
            'pulse_intervals': [],
            'pulse_durations': [],
            'pulse_peak_intensities': [],
            'pulse_energies': [],
            'pulse_rise_rates': [],
            'pulse_fall_rates': []
        }

        # 如果检测到脉冲，提取详细特征
        if len(pulses) > 0:
            for pulse in pulses:
                start_idx, end_idx, peak_idx = pulse

                # 脉冲持续时间
                duration = (end_idx - start_idx) * self.dt
                features['pulse_durations'].append(duration)

                # 脉冲峰值强度
                peak_intensity = uv_series[peak_idx]
                features['pulse_peak_intensities'].append(peak_intensity)

                # 脉冲能量（积分）
                pulse_segment = uv_series[start_idx:end_idx+1]
                pulse_energy = integrate.trapz(pulse_segment, dx=self.dt)
                features['pulse_energies'].append(pulse_energy)

                # 上升沿斜率 dI/dt
                if peak_idx > start_idx:
                    rise_segment = uv_series[start_idx:peak_idx+1]
                    rise_rate = np.max(np.diff(rise_segment)) / self.dt
                else:
                    rise_rate = 0.0
                features['pulse_rise_rates'].append(rise_rate)

                # 下降沿斜率
                if end_idx > peak_idx:
                    fall_segment = uv_series[peak_idx:end_idx+1]
                    fall_rate = np.min(np.diff(fall_segment)) / self.dt
                else:
                    fall_rate = 0.0
                features['pulse_fall_rates'].append(fall_rate)

            # 脉冲间隔
            if len(pulses) > 1:
                for i in range(len(pulses) - 1):
                    interval = (pulses[i+1][0] - pulses[i][1]) * self.dt
                    features['pulse_intervals'].append(interval)

            # 统计特征（均值、标准差、最大值）
            features['mean_pulse_duration'] = float(np.mean(features['pulse_durations']))
            features['max_pulse_duration'] = float(np.max(features['pulse_durations']))
            features['mean_pulse_peak'] = float(np.mean(features['pulse_peak_intensities']))
            features['max_pulse_peak'] = float(np.max(features['pulse_peak_intensities']))
            features['mean_pulse_energy'] = float(np.mean(features['pulse_energies']))
            features['max_rise_rate'] = float(np.max(features['pulse_rise_rates']))
            features['mean_rise_rate'] = float(np.mean(features['pulse_rise_rates']))

            if len(features['pulse_intervals']) > 0:
                features['mean_pulse_interval'] = float(np.mean(features['pulse_intervals']))
                features['min_pulse_interval'] = float(np.min(features['pulse_intervals']))
            else:
                features['mean_pulse_interval'] = 0.0
                features['min_pulse_interval'] = 0.0
        else:
            # 没有检测到脉冲，填充默认值
            features['mean_pulse_duration'] = 0.0
            features['max_pulse_duration'] = 0.0
            features['mean_pulse_peak'] = 0.0
            features['max_pulse_peak'] = 0.0
            features['mean_pulse_energy'] = 0.0
            features['max_rise_rate'] = 0.0
            features['mean_rise_rate'] = 0.0
            features['mean_pulse_interval'] = 0.0
            features['min_pulse_interval'] = 0.0

        return features

    def _compute_threshold(
        self,
        uv_series: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        计算背景值和检测阈值

        使用前10%的数据作为背景估计
        """
        n_background = max(10, int(len(uv_series) * 0.1))
        background_data = uv_series[:n_background]

        background_mean = np.mean(background_data)
        background_std = np.std(background_data)

        # 阈值 = 背景均值 + threshold_factor × 背景标准差
        threshold = background_mean + self.threshold_factor * background_std

        return background_mean, background_std, threshold

    def _detect_pulses(
        self,
        uv_series: np.ndarray,
        threshold: float
    ) -> List[Tuple[int, int, int]]:
        """
        检测脉冲

        返回:
            pulses: [(start_idx, end_idx, peak_idx), ...]
        """
        # 二值化
        above_threshold = uv_series > threshold

        # 查找连续区域
        pulses = []
        in_pulse = False
        start_idx = 0

        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_pulse:
                # 脉冲开始
                start_idx = i
                in_pulse = True
            elif not above_threshold[i] and in_pulse:
                # 脉冲结束
                end_idx = i - 1

                # 检查脉冲持续时间
                duration = (end_idx - start_idx + 1) * self.dt
                if duration >= self.min_pulse_duration:
                    # 找到峰值位置
                    peak_idx = start_idx + np.argmax(uv_series[start_idx:end_idx+1])
                    pulses.append((start_idx, end_idx, peak_idx))

                in_pulse = False

        # 处理最后一个脉冲
        if in_pulse:
            end_idx = len(uv_series) - 1
            duration = (end_idx - start_idx + 1) * self.dt
            if duration >= self.min_pulse_duration:
                peak_idx = start_idx + np.argmax(uv_series[start_idx:end_idx+1])
                pulses.append((start_idx, end_idx, peak_idx))

        return pulses

    def process_csv(
        self,
        csv_file: Path
    ) -> Dict:
        """
        从 CSV 文件提取 UV 特征

        参数:
            csv_file: CSV 文件路径

        返回:
            features: 特征字典
        """
        try:
            # 读取 CSV
            df = pd.read_csv(csv_file)

            # 检查必要列
            if 'uv_360nm' not in df.columns:
                raise ValueError("CSV must contain 'uv_360nm' column")

            # 提取 UV 时间序列
            uv_series = df['uv_360nm'].values

            # 提取时间序列（如果有）
            if 'time' in df.columns:
                time_series = pd.to_datetime(df['time']).values
                time_series = (time_series - time_series[0]).astype('timedelta64[ms]').astype(float) / 1000.0
            else:
                time_series = None

            # 提取特征
            features = self.extract_features(uv_series, time_series)

            # 添加元信息
            features['file_name'] = csv_file.stem

            # 解析文件名获取 sn 和 test_id
            parts = csv_file.stem.split('_')
            if len(parts) >= 3:
                features['sn'] = int(parts[2].replace('SN', ''))
                features['test_id'] = int(parts[0])
            else:
                features['sn'] = 0
                features['test_id'] = 0

            # 提取真实标签（如果有）
            if 'ton' in df.columns:
                ignition_data = df[df['ton'] == 1]
                if len(ignition_data) > 0:
                    features['true_ignition_time'] = ignition_data.iloc[0]['time']
                    features['true_thrust'] = ignition_data['thrust'].mean()
                    features['true_duration'] = len(ignition_data) * self.dt
                else:
                    features['true_ignition_time'] = None
                    features['true_thrust'] = 0.0
                    features['true_duration'] = 0.0

            return features

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None


def batch_extract_features(
    input_dir: Path,
    output_file: Path,
    extractor: UVFeatureExtractor = None
) -> pd.DataFrame:
    """
    批量提取特征

    参数:
        input_dir: 输入目录
        output_file: 输出 CSV 文件
        extractor: 特征提取器

    返回:
        df: 特征 DataFrame
    """
    if extractor is None:
        extractor = UVFeatureExtractor()

    # 获取所有 CSV 文件
    csv_files = sorted(Path(input_dir).glob('*.csv'))

    print(f"Extracting UV features from {len(csv_files)} files...")

    all_features = []
    failed_count = 0

    for i, csv_file in enumerate(csv_files):
        features = extractor.process_csv(csv_file)

        if features is not None:
            all_features.append(features)
        else:
            failed_count += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(csv_files)} files...")

    print(f"Completed: {len(all_features)} success, {failed_count} failed")

    # 转换为 DataFrame
    df = pd.DataFrame(all_features)

    # 保存
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return df


if __name__ == '__main__':
    print("=" * 70)
    print("第二阶段：UV 特征工程")
    print("=" * 70)

    # 创建特征提取器
    extractor = UVFeatureExtractor(
        threshold_factor=3.0,
        min_pulse_duration=0.1,
        sampling_rate=100.0
    )

    print(f"\n特征提取器参数：")
    print(f"  - 阈值因子: {extractor.threshold_factor}")
    print(f"  - 最小脉冲持续时间: {extractor.min_pulse_duration} 秒")
    print(f"  - 采样率: {extractor.sampling_rate} Hz")
    print()

    # 提取训练集特征
    print("\n提取训练集特征...")
    train_features = batch_extract_features(
        input_dir='data/train_with_uv',
        output_file='data/uv_features_train.csv',
        extractor=extractor
    )

    # 提取测试集特征
    print("\n提取测试集特征...")
    test_features = batch_extract_features(
        input_dir='data/test_with_uv',
        output_file='data/uv_features_test.csv',
        extractor=extractor
    )

    print("\n" + "=" * 70)
    print("第二阶段完成！")
    print("=" * 70)
    print(f"训练集特征: {len(train_features)} 个样本")
    print(f"测试集特征: {len(test_features)} 个样本")
    print(f"特征维度: {len(train_features.columns)} 列")
    print(f"输出文件: data/uv_features_train.csv, data/uv_features_test.csv")
    print("=" * 70)
