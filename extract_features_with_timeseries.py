"""
改进的特征提取脚本 - 包含时序特征
===========================================

提取特征:
- 统计特征: P (峰值), T (持续时间), R (频域强度比)
- 时序特征: 300点推力序列 (固定长度采样)
- 标签信息: ignition_time, true_thrust, is_anomalous

作者: Claude Code
日期: 2026-01-23
"""

import pandas as pd
import numpy as np
from scipy import signal, interpolate
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_fft_ratio(thrust_sequence, sampling_rate=100):
    """
    计算频域强度比 R = 基频幅值 / 二次谐波幅值

    参数:
        thrust_sequence: 推力时序数据
        sampling_rate: 采样率 (Hz)

    返回:
        R: 频域强度比
    """
    if len(thrust_sequence) < 10:
        return np.nan

    # 去除直流分量
    thrust_detrend = thrust_sequence - np.mean(thrust_sequence)

    # FFT
    fft_result = np.fft.fft(thrust_detrend)
    fft_freq = np.fft.fftfreq(len(thrust_sequence), 1/sampling_rate)

    # 只取正频率部分
    positive_freq_idx = fft_freq > 0
    fft_magnitude = np.abs(fft_result[positive_freq_idx])
    fft_freq_positive = fft_freq[positive_freq_idx]

    if len(fft_magnitude) < 2:
        return np.nan

    # 找基频（最大幅值对应的频率）
    fundamental_idx = np.argmax(fft_magnitude)
    fundamental_freq = fft_freq_positive[fundamental_idx]
    fundamental_magnitude = fft_magnitude[fundamental_idx]

    # 找二次谐波（2倍基频附近）
    second_harmonic_freq = 2 * fundamental_freq
    second_harmonic_idx = np.argmin(np.abs(fft_freq_positive - second_harmonic_freq))
    second_harmonic_magnitude = fft_magnitude[second_harmonic_idx]

    # 计算比值
    if second_harmonic_magnitude > 0:
        R = fundamental_magnitude / second_harmonic_magnitude
    else:
        R = np.nan

    return R


def sample_fixed_length(sequence, target_len=300, method='linear'):
    """
    将不同长度的序列采样到固定长度

    参数:
        sequence: 输入序列
        target_len: 目标长度
        method: 插值方法 ('linear', 'cubic')

    返回:
        采样后的固定长度序列
    """
    sequence = np.array(sequence)
    current_len = len(sequence)

    if current_len == target_len:
        return sequence

    if current_len < 2:
        # 序列太短，用零填充
        return np.zeros(target_len)

    # 插值到目标长度
    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, target_len)

    try:
        if method == 'linear':
            interpolator = interpolate.interp1d(x_old, sequence, kind='linear')
        else:
            interpolator = interpolate.interp1d(x_old, sequence, kind='cubic')

        sampled_sequence = interpolator(x_new)
    except Exception as e:
        # 插值失败，使用简单的重采样
        indices = np.linspace(0, current_len - 1, target_len).astype(int)
        sampled_sequence = sequence[indices]

    return sampled_sequence


def extract_features_with_timeseries(csv_file, target_len=300):
    """
    从CSV文件提取完整特征（统计特征 + 时序特征）

    参数:
        csv_file: CSV文件路径
        target_len: 时序特征的目标长度

    返回:
        特征字典
    """
    try:
        # 读取CSV
        df = pd.read_csv(csv_file)

        # 提取元信息
        file_name = Path(csv_file).stem

        # 解析文件名获取sn和test_id (格式: XXXXX_YYY_SNZZ_...)
        parts = file_name.split('_')
        if len(parts) >= 3:
            sn_str = parts[2]  # 如 'SN13'
            sn = int(sn_str.replace('SN', ''))
            test_id = int(parts[0])  # 测试ID
        else:
            sn = 0
            test_id = 0

        # 提取点火时段数据 (ton=1)
        ignition_data = df[df['ton'] == 1].copy()

        if len(ignition_data) == 0:
            return None  # 无有效点火数据

        # 提取背景数据 (ton=0)
        background_data = df[df['ton'] == 0]

        # 计算背景值和标准差
        if len(background_data) > 0:
            background_thrust = background_data['thrust'].mean()
            background_std = background_data['thrust'].std()
        else:
            background_thrust = 0
            background_std = 0.001

        # 统计特征提取
        thrust_values = ignition_data['thrust'].values

        # P: 峰值推力（去除背景）
        P = thrust_values.max() - background_thrust

        # T: 持续时间（秒）
        T = len(ignition_data) * 0.01  # 100Hz采样

        # R: 频域强度比
        R = compute_fft_ratio(thrust_values)

        # 有效性检查
        is_valid = (P > 0) and (T >= 0.1) and (not np.isnan(R))

        # 点火时刻（第一个ton=1的时刻）
        if len(ignition_data) > 0:
            ignition_time = ignition_data.iloc[0]['time']
        else:
            ignition_time = None

        # 平均推力
        true_thrust = thrust_values.mean()

        # 异常检测（基于anomaly_code）
        if 'anomaly_code' in df.columns:
            anomaly_codes = df[df['ton'] == 1]['anomaly_code'].dropna()
            is_anomalous = len(anomaly_codes) > 0 and any(anomaly_codes != '')
        else:
            is_anomalous = False

        # 时序特征提取（关键改进！）
        thrust_sequence = sample_fixed_length(thrust_values, target_len=target_len)

        # 同时提取质量流率时序（辅助特征）
        if 'mfr' in ignition_data.columns:
            mfr_values = ignition_data['mfr'].values
            mfr_sequence = sample_fixed_length(mfr_values, target_len=target_len)
        else:
            mfr_sequence = np.zeros(target_len)

        # 构建特征字典
        features = {
            'file_name': file_name,
            'sn': sn,
            'test_id': test_id,
            'P': float(P),
            'T': float(T),
            'R': float(R) if not np.isnan(R) else 0.0,
            'ignition_time': str(ignition_time),
            'true_thrust': float(true_thrust),
            'is_anomalous': bool(is_anomalous),
            'is_valid': bool(is_valid),
            'thrust_sequence': thrust_sequence.tolist(),
            'mfr_sequence': mfr_sequence.tolist(),
            'sequence_length': target_len,
            'background_thrust': float(background_thrust),
            'background_std': float(background_std),
            'signal_to_noise_ratio': float(P / background_std) if background_std > 0 else 0.0
        }

        return features

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def process_dataset(data_dir, output_dir, split='train', target_len=300):
    """
    批量处理数据集

    参数:
        data_dir: 数据目录（包含CSV文件）
        output_dir: 输出目录
        split: 数据集划分 ('train' or 'test')
        target_len: 时序特征长度
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有CSV文件
    csv_files = sorted(data_path.glob('*.csv'))

    print(f"Processing {split} dataset: {len(csv_files)} files")

    # 存储所有特征
    all_features = []
    failed_files = []

    # 批量处理
    for csv_file in tqdm(csv_files, desc=f"Extracting {split} features"):
        features = extract_features_with_timeseries(csv_file, target_len=target_len)

        if features is not None and features['is_valid']:
            all_features.append(features)
        else:
            failed_files.append(str(csv_file))

    print(f"\nSuccessfully processed: {len(all_features)} samples")
    print(f"Failed/Invalid: {len(failed_files)} samples")

    # 转换为numpy数组（方便后续训练）
    if len(all_features) > 0:
        # 分离统计特征和时序特征
        statistical_features = np.array([
            [f['P'], f['T'], f['R']] for f in all_features
        ])

        thrust_sequences = np.array([
            f['thrust_sequence'] for f in all_features
        ])

        mfr_sequences = np.array([
            f['mfr_sequence'] for f in all_features
        ])

        labels_anomaly = np.array([
            f['is_anomalous'] for f in all_features
        ], dtype=np.int32)

        labels_thrust = np.array([
            f['true_thrust'] for f in all_features
        ], dtype=np.float32)

        # 元信息
        metadata = pd.DataFrame([
            {
                'file_name': f['file_name'],
                'sn': f['sn'],
                'test_id': f['test_id'],
                'ignition_time': f['ignition_time'],
                'P': f['P'],
                'T': f['T'],
                'R': f['R'],
                'true_thrust': f['true_thrust'],
                'is_anomalous': f['is_anomalous'],
                'snr': f['signal_to_noise_ratio']
            }
            for f in all_features
        ])

        # 保存为NPY格式（高效存储）
        np.save(output_path / f'{split}_statistical_features.npy', statistical_features)
        np.save(output_path / f'{split}_thrust_sequences.npy', thrust_sequences)
        np.save(output_path / f'{split}_mfr_sequences.npy', mfr_sequences)
        np.save(output_path / f'{split}_labels_anomaly.npy', labels_anomaly)
        np.save(output_path / f'{split}_labels_thrust.npy', labels_thrust)

        # 保存元数据为CSV
        metadata.to_csv(output_path / f'{split}_metadata.csv', index=False)

        # 保存失败文件列表
        if failed_files:
            with open(output_path / f'{split}_failed_files.txt', 'w') as f:
                f.write('\n'.join(failed_files))

        # 保存数据集信息
        dataset_info = {
            'split': split,
            'total_samples': len(all_features),
            'failed_samples': len(failed_files),
            'feature_dimensions': {
                'statistical': list(statistical_features.shape),
                'thrust_sequence': list(thrust_sequences.shape),
                'mfr_sequence': list(mfr_sequences.shape)
            },
            'sequence_length': target_len,
            'anomaly_count': int(labels_anomaly.sum()),
            'normal_count': int(len(labels_anomaly) - labels_anomaly.sum())
        }

        with open(output_path / f'{split}_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\nDataset saved to {output_path}")
        print(f"  - Statistical features: {statistical_features.shape}")
        print(f"  - Thrust sequences: {thrust_sequences.shape}")
        print(f"  - MFR sequences: {mfr_sequences.shape}")
        print(f"  - Anomaly samples: {labels_anomaly.sum()}/{len(labels_anomaly)}")

    return len(all_features), len(failed_files)


if __name__ == '__main__':
    # 处理训练集
    print("="*70)
    print("Task 1: Extract features with time series")
    print("="*70)

    train_count, train_failed = process_dataset(
        data_dir='data/train',
        output_dir='data/features_timeseries',
        split='train',
        target_len=300
    )

    print("\n" + "="*70)

    # 处理测试集
    test_count, test_failed = process_dataset(
        data_dir='data/test',
        output_dir='data/features_timeseries',
        split='test',
        target_len=300
    )

    print("\n" + "="*70)
    print("Feature extraction completed!")
    print("="*70)
    print(f"Training set: {train_count} samples ({train_failed} failed)")
    print(f"Test set: {test_count} samples ({test_failed} failed)")
    print(f"Output directory: data/features_timeseries/")
