#!/usr/bin/env python3
"""
数据增强模块使用示例

演示如何使用数据增强模块解决STFT数据集的四个局限性
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_augmentation import (
    DataAugmentationPipeline,
    AugmentationConfig,
    create_augmentation_pipeline,
    RealisticNoiseModel,
    DistanceAttenuationModel,
    ManeuverTypeLabelMapper,
    ManeuverType,
    TimeScaleAdapter,
    LongDurationSimulator
)


def demo_noise_model():
    """演示真实噪声模型"""
    print("=" * 60)
    print("1. 真实噪声模型演示")
    print("=" * 60)

    # 创建干净信号
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    clean_signal = 1000 * np.exp(-0.5 * ((t - 5) / 1) ** 2)  # 高斯脉冲

    # 创建噪声模型
    noise_model = RealisticNoiseModel(
        snr_target=5.0,  # 5dB信噪比
        atmospheric_params={'cn2': 1e-14},
        background_params={'background_level': 100, 'clutter_density': 0.02},
        seed=42
    )

    # 应用噪声
    noisy_signal, noise_components = noise_model.apply_noise(
        clean_signal, sampling_rate=100.0
    )

    print(f"原始信号峰值: {np.max(clean_signal):.2f}")
    print(f"带噪信号峰值: {np.max(noisy_signal):.2f}")
    print(f"大气闪烁标准差: {np.std(noise_components['atmospheric']):.4f}")
    print(f"背景杂波标准差: {np.std(noise_components['background']):.2f}")
    print(f"探测器噪声标准差: {np.std(noise_components['detector']):.2f}")

    # 计算实际SNR
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    actual_snr = 10 * np.log10(signal_power / noise_power)
    print(f"实际信噪比: {actual_snr:.2f} dB")

    return clean_signal, noisy_signal, t


def demo_distance_attenuation():
    """演示距离衰减模型"""
    print("\n" + "=" * 60)
    print("2. 距离衰减模型演示")
    print("=" * 60)

    # 创建源信号
    n_samples = 1000
    source_intensity = np.full(n_samples, 10000.0)  # 恒定源强度

    # 创建距离衰减模型
    distance_model = DistanceAttenuationModel(
        reference_distance=500e3,  # 500km参考距离
        atmospheric_extinction=0.1,
        seed=42
    )

    # 模拟不同距离
    distances = [400e3, 800e3, 1200e3, 1600e3, 2000e3]

    print("距离衰减效果:")
    print("-" * 40)
    for dist in distances:
        attenuated, params = distance_model.apply_attenuation(
            source_intensity,
            distance=np.full(n_samples, dist),
            include_atmospheric=True
        )
        attn_factor = np.mean(params['attenuation_factor'])
        print(f"距离 {dist/1e3:.0f} km: 衰减因子 = {attn_factor:.6f}, "
              f"观测强度 = {np.mean(attenuated):.2f}")

    return distances


def demo_maneuver_type_mapping():
    """演示变轨类型标签映射"""
    print("\n" + "=" * 60)
    print("3. 变轨类型标签映射演示")
    print("=" * 60)

    mapper = ManeuverTypeLabelMapper(seed=42)

    # 测试不同test_mode和test_pressure组合
    test_cases = [
        ('ssf', 24.0),
        ('ssf', 9.0),
        ('ramp1', 24.0),
        ('ramp4', 24.0),
        ('health_check', 18.0),
        ('random_short', 21.0),
        ('random_long', 24.0),
        ('onmod', 15.0),
    ]

    print("test_mode + test_pressure -> 变轨类型映射:")
    print("-" * 60)
    for test_mode, test_pressure in test_cases:
        maneuver_type = mapper.map_to_maneuver_type(test_mode, test_pressure)
        print(f"{test_mode:15s} + {test_pressure:5.1f} bar -> {maneuver_type.name}")

    # 演示delta_v估计
    print("\n变轨类型特征:")
    print("-" * 60)
    for mt in [ManeuverType.STATION_KEEPING, ManeuverType.HOHMANN_TRANSFER,
               ManeuverType.PHASE_ADJUSTMENT, ManeuverType.COLLISION_AVOIDANCE]:
        from src.data_augmentation import MANEUVER_CHARACTERISTICS
        chars = MANEUVER_CHARACTERISTICS[mt]
        print(f"{mt.name:20s}: Δv={chars.delta_v_range}, "
              f"T={chars.duration_range}, profile={chars.thrust_profile}")


def demo_time_scale_adaptation():
    """演示时间尺度适配"""
    print("\n" + "=" * 60)
    print("4. 时间尺度适配演示")
    print("=" * 60)

    # 创建短脉冲信号（100Hz采样，3秒）
    original_rate = 100.0
    original_duration = 3.0
    n_samples = int(original_rate * original_duration)
    t = np.arange(n_samples) / original_rate

    # 高斯脉冲
    short_signal = np.exp(-0.5 * ((t - 1.5) / 0.3) ** 2)

    print(f"原始信号: {original_duration}秒, {original_rate}Hz采样, "
          f"{n_samples}个样本")

    # 时间尺度适配
    adapter = TimeScaleAdapter(seed=42)

    # 拉伸到不同时间尺度
    target_durations = [30.0, 300.0, 1800.0]  # 30秒, 5分钟, 30分钟

    print("\n时间尺度拉伸:")
    print("-" * 50)
    for target_duration in target_durations:
        stretched, new_rate = adapter.adapt_to_maneuver_duration(
            short_signal, original_duration, target_duration, original_rate
        )
        print(f"目标时长 {target_duration:6.0f}秒: "
              f"新采样率 {new_rate:.4f}Hz, "
              f"样本数 {len(stretched)}")

    # 长时间变轨模拟
    print("\n长时间变轨模拟:")
    print("-" * 50)
    simulator = LongDurationSimulator(seed=42)

    for profile in ['constant', 'bell', 'trapezoidal']:
        data = simulator.simulate_continuous_thrust(
            duration=600.0,  # 10分钟
            sampling_rate=1.0,
            thrust_profile=profile,
            thrust_level=1.0
        )
        print(f"推力曲线 {profile:12s}: "
              f"峰值={np.max(data['thrust']):.3f}, "
              f"均值={np.mean(data['thrust']):.3f}")


def demo_full_pipeline():
    """演示完整增强流水线"""
    print("\n" + "=" * 60)
    print("5. 完整增强流水线演示")
    print("=" * 60)

    # 创建流水线
    pipeline = create_augmentation_pipeline(snr_db=5.0, enable_all=True, seed=42)

    # 打印配置摘要
    summary = pipeline.get_augmentation_summary()
    print("流水线配置:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 创建测试信号
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    test_signal = 1000 * np.exp(-0.5 * ((t - 5) / 1) ** 2)

    # 增强信号
    result = pipeline.augment_signal(
        test_signal,
        sampling_rate=100.0,
        maneuver_type='station_keeping'
    )

    print("\n信号增强结果:")
    print(f"  原始信号峰值: {np.max(result['original_signal']):.2f}")
    print(f"  增强信号峰值: {np.max(result['augmented_signal']):.2f}")

    if 'scale_invariant_features' in result:
        print("\n尺度不变特征:")
        for key, value in result['scale_invariant_features'].items():
            print(f"  {key}: {value:.4f}")

    # 模拟真实观测
    print("\n模拟真实观测场景:")
    obs_data = pipeline.simulate_realistic_observation(
        source_intensity=10000.0,
        duration=600.0,  # 10分钟
        maneuver_type='hohmann_transfer',
        sampling_rate=1.0
    )
    print(f"  源信号峰值: {np.max(obs_data['source_signal']):.2f}")
    print(f"  观测信号峰值: {np.max(obs_data['observed_signal']):.2f}")
    print(f"  距离范围: {np.min(obs_data['distance'])/1e3:.0f} - "
          f"{np.max(obs_data['distance'])/1e3:.0f} km")


def demo_dataset_augmentation():
    """演示数据集增强"""
    print("\n" + "=" * 60)
    print("6. 数据集增强演示")
    print("=" * 60)

    # 加载数据
    data_dir = Path(__file__).parent.parent / 'data'
    feature_path = data_dir / 'feature_dataset.csv'
    metadata_path = data_dir / 'metadata.csv'

    if not feature_path.exists():
        print(f"特征数据集不存在: {feature_path}")
        return

    feature_df = pd.read_csv(feature_path)
    metadata_df = pd.read_csv(metadata_path)

    print(f"原始数据集大小: {len(feature_df)}")

    # 创建流水线
    pipeline = create_augmentation_pipeline(snr_db=5.0, seed=42)

    # 生成变轨类型标签
    labels_df = pipeline.generate_maneuver_labels(metadata_df, feature_df)
    print(f"\n变轨类型分布:")
    print(labels_df['maneuver_type'].value_counts())

    # 增强数据集（只取前100条演示）
    sample_df = feature_df.head(100)
    sample_meta = metadata_df[metadata_df['uid'].isin(sample_df['uid'])]

    augmented_df = pipeline.augment_dataset(
        sample_df,
        sample_meta,
        augmentation_factor=2
    )

    print(f"\n增强后数据集大小: {len(augmented_df)}")
    print(f"增强类型分布:")
    print(augmented_df['augmentation_type'].value_counts())


def main():
    """主函数"""
    print("STFT数据集局限性解决方案演示")
    print("=" * 60)
    print("解决的问题:")
    print("1. 缺少真实噪声特性 -> RealisticNoiseModel")
    print("2. 无距离衰减 -> DistanceAttenuationModel")
    print("3. 无变轨类型标签 -> ManeuverTypeLabelMapper")
    print("4. 时间尺度差异 -> TimeScaleAdapter")
    print("=" * 60)

    # 运行各演示
    demo_noise_model()
    demo_distance_attenuation()
    demo_maneuver_type_mapping()
    demo_time_scale_adaptation()
    demo_full_pipeline()
    demo_dataset_augmentation()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
