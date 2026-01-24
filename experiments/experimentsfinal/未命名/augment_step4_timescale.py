#!/usr/bin/env python3
"""
STFT数据集增强脚本 - 第四部分：时间尺度适配

解决局限性4：时间尺度差异
将100Hz短脉冲数据适配到实际变轨的分钟到小时级时间尺度
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_augmentation import (
    TimeScaleAdapter,
    TimeScaleConfig,
    MultiScaleProcessor,
    LongDurationSimulator,
    TemporalFeatureExtractor,
    MANEUVER_CHARACTERISTICS,
    ManeuverType
)


def load_step3_data():
    """加载步骤3的结果"""
    data_path = PROJECT_ROOT / 'data' / 'augmented' / 'step3_realistic_noise.csv'
    df = pd.read_csv(data_path)
    logger.info(f"加载步骤3数据: {len(df)} 条")
    return df


def apply_time_scale_adaptation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    应用时间尺度适配

    根据变轨类型将原始时间尺度（100Hz采样的短脉冲）
    适配到实际变轨的时间尺度（秒到小时级）
    """
    logger.info("开始应用时间尺度适配...")

    rng = np.random.default_rng(seed)

    # 变轨类型的时间尺度范围（秒）
    TIME_SCALE_RANGES = {
        'STATION_KEEPING': (1.0, 60.0),
        'ATTITUDE_CONTROL': (0.1, 10.0),
        'COLLISION_AVOIDANCE': (5.0, 120.0),
        'PHASE_ADJUSTMENT': (10.0, 300.0),
        'ORBIT_RAISING': (30.0, 1800.0),
        'ORBIT_LOWERING': (30.0, 1800.0),
        'HOHMANN_TRANSFER': (60.0, 3600.0),
        'PLANE_CHANGE': (120.0, 7200.0),
        'DEORBIT': (300.0, 3600.0),
        'UNKNOWN': (1.0, 60.0),
    }

    # 原始采样率和参考持续时间
    ORIGINAL_SAMPLING_RATE = 100.0  # Hz
    ORIGINAL_DURATION = 3.0  # 秒（假设原始数据约3秒）

    augmented_rows = []
    total = len(df)

    for idx, row in df.iterrows():
        if idx % 2000 == 0:
            logger.info(f"处理进度: {idx}/{total}")

        # 获取变轨类型
        maneuver_type = row.get('maneuver_type', 'UNKNOWN')
        time_range = TIME_SCALE_RANGES.get(maneuver_type, (1.0, 60.0))

        # 原始持续时间T（来自数据集，单位：采样点数/100Hz）
        T_original = row['T']  # 这是采样点数
        T_original_seconds = T_original / ORIGINAL_SAMPLING_RATE if T_original > 0 else ORIGINAL_DURATION

        # 生成目标持续时间（根据变轨类型）
        target_duration = rng.uniform(time_range[0], time_range[1])

        # 计算时间拉伸因子
        stretch_factor = target_duration / T_original_seconds if T_original_seconds > 0 else 1.0
        stretch_factor = np.clip(stretch_factor, 0.1, 1000.0)  # 限制拉伸范围

        # 计算新的采样率（保持样本数不变时）
        new_sampling_rate = ORIGINAL_SAMPLING_RATE / stretch_factor

        # 适配后的持续时间（秒）
        T_adapted = target_duration

        # 计算尺度不变特征
        # 峰值归一化（不受时间尺度影响）
        P = row['P']
        P_normalized = P / (np.max([P, 1e-10]))

        # 能量密度（单位时间能量）
        energy_density = P / T_adapted if T_adapted > 0 else 0

        # 推力曲线类型
        thrust_profile = row.get('thrust_profile', 'unknown')

        # 根据推力曲线类型计算形状因子
        if thrust_profile == 'impulsive':
            shape_factor = 0.9  # 脉冲式，能量集中
        elif thrust_profile == 'continuous':
            shape_factor = 0.5  # 连续式，能量分散
        elif thrust_profile == 'pulsed':
            shape_factor = 0.7  # 脉冲式，中等集中
        else:
            shape_factor = 0.5

        # 创建增强样本
        new_row = row.to_dict()
        new_row['T_original_samples'] = T_original
        new_row['T_original_seconds'] = T_original_seconds
        new_row['T_adapted_seconds'] = T_adapted
        new_row['T'] = T_adapted  # 更新T为适配后的秒数
        new_row['stretch_factor'] = stretch_factor
        new_row['new_sampling_rate'] = new_sampling_rate
        new_row['time_scale_min'] = time_range[0]
        new_row['time_scale_max'] = time_range[1]
        new_row['P_normalized'] = P_normalized
        new_row['energy_density'] = energy_density
        new_row['shape_factor'] = shape_factor

        augmented_rows.append(new_row)

    result_df = pd.DataFrame(augmented_rows)

    # 统计
    logger.info(f"\n增强后数据量: {len(result_df)} 条")

    logger.info("\n时间尺度统计（按变轨类型）:")
    for mt in TIME_SCALE_RANGES.keys():
        subset = result_df[result_df['maneuver_type'] == mt]
        if len(subset) > 0:
            logger.info(f"  {mt}:")
            logger.info(f"    样本数: {len(subset)}")
            logger.info(f"    持续时间范围: {subset['T_adapted_seconds'].min():.1f} - {subset['T_adapted_seconds'].max():.1f} 秒")
            logger.info(f"    平均拉伸因子: {subset['stretch_factor'].mean():.2f}")

    return result_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("步骤4: 时间尺度适配")
    logger.info("=" * 60)

    # 加载步骤3数据
    df = load_step3_data()

    # 应用时间尺度适配
    augmented_df = apply_time_scale_adaptation(df)

    # 保存
    output_dir = PROJECT_ROOT / 'data' / 'augmented'
    output_path = output_dir / 'step4_time_scale.csv'
    augmented_df.to_csv(output_path, index=False)
    logger.info(f"\n保存到: {output_path}")

    # 打印样本
    logger.info("\n样本数据预览:")
    cols = ['uid', 'maneuver_type', 'T_original_seconds', 'T_adapted_seconds', 'stretch_factor', 'energy_density']
    print(augmented_df[cols].head(10))

    return augmented_df


if __name__ == '__main__':
    result_df = main()
