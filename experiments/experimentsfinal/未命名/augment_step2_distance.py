#!/usr/bin/env python3
"""
STFT数据集增强脚本 - 第二部分：距离衰减增强

解决局限性2：无距离衰减
模拟真实场景中辐射强度随距离的平方反比衰减
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
    DistanceAttenuationModel,
    DistanceVariationSimulator,
    OrbitParameters
)


def load_step1_data():
    """加载步骤1的结果"""
    data_path = PROJECT_ROOT / 'data' / 'augmented' / 'step1_maneuver_labels.csv'
    df = pd.read_csv(data_path)
    logger.info(f"加载步骤1数据: {len(df)} 条")
    return df


def apply_distance_attenuation(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    应用距离衰减增强

    为每个样本模拟不同的观测距离，并计算衰减后的特征
    """
    logger.info("开始应用距离衰减增强...")

    rng = np.random.default_rng(seed)

    # 创建距离衰减模型
    orbit_params = OrbitParameters(
        semi_major_axis=7000e3,  # 约630km高度
        eccentricity=0.001,
        inclination=98.0,
        observer_altitude=0.0,
        observer_latitude=40.0
    )

    distance_model = DistanceAttenuationModel(
        orbit_params=orbit_params,
        atmospheric_extinction=0.1,
        reference_distance=500e3,  # 500km参考距离
        seed=seed
    )

    distance_simulator = DistanceVariationSimulator(seed=seed)

    # 距离范围配置
    DISTANCE_RANGES = {
        'near': (400e3, 600e3),      # 近距离: 400-600km
        'medium': (600e3, 1000e3),   # 中距离: 600-1000km
        'far': (1000e3, 1500e3),     # 远距离: 1000-1500km
        'very_far': (1500e3, 2500e3) # 极远距离: 1500-2500km
    }

    # 过境类型
    PASS_TYPES = ['overhead', 'horizon', 'random']

    augmented_rows = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"处理进度: {idx}/{len(df)}")

        # 原始特征
        P_original = row['P']
        T_original = row['T']
        R_original = row['R']

        # 为每个样本生成多个距离场景
        for dist_type, (dist_min, dist_max) in DISTANCE_RANGES.items():
            # 随机选择过境类型
            pass_type = rng.choice(PASS_TYPES)

            # 生成观测距离
            distance = rng.uniform(dist_min, dist_max)

            # 计算衰减因子
            attenuation_factor = (500e3 / distance) ** 2

            # 计算大气消光（简化模型）
            # 天顶角估计
            zenith_angle = np.arccos(500e3 / distance) if distance > 500e3 else 0
            airmass = 1 / (np.cos(zenith_angle) + 0.01)
            atmospheric_factor = 10 ** (-0.4 * 0.1 * airmass)

            # 总衰减
            total_attenuation = attenuation_factor * atmospheric_factor

            # 应用衰减到特征
            P_attenuated = P_original * total_attenuation
            # T不受距离影响
            T_attenuated = T_original
            # R比值基本不变（两个波长同样衰减）
            R_attenuated = R_original * (1 + rng.normal(0, 0.05))  # 添加小扰动

            # 创建增强样本
            new_row = row.to_dict()
            new_row['P_original'] = P_original
            new_row['P'] = P_attenuated
            new_row['T_original'] = T_original
            new_row['T'] = T_attenuated
            new_row['R_original'] = R_original
            new_row['R'] = R_attenuated
            new_row['distance_km'] = distance / 1e3
            new_row['distance_type'] = dist_type
            new_row['pass_type'] = pass_type
            new_row['attenuation_factor'] = total_attenuation
            new_row['atmospheric_factor'] = atmospheric_factor
            new_row['zenith_angle_deg'] = np.degrees(zenith_angle)

            augmented_rows.append(new_row)

    result_df = pd.DataFrame(augmented_rows)

    # 统计
    logger.info(f"\n增强后数据量: {len(result_df)} 条 (原始 {len(df)} 条)")
    logger.info("\n距离类型分布:")
    for dist_type in DISTANCE_RANGES.keys():
        count = len(result_df[result_df['distance_type'] == dist_type])
        logger.info(f"  {dist_type}: {count}")

    logger.info("\n衰减因子统计:")
    logger.info(f"  最小: {result_df['attenuation_factor'].min():.6f}")
    logger.info(f"  最大: {result_df['attenuation_factor'].max():.6f}")
    logger.info(f"  均值: {result_df['attenuation_factor'].mean():.6f}")

    return result_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("步骤2: 距离衰减增强")
    logger.info("=" * 60)

    # 加载步骤1数据
    df = load_step1_data()

    # 应用距离衰减
    augmented_df = apply_distance_attenuation(df)

    # 保存
    output_dir = PROJECT_ROOT / 'data' / 'augmented'
    output_path = output_dir / 'step2_distance_attenuation.csv'
    augmented_df.to_csv(output_path, index=False)
    logger.info(f"\n保存到: {output_path}")

    # 打印样本
    logger.info("\n样本数据预览:")
    cols = ['uid', 'P_original', 'P', 'distance_km', 'distance_type', 'attenuation_factor']
    print(augmented_df[cols].head(10))

    return augmented_df


if __name__ == '__main__':
    result_df = main()
