#!/usr/bin/env python3
"""
STFT数据集增强脚本 - 第三部分：真实噪声添加

解决局限性1：缺少真实噪声特性
添加大气闪烁、背景杂波、探测器噪声
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
    RealisticNoiseModel,
    AtmosphericScintillation,
    BackgroundClutter,
    DetectorNoise
)


def load_step2_data():
    """加载步骤2的结果"""
    data_path = PROJECT_ROOT / 'data' / 'augmented' / 'step2_distance_attenuation.csv'
    df = pd.read_csv(data_path)
    logger.info(f"加载步骤2数据: {len(df)} 条")
    return df


def apply_realistic_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    应用真实噪声增强

    噪声类型：
    1. 大气闪烁（乘性噪声）- 基于Rytov方差
    2. 背景杂波（加性噪声）- 泊松过程
    3. 探测器噪声（加性噪声）- 散粒噪声+暗电流+读出噪声
    """
    logger.info("开始应用真实噪声增强...")

    rng = np.random.default_rng(seed)

    # 信噪比配置（对应不同观测条件）
    SNR_CONFIGS = {
        'excellent': {'snr_db': 15.0, 'cn2': 1e-15, 'clutter': 0.005},  # 优秀条件
        'good': {'snr_db': 10.0, 'cn2': 5e-15, 'clutter': 0.01},       # 良好条件
        'moderate': {'snr_db': 5.0, 'cn2': 1e-14, 'clutter': 0.02},    # 中等条件
        'poor': {'snr_db': 3.0, 'cn2': 5e-14, 'clutter': 0.05},        # 较差条件
        'very_poor': {'snr_db': 1.0, 'cn2': 1e-13, 'clutter': 0.1},    # 极差条件
    }

    augmented_rows = []
    total = len(df)

    for idx, row in df.iterrows():
        if idx % 2000 == 0:
            logger.info(f"处理进度: {idx}/{total}")

        # 获取当前特征
        P = row['P']
        T = row['T']
        R = row['R']
        distance_type = row['distance_type']

        # 根据距离类型选择噪声条件
        # 远距离通常意味着更差的信噪比
        if distance_type == 'near':
            snr_weights = [0.4, 0.3, 0.2, 0.08, 0.02]
        elif distance_type == 'medium':
            snr_weights = [0.2, 0.3, 0.3, 0.15, 0.05]
        elif distance_type == 'far':
            snr_weights = [0.1, 0.2, 0.35, 0.25, 0.1]
        else:  # very_far
            snr_weights = [0.05, 0.1, 0.3, 0.35, 0.2]

        # 随机选择噪声条件
        snr_type = rng.choice(list(SNR_CONFIGS.keys()), p=snr_weights)
        snr_config = SNR_CONFIGS[snr_type]

        # 创建噪声模型
        noise_model = RealisticNoiseModel(
            snr_target=snr_config['snr_db'],
            atmospheric_params={'cn2': snr_config['cn2']},
            background_params={
                'background_level': 100.0,
                'clutter_density': snr_config['clutter']
            },
            seed=seed + idx
        )

        # 模拟噪声对特征的影响
        # 大气闪烁（乘性）
        scintillation_factor = 1.0 + rng.normal(0, np.sqrt(snr_config['cn2'] * 1e12))
        scintillation_factor = np.clip(scintillation_factor, 0.5, 2.0)

        # 背景杂波（加性）
        background_noise = rng.normal(0, snr_config['clutter'] * P) if P > 0 else 0

        # 探测器噪声（加性）
        detector_noise = rng.normal(0, 0.01 * P) if P > 0 else 0

        # 应用噪声到P
        P_noisy = P * scintillation_factor + background_noise + detector_noise
        P_noisy = max(0, P_noisy)  # 确保非负

        # R受噪声影响（两个波长的噪声不完全相关）
        R_noise_factor = 1.0 + rng.normal(0, 0.1 / (snr_config['snr_db'] + 1))
        R_noisy = R * R_noise_factor if pd.notna(R) else R

        # T基本不受噪声影响（时间测量相对稳定）
        T_noisy = T

        # 计算实际信噪比
        if P > 0:
            noise_power = (P_noisy - P) ** 2
            signal_power = P ** 2
            actual_snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        else:
            actual_snr = 0

        # 创建增强样本
        new_row = row.to_dict()
        new_row['P_before_noise'] = P
        new_row['P'] = P_noisy
        new_row['R_before_noise'] = R
        new_row['R'] = R_noisy
        new_row['T'] = T_noisy
        new_row['snr_condition'] = snr_type
        new_row['target_snr_db'] = snr_config['snr_db']
        new_row['actual_snr_db'] = actual_snr
        new_row['scintillation_factor'] = scintillation_factor
        new_row['cn2'] = snr_config['cn2']

        augmented_rows.append(new_row)

    result_df = pd.DataFrame(augmented_rows)

    # 统计
    logger.info(f"\n增强后数据量: {len(result_df)} 条")
    logger.info("\n信噪比条件分布:")
    for snr_type in SNR_CONFIGS.keys():
        count = len(result_df[result_df['snr_condition'] == snr_type])
        logger.info(f"  {snr_type}: {count} ({count/len(result_df)*100:.1f}%)")

    logger.info("\n实际信噪比统计:")
    logger.info(f"  最小: {result_df['actual_snr_db'].min():.2f} dB")
    logger.info(f"  最大: {result_df['actual_snr_db'].max():.2f} dB")
    logger.info(f"  均值: {result_df['actual_snr_db'].mean():.2f} dB")

    return result_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("步骤3: 真实噪声添加")
    logger.info("=" * 60)

    # 加载步骤2数据
    df = load_step2_data()

    # 应用噪声
    augmented_df = apply_realistic_noise(df)

    # 保存
    output_dir = PROJECT_ROOT / 'data' / 'augmented'
    output_path = output_dir / 'step3_realistic_noise.csv'
    augmented_df.to_csv(output_path, index=False)
    logger.info(f"\n保存到: {output_path}")

    # 打印样本
    logger.info("\n样本数据预览:")
    cols = ['uid', 'P_before_noise', 'P', 'snr_condition', 'target_snr_db', 'actual_snr_db']
    print(augmented_df[cols].head(10))

    return augmented_df


if __name__ == '__main__':
    result_df = main()
