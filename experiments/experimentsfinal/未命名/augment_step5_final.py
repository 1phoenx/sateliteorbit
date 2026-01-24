#!/usr/bin/env python3
"""
STFT数据集增强脚本 - 第五部分：综合增强与最终数据集生成

整合所有增强步骤，生成最终的增强数据集
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_step4_data():
    """加载步骤4的结果"""
    data_path = PROJECT_ROOT / 'data' / 'augmented' / 'step4_time_scale.csv'
    df = pd.read_csv(data_path)
    logger.info(f"加载步骤4数据: {len(df)} 条")
    return df


def create_final_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    创建最终增强数据集

    整理列名，添加派生特征，确保数据质量
    """
    logger.info("创建最终增强数据集...")

    rng = np.random.default_rng(seed)

    # 选择和重命名核心列
    final_columns = {
        # 标识列
        'uid': 'uid',
        'sn': 'sn',
        'test_id': 'test_id',
        'split': 'split',

        # 核心特征（增强后）
        'P': 'P',  # 辐射强度峰值（经过距离衰减和噪声）
        'T': 'T',  # 持续时间（适配后的秒数）
        'R': 'R',  # 频域强度比（经过噪声）

        # 原始特征（用于对比）
        'P_original': 'P_original',
        'T_original_seconds': 'T_original',
        'R_original': 'R_original',

        # 变轨类型标签
        'maneuver_type': 'maneuver_type',
        'maneuver_type_id': 'maneuver_type_id',
        'thrust_profile': 'thrust_profile',
        'estimated_delta_v': 'estimated_delta_v',

        # 距离相关
        'distance_km': 'distance_km',
        'distance_type': 'distance_type',
        'attenuation_factor': 'attenuation_factor',

        # 噪声相关
        'snr_condition': 'snr_condition',
        'target_snr_db': 'target_snr_db',
        'actual_snr_db': 'actual_snr_db',

        # 时间尺度相关
        'stretch_factor': 'stretch_factor',
        'energy_density': 'energy_density',
        'shape_factor': 'shape_factor',

        # 标签
        'is_anomalous': 'is_anomalous',
        'is_valid': 'is_valid',
        'true_thrust': 'true_thrust',
        'ignition_time': 'ignition_time',
    }

    # 创建最终数据框
    final_df = pd.DataFrame()

    for old_col, new_col in final_columns.items():
        if old_col in df.columns:
            final_df[new_col] = df[old_col]

    # 添加派生特征
    logger.info("计算派生特征...")

    # 1. 信号质量指标
    final_df['signal_quality'] = final_df['actual_snr_db'].apply(
        lambda x: 'high' if x > 10 else ('medium' if x > 5 else 'low')
    )

    # 2. 检测难度评估
    def assess_detection_difficulty(row):
        difficulty = 0
        # 距离因素
        if row['distance_type'] == 'very_far':
            difficulty += 3
        elif row['distance_type'] == 'far':
            difficulty += 2
        elif row['distance_type'] == 'medium':
            difficulty += 1

        # 信噪比因素
        if row['snr_condition'] == 'very_poor':
            difficulty += 3
        elif row['snr_condition'] == 'poor':
            difficulty += 2
        elif row['snr_condition'] == 'moderate':
            difficulty += 1

        # 持续时间因素（太短或太长都难）
        T = row['T']
        if T < 1 or T > 1000:
            difficulty += 1

        return 'hard' if difficulty >= 4 else ('medium' if difficulty >= 2 else 'easy')

    final_df['detection_difficulty'] = final_df.apply(assess_detection_difficulty, axis=1)

    # 3. 归一化特征（用于模型输入）
    # P归一化
    P_mean = final_df['P'].mean()
    P_std = final_df['P'].std()
    final_df['P_normalized'] = (final_df['P'] - P_mean) / (P_std + 1e-10)

    # T归一化（对数变换后归一化）
    final_df['T_log'] = np.log1p(final_df['T'])
    T_log_mean = final_df['T_log'].mean()
    T_log_std = final_df['T_log'].std()
    final_df['T_normalized'] = (final_df['T_log'] - T_log_mean) / (T_log_std + 1e-10)

    # R归一化
    R_valid = final_df['R'].dropna()
    R_mean = R_valid.mean()
    R_std = R_valid.std()
    final_df['R_normalized'] = (final_df['R'] - R_mean) / (R_std + 1e-10)

    # 4. 添加数据集元信息
    final_df['augmentation_version'] = '1.0'
    final_df['augmentation_date'] = datetime.now().strftime('%Y-%m-%d')

    # 5. 过滤无效样本
    valid_mask = (
        (final_df['P'] > 0) &
        (final_df['T'] > 0) &
        (final_df['is_valid'] == 1)
    )
    final_df['is_valid_augmented'] = valid_mask.astype(int)

    logger.info(f"有效增强样本: {valid_mask.sum()} / {len(final_df)}")

    return final_df


def generate_dataset_report(df: pd.DataFrame) -> str:
    """生成数据集报告"""
    report = []
    report.append("=" * 70)
    report.append("STFT增强数据集报告")
    report.append("=" * 70)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append("\n\n1. 数据集规模")
    report.append("-" * 40)
    report.append(f"总样本数: {len(df)}")
    report.append(f"有效样本数: {df['is_valid_augmented'].sum()}")
    report.append(f"训练集样本: {len(df[df['split'] == 'train'])}")
    report.append(f"测试集样本: {len(df[df['split'] == 'test'])}")

    report.append("\n\n2. 变轨类型分布")
    report.append("-" * 40)
    type_counts = df['maneuver_type'].value_counts()
    for mt, count in type_counts.items():
        report.append(f"  {mt}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n\n3. 距离分布")
    report.append("-" * 40)
    dist_counts = df['distance_type'].value_counts()
    for dt, count in dist_counts.items():
        report.append(f"  {dt}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n\n4. 信噪比条件分布")
    report.append("-" * 40)
    snr_counts = df['snr_condition'].value_counts()
    for sc, count in snr_counts.items():
        report.append(f"  {sc}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n\n5. 检测难度分布")
    report.append("-" * 40)
    diff_counts = df['detection_difficulty'].value_counts()
    for dc, count in diff_counts.items():
        report.append(f"  {dc}: {count} ({count/len(df)*100:.1f}%)")

    report.append("\n\n6. 特征统计")
    report.append("-" * 40)
    for col in ['P', 'T', 'R', 'distance_km', 'actual_snr_db']:
        if col in df.columns:
            report.append(f"\n  {col}:")
            report.append(f"    最小值: {df[col].min():.4f}")
            report.append(f"    最大值: {df[col].max():.4f}")
            report.append(f"    均值: {df[col].mean():.4f}")
            report.append(f"    标准差: {df[col].std():.4f}")

    report.append("\n\n7. 解决的局限性")
    report.append("-" * 40)
    report.append("  [√] 局限性1: 缺少真实噪声特性")
    report.append("      - 添加了大气闪烁、背景杂波、探测器噪声")
    report.append(f"      - 信噪比范围: {df['actual_snr_db'].min():.1f} - {df['actual_snr_db'].max():.1f} dB")

    report.append("\n  [√] 局限性2: 无距离衰减")
    report.append("      - 应用了平方反比定律和大气消光")
    report.append(f"      - 距离范围: {df['distance_km'].min():.0f} - {df['distance_km'].max():.0f} km")

    report.append("\n  [√] 局限性3: 无变轨类型标签")
    report.append(f"      - 定义了 {df['maneuver_type'].nunique()} 种变轨类型")
    report.append("      - 包含推力曲线类型和delta_v估计")

    report.append("\n  [√] 局限性4: 时间尺度差异")
    report.append(f"      - 持续时间范围: {df['T'].min():.1f} - {df['T'].max():.1f} 秒")
    report.append(f"      - 拉伸因子范围: {df['stretch_factor'].min():.1f} - {df['stretch_factor'].max():.1f}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("步骤5: 综合增强与最终数据集生成")
    logger.info("=" * 60)

    # 加载步骤4数据
    df = load_step4_data()

    # 创建最终数据集
    final_df = create_final_dataset(df)

    # 保存最终数据集
    output_dir = PROJECT_ROOT / 'data' / 'augmented'

    # 完整数据集
    final_path = output_dir / 'enhanced_dataset_full.csv'
    final_df.to_csv(final_path, index=False)
    logger.info(f"保存完整数据集: {final_path}")

    # 仅有效样本
    valid_df = final_df[final_df['is_valid_augmented'] == 1]
    valid_path = output_dir / 'enhanced_dataset_valid.csv'
    valid_df.to_csv(valid_path, index=False)
    logger.info(f"保存有效样本数据集: {valid_path}")

    # 训练集
    train_df = valid_df[valid_df['split'] == 'train']
    train_path = output_dir / 'enhanced_dataset_train.csv'
    train_df.to_csv(train_path, index=False)
    logger.info(f"保存训练集: {train_path} ({len(train_df)} 条)")

    # 测试集
    test_df = valid_df[valid_df['split'] == 'test']
    test_path = output_dir / 'enhanced_dataset_test.csv'
    test_df.to_csv(test_path, index=False)
    logger.info(f"保存测试集: {test_path} ({len(test_df)} 条)")

    # 生成报告
    report = generate_dataset_report(final_df)
    report_path = output_dir / 'dataset_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"保存数据集报告: {report_path}")

    # 打印报告
    print("\n" + report)

    # 打印样本
    logger.info("\n最终数据集样本预览:")
    cols = ['uid', 'maneuver_type', 'P', 'T', 'R', 'distance_km', 'actual_snr_db', 'detection_difficulty']
    print(final_df[cols].head(10))

    return final_df


if __name__ == '__main__':
    result_df = main()
