"""
完整的原始CSV数据扩充实验
流程: 原始CSV → 学习分布 → 生成合成CSV → 提取P/T/R → 合并数据集
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_csv_augmentation(
    n_synthetic: int = 1000,
    anomaly_ratio: float = 0.1,
    output_dir: str = 'data/augmented_csv'
):
    """
    运行CSV数据扩充

    Args:
        n_synthetic: 生成的合成样本数量
        anomaly_ratio: 异常样本比例
        output_dir: 输出目录
    """
    from src.timeseries_gan import ThrusterDataAugmentor

    print("=" * 70)
    print("原始CSV数据扩充实验")
    print("=" * 70)
    print(f"生成样本数: {n_synthetic}")
    print(f"异常比例: {anomaly_ratio * 100}%")
    print(f"输出目录: {output_dir}")

    # 创建扩充器
    augmentor = ThrusterDataAugmentor(
        data_dir='data',
        output_dir=output_dir,
        sampling_rate=100.0
    )

    # Step 1: 学习原始数据分布
    print("\n" + "-" * 70)
    print("Step 1: 学习原始数据分布")
    print("-" * 70)
    params = augmentor.learn_distribution(n_samples=200)

    # Step 2: 生成合成CSV文件
    print("\n" + "-" * 70)
    print("Step 2: 生成合成CSV文件")
    print("-" * 70)
    generated_files = augmentor.augment_dataset(
        n_synthetic=n_synthetic,
        anomaly_ratio=anomaly_ratio,
        prefix='synthetic'
    )

    # Step 3: 从生成的CSV提取P/T/R特征
    print("\n" + "-" * 70)
    print("Step 3: 从生成数据提取P/T/R特征")
    print("-" * 70)
    synthetic_features = augmentor.extract_features_from_generated(
        output_path='data/synthetic_features.csv'
    )

    # Step 4: 合并原始特征和合成特征
    print("\n" + "-" * 70)
    print("Step 4: 合并数据集")
    print("-" * 70)

    # 加载原始特征
    original_features = pd.read_csv('data/feature_dataset.csv')
    original_features['is_synthetic'] = 0

    # 为合成数据添加必要字段
    synthetic_features['uid'] = [f'synthetic_{i}' for i in range(len(synthetic_features))]
    synthetic_features['sn'] = 'SYN'
    synthetic_features['test_id'] = range(len(synthetic_features))
    synthetic_features['split'] = 'train'

    # 选择共同字段
    common_cols = ['uid', 'sn', 'test_id', 'split', 'P', 'T', 'R',
                   'ignition_time', 'true_thrust', 'is_anomalous', 'is_valid', 'is_synthetic']

    # 确保原始数据有所有字段
    for col in common_cols:
        if col not in original_features.columns:
            if col == 'is_synthetic':
                original_features[col] = 0
            else:
                original_features[col] = np.nan

    # 合并
    combined = pd.concat([
        original_features[common_cols],
        synthetic_features[common_cols]
    ], ignore_index=True)

    # 过滤有效样本
    combined_valid = combined[combined['is_valid'] == 1].copy()

    # 填充NaN
    combined_valid['R'] = combined_valid['R'].fillna(combined_valid['R'].median())

    # 保存
    combined.to_csv('data/feature_dataset_with_synthetic.csv', index=False)
    combined_valid.to_csv('data/feature_dataset_augmented_v2.csv', index=False)

    print(f"\n合并后数据集:")
    print(f"  原始样本: {len(original_features)}")
    print(f"  合成样本: {len(synthetic_features)}")
    print(f"  总样本数: {len(combined)}")
    print(f"  有效样本: {len(combined_valid)}")

    # Step 5: 数据质量验证
    print("\n" + "-" * 70)
    print("Step 5: 数据质量验证")
    print("-" * 70)

    # 对比原始和合成数据的分布
    print("\n原始数据 P/T/R 统计:")
    orig_valid = original_features[original_features['is_valid'] == 1]
    print(orig_valid[['P', 'T', 'R']].describe())

    print("\n合成数据 P/T/R 统计:")
    syn_valid = synthetic_features[synthetic_features['is_valid'] == 1]
    print(syn_valid[['P', 'T', 'R']].describe())

    # 计算分布偏差
    print("\n分布偏差分析:")
    for col in ['P', 'T', 'R']:
        orig_mean = orig_valid[col].mean()
        syn_mean = syn_valid[col].mean()
        deviation = abs(syn_mean - orig_mean) / orig_mean * 100
        status = "✓" if deviation < 20 else "⚠"
        print(f"  {col}: 原始均值={orig_mean:.4f}, 合成均值={syn_mean:.4f}, 偏差={deviation:.1f}% {status}")

    print("\n" + "=" * 70)
    print("扩充完成!")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  1. {output_dir}/ - 合成CSV文件目录")
    print(f"  2. data/synthetic_features.csv - 合成数据特征")
    print(f"  3. data/feature_dataset_augmented_v2.csv - 合并后的完整数据集")

    return combined_valid


def compare_augmentation_methods():
    """对比两种扩充方法"""
    print("\n" + "=" * 70)
    print("扩充方法对比")
    print("=" * 70)

    # 方法1: 直接扩充PTR (原方法)
    print("\n方法1: 直接扩充PTR特征")
    ptr_augmented = pd.read_csv('data/augmented_dataset.csv')
    print(f"  样本数: {len(ptr_augmented)}")
    print(f"  特征: P, T, R")
    print(f"  优点: 简单快速")
    print(f"  缺点: 可能不符合物理规律")

    # 方法2: 扩充原始CSV (新方法)
    print("\n方法2: 扩充原始CSV时序数据")
    if os.path.exists('data/feature_dataset_augmented_v2.csv'):
        csv_augmented = pd.read_csv('data/feature_dataset_augmented_v2.csv')
        print(f"  样本数: {len(csv_augmented)}")
        print(f"  特征: 从时序数据重新提取的P, T, R")
        print(f"  优点: 保留物理规律，特征更真实")
        print(f"  缺点: 计算量较大")
    else:
        print("  (尚未运行，请先执行 run_csv_augmentation)")

    print("\n建议: 使用方法2进行数据扩充，确保生成数据的物理合理性")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='原始CSV数据扩充')
    parser.add_argument('--n_synthetic', type=int, default=1000,
                        help='生成的合成样本数量')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1,
                        help='异常样本比例')
    parser.add_argument('--compare', action='store_true',
                        help='对比两种扩充方法')

    args = parser.parse_args()

    if args.compare:
        compare_augmentation_methods()
    else:
        run_csv_augmentation(
            n_synthetic=args.n_synthetic,
            anomaly_ratio=args.anomaly_ratio
        )


if __name__ == '__main__':
    main()
