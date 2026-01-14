"""
数据集生成主程序
按照innovation.md文档中的方法构建完整数据集
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from data_generation import (
    DataFusion,
    DataValidator
)


def set_seed(seed: int = 42):
    """固定随机种子"""
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='卫星变轨检测数据集生成')
    parser.add_argument('--n_events', type=int, default=100, help='事件数量')
    parser.add_argument('--maneuver_ratio', type=float, default=0.3, help='变轨比例')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("卫星变轨检测数据集生成")
    print("=" * 50)

    fusion = DataFusion(seed=args.seed)
    validator = DataValidator()

    # 生成GEO光学数据集
    print("\n[1/4] 生成GEO光学融合数据集...")
    geo_df = fusion.generate_fused_geo_dataset(
        n_events=args.n_events,
        maneuver_ratio=args.maneuver_ratio
    )
    print(f"  - 样本数: {len(geo_df)}")
    print(f"  - 变轨事件: {geo_df['maneuver_label'].sum() // len(geo_df) * args.n_events}")

    # 生成LEO雷达数据集
    print("\n[2/4] 生成LEO雷达融合数据集...")
    leo_df = fusion.generate_fused_leo_dataset(
        n_events=args.n_events,
        maneuver_ratio=args.maneuver_ratio
    )
    print(f"  - 样本数: {len(leo_df)}")

    # 验证数据
    print("\n[3/4] 验证数据一致性...")
    geo_result = validator.validate_all(geo_df)
    leo_result = validator.validate_all(leo_df)
    print(f"  - GEO数据验证: {'通过' if geo_result['passed'] else '失败'}")
    print(f"  - LEO数据验证: {'通过' if leo_result['passed'] else '失败'}")

    # 保存数据
    print("\n[4/4] 保存数据集...")
    geo_df.to_csv(output_dir / 'geo_optical_dataset.csv', index=False)
    leo_df.to_csv(output_dir / 'leo_radar_dataset.csv', index=False)
    print(f"  - 已保存至: {output_dir}")

    print("\n" + "=" * 50)
    print("数据集生成完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
