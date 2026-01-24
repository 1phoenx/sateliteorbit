#!/usr/bin/env python3
"""
STFT数据集增强脚本 - 第一部分：变轨类型标签生成

解决局限性3：无变轨类型标签
基于test_mode和test_pressure组合定义变轨类型
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

from src.data_augmentation import (
    ManeuverTypeLabelMapper,
    ManeuverType,
    MANEUVER_CHARACTERISTICS
)


def load_data():
    """加载原始数据"""
    data_dir = PROJECT_ROOT / 'data'

    feature_df = pd.read_csv(data_dir / 'feature_dataset.csv')
    metadata_df = pd.read_csv(data_dir / 'metadata.csv')

    logger.info(f"加载特征数据: {len(feature_df)} 条")
    logger.info(f"加载元数据: {len(metadata_df)} 条")

    return feature_df, metadata_df


def generate_maneuver_labels(feature_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成变轨类型标签

    映射规则：
    - ssf (稳态点火): 位置保持/姿态控制
    - health_check: 位置保持
    - ramp1/ramp2: 相位调整/升轨
    - ramp3/ramp4: 霍曼转移/轨道面变换
    - onmod/offmod: 位置保持/碰撞规避
    - random_short: 姿态控制/碰撞规避
    - random_long: 升轨/降轨
    - random_mixed: 相位调整/位置保持
    """
    logger.info("开始生成变轨类型标签...")

    mapper = ManeuverTypeLabelMapper(seed=42)

    # 合并特征和元数据
    merged_df = feature_df.merge(
        metadata_df[['uid', 'test_mode', 'test_pressure', 'anomaly_code']],
        on='uid',
        how='left'
    )

    # 生成标签
    labels = []
    for idx, row in merged_df.iterrows():
        test_mode = row.get('test_mode', 'unknown')
        test_pressure = row.get('test_pressure', 18.0)
        duration = row.get('T', None)
        thrust = row.get('true_thrust', None)

        # 映射变轨类型
        maneuver_type = mapper.map_to_maneuver_type(
            str(test_mode) if pd.notna(test_mode) else 'unknown',
            float(test_pressure) if pd.notna(test_pressure) else 18.0,
            duration=float(duration) if pd.notna(duration) else None,
            thrust=float(thrust) if pd.notna(thrust) else None
        )

        # 获取变轨特征
        chars = MANEUVER_CHARACTERISTICS.get(maneuver_type)

        # 估计delta_v
        if chars and pd.notna(thrust) and pd.notna(duration):
            # 简化估计：delta_v = thrust * duration / mass
            estimated_delta_v = mapper.estimate_delta_v(
                maneuver_type,
                float(thrust),
                float(duration),
                mass=500.0  # 假设卫星质量500kg
            )
        else:
            estimated_delta_v = np.nan

        labels.append({
            'uid': row['uid'],
            'maneuver_type': maneuver_type.name,
            'maneuver_type_id': maneuver_type.value,
            'thrust_profile': chars.thrust_profile if chars else 'unknown',
            'estimated_delta_v': estimated_delta_v,
            'delta_v_min': chars.delta_v_range[0] if chars else np.nan,
            'delta_v_max': chars.delta_v_range[1] if chars else np.nan,
            'duration_min': chars.duration_range[0] if chars else np.nan,
            'duration_max': chars.duration_range[1] if chars else np.nan,
        })

    labels_df = pd.DataFrame(labels)

    # 统计
    logger.info("\n变轨类型分布:")
    type_counts = labels_df['maneuver_type'].value_counts()
    for mt, count in type_counts.items():
        logger.info(f"  {mt}: {count} ({count/len(labels_df)*100:.1f}%)")

    return labels_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("步骤1: 变轨类型标签生成")
    logger.info("=" * 60)

    # 加载数据
    feature_df, metadata_df = load_data()

    # 生成变轨类型标签
    labels_df = generate_maneuver_labels(feature_df, metadata_df)

    # 合并到特征数据
    enhanced_df = feature_df.merge(labels_df, on='uid', how='left')

    # 保存
    output_dir = PROJECT_ROOT / 'data' / 'augmented'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'step1_maneuver_labels.csv'
    enhanced_df.to_csv(output_path, index=False)
    logger.info(f"\n保存到: {output_path}")

    # 打印样本
    logger.info("\n样本数据预览:")
    print(enhanced_df[['uid', 'sn', 'P', 'T', 'R', 'maneuver_type', 'thrust_profile', 'estimated_delta_v']].head(10))

    return enhanced_df


if __name__ == '__main__':
    result_df = main()
