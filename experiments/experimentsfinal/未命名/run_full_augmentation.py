#!/usr/bin/env python3
"""
STFT数据集一键增强脚本

运行所有增强步骤，生成符合项目需求的增强数据集

解决的四个局限性：
1. 缺少真实噪声特性 -> 添加大气闪烁、背景杂波、探测器噪声
2. 无距离衰减 -> 应用平方反比定律和大气消光
3. 无变轨类型标签 -> 基于test_mode和test_pressure映射变轨类型
4. 时间尺度差异 -> 将100Hz短脉冲适配到实际变轨时间尺度

使用方法：
    python experiments/run_full_augmentation.py

输出文件：
    data/augmented/enhanced_dataset_full.csv   - 完整增强数据集
    data/augmented/enhanced_dataset_valid.csv  - 仅有效样本
    data/augmented/enhanced_dataset_train.csv  - 训练集
    data/augmented/enhanced_dataset_test.csv   - 测试集
    data/augmented/dataset_report.txt          - 数据集报告
"""
import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent


def run_step(step_name: str, script_name: str) -> bool:
    """运行单个增强步骤"""
    print(f"\n{'='*60}")
    print(f"执行: {step_name}")
    print(f"{'='*60}")

    script_path = PROJECT_ROOT / 'experiments' / script_name
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )

    elapsed = time.time() - start_time
    print(f"\n耗时: {elapsed:.2f} 秒")

    if result.returncode != 0:
        print(f"错误: {step_name} 执行失败!")
        return False

    return True


def main():
    """主函数"""
    print("=" * 70)
    print("STFT数据集增强流水线")
    print("=" * 70)
    print("\n解决的局限性:")
    print("  1. 缺少真实噪声特性 -> 添加大气闪烁、背景杂波、探测器噪声")
    print("  2. 无距离衰减 -> 应用平方反比定律和大气消光")
    print("  3. 无变轨类型标签 -> 基于test_mode和test_pressure映射变轨类型")
    print("  4. 时间尺度差异 -> 将100Hz短脉冲适配到实际变轨时间尺度")

    total_start = time.time()

    steps = [
        ("步骤1: 变轨类型标签生成", "augment_step1_maneuver_labels.py"),
        ("步骤2: 距离衰减增强", "augment_step2_distance.py"),
        ("步骤3: 真实噪声添加", "augment_step3_noise.py"),
        ("步骤4: 时间尺度适配", "augment_step4_timescale.py"),
        ("步骤5: 综合增强与最终数据集生成", "augment_step5_final.py"),
    ]

    for step_name, script_name in steps:
        if not run_step(step_name, script_name):
            print("\n增强流水线执行失败!")
            return 1

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("增强流水线执行完成!")
    print("=" * 70)
    print(f"\n总耗时: {total_elapsed:.2f} 秒")

    # 显示输出文件
    output_dir = PROJECT_ROOT / 'data' / 'augmented'
    print(f"\n输出文件目录: {output_dir}")
    print("\n生成的文件:")
    for f in sorted(output_dir.glob('*.csv')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

    # 显示报告
    report_path = output_dir / 'dataset_report.txt'
    if report_path.exists():
        print(f"\n数据集报告: {report_path}")
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())

    return 0


if __name__ == '__main__':
    sys.exit(main())
