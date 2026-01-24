"""
点火时刻定位优化验证实验
对比优化前后的点火时刻检测精度
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_sample_data(data_dir: str, metadata_path: str, n_samples: int = 100) -> List[Dict]:
    """加载样本数据"""
    metadata = pd.read_csv(metadata_path)

    # 随机选择样本
    if len(metadata) > n_samples:
        metadata = metadata.sample(n=n_samples, random_state=42)

    samples = []
    for _, row in metadata.iterrows():
        filename = row['filename']
        train_path = Path(data_dir) / 'train' / filename
        test_path = Path(data_dir) / 'test' / filename

        file_path = train_path if train_path.exists() else test_path
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        if 'thrust' not in df.columns or 'ton' not in df.columns:
            continue

        samples.append({
            'uid': row['uid'],
            'filename': filename,
            'thrust': df['thrust'].values,
            'ton': df['ton'].values,
            'time': df['time'].values if 'time' in df.columns else np.arange(len(df)) / 100.0
        })

    return samples


def detect_ignition_baseline(thrust: np.ndarray, ton: np.ndarray, sampling_rate: float = 100.0) -> float:
    """基线方法: 简单阈值检测"""
    # 计算背景
    background_mask = ton == 0
    if np.sum(background_mask) > 0:
        B_thrust = np.mean(thrust[background_mask])
        sigma = np.std(thrust[background_mask])
    else:
        B_thrust = np.percentile(thrust, 10)
        sigma = np.std(thrust) * 0.1

    threshold = B_thrust + 3 * sigma
    above_threshold = thrust > threshold

    if not np.any(above_threshold):
        return np.nan

    first_idx = np.argmax(above_threshold)
    return first_idx / sampling_rate


def detect_ignition_optimized(thrust: np.ndarray, ton: np.ndarray, sampling_rate: float = 100.0) -> float:
    """优化方法: 集成检测"""
    from src.ignition_detector import detect_ignition_ensemble

    result = detect_ignition_ensemble(thrust, ton, sampling_rate)
    return result['ignition_time']


def get_ground_truth_ignition(ton: np.ndarray, sampling_rate: float = 100.0) -> float:
    """获取真实点火时刻 (基于ton信号)"""
    # ton=1 表示推力器开启
    firing_mask = ton == 1
    if not np.any(firing_mask):
        return np.nan

    first_firing_idx = np.argmax(firing_mask)
    return first_firing_idx / sampling_rate


def evaluate_ignition_detection(
    samples: List[Dict],
    sampling_rate: float = 100.0
) -> Dict:
    """评估点火时刻检测性能"""

    baseline_errors = []
    optimized_errors = []
    baseline_times = []
    optimized_times = []

    for sample in samples:
        thrust = sample['thrust']
        ton = sample['ton']

        # 获取真实点火时刻
        true_ignition = get_ground_truth_ignition(ton, sampling_rate)
        if np.isnan(true_ignition):
            continue

        # 基线方法
        start = time.time()
        baseline_pred = detect_ignition_baseline(thrust, ton, sampling_rate)
        baseline_times.append(time.time() - start)

        if not np.isnan(baseline_pred):
            baseline_errors.append(abs(baseline_pred - true_ignition))

        # 优化方法
        start = time.time()
        optimized_pred = detect_ignition_optimized(thrust, ton, sampling_rate)
        optimized_times.append(time.time() - start)

        if not np.isnan(optimized_pred):
            optimized_errors.append(abs(optimized_pred - true_ignition))

    # 计算统计指标
    results = {
        'baseline': {
            'mae': np.mean(baseline_errors) if baseline_errors else np.nan,
            'rmse': np.sqrt(np.mean(np.array(baseline_errors)**2)) if baseline_errors else np.nan,
            'median_error': np.median(baseline_errors) if baseline_errors else np.nan,
            'p90_error': np.percentile(baseline_errors, 90) if baseline_errors else np.nan,
            'hit_rate_1s': np.mean(np.array(baseline_errors) <= 1.0) if baseline_errors else 0,
            'hit_rate_0.5s': np.mean(np.array(baseline_errors) <= 0.5) if baseline_errors else 0,
            'avg_time': np.mean(baseline_times) if baseline_times else np.nan,
            'n_valid': len(baseline_errors)
        },
        'optimized': {
            'mae': np.mean(optimized_errors) if optimized_errors else np.nan,
            'rmse': np.sqrt(np.mean(np.array(optimized_errors)**2)) if optimized_errors else np.nan,
            'median_error': np.median(optimized_errors) if optimized_errors else np.nan,
            'p90_error': np.percentile(optimized_errors, 90) if optimized_errors else np.nan,
            'hit_rate_1s': np.mean(np.array(optimized_errors) <= 1.0) if optimized_errors else 0,
            'hit_rate_0.5s': np.mean(np.array(optimized_errors) <= 0.5) if optimized_errors else 0,
            'avg_time': np.mean(optimized_times) if optimized_times else np.nan,
            'n_valid': len(optimized_errors)
        }
    }

    return results


def print_results(results: Dict):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print("点火时刻定位精度对比")
    print("=" * 70)

    print(f"\n{'指标':<20} {'基线方法':<15} {'优化方法':<15} {'提升':<15}")
    print("-" * 70)

    baseline = results['baseline']
    optimized = results['optimized']

    # MAE
    improvement = (baseline['mae'] - optimized['mae']) / baseline['mae'] * 100 if baseline['mae'] > 0 else 0
    print(f"{'MAE (秒)':<20} {baseline['mae']:<15.4f} {optimized['mae']:<15.4f} {improvement:>+.1f}%")

    # RMSE
    improvement = (baseline['rmse'] - optimized['rmse']) / baseline['rmse'] * 100 if baseline['rmse'] > 0 else 0
    print(f"{'RMSE (秒)':<20} {baseline['rmse']:<15.4f} {optimized['rmse']:<15.4f} {improvement:>+.1f}%")

    # Median Error
    improvement = (baseline['median_error'] - optimized['median_error']) / baseline['median_error'] * 100 if baseline['median_error'] > 0 else 0
    print(f"{'中位数误差 (秒)':<20} {baseline['median_error']:<15.4f} {optimized['median_error']:<15.4f} {improvement:>+.1f}%")

    # P90 Error
    improvement = (baseline['p90_error'] - optimized['p90_error']) / baseline['p90_error'] * 100 if baseline['p90_error'] > 0 else 0
    print(f"{'P90误差 (秒)':<20} {baseline['p90_error']:<15.4f} {optimized['p90_error']:<15.4f} {improvement:>+.1f}%")

    # Hit Rate @ 1s
    improvement = (optimized['hit_rate_1s'] - baseline['hit_rate_1s']) * 100
    print(f"{'命中率@1秒':<20} {baseline['hit_rate_1s']*100:<14.1f}% {optimized['hit_rate_1s']*100:<14.1f}% {improvement:>+.1f}pp")

    # Hit Rate @ 0.5s
    improvement = (optimized['hit_rate_0.5s'] - baseline['hit_rate_0.5s']) * 100
    print(f"{'命中率@0.5秒':<20} {baseline['hit_rate_0.5s']*100:<14.1f}% {optimized['hit_rate_0.5s']*100:<14.1f}% {improvement:>+.1f}pp")

    # Inference Time
    print(f"{'平均推理时间 (ms)':<20} {baseline['avg_time']*1000:<15.2f} {optimized['avg_time']*1000:<15.2f}")

    print("-" * 70)
    print(f"有效样本数: 基线={baseline['n_valid']}, 优化={optimized['n_valid']}")

    # 判断是否达标
    print("\n" + "=" * 70)
    print("目标达成情况")
    print("=" * 70)

    target_mae = 1.0  # 目标: MAE ≤ 1秒
    if optimized['mae'] <= target_mae:
        print(f"✓ 点火时刻MAE: {optimized['mae']:.4f}秒 ≤ {target_mae}秒 [达标]")
    else:
        print(f"✗ 点火时刻MAE: {optimized['mae']:.4f}秒 > {target_mae}秒 [未达标]")

    target_hit_rate = 0.9  # 目标: 命中率@1秒 ≥ 90%
    if optimized['hit_rate_1s'] >= target_hit_rate:
        print(f"✓ 命中率@1秒: {optimized['hit_rate_1s']*100:.1f}% ≥ {target_hit_rate*100}% [达标]")
    else:
        print(f"✗ 命中率@1秒: {optimized['hit_rate_1s']*100:.1f}% < {target_hit_rate*100}% [未达标]")


def main():
    """主函数"""
    print("=" * 70)
    print("点火时刻定位优化验证实验")
    print("=" * 70)

    # 配置
    data_dir = "data"
    metadata_path = "data/metadata.csv"
    n_samples = 500  # 测试样本数

    # 检查数据
    if not os.path.exists(metadata_path):
        print(f"错误: 找不到元数据文件 {metadata_path}")
        return

    # 加载数据
    print(f"\n加载样本数据 (n={n_samples})...")
    samples = load_sample_data(data_dir, metadata_path, n_samples)
    print(f"成功加载 {len(samples)} 个样本")

    if len(samples) == 0:
        print("错误: 没有可用的样本数据")
        return

    # 评估
    print("\n开始评估...")
    results = evaluate_ignition_detection(samples)

    # 打印结果
    print_results(results)

    # 保存结果
    results_df = pd.DataFrame({
        '指标': ['MAE', 'RMSE', '中位数误差', 'P90误差', '命中率@1秒', '命中率@0.5秒', '平均推理时间(ms)'],
        '基线方法': [
            results['baseline']['mae'],
            results['baseline']['rmse'],
            results['baseline']['median_error'],
            results['baseline']['p90_error'],
            results['baseline']['hit_rate_1s'],
            results['baseline']['hit_rate_0.5s'],
            results['baseline']['avg_time'] * 1000
        ],
        '优化方法': [
            results['optimized']['mae'],
            results['optimized']['rmse'],
            results['optimized']['median_error'],
            results['optimized']['p90_error'],
            results['optimized']['hit_rate_1s'],
            results['optimized']['hit_rate_0.5s'],
            results['optimized']['avg_time'] * 1000
        ]
    })

    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ignition_optimization_results.csv', index=False)
    print(f"\n结果已保存至 results/ignition_optimization_results.csv")


if __name__ == '__main__':
    main()
