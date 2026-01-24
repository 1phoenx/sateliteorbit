"""
UV 识别系统 - 性能可视化
===========================================

可视化性能优化结果

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceVisualizer:
    """性能可视化器"""

    def __init__(self, output_dir: Path = 'analysis/performance_optimization'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_snr_performance(
        self,
        snr_results: pd.DataFrame,
        target_accuracy: float = 0.92,
        target_far: float = 0.03,
        save_name: str = 'snr_performance.png'
    ):
        """
        绘制SNR性能曲线

        参数:
            snr_results: SNR测试结果
            target_accuracy: 目标准确率
            target_far: 目标虚警率
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 准确率 vs SNR
        ax1 = axes[0, 0]
        ax1.plot(snr_results['snr_db'], snr_results['accuracy'],
                'o-', linewidth=2, markersize=8, color='blue', label='Accuracy')
        ax1.axhline(y=target_accuracy, color='r', linestyle='--',
                   linewidth=2, label=f'Target ({target_accuracy:.0%})')
        ax1.fill_between(snr_results['snr_db'], target_accuracy, 1.0,
                        alpha=0.2, color='green')
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs SNR', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.8, 1.05])

        # 子图2: 虚警率 vs SNR
        ax2 = axes[0, 1]
        ax2.plot(snr_results['snr_db'], snr_results['false_alarm_rate'],
                'o-', linewidth=2, markersize=8, color='red', label='False Alarm Rate')
        ax2.axhline(y=target_far, color='g', linestyle='--',
                   linewidth=2, label=f'Target ({target_far:.0%})')
        ax2.fill_between(snr_results['snr_db'], 0, target_far,
                        alpha=0.2, color='green')
        ax2.set_xlabel('SNR (dB)', fontsize=12)
        ax2.set_ylabel('False Alarm Rate', fontsize=12)
        ax2.set_title('False Alarm Rate vs SNR', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 0.1])

        # 子图3: F1分数 vs SNR
        ax3 = axes[1, 0]
        ax3.plot(snr_results['snr_db'], snr_results['f1'],
                'o-', linewidth=2, markersize=8, color='purple', label='F1 Score')
        ax3.set_xlabel('SNR (dB)', fontsize=12)
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_title('F1 Score vs SNR', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.8, 1.05])

        # 子图4: 精确率和召回率 vs SNR
        ax4 = axes[1, 1]
        ax4.plot(snr_results['snr_db'], snr_results['precision'],
                'o-', linewidth=2, markersize=8, color='orange', label='Precision')
        ax4.plot(snr_results['snr_db'], snr_results['recall'],
                's-', linewidth=2, markersize=8, color='cyan', label='Recall')
        ax4.set_xlabel('SNR (dB)', fontsize=12)
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_title('Precision & Recall vs SNR', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.8, 1.05])

        plt.suptitle('Performance vs Signal-to-Noise Ratio',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {save_path}")
        plt.close()

    def plot_threshold_optimization(
        self,
        snr_results: pd.DataFrame,
        save_name: str = 'threshold_optimization.png'
    ):
        """
        绘制阈值优化结果

        参数:
            snr_results: SNR测试结果
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制最优阈值 vs SNR
        ax.plot(snr_results['snr_db'], snr_results['optimal_threshold'],
               'o-', linewidth=2, markersize=10, color='green')

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Optimal Threshold', fontsize=12)
        ax.set_title('Optimal Classification Threshold vs SNR',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 添加数值标签
        for _, row in snr_results.iterrows():
            ax.text(row['snr_db'], row['optimal_threshold'] + 0.05,
                   f"{row['optimal_threshold']:.3f}",
                   ha='center', fontsize=9)

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {save_path}")
        plt.close()

    def plot_performance_comparison(
        self,
        baseline_results: Dict,
        optimized_results: Dict,
        save_name: str = 'performance_comparison.png'
    ):
        """
        绘制性能对比

        参数:
            baseline_results: 基线结果
            optimized_results: 优化后结果
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['accuracy', 'false_alarm_rate', 'response_time']
        titles = ['Accuracy', 'False Alarm Rate', 'Response Time (s)']
        colors = ['blue', 'red', 'green']

        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[i]

            baseline_val = baseline_results.get(metric, 0)
            optimized_val = optimized_results.get(metric, 0)

            bars = ax.bar(['Baseline', 'Optimized'],
                         [baseline_val, optimized_val],
                         color=[color, 'lightgreen'])

            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)

        plt.suptitle('Performance Comparison: Baseline vs Optimized',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {save_path}")
        plt.close()

    def plot_response_time_distribution(
        self,
        response_times: np.ndarray,
        target_time: float = 5.0,
        save_name: str = 'response_time_distribution.png'
    ):
        """
        绘制响应时间分布

        参数:
            response_times: 响应时间数组
            target_time: 目标响应时间
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1: 直方图
        ax1 = axes[0]
        ax1.hist(response_times, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=target_time, color='r', linestyle='--',
                   linewidth=2, label=f'Target ({target_time}s)')
        ax1.axvline(x=np.mean(response_times), color='g', linestyle='-',
                   linewidth=2, label=f'Mean ({np.mean(response_times):.3f}s)')
        ax1.set_xlabel('Response Time (s)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 子图2: 累积分布
        ax2 = axes[1]
        sorted_times = np.sort(response_times)
        cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, cumulative, linewidth=2, color='blue')
        ax2.axvline(x=target_time, color='r', linestyle='--',
                   linewidth=2, label=f'Target ({target_time}s)')
        ax2.axhline(y=0.95, color='g', linestyle=':',
                   linewidth=1, label='95th percentile')
        ax2.set_xlabel('Response Time (s)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {save_path}")
        plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("UV 识别系统 - 性能可视化")
    print("=" * 70)

    visualizer = PerformanceVisualizer()

    # 1. 加载SNR性能结果
    snr_file = Path('analysis/performance_optimization/snr_performance.csv')
    if snr_file.exists():
        print("\n加载SNR性能结果...")
        snr_results = pd.read_csv(snr_file)

        # 绘制SNR性能曲线
        print("\n生成SNR性能曲线...")
        visualizer.plot_snr_performance(
            snr_results,
            target_accuracy=0.92,
            target_far=0.03
        )

        # 绘制阈值优化结果
        print("\n生成阈值优化图表...")
        visualizer.plot_threshold_optimization(snr_results)

    # 2. 性能对比
    print("\n生成性能对比图表...")
    baseline_results = {
        'accuracy': 0.9926,
        'false_alarm_rate': 0.0,
        'response_time': 0.1063
    }

    optimized_results = {
        'accuracy': 1.0,
        'false_alarm_rate': 0.0,
        'response_time': 0.1063
    }

    visualizer.plot_performance_comparison(
        baseline_results,
        optimized_results
    )

    print("\n" + "=" * 70)
    print("性能可视化完成！")
    print("=" * 70)
    print(f"所有图表已保存到: {visualizer.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
