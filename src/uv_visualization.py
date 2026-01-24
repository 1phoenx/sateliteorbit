"""
UV 识别系统可视化工具
===========================================

可视化内容：
1. UV 映射效果（推力/质量流率 → UV 强度）
2. 特征提取结果（脉冲检测）
3. 识别结果对比
4. 性能评估图表

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class UVVisualizer:
    """UV 识别系统可视化器"""

    def __init__(self, output_dir: Union[str, Path] = 'figures/uv_recognition'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_uv_mapping(
        self,
        csv_file: Union[str, Path],
        save_name: str = 'uv_mapping_example.png'
    ):
        """
        可视化 UV 映射效果

        参数:
            csv_file: 包含 thrust, mfr, uv_360nm 的 CSV 文件
            save_name: 保存文件名
        """
        # 读取数据
        df = pd.read_csv(csv_file)

        # 提取时间序列
        time = np.arange(len(df)) * 0.01  # 100 Hz 采样

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # 子图1: 推力
        axes[0].plot(time, df['thrust'], 'b-', linewidth=1, label='Thrust')
        axes[0].set_ylabel('Thrust (N)', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Thrust → UV Mapping Visualization', fontsize=14, fontweight='bold')

        # 子图2: 质量流率
        axes[1].plot(time, df['mfr'], 'g-', linewidth=1, label='Mass Flow Rate')
        axes[1].set_ylabel('MFR (kg/s)', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # 子图3: UV 强度
        axes[2].plot(time, df['uv_360nm'], 'r-', linewidth=1, label='UV 360nm Intensity')
        axes[2].set_ylabel('UV Intensity (a.u.)', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # 子图4: 点火指令
        if 'ton' in df.columns:
            axes[3].plot(time, df['ton'], 'k-', linewidth=2, label='Ignition Command')
            axes[3].set_ylabel('Ignition (on/off)', fontsize=12)
            axes[3].set_ylim([-0.1, 1.1])
            axes[3].legend(loc='upper right')
            axes[3].grid(True, alpha=0.3)

        axes[3].set_xlabel('Time (s)', fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_pulse_detection(
        self,
        csv_file: Union[str, Path],
        threshold_factor: float = 3.0,
        save_name: str = 'pulse_detection_example.png'
    ):
        """
        可视化脉冲检测结果

        参数:
            csv_file: 包含 uv_360nm 的 CSV 文件
            threshold_factor: 阈值因子
            save_name: 保存文件名
        """
        # 读取数据
        df = pd.read_csv(csv_file)
        uv_series = df['uv_360nm'].values
        time = np.arange(len(uv_series)) * 0.01

        # 计算阈值
        n_background = max(10, int(len(uv_series) * 0.1))
        background_mean = np.mean(uv_series[:n_background])
        background_std = np.std(uv_series[:n_background])
        threshold = background_mean + threshold_factor * background_std

        # 检测脉冲
        above_threshold = uv_series > threshold

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 子图1: UV 强度 + 阈值
        axes[0].plot(time, uv_series, 'b-', linewidth=1, label='UV Intensity', alpha=0.7)
        axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold_factor}σ)')
        axes[0].axhline(y=background_mean, color='g', linestyle=':', linewidth=1, label='Background Mean')
        axes[0].fill_between(time, background_mean - background_std, background_mean + background_std,
                            color='g', alpha=0.2, label='Background ±1σ')
        axes[0].set_ylabel('UV Intensity (a.u.)', fontsize=12)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Pulse Detection Visualization', fontsize=14, fontweight='bold')

        # 子图2: 脉冲检测结果
        axes[1].fill_between(time, 0, above_threshold.astype(int), color='orange', alpha=0.5, label='Detected Pulses')
        if 'ton' in df.columns:
            axes[1].plot(time, df['ton'], 'k-', linewidth=2, label='True Ignition', alpha=0.7)
        axes[1].set_ylabel('Pulse Detection', fontsize=12)
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Time (s)', fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_recognition_results(
        self,
        results_file: Union[str, Path],
        save_name: str = 'recognition_performance.png'
    ):
        """
        可视化识别结果

        参数:
            results_file: 识别结果 CSV 文件
            save_name: 保存文件名
        """
        # 读取结果
        df = pd.read_csv(results_file)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 推力估计 vs 真实推力
        if 'true_thrust' in df.columns:
            axes[0, 0].scatter(df['true_thrust'], df['thrust_estimate'], alpha=0.5, s=20)
            axes[0, 0].plot([df['true_thrust'].min(), df['true_thrust'].max()],
                           [df['true_thrust'].min(), df['true_thrust'].max()],
                           'r--', linewidth=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('True Thrust (N)', fontsize=12)
            axes[0, 0].set_ylabel('Estimated Thrust (N)', fontsize=12)
            axes[0, 0].set_title('Thrust Estimation Performance', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 计算 MAE
            mae = np.abs(df['thrust_estimate'] - df['true_thrust']).mean()
            axes[0, 0].text(0.05, 0.95, f'MAE = {mae:.4f} N',
                           transform=axes[0, 0].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 子图2: 变轨检测置信度分布
        axes[0, 1].hist(df['maneuver_probability'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Maneuver Probability', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Maneuver Detection Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 子图3: 变轨类型分布
        type_counts = df['maneuver_type_name'].value_counts()
        axes[1, 0].bar(range(len(type_counts)), type_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_xticks(range(len(type_counts)))
        axes[1, 0].set_xticklabels(type_counts.index, rotation=15, ha='right')
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Maneuver Type Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 子图4: 点火时刻检测置信度
        axes[1, 1].hist(df['ignition_confidence'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Ignition Detection Confidence', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Ignition Detection Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_feature_importance(
        self,
        model_file: Union[str, Path],
        save_name: str = 'feature_importance.png'
    ):
        """
        可视化特征重要性

        参数:
            model_file: 模型文件（pkl）
            save_name: 保存文件名
        """
        import pickle

        # 加载模型
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            feature_names = data.get('feature_names', [f'Feature {i}' for i in range(len(model.feature_importances_))])

        # 获取特征重要性
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制条形图
        ax.barh(range(len(importances)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def main():
    """主函数：生成所有可视化图表"""
    print("=" * 70)
    print("UV 识别系统可视化")
    print("=" * 70)

    visualizer = UVVisualizer(output_dir='figures/uv_recognition')

    # 1. UV 映射可视化
    print("\n1. 生成 UV 映射可视化...")
    uv_files = list(Path('data/train_with_uv').glob('*.csv'))
    if len(uv_files) > 0:
        visualizer.plot_uv_mapping(uv_files[0], 'uv_mapping_example.png')

    # 2. 脉冲检测可视化
    print("\n2. 生成脉冲检测可视化...")
    if len(uv_files) > 0:
        visualizer.plot_pulse_detection(uv_files[0], threshold_factor=3.0, save_name='pulse_detection_example.png')

    # 3. 识别结果可视化
    print("\n3. 生成识别结果可视化...")
    results_file = Path('results/uv_recognition_results.csv')
    if results_file.exists():
        visualizer.plot_recognition_results(results_file, 'recognition_performance.png')

    # 4. 特征重要性可视化
    print("\n4. 生成特征重要性可视化...")
    model_files = [
        ('models/uv_recognition/maneuver_classifier.pkl', 'feature_importance_maneuver.png'),
        ('models/uv_recognition/thrust_regressor.pkl', 'feature_importance_thrust.png'),
        ('models/uv_recognition/maneuver_type_classifier.pkl', 'feature_importance_type.png')
    ]

    for model_file, save_name in model_files:
        if Path(model_file).exists():
            visualizer.plot_feature_importance(model_file, save_name)

    print("\n" + "=" * 70)
    print("可视化完成！")
    print(f"所有图表已保存到: {visualizer.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
