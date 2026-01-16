"""
召回率优化实验可视化
生成召回率优化相关的图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs("figures", exist_ok=True)


def plot_recall_comparison():
    """绘制召回率对比柱状图"""
    methods = ['基线方法', '自定义权重', '阈值调整', 'SMOTE过采样', '综合优化']
    recall = [0.8909, 0.8909, 0.9280, 0.9012, 0.9342]
    far = [0.0034, 0.0108, 0.0255, 0.0047, 0.0293]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, [r * 100 for r in recall], width,
                    label='Recall (%)', color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Recall (%)', fontsize=12)
    ax1.set_ylim(85, 100)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, [f * 100 for f in far], width,
                    label='FAR (%)', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('False Alarm Rate (%)', fontsize=12)
    ax2.set_ylim(0, 5)
    ax2.axhline(y=3, color='red', linestyle='--', linewidth=1, label='FAR Threshold (3%)')

    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_title('Recall Optimization: Recall vs False Alarm Rate', fontsize=14)

    # 添加数值标签
    for bar, val in zip(bars1, recall):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, far):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('figures/recall_optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/recall_optimization_comparison.png")


def plot_threshold_tuning():
    """绘制阈值调整曲线"""
    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    recall = [0.8909, 0.9012, 0.9136, 0.9177, 0.9280, 0.9362, 0.9424]
    far = [0.0034, 0.0128, 0.0169, 0.0214, 0.0255, 0.0318, 0.0437]
    accuracy = [0.9862, 0.9787, 0.9762, 0.9726, 0.9699, 0.9651, 0.9549]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, [r * 100 for r in recall], 'g-o', linewidth=2,
             markersize=8, label='Recall (%)')
    ax1.plot(thresholds, [a * 100 for a in accuracy], 'b-s', linewidth=2,
             markersize=8, label='Accuracy (%)')
    ax1.set_xlabel('Classification Threshold', fontsize=12)
    ax1.set_ylabel('Recall / Accuracy (%)', fontsize=12)
    ax1.set_ylim(90, 100)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, [f * 100 for f in far], 'r-^', linewidth=2,
             markersize=8, label='FAR (%)')
    ax2.axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('False Alarm Rate (%)', fontsize=12)
    ax2.set_ylim(0, 5)

    # 标记最优阈值
    ax1.axvline(x=0.2, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.annotate('Optimal\nThreshold=0.2', xy=(0.2, 92.8), xytext=(0.28, 91.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

    ax1.set_title('Threshold Tuning: Trade-off between Recall and FAR', fontsize=14)
    ax1.invert_xaxis()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')

    plt.tight_layout()
    plt.savefig('figures/threshold_tuning_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/threshold_tuning_curve.png")


def plot_confusion_matrix_comparison():
    """绘制混淆矩阵对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 基线方法混淆矩阵
    cm_baseline = np.array([[4421, 15], [53, 433]])
    im1 = axes[0].imshow(cm_baseline, cmap='Blues')
    axes[0].set_title('Baseline (threshold=0.5)\nRecall=89.09%', fontsize=12)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Normal', 'Maneuver'])
    axes[0].set_yticklabels(['Normal', 'Maneuver'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    for i in range(2):
        for j in range(2):
            text = axes[0].text(j, i, cm_baseline[i, j], ha='center', va='center',
                               fontsize=14, color='white' if cm_baseline[i,j] > 2000 else 'black')

    # 优化后混淆矩阵
    cm_optimized = np.array([[4306, 130], [32, 454]])
    im2 = axes[1].imshow(cm_optimized, cmap='Greens')
    axes[1].set_title('Optimized (SMOTE + threshold=0.2)\nRecall=93.42%', fontsize=12)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Normal', 'Maneuver'])
    axes[1].set_yticklabels(['Normal', 'Maneuver'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    for i in range(2):
        for j in range(2):
            text = axes[1].text(j, i, cm_optimized[i, j], ha='center', va='center',
                               fontsize=14, color='white' if cm_optimized[i,j] > 2000 else 'black')

    plt.suptitle('Confusion Matrix Comparison: Before vs After Optimization', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/confusion_matrix_comparison.png")


def plot_class_distribution():
    """绘制类别分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 原始数据分布
    labels = ['Normal', 'Maneuver']
    sizes_orig = [1911, 221]
    colors = ['#3498db', '#e74c3c']

    axes[0].pie(sizes_orig, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0, 0.1))
    axes[0].set_title(f'Original Dataset\n(Total: {sum(sizes_orig)})', fontsize=12)

    # SMOTE后分布
    sizes_smote = [17742, 8871]
    axes[1].pie(sizes_smote, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0, 0.05))
    axes[1].set_title(f'After SMOTE\n(Total: {sum(sizes_smote)})', fontsize=12)

    plt.suptitle('Class Distribution: Before vs After SMOTE Oversampling', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/class_distribution_smote.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/class_distribution_smote.png")


if __name__ == '__main__':
    print("生成召回率优化可视化图表...")
    plot_recall_comparison()
    plot_threshold_tuning()
    plot_confusion_matrix_comparison()
    plot_class_distribution()
    print("\n所有图表生成完成!")
