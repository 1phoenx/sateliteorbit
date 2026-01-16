"""
创新点可视化脚本
生成论文所需的架构图和流程图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)


def plot_system_architecture():
    """绘制系统整体架构图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 颜色定义
    colors = {
        'input': '#E3F2FD',
        'feature': '#FFF3E0',
        'gan': '#E8F5E9',
        'model': '#FCE4EC',
        'output': '#F3E5F5'
    }

    # 1. 输入层
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['input'], edgecolor='#1976D2', linewidth=2))
    ax.text(2, 8.25, 'Raw Time Series\n(thrust, mfr)', ha='center', va='center', fontsize=11, fontweight='bold')

    # 2. 特征提取
    ax.add_patch(FancyBboxPatch((0.5, 5), 3, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['feature'], edgecolor='#F57C00', linewidth=2))
    ax.text(2, 6, 'Feature Extraction\n(P, T, R)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2, 5.3, '• Peak Intensity (P)\n• Duration (T)\n• Frequency Ratio (R)', ha='center', va='center', fontsize=9)

    # 3. 特征增强
    ax.add_patch(FancyBboxPatch((4.5, 5), 3, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['feature'], edgecolor='#F57C00', linewidth=2))
    ax.text(6, 6, 'Feature Engineering\n(20-dim)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 5.3, '• Interaction: P*T, P/R\n• Polynomial: P², T²\n• Log: log(P), log(T)', ha='center', va='center', fontsize=9)

    # 4. GAN扩充
    ax.add_patch(FancyBboxPatch((8.5, 5), 3, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['gan'], edgecolor='#388E3C', linewidth=2))
    ax.text(10, 6, 'GAN Augmentation\n(10x samples)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10, 5.3, '• Generator Network\n• Discriminator Network\n• Feature-level synthesis', ha='center', va='center', fontsize=9)

    # 5. RF分类器
    ax.add_patch(FancyBboxPatch((2, 2), 4, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['model'], edgecolor='#C2185B', linewidth=2))
    ax.text(4, 3, 'RandomForest Classifier\n(1000 trees)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4, 2.3, 'Maneuver Detection\nAcc: 98.50%', ha='center', va='center', fontsize=10)

    # 6. Transformer回归器
    ax.add_patch(FancyBboxPatch((7, 2), 5, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['model'], edgecolor='#C2185B', linewidth=2))
    ax.text(9.5, 3, 'Attention Transformer\n(8 heads, 6 layers)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9.5, 2.3, 'Ignition Time Estimation\nMAE: 5.08s', ha='center', va='center', fontsize=10)

    # 7. 输出
    ax.add_patch(FancyBboxPatch((4, 0), 6, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='#7B1FA2', linewidth=2))
    ax.text(7, 0.6, 'Output: Maneuver Detection + Ignition Time', ha='center', va='center', fontsize=11, fontweight='bold')

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#424242', lw=2)
    ax.annotate('', xy=(2, 7.5), xytext=(2, 7), arrowprops=arrow_style)
    ax.annotate('', xy=(3.5, 6), xytext=(4.5, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 6), xytext=(8.5, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 5), xytext=(4, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(9.5, 5), xytext=(9.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 2), xytext=(5.5, 1.2), arrowprops=arrow_style)
    ax.annotate('', xy=(9.5, 2), xytext=(8.5, 1.2), arrowprops=arrow_style)

    # 标题
    ax.text(7, 9.5, 'System Architecture: RF + Attention Transformer', ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/system_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/system_architecture.png")


def plot_gan_augmentation():
    """绘制GAN数据扩充流程图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 原始数据
    ax.add_patch(FancyBboxPatch((0.5, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='#BBDEFB', edgecolor='#1976D2', linewidth=2))
    ax.text(1.75, 3, 'Original Data\n2,612 samples', ha='center', va='center', fontsize=11, fontweight='bold')

    # Generator
    ax.add_patch(FancyBboxPatch((4, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2))
    ax.text(5, 4.25, 'Generator\nG(z)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Discriminator
    ax.add_patch(FancyBboxPatch((4, 1), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFCDD2', edgecolor='#D32F2F', linewidth=2))
    ax.text(5, 1.75, 'Discriminator\nD(x)', ha='center', va='center', fontsize=10, fontweight='bold')

    # 生成数据
    ax.add_patch(FancyBboxPatch((7, 2), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=2))
    ax.text(8.25, 3, 'Synthetic Data\n21,998 samples', ha='center', va='center', fontsize=11, fontweight='bold')

    # 扩充后数据
    ax.add_patch(FancyBboxPatch((10, 2), 1.8, 2, boxstyle="round,pad=0.1",
                                 facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=2))
    ax.text(10.9, 3, 'Augmented\n24,610\nsamples', ha='center', va='center', fontsize=10, fontweight='bold')

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#424242', lw=2)
    ax.annotate('', xy=(3, 3.5), xytext=(4, 4.25), arrowprops=arrow_style)
    ax.annotate('', xy=(3, 2.5), xytext=(4, 1.75), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 4.25), xytext=(7, 3.5), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 1.75), xytext=(7, 2.5), arrowprops=arrow_style)
    ax.annotate('', xy=(9.5, 3), xytext=(10, 3), arrowprops=arrow_style)

    # 对抗训练箭头
    ax.annotate('', xy=(5, 3.5), xytext=(5, 2.5), arrowprops=dict(arrowstyle='<->', color='#FF5722', lw=2))
    ax.text(5.5, 3, 'Adversarial\nTraining', ha='left', va='center', fontsize=9, color='#FF5722')

    # 标题
    ax.text(6, 5.5, 'GAN Feature-Level Data Augmentation (10x)', ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/gan_augmentation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/gan_augmentation.png")


def plot_transformer_architecture():
    """绘制Transformer架构图"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 输入
    ax.add_patch(FancyBboxPatch((3, 10.5), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(5, 10.9, 'Input Features (20-dim)', ha='center', va='center', fontsize=11, fontweight='bold')

    # Input Projection
    ax.add_patch(FancyBboxPatch((3, 9.2), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    ax.text(5, 9.6, 'Input Projection (d=128)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Position Encoding
    ax.add_patch(FancyBboxPatch((3, 8), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(5, 8.4, '+ Positional Encoding', ha='center', va='center', fontsize=10, fontweight='bold')

    # Transformer Blocks (6 layers)
    for i in range(3):
        y = 6.5 - i * 1.8
        # Multi-Head Attention
        ax.add_patch(FancyBboxPatch((1.5, y), 3, 0.7, boxstyle="round,pad=0.05",
                                     facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=1.5))
        ax.text(3, y + 0.35, f'Multi-Head Attention\n(8 heads)', ha='center', va='center', fontsize=9)

        # Feed Forward
        ax.add_patch(FancyBboxPatch((5.5, y), 3, 0.7, boxstyle="round,pad=0.05",
                                     facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=1.5))
        ax.text(7, y + 0.35, f'Feed Forward\n(d_ff=512)', ha='center', va='center', fontsize=9)

        # Layer label
        ax.text(0.8, y + 0.35, f'Layer\n{i*2+1}-{i*2+2}', ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows
        if i < 2:
            ax.annotate('', xy=(5, y), xytext=(5, y - 0.5), arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))

    # Output Layer
    ax.add_patch(FancyBboxPatch((3, 1.5), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#FFECB3', edgecolor='#FFA000', linewidth=2))
    ax.text(5, 1.9, 'Output Layer (Linear)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Output
    ax.add_patch(FancyBboxPatch((3, 0.3), 4, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#B2DFDB', edgecolor='#00796B', linewidth=2))
    ax.text(5, 0.7, 'Ignition Time Prediction', ha='center', va='center', fontsize=11, fontweight='bold')

    # 连接箭头
    arrow_style = dict(arrowstyle='->', color='#424242', lw=1.5)
    ax.annotate('', xy=(5, 10.5), xytext=(5, 10), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 9.2), xytext=(5, 8.8), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 8), xytext=(5, 7.2), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 2.9), xytext=(5, 2.3), arrowprops=arrow_style)
    ax.annotate('', xy=(5, 1.5), xytext=(5, 1.1), arrowprops=arrow_style)

    # 残差连接
    ax.annotate('', xy=(0.5, 6.85), xytext=(0.5, 3.35),
                arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.5, connectionstyle='arc3,rad=0.3'))
    ax.text(0.2, 5, 'Residual\nConnections', ha='center', va='center', fontsize=8, color='#FF5722', rotation=90)

    # 标题
    ax.text(5, 11.5, 'Attention Transformer Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/transformer_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/transformer_architecture.png")


def plot_feature_engineering():
    """绘制特征工程流程图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 原始特征
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(1.75, 6.5, 'Base Features\n(3-dim)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.75, 5.9, 'P, T, R', ha='center', va='center', fontsize=10)

    # 交互特征
    ax.add_patch(FancyBboxPatch((4, 6), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    ax.text(5.5, 6.75, 'Interaction Features', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 6.25, 'P*T, P/R, T/R, P*T/R', ha='center', va='center', fontsize=9)

    # 多项式特征
    ax.add_patch(FancyBboxPatch((4, 4), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(5.5, 4.75, 'Polynomial Features', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 4.25, 'P², T², R², √P, √T', ha='center', va='center', fontsize=9)

    # 对数特征
    ax.add_patch(FancyBboxPatch((4, 2), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2))
    ax.text(5.5, 2.75, 'Log Features', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 2.25, 'log(P), log(T), log(R)', ha='center', va='center', fontsize=9)

    # 时间相关特征
    ax.add_patch(FancyBboxPatch((4, 0), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#E1BEE7', edgecolor='#7B1FA2', linewidth=2))
    ax.text(5.5, 0.75, 'Time-Related Features', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 0.25, 'T*R, P/T, exp(-T)', ha='center', va='center', fontsize=9)

    # 增强特征
    ax.add_patch(FancyBboxPatch((8.5, 3), 3, 2, boxstyle="round,pad=0.1",
                                 facecolor='#B2DFDB', edgecolor='#00796B', linewidth=2))
    ax.text(10, 4, 'Enhanced Features\n(20-dim)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(10, 3.3, 'Concatenation of\nall feature groups', ha='center', va='center', fontsize=9)

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#424242', lw=2)
    ax.annotate('', xy=(3, 6.5), xytext=(4, 6.75), arrowprops=arrow_style)
    ax.annotate('', xy=(3, 6.5), xytext=(4, 4.75), arrowprops=arrow_style)
    ax.annotate('', xy=(3, 6.5), xytext=(4, 2.75), arrowprops=arrow_style)
    ax.annotate('', xy=(3, 6.5), xytext=(4, 0.75), arrowprops=arrow_style)

    ax.annotate('', xy=(7, 6.75), xytext=(8.5, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 4.75), xytext=(8.5, 4.2), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 2.75), xytext=(8.5, 3.8), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 0.75), xytext=(8.5, 3.5), arrowprops=arrow_style)

    # 标题
    ax.text(6, 7.5, '20-Dimensional Feature Engineering', ha='center', va='center',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/feature_engineering.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/feature_engineering.png")


def plot_dual_stage():
    """绘制双阶段识别机制"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Stage 1
    ax.add_patch(FancyBboxPatch((0.5, 2), 5, 3, boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(3, 4.5, 'Stage 1: Classification', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 3.8, 'RandomForest Classifier', ha='center', va='center', fontsize=10)
    ax.text(3, 3.2, '• 1000 decision trees', ha='center', va='center', fontsize=9)
    ax.text(3, 2.7, '• Class-balanced weights', ha='center', va='center', fontsize=9)
    ax.text(3, 2.2, '• Output: Maneuver/Normal', ha='center', va='center', fontsize=9)

    # Stage 2
    ax.add_patch(FancyBboxPatch((6.5, 2), 5, 3, boxstyle="round,pad=0.1",
                                 facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2))
    ax.text(9, 4.5, 'Stage 2: Regression', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(9, 3.8, 'Attention Transformer', ha='center', va='center', fontsize=10)
    ax.text(9, 3.2, '• 8-head self-attention', ha='center', va='center', fontsize=9)
    ax.text(9, 2.7, '• 6 transformer layers', ha='center', va='center', fontsize=9)
    ax.text(9, 2.2, '• Output: Ignition Time', ha='center', va='center', fontsize=9)

    # 箭头
    ax.annotate('', xy=(5.5, 3.5), xytext=(6.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=3))

    # 标题
    ax.text(6, 5.5, 'Dual-Stage Recognition Mechanism', ha='center', va='center',
            fontsize=14, fontweight='bold')

    # 性能指标
    ax.text(3, 1.3, 'Accuracy: 98.50%\nFalse Alarm: 0.31%', ha='center', va='center',
            fontsize=10, color='#1976D2', fontweight='bold')
    ax.text(9, 1.3, 'MAE: 5.08s', ha='center', va='center',
            fontsize=10, color='#C2185B', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/dual_stage.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: figures/dual_stage.png")


def plot_performance_comparison():
    """绘制性能对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 数据
    methods = ['Before\nGAN', 'After\nGAN']
    accuracy = [89.93, 98.50]
    mae = [5.46, 5.08]

    # 准确率对比
    colors = ['#90CAF9', '#1976D2']
    bars1 = axes[0].bar(methods, accuracy, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=92, color='red', linestyle='--', linewidth=2, label='Target (92%)')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([85, 100])
    axes[0].legend(fontsize=10)
    for bar, val in zip(bars1, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # MAE对比
    bars2 = axes[1].bar(methods, mae, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('MAE (seconds)', fontsize=12)
    axes[1].set_title('Ignition Time Estimation Error', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 8])
    for bar, val in zip(bars2, mae):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/performance_comparison.png")


def main():
    """生成所有创新点可视化图表"""
    print("Generating innovation visualization figures...")
    print("=" * 50)

    plot_system_architecture()
    plot_gan_augmentation()
    plot_transformer_architecture()
    plot_feature_engineering()
    plot_dual_stage()
    plot_performance_comparison()

    print("=" * 50)
    print("All innovation figures saved to figures/ directory")


if __name__ == '__main__':
    main()
