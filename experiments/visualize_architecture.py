"""
系统架构图和网络架构图美化脚本
生成清晰、专业的架构可视化图
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_system_architecture(save_path: str = 'figures/system_architecture_v2.png'):
    """创建美化的系统总体架构图"""
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # 颜色定义
    colors = {
        'input': '#E3F2FD',      # 浅蓝
        'preprocess': '#E8F5E9', # 浅绿
        'feature': '#FFF3E0',    # 浅橙
        'model': '#F3E5F5',      # 浅紫
        'output': '#FFEBEE',     # 浅红
        'arrow': '#455A64',      # 深灰
        'border': '#37474F'      # 边框
    }

    # 标题
    ax.text(9, 13.5, '小推力变轨检测系统架构',
            fontsize=20, ha='center', fontweight='bold', color='#1A237E')
    ax.text(9, 13, 'Small Thrust Maneuver Detection System Architecture',
            fontsize=12, ha='center', style='italic', color='#5C6BC0')

    # ==================== 第一层: 数据输入 ====================
    input_box = FancyBboxPatch((1, 11), 4, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(3, 11.75, '数据输入层', fontsize=12, ha='center', fontweight='bold')
    ax.text(3, 11.3, 'Thruster Test Data\n(thrust, ton, mfr)', fontsize=9, ha='center')

    # ==================== 第二层: 数据预处理 ====================
    preprocess_box = FancyBboxPatch((1, 8.5), 4, 2, boxstyle="round,pad=0.05",
                                     facecolor=colors['preprocess'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(3, 9.75, '数据预处理', fontsize=12, ha='center', fontweight='bold')
    ax.text(3, 9.2, '• 移动平均滤波 (N=5)\n• 基线校准\n• 异常值过滤', fontsize=9, ha='center')

    # 箭头
    ax.annotate('', xy=(3, 10.5), xytext=(3, 11),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # ==================== 第三层: 特征提取 ====================
    # P特征
    p_box = FancyBboxPatch((0.5, 5.5), 3, 2.5, boxstyle="round,pad=0.05",
                            facecolor=colors['feature'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(p_box)
    ax.text(2, 7.5, 'P 特征', fontsize=11, ha='center', fontweight='bold')
    ax.text(2, 6.9, '辐射强度峰值', fontsize=9, ha='center')
    ax.text(2, 6.3, 'P = max(thrust) - B', fontsize=8, ha='center', family='monospace')
    ax.text(2, 5.8, '有效: P > 3σ', fontsize=8, ha='center')

    # T特征
    t_box = FancyBboxPatch((4, 5.5), 3, 2.5, boxstyle="round,pad=0.05",
                            facecolor=colors['feature'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(t_box)
    ax.text(5.5, 7.5, 'T 特征', fontsize=11, ha='center', fontweight='bold')
    ax.text(5.5, 6.9, '持续时间', fontsize=9, ha='center')
    ax.text(5.5, 6.3, 'T = (t_end - t_start)', fontsize=8, ha='center', family='monospace')
    ax.text(5.5, 5.8, '有效: T ≥ 0.1s', fontsize=8, ha='center')

    # R特征
    r_box = FancyBboxPatch((7.5, 5.5), 3, 2.5, boxstyle="round,pad=0.05",
                            facecolor=colors['feature'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(r_box)
    ax.text(9, 7.5, 'R 特征', fontsize=11, ha='center', fontweight='bold')
    ax.text(9, 6.9, '频域强度比', fontsize=9, ha='center')
    ax.text(9, 6.3, 'R = FFT_f0 / FFT_2f0', fontsize=8, ha='center', family='monospace')
    ax.text(9, 5.8, '有效: 0.1 < R < 10', fontsize=8, ha='center')

    # 箭头从预处理到特征
    ax.annotate('', xy=(2, 8), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(5.5, 8), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(9, 8), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # ==================== 第四层: 数据扩充 ====================
    gan_box = FancyBboxPatch((11.5, 5.5), 5, 2.5, boxstyle="round,pad=0.05",
                              facecolor='#E1F5FE', edgecolor=colors['border'], linewidth=2)
    ax.add_patch(gan_box)
    ax.text(14, 7.5, 'GAN 数据扩充', fontsize=11, ha='center', fontweight='bold')
    ax.text(14, 6.9, 'LayerNorm + GELU', fontsize=9, ha='center')
    ax.text(14, 6.3, '扩充倍数: 10×', fontsize=9, ha='center')
    ax.text(14, 5.8, '生成器 → 判别器', fontsize=8, ha='center')

    # 箭头从特征到GAN
    ax.annotate('', xy=(11.5, 6.75), xytext=(10.5, 6.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # ==================== 第五层: 模型训练 ====================
    # DPC聚类
    dpc_box = FancyBboxPatch((1, 2.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                              facecolor=colors['model'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(dpc_box)
    ax.text(2.75, 4.5, 'DPC 聚类', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.75, 3.9, '密度峰值聚类', fontsize=9, ha='center')
    ax.text(2.75, 3.3, '去除冗余样本', fontsize=9, ha='center')
    ax.text(2.75, 2.8, 'ρ=0.2, δ=8', fontsize=8, ha='center', family='monospace')

    # PCA降维
    pca_box = FancyBboxPatch((5, 2.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                              facecolor=colors['model'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(pca_box)
    ax.text(6.75, 4.5, 'PCA 降维', fontsize=11, ha='center', fontweight='bold')
    ax.text(6.75, 3.9, '主成分分析', fontsize=9, ha='center')
    ax.text(6.75, 3.3, '降维: 20 → 3', fontsize=9, ha='center')
    ax.text(6.75, 2.8, '保留95%方差', fontsize=8, ha='center')

    # RF分类器
    rf_box = FancyBboxPatch((9, 2.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                             facecolor=colors['model'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(rf_box)
    ax.text(10.75, 4.5, 'RF 分类器', fontsize=11, ha='center', fontweight='bold')
    ax.text(10.75, 3.9, 'Random Forest', fontsize=9, ha='center')
    ax.text(10.75, 3.3, 'n_trees=1000', fontsize=9, ha='center')
    ax.text(10.75, 2.8, '变轨检测', fontsize=8, ha='center')

    # 回归器
    reg_box = FancyBboxPatch((13, 2.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                              facecolor=colors['model'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(reg_box)
    ax.text(14.75, 4.5, 'RF 回归器', fontsize=11, ha='center', fontweight='bold')
    ax.text(14.75, 3.9, 'Random Forest', fontsize=9, ha='center')
    ax.text(14.75, 3.3, 'n_trees=500', fontsize=9, ha='center')
    ax.text(14.75, 2.8, '点火时刻估计', fontsize=8, ha='center')

    # 箭头连接
    ax.annotate('', xy=(2.75, 5), xytext=(5.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(6.75, 5), xytext=(14, 5.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(4.5, 3.75), xytext=(5, 3.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(8.5, 3.75), xytext=(9, 3.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(12.5, 3.75), xytext=(13, 3.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # ==================== 第六层: 输出 ====================
    output_box = FancyBboxPatch((5, 0.3), 8, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 1.3, '输出结果', fontsize=12, ha='center', fontweight='bold')
    ax.text(9, 0.7, '变轨检测 (准确率≥92%) | 点火时刻 (误差≤1s) | 推力估计 (Δv<0.1m/s)',
            fontsize=9, ha='center')

    # 箭头到输出
    ax.annotate('', xy=(9, 1.8), xytext=(10.75, 2.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
    ax.annotate('', xy=(9, 1.8), xytext=(14.75, 2.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"系统架构图已保存至 {save_path}")


def create_network_architecture(save_path: str = 'figures/network_architecture_v2.png'):
    """创建美化的网络架构图"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # ==================== 左图: GAN架构 ====================
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('GAN Network Architecture\n(生成对抗网络架构)', fontsize=14, fontweight='bold', pad=20)

    # Generator
    gen_layers = [
        ('Input\nz~N(0,1)', 'latent_dim=100', '#E3F2FD'),
        ('Linear\n+ LayerNorm\n+ GELU', '100→128', '#BBDEFB'),
        ('Linear\n+ LayerNorm\n+ GELU', '128→256', '#90CAF9'),
        ('Residual\nBlock', '256→256', '#64B5F6'),
        ('Linear\n+ LayerNorm\n+ GELU', '256→512', '#42A5F5'),
        ('Residual\nBlock', '512→512', '#2196F3'),
        ('Linear\n+ Tanh', '512→3', '#1E88E5'),
        ('Output\n(P, T, R)', 'dim=3', '#1565C0')
    ]

    ax1.text(2.5, 11.5, 'Generator', fontsize=12, ha='center', fontweight='bold', color='#1565C0')

    for i, (name, dim, color) in enumerate(gen_layers):
        y = 10.5 - i * 1.2
        box = FancyBboxPatch((0.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#0D47A1', linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(2.5, y, name, fontsize=8, ha='center', va='center')
        ax1.text(4.7, y, dim, fontsize=7, ha='left', va='center', color='#666')

        if i < len(gen_layers) - 1:
            ax1.annotate('', xy=(2.5, y-0.4), xytext=(2.5, y-0.8),
                        arrowprops=dict(arrowstyle='->', color='#455A64', lw=1))

    # Discriminator
    disc_layers = [
        ('Input\n(P, T, R)', 'dim=3', '#FFEBEE'),
        ('Linear\n+ LayerNorm\n+ GELU', '3→512', '#FFCDD2'),
        ('Dropout(0.3)', '', '#EF9A9A'),
        ('Linear\n+ LayerNorm\n+ GELU', '512→256', '#E57373'),
        ('Residual\nBlock', '256→256', '#EF5350'),
        ('Linear\n+ Sigmoid', '256→1', '#F44336'),
        ('Output\nReal/Fake', 'prob', '#D32F2F')
    ]

    ax1.text(7.5, 11.5, 'Discriminator', fontsize=12, ha='center', fontweight='bold', color='#C62828')

    for i, (name, dim, color) in enumerate(disc_layers):
        y = 10.5 - i * 1.4
        box = FancyBboxPatch((5.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#B71C1C', linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(7.5, y, name, fontsize=8, ha='center', va='center')
        if dim:
            ax1.text(9.7, y, dim, fontsize=7, ha='left', va='center', color='#666')

        if i < len(disc_layers) - 1:
            ax1.annotate('', xy=(7.5, y-0.4), xytext=(7.5, y-1),
                        arrowprops=dict(arrowstyle='->', color='#455A64', lw=1))

    # ==================== 右图: 分类器架构 ====================
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Classifier Network Architecture\n(分类器网络架构)', fontsize=14, fontweight='bold', pad=20)

    clf_layers = [
        ('Input Features', '(P, T, R) + 17 Enhanced\ndim=20', '#E8F5E9'),
        ('Linear + LayerNorm\n+ GELU + Dropout(0.3)', '20→128', '#C8E6C9'),
        ('Linear + LayerNorm\n+ GELU + Dropout(0.3)', '128→256', '#A5D6A7'),
        ('Linear + LayerNorm\n+ GELU + Dropout(0.2)', '256→128', '#81C784'),
        ('Linear', '128→2', '#66BB6A'),
        ('Softmax', 'Classification', '#4CAF50'),
        ('Output', '0: Normal\n1: Maneuver', '#388E3C')
    ]

    for i, (name, dim, color) in enumerate(clf_layers):
        y = 10.5 - i * 1.4
        box = FancyBboxPatch((2.5, y-0.5), 5, 1, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='#1B5E20', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(5, y, name, fontsize=9, ha='center', va='center')
        ax2.text(7.7, y, dim, fontsize=7, ha='left', va='center', color='#666')

        if i < len(clf_layers) - 1:
            ax2.annotate('', xy=(5, y-0.5), xytext=(5, y-0.9),
                        arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"网络架构图已保存至 {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("生成美化架构图")
    print("=" * 60)

    create_system_architecture('figures/system_architecture_v2.png')
    create_network_architecture('figures/network_architecture_v2.png')

    print("\n完成!")


if __name__ == '__main__':
    main()
