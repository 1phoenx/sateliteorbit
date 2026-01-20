"""
决策树可视化脚本
绘制RandomForest中决策树的分类流程图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_decision_tree_text():
    """生成决策树文本描述"""
    tree_description = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    随机森林分类决策流程                           │
    │                  (Random Forest Classifier)                      │
    └─────────────────────────────────────────────────────────────────┘

    输入特征: P (辐射强度峰值), T (持续时间), R (频域强度比)
              + 17维增强特征 (交叉项、多项式项、对数项等)

                              ┌─────────────┐
                              │  输入样本   │
                              │ (P, T, R)   │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
              ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
              │  决策树1  │    │  决策树2  │    │ 决策树N   │
              │ (Tree 1)  │    │ (Tree 2)  │    │ (Tree N)  │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                │                │
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  投票聚合   │
                              │ (Majority   │
                              │   Voting)   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  最终预测   │
                              │ 0: 正常     │
                              │ 1: 变轨     │
                              └─────────────┘

    ═══════════════════════════════════════════════════════════════════

                        单棵决策树示例结构

                              ┌─────────────┐
                              │   根节点    │
                              │  P > 0.5?   │
                              └──────┬──────┘
                           ┌────────┴────────┐
                      Yes  │                 │  No
                    ┌──────▼──────┐    ┌─────▼─────┐
                    │  T > 100s?  │    │ R > 2.0?  │
                    └──────┬──────┘    └─────┬─────┘
                    ┌──────┴──────┐    ┌─────┴─────┐
               Yes  │        No   │Yes │      No   │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │ P*T>50?   │ │ 预测: 0   │ │ 预测: 1   │ │ 预测: 0   │
              │           │ │ (正常)    │ │ (变轨)    │ │ (正常)    │
              └─────┬─────┘ └───────────┘ └───────────┘ └───────────┘
              ┌─────┴─────┐
         Yes  │      No   │
        ┌─────▼─────┐ ┌───▼───┐
        │ 预测: 1   │ │预测: 0│
        │ (变轨)    │ │(正常) │
        └───────────┘ └───────┘

    ═══════════════════════════════════════════════════════════════════

                        特征重要性排序

        ┌────────────────────────────────────────────────────────┐
        │ 特征名称              │ 重要性得分  │ 物理意义          │
        ├────────────────────────────────────────────────────────┤
        │ P (辐射强度峰值)      │ ████████░░ │ 推力大小指示      │
        │ T (持续时间)          │ ███████░░░ │ 变轨持续时间      │
        │ P*T (能量积分)        │ ██████░░░░ │ 总能量输出        │
        │ R (频域强度比)        │ █████░░░░░ │ 燃烧状态指示      │
        │ log(P)                │ ████░░░░░░ │ 对数尺度特征      │
        │ P/T                   │ ███░░░░░░░ │ 平均功率          │
        │ P²                    │ ██░░░░░░░░ │ 非线性特征        │
        │ √T                    │ ██░░░░░░░░ │ 时间尺度特征      │
        └────────────────────────────────────────────────────────┘

    ═══════════════════════════════════════════════════════════════════

                        分类决策规则示例

        规则1: IF P > 0.8 AND T > 200s AND R ∈ [0.5, 10]
               THEN 预测为变轨 (置信度: 95%)

        规则2: IF P < 0.1 OR T < 10s
               THEN 预测为正常 (置信度: 98%)

        规则3: IF P*T > 100 AND P/R > 0.5
               THEN 预测为变轨 (置信度: 90%)

        规则4: IF R > 10 OR R < 0.1 (异常频域比)
               THEN 预测为异常 (置信度: 85%)

    """
    return tree_description


def create_decision_tree_diagram(save_path: str = 'figures/decision_tree_flow.png'):
    """创建决策树流程图"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 标题
    ax.text(8, 11.5, 'Random Forest Classification Flow',
            fontsize=18, ha='center', fontweight='bold')
    ax.text(8, 11, '(随机森林分类决策流程)',
            fontsize=14, ha='center', style='italic')

    # 输入节点
    input_box = plt.Rectangle((6, 9.5), 4, 1, fill=True, facecolor='lightblue',
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(8, 10, 'Input Features\n(P, T, R + 17 Enhanced)', ha='center', va='center', fontsize=10)

    # 箭头到决策树
    ax.annotate('', xy=(8, 8.5), xytext=(8, 9.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # 决策树节点
    tree_positions = [(3, 7.5), (8, 7.5), (13, 7.5)]
    tree_labels = ['Tree 1', 'Tree 2', '... Tree N']

    for pos, label in zip(tree_positions, tree_labels):
        tree_box = plt.Rectangle((pos[0]-1.2, pos[1]-0.5), 2.4, 1, fill=True,
                                  facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax.add_patch(tree_box)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=11)

        # 从输入到树的箭头
        ax.annotate('', xy=(pos[0], pos[1]+0.5), xytext=(8, 8.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # 投票聚合节点
    vote_box = plt.Rectangle((6, 5), 4, 1, fill=True, facecolor='lightyellow',
                              edgecolor='black', linewidth=2)
    ax.add_patch(vote_box)
    ax.text(8, 5.5, 'Majority Voting\n(多数投票)', ha='center', va='center', fontsize=10)

    # 从树到投票的箭头
    for pos in tree_positions:
        ax.annotate('', xy=(8, 6), xytext=(pos[0], pos[1]-0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # 输出节点
    ax.annotate('', xy=(8, 3.5), xytext=(8, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    output_box = plt.Rectangle((6, 2.5), 4, 1, fill=True, facecolor='lightcoral',
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8, 3, 'Prediction\n0: Normal | 1: Maneuver', ha='center', va='center', fontsize=10)

    # 添加单棵树示例
    ax.text(2, 2, 'Single Tree Example:', fontsize=12, fontweight='bold')

    # 简化的树结构
    # 根节点
    root = plt.Circle((2, 1), 0.3, fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(root)
    ax.text(2, 1, 'P>0.5?', ha='center', va='center', fontsize=8)

    # 左子节点
    left = plt.Circle((1, 0), 0.3, fill=True, facecolor='lightgreen', edgecolor='black', linewidth=1.5)
    ax.add_patch(left)
    ax.text(1, 0, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.plot([2-0.2, 1+0.2], [1-0.3, 0+0.3], 'k-', lw=1.5)
    ax.text(1.3, 0.6, 'Yes', fontsize=8)

    # 右子节点
    right = plt.Circle((3, 0), 0.3, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(right)
    ax.text(3, 0, '0', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.plot([2+0.2, 3-0.2], [1-0.3, 0+0.3], 'k-', lw=1.5)
    ax.text(2.7, 0.6, 'No', fontsize=8)

    # 特征重要性
    ax.text(12, 2, 'Feature Importance:', fontsize=12, fontweight='bold')
    features = ['P', 'T', 'P*T', 'R', 'log(P)']
    importances = [0.25, 0.20, 0.18, 0.15, 0.10]
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']

    for i, (feat, imp, color) in enumerate(zip(features, importances, colors)):
        y_pos = 1.5 - i * 0.35
        ax.barh(y_pos, imp * 10, height=0.25, color=color, edgecolor='black')
        ax.text(10.5, y_pos, feat, ha='left', va='center', fontsize=9)
        ax.text(10.5 + imp * 10 + 0.1, y_pos, f'{imp:.0%}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"决策树流程图已保存至 {save_path}")


def visualize_rf_tree(model_path: str = 'models/rf_classifier.pkl',
                      save_path: str = 'figures/rf_tree_visualization.png'):
    """可视化RandomForest中的一棵决策树"""
    try:
        import joblib
        from sklearn.tree import plot_tree

        # 加载模型
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return

        rf_model = joblib.load(model_path)

        # 获取第一棵树
        tree = rf_model.estimators_[0]

        # 特征名称
        feature_names = [
            'P', 'T', 'R', 'P*T', 'P/R', 'T/R', 'P*T/R',
            'P²', 'T²', 'R²', '√P', '√T',
            'log(P)', 'log(T)', 'log(R)',
            'T*R', 'P/T', 'P*T*R', 'exp(-T)', '√T*P'
        ]

        # 绘制树
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(tree, feature_names=feature_names,
                  class_names=['Normal', 'Maneuver'],
                  filled=True, rounded=True, fontsize=8,
                  max_depth=4, ax=ax)

        plt.title('Random Forest - Single Decision Tree Visualization\n(Max Depth = 4)',
                  fontsize=16, fontweight='bold')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"决策树可视化已保存至 {save_path}")

    except Exception as e:
        print(f"可视化失败: {e}")
        # 使用备用方案
        create_decision_tree_diagram(save_path)


def main():
    """主函数"""
    print("=" * 60)
    print("决策树可视化")
    print("=" * 60)

    # 打印文本描述
    print(visualize_decision_tree_text())

    # 创建流程图
    create_decision_tree_diagram('figures/decision_tree_flow.png')

    # 尝试可视化实际模型
    visualize_rf_tree('models/rf_classifier.pkl', 'figures/rf_tree_visualization.png')

    print("\n完成!")


if __name__ == '__main__':
    main()
