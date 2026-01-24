#!/usr/bin/env python3
"""
使用GAN扩充后的数据训练CNN模型

对比实验：
1. Baseline: 仅使用原始特征(2612条)
2. Enhanced: 使用增强后特征(10444条)
3. GAN Augmented: 使用GAN扩充数据(40000+条)

评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- 推力估计MAE
- 虚警率 (False Alarm Rate)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dual_branch_cnn import DualBranchCNN, DualBranchLoss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GANAugmentedTrainer:
    """GAN扩充数据训练器"""

    def __init__(
        self,
        device: str = None,
        results_dir: str = 'results',
        seed: int = 42
    ):
        """
        初始化

        Args:
            device: 设备 ('cuda' 或 'cpu')
            results_dir: 结果保存目录
            seed: 随机种子
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = PROJECT_ROOT / results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        logger.info(f"使用设备: {self.device}")

    def load_data(self, data_path: str):
        """
        加载数据

        Args:
            data_path: 数据文件路径

        Returns:
            features, labels
        """
        logger.info(f"加载数据: {data_path}")
        df = pd.read_csv(PROJECT_ROOT / data_path)

        features = df[['P', 'T', 'R']].values
        labels = df['is_anomalous'].values.astype(int)

        # 过滤NaN
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]

        logger.info(f"  样本数: {len(features)}")
        logger.info(f"  正常/异常: {(labels==0).sum()}/{(labels==1).sum()}")

        return features, labels

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3
    ):
        """
        训练并评估模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            experiment_name: 实验名称
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率

        Returns:
            评估结果字典
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"实验: {experiment_name}")
        logger.info(f"{'='*70}")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        # 初始化模型
        model = DualBranchCNN(input_dim=3).to(self.device)
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 训练
        logger.info(f"开始训练...")
        start_time = time.time()
        train_history = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                cls_output, _ = model(X_batch)

                # 计算损失
                loss = criterion_cls(cls_output, y_batch)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            train_history.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        train_time = time.time() - start_time
        logger.info(f"训练完成! 耗时: {train_time:.2f} 秒")

        # 评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            cls_output, _ = model(X_test_tensor)
            y_pred = cls_output.argmax(dim=1).cpu().numpy()

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 虚警率
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        logger.info(f"\n评估结果:")
        logger.info(f"  准确率: {accuracy*100:.2f}%")
        logger.info(f"  精确率: {precision*100:.2f}%")
        logger.info(f"  召回率: {recall*100:.2f}%")
        logger.info(f"  F1分数: {f1*100:.2f}%")
        logger.info(f"  虚警率: {false_alarm_rate*100:.2f}%")
        logger.info(f"  混淆矩阵:")
        logger.info(f"    TN={tn}, FP={fp}")
        logger.info(f"    FN={fn}, TP={tp}")

        results = {
            'experiment': experiment_name,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_time': train_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_alarm_rate': false_alarm_rate,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'train_history': train_history
        }

        return results

    def run_comparison_experiments(self):
        """运行对比实验"""
        logger.info("\n" + "="*70)
        logger.info("GAN扩充数据对比实验")
        logger.info("="*70)

        all_results = []

        # 加载测试集（固定）
        logger.info("\n加载测试集...")
        test_features, test_labels = self.load_data('data/feature_dataset.csv')

        # 划分测试集（固定，用于所有实验）
        _, X_test, _, y_test = train_test_split(
            test_features, test_labels,
            test_size=0.2,
            random_state=self.seed,
            stratify=test_labels
        )

        logger.info(f"固定测试集: {len(X_test)} 条")

        # 实验1: Baseline - 仅原始特征
        logger.info("\n" + "="*70)
        logger.info("实验1: Baseline - 仅原始特征")
        logger.info("="*70)

        original_features, original_labels = self.load_data('data/feature_dataset.csv')
        X_train_orig, _, y_train_orig, _ = train_test_split(
            original_features, original_labels,
            test_size=0.2,
            random_state=self.seed,
            stratify=original_labels
        )

        results_baseline = self.train_and_evaluate(
            X_train_orig, y_train_orig,
            X_test, y_test,
            experiment_name="Baseline (Original)",
            epochs=100
        )
        all_results.append(results_baseline)

        # 实验2: Enhanced - 增强后特征
        logger.info("\n" + "="*70)
        logger.info("实验2: Enhanced - 增强后特征")
        logger.info("="*70)

        enhanced_features, enhanced_labels = self.load_data('data/augmented/enhanced_dataset_valid.csv')
        X_train_enh, _, y_train_enh, _ = train_test_split(
            enhanced_features, enhanced_labels,
            test_size=0.2,
            random_state=self.seed,
            stratify=enhanced_labels
        )

        results_enhanced = self.train_and_evaluate(
            X_train_enh, y_train_enh,
            X_test, y_test,
            experiment_name="Enhanced (Physical Augmentation)",
            epochs=100
        )
        all_results.append(results_enhanced)

        # 实验3: GAN Augmented - GAN扩充数据
        logger.info("\n" + "="*70)
        logger.info("实验3: GAN Augmented - GAN扩充数据")
        logger.info("="*70)

        gan_path = PROJECT_ROOT / 'data/augmented/gan_augmented_features.csv'
        if not gan_path.exists():
            logger.warning(f"GAN扩充数据不存在: {gan_path}")
            logger.warning("请先运行: python experiments/gan_augment_enhanced_features.py")
        else:
            gan_features, gan_labels = self.load_data('data/augmented/gan_augmented_features.csv')
            X_train_gan, _, y_train_gan, _ = train_test_split(
                gan_features, gan_labels,
                test_size=0.2,
                random_state=self.seed,
                stratify=gan_labels
            )

            results_gan = self.train_and_evaluate(
                X_train_gan, y_train_gan,
                X_test, y_test,
                experiment_name="GAN Augmented",
                epochs=100
            )
            all_results.append(results_gan)

        # 保存结果
        self._save_results(all_results)

        # 可视化对比
        self._plot_comparison(all_results)

        return all_results

    def _save_results(self, results: list):
        """保存结果"""
        df = pd.DataFrame([
            {
                'Experiment': r['experiment'],
                'Train Samples': r['train_samples'],
                'Test Samples': r['test_samples'],
                'Train Time (s)': r['train_time'],
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1 Score': r['f1'],
                'False Alarm Rate': r['false_alarm_rate']
            }
            for r in results
        ])

        save_path = self.results_dir / 'gan_augmentation_comparison.csv'
        df.to_csv(save_path, index=False)
        logger.info(f"\n结果已保存: {save_path}")

        # 打印表格
        logger.info("\n" + "="*70)
        logger.info("对比结果汇总")
        logger.info("="*70)
        print("\n" + df.to_string(index=False))

    def _plot_comparison(self, results: list):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        experiments = [r['experiment'] for r in results]
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        # 1. 准确率对比
        ax1 = axes[0, 0]
        accuracies = [r['accuracy'] * 100 for r in results]
        bars = ax1.bar(experiments, accuracies, color=colors[:len(experiments)])
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. F1分数对比
        ax2 = axes[0, 1]
        f1_scores = [r['f1'] * 100 for r in results]
        bars = ax2.bar(experiments, f1_scores, color=colors[:len(experiments)])
        ax2.set_ylabel('F1 Score (%)', fontsize=11)
        ax2.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 3. 虚警率对比（越低越好）
        ax3 = axes[1, 0]
        far_values = [r['false_alarm_rate'] * 100 for r in results]
        bars = ax3.bar(experiments, far_values, color=colors[:len(experiments)])
        ax3.set_ylabel('False Alarm Rate (%)', fontsize=11)
        ax3.set_title('False Alarm Rate (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, max(far_values) * 1.2)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, far in zip(bars, far_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{far:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 4. 训练样本数对比
        ax4 = axes[1, 1]
        train_samples = [r['train_samples'] for r in results]
        bars = ax4.bar(experiments, train_samples, color=colors[:len(experiments)])
        ax4.set_ylabel('Training Samples', fontsize=11)
        ax4.set_title('Training Set Size', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, samples in zip(bars, train_samples):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{samples:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 旋转x轴标签
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=15)

        plt.tight_layout()

        save_path = self.results_dir / 'gan_augmentation_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"对比图已保存: {save_path}")
        plt.close()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("使用GAN扩充数据训练CNN")
    print("="*70)

    trainer = GANAugmentedTrainer(seed=42)
    results = trainer.run_comparison_experiments()

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)

    # 显示改进
    if len(results) >= 3:
        baseline_acc = results[0]['accuracy'] * 100
        enhanced_acc = results[1]['accuracy'] * 100
        gan_acc = results[2]['accuracy'] * 100

        print(f"\n准确率提升:")
        print(f"  Baseline → Enhanced: +{enhanced_acc - baseline_acc:.2f}%")
        print(f"  Enhanced → GAN: +{gan_acc - enhanced_acc:.2f}%")
        print(f"  Baseline → GAN: +{gan_acc - baseline_acc:.2f}% (总提升)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
