#!/usr/bin/env python3
"""
对增强后的特征进行GAN扩充

流程：
1. 加载enhanced_dataset (10,444条增强后的P/T/R特征)
2. 训练FeatureGAN学习增强后的分布
3. 生成合成样本（可配置倍数）
4. 验证生成质量（分布对比、统计检验）
5. 保存扩充后的数据集供CNN训练使用

输出：
- data/augmented/gan_augmented_features.csv (完整扩充数据集)
- models/gan_enhanced_features.pth (训练好的GAN模型)
- results/gan_augmentation_report.txt (质量验证报告)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
import logging

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gan import FeatureGAN
from src.config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFeatureGANAugmentor:
    """增强特征GAN扩充器"""

    def __init__(
        self,
        enhanced_data_path: str = 'data/augmented/enhanced_dataset_valid.csv',
        original_data_path: str = 'data/feature_dataset.csv',
        output_dir: str = 'data/augmented',
        model_dir: str = 'models',
        results_dir: str = 'results',
        seed: int = 42
    ):
        """
        初始化

        Args:
            enhanced_data_path: 增强后特征数据路径
            original_data_path: 原始特征数据路径（可选，用于混合策略）
            output_dir: 输出目录
            model_dir: 模型保存目录
            results_dir: 结果保存目录
            seed: 随机种子
        """
        self.enhanced_data_path = PROJECT_ROOT / enhanced_data_path
        self.original_data_path = PROJECT_ROOT / original_data_path
        self.output_dir = PROJECT_ROOT / output_dir
        self.model_dir = PROJECT_ROOT / model_dir
        self.results_dir = PROJECT_ROOT / results_dir
        self.seed = seed

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 数据容器
        self.enhanced_features = None
        self.enhanced_labels = None
        self.enhanced_metadata = None
        self.original_features = None

        # GAN模型
        self.gan = None

        # 设置随机种子
        np.random.seed(seed)

    def load_data(self):
        """加载增强后的特征数据"""
        logger.info("=" * 70)
        logger.info("步骤1: 加载数据")
        logger.info("=" * 70)

        # 加载增强后的数据
        logger.info(f"加载增强后的数据: {self.enhanced_data_path}")
        df = pd.read_csv(self.enhanced_data_path)
        logger.info(f"  总样本数: {len(df)}")

        # 提取P/T/R特征（增强后的）
        self.enhanced_features = df[['P', 'T', 'R']].values
        self.enhanced_labels = df['is_anomalous'].values.astype(int)

        # 保存元数据
        metadata_cols = ['uid', 'sn', 'test_id', 'split', 'maneuver_type',
                        'distance_km', 'actual_snr_db', 'detection_difficulty']
        self.enhanced_metadata = df[[col for col in metadata_cols if col in df.columns]]

        # 处理NaN值
        nan_mask = np.isnan(self.enhanced_features).any(axis=1)
        if nan_mask.sum() > 0:
            logger.warning(f"  发现 {nan_mask.sum()} 条含有NaN的样本，将被过滤")
            self.enhanced_features = self.enhanced_features[~nan_mask]
            self.enhanced_labels = self.enhanced_labels[~nan_mask]
            self.enhanced_metadata = self.enhanced_metadata[~nan_mask]

        logger.info(f"  有效样本数: {len(self.enhanced_features)}")
        logger.info(f"  特征维度: {self.enhanced_features.shape[1]}")

        # 显示特征统计
        logger.info("\n增强后特征统计:")
        for i, feat_name in enumerate(['P', 'T', 'R']):
            feat_values = self.enhanced_features[:, i]
            logger.info(f"  {feat_name}:")
            logger.info(f"    范围: {feat_values.min():.4f} ~ {feat_values.max():.4f}")
            logger.info(f"    均值: {feat_values.mean():.4f} ± {feat_values.std():.4f}")

        # 可选：加载原始特征（用于混合策略）
        if self.original_data_path.exists():
            logger.info(f"\n加载原始特征数据: {self.original_data_path}")
            orig_df = pd.read_csv(self.original_data_path)
            self.original_features = orig_df[['P', 'T', 'R']].values

            # 过滤NaN
            orig_nan_mask = np.isnan(self.original_features).any(axis=1)
            self.original_features = self.original_features[~orig_nan_mask]

            logger.info(f"  原始特征样本数: {len(self.original_features)}")

    def train_gan(
        self,
        epochs: int = 400,
        batch_size: int = 32,
        latent_dim: int = 100,
        verbose: bool = True
    ):
        """
        训练GAN

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            latent_dim: 潜在空间维度
            verbose: 是否显示详细信息
        """
        logger.info("\n" + "=" * 70)
        logger.info("步骤2: 训练GAN")
        logger.info("=" * 70)

        # 创建GAN配置
        config = Config()
        config.GAN_CONFIG['epochs'] = epochs
        config.GAN_CONFIG['batch_size'] = batch_size
        config.GAN_CONFIG['latent_dim'] = latent_dim

        # 初始化GAN
        self.gan = FeatureGAN(config=config)

        logger.info(f"GAN配置:")
        logger.info(f"  训练轮数: {epochs}")
        logger.info(f"  批次大小: {batch_size}")
        logger.info(f"  潜在维度: {latent_dim}")
        logger.info(f"  训练样本: {len(self.enhanced_features)} 条")

        # 训练
        start_time = time.time()
        history = self.gan.train(
            real_features=self.enhanced_features,
            epochs=epochs,
            verbose=verbose
        )
        train_time = time.time() - start_time

        logger.info(f"\nGAN训练完成!")
        logger.info(f"  训练耗时: {train_time:.2f} 秒 ({train_time/60:.2f} 分钟)")
        logger.info(f"  最终生成器损失: {history['g_loss'][-1]:.4f}")
        logger.info(f"  最终判别器损失: {history['d_loss'][-1]:.4f}")

        # 保存模型
        model_path = self.model_dir / 'gan_enhanced_features.pth'
        self.gan.save_model(str(model_path))
        logger.info(f"  模型已保存: {model_path}")

        # 绘制训练曲线
        self._plot_training_history(history)

        return history

    def generate_samples(self, expansion_factor: int = 4):
        """
        生成合成样本

        Args:
            expansion_factor: 扩充倍数（总样本 = 原始样本 × expansion_factor）

        Returns:
            synthetic_features: 生成的特征
        """
        logger.info("\n" + "=" * 70)
        logger.info("步骤3: 生成合成样本")
        logger.info("=" * 70)

        n_real = len(self.enhanced_features)
        n_synthetic = n_real * (expansion_factor - 1)

        logger.info(f"扩充倍数: {expansion_factor}x")
        logger.info(f"  真实样本: {n_real}")
        logger.info(f"  待生成样本: {n_synthetic}")
        logger.info(f"  总样本数: {n_real + n_synthetic}")

        # 生成
        start_time = time.time()
        synthetic_features = self.gan.generate_samples(n_synthetic)
        gen_time = time.time() - start_time

        logger.info(f"\n生成完成!")
        logger.info(f"  生成耗时: {gen_time:.2f} 秒")
        logger.info(f"  平均速度: {n_synthetic/gen_time:.0f} 样本/秒")

        return synthetic_features

    def validate_quality(self, synthetic_features: np.ndarray):
        """
        验证生成质量

        Args:
            synthetic_features: 生成的特征
        """
        logger.info("\n" + "=" * 70)
        logger.info("步骤4: 验证生成质量")
        logger.info("=" * 70)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("GAN生成质量验证报告")
        report_lines.append("=" * 70)
        report_lines.append(f"\n生成时间: {pd.Timestamp.now()}")

        # 1. 统计特性对比
        logger.info("\n1. 统计特性对比:")
        report_lines.append("\n1. 统计特性对比")
        report_lines.append("-" * 70)

        feature_names = ['P', 'T', 'R']
        stats_comparison = []

        for i, feat_name in enumerate(feature_names):
            real_feat = self.enhanced_features[:, i]
            syn_feat = synthetic_features[:, i]

            real_mean, real_std = real_feat.mean(), real_feat.std()
            syn_mean, syn_std = syn_feat.mean(), syn_feat.std()

            mean_diff = abs(syn_mean - real_mean) / (real_mean + 1e-10) * 100
            std_diff = abs(syn_std - real_std) / (real_std + 1e-10) * 100

            logger.info(f"\n  {feat_name}:")
            logger.info(f"    真实: μ={real_mean:.4f}, σ={real_std:.4f}")
            logger.info(f"    生成: μ={syn_mean:.4f}, σ={syn_std:.4f}")
            logger.info(f"    偏差: 均值 {mean_diff:.2f}%, 标准差 {std_diff:.2f}%")

            report_lines.append(f"\n{feat_name}:")
            report_lines.append(f"  真实数据: μ={real_mean:.4f}, σ={real_std:.4f}")
            report_lines.append(f"  生成数据: μ={syn_mean:.4f}, σ={syn_std:.4f}")
            report_lines.append(f"  均值偏差: {mean_diff:.2f}%")
            report_lines.append(f"  标准差偏差: {std_diff:.2f}%")

            stats_comparison.append({
                'feature': feat_name,
                'real_mean': real_mean,
                'real_std': real_std,
                'syn_mean': syn_mean,
                'syn_std': syn_std,
                'mean_diff_pct': mean_diff,
                'std_diff_pct': std_diff
            })

        # 2. KS检验（分布一致性）
        logger.info("\n2. Kolmogorov-Smirnov分布检验:")
        report_lines.append("\n\n2. Kolmogorov-Smirnov分布检验")
        report_lines.append("-" * 70)

        for i, feat_name in enumerate(feature_names):
            real_feat = self.enhanced_features[:, i]
            syn_feat = synthetic_features[:, i]

            statistic, p_value = ks_2samp(real_feat, syn_feat)

            # p_value > 0.05 表示两分布无显著差异（好）
            result = "通过 ✓" if p_value > 0.05 else "未通过 ✗"

            logger.info(f"  {feat_name}: KS统计量={statistic:.4f}, p值={p_value:.4f} - {result}")
            report_lines.append(f"{feat_name}: KS={statistic:.4f}, p={p_value:.4f} - {result}")

        # 3. 合理性检查
        logger.info("\n3. 物理合理性检查:")
        report_lines.append("\n\n3. 物理合理性检查")
        report_lines.append("-" * 70)

        # P应该>0
        invalid_P = (synthetic_features[:, 0] < 0).sum()
        pct_invalid_P = invalid_P / len(synthetic_features) * 100
        logger.info(f"  P < 0 的样本: {invalid_P} ({pct_invalid_P:.2f}%)")
        report_lines.append(f"P < 0 的样本: {invalid_P} ({pct_invalid_P:.2f}%)")

        # T应该>0.1秒
        invalid_T = (synthetic_features[:, 1] < 0.1).sum()
        pct_invalid_T = invalid_T / len(synthetic_features) * 100
        logger.info(f"  T < 0.1秒 的样本: {invalid_T} ({pct_invalid_T:.2f}%)")
        report_lines.append(f"T < 0.1秒 的样本: {invalid_T} ({pct_invalid_T:.2f}%)")

        # R应该>0
        invalid_R = (synthetic_features[:, 2] < 0).sum()
        pct_invalid_R = invalid_R / len(synthetic_features) * 100
        logger.info(f"  R < 0 的样本: {invalid_R} ({pct_invalid_R:.2f}%)")
        report_lines.append(f"R < 0 的样本: {invalid_R} ({pct_invalid_R:.2f}%)")

        total_invalid = invalid_P + invalid_T + invalid_R
        pct_valid = (1 - total_invalid / (len(synthetic_features) * 3)) * 100
        logger.info(f"\n  总体有效率: {pct_valid:.2f}%")
        report_lines.append(f"\n总体有效率: {pct_valid:.2f}%")

        # 4. 质量评分
        report_lines.append("\n\n4. 质量评分")
        report_lines.append("-" * 70)

        # 计算总分
        mean_diff_avg = np.mean([s['mean_diff_pct'] for s in stats_comparison])
        std_diff_avg = np.mean([s['std_diff_pct'] for s in stats_comparison])

        score = 100
        if mean_diff_avg > 10:
            score -= 20
        elif mean_diff_avg > 5:
            score -= 10

        if std_diff_avg > 15:
            score -= 15
        elif std_diff_avg > 10:
            score -= 10

        if pct_valid < 95:
            score -= 20
        elif pct_valid < 98:
            score -= 10

        report_lines.append(f"均值偏差: {mean_diff_avg:.2f}%")
        report_lines.append(f"标准差偏差: {std_diff_avg:.2f}%")
        report_lines.append(f"有效率: {pct_valid:.2f}%")
        report_lines.append(f"\n最终评分: {score}/100")

        if score >= 90:
            grade = "优秀 ★★★★★"
        elif score >= 80:
            grade = "良好 ★★★★"
        elif score >= 70:
            grade = "中等 ★★★"
        else:
            grade = "需改进 ★★"

        report_lines.append(f"质量等级: {grade}")

        logger.info(f"\n质量评分: {score}/100 - {grade}")

        # 保存报告
        report_path = self.results_dir / 'gan_augmentation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"\n质量报告已保存: {report_path}")

        # 绘制分布对比图
        self._plot_distribution_comparison(synthetic_features)

        return score

    def save_augmented_dataset(
        self,
        synthetic_features: np.ndarray,
        strategy: str = 'enhanced_only'
    ):
        """
        保存扩充后的数据集

        Args:
            synthetic_features: 生成的特征
            strategy: 保存策略
                - 'enhanced_only': 仅增强后数据 + GAN生成
                - 'mixed': 原始 + 增强 + GAN生成
        """
        logger.info("\n" + "=" * 70)
        logger.info("步骤5: 保存扩充数据集")
        logger.info("=" * 70)

        logger.info(f"保存策略: {strategy}")

        # 准备数据
        if strategy == 'mixed' and self.original_features is not None:
            # 混合策略：原始 + 增强 + GAN
            all_features = np.vstack([
                self.original_features,
                self.enhanced_features,
                synthetic_features
            ])

            n_original = len(self.original_features)
            n_enhanced = len(self.enhanced_features)
            n_synthetic = len(synthetic_features)

            # 创建标记
            source_labels = np.concatenate([
                np.full(n_original, 'original'),
                np.full(n_enhanced, 'enhanced'),
                np.full(n_synthetic, 'gan_synthetic')
            ])

            logger.info(f"混合数据集构成:")
            logger.info(f"  原始特征: {n_original} 条")
            logger.info(f"  增强特征: {n_enhanced} 条")
            logger.info(f"  GAN生成: {n_synthetic} 条")
            logger.info(f"  总计: {len(all_features)} 条")

        else:
            # 仅增强后数据 + GAN
            all_features = np.vstack([
                self.enhanced_features,
                synthetic_features
            ])

            n_enhanced = len(self.enhanced_features)
            n_synthetic = len(synthetic_features)

            source_labels = np.concatenate([
                np.full(n_enhanced, 'enhanced'),
                np.full(n_synthetic, 'gan_synthetic')
            ])

            logger.info(f"数据集构成:")
            logger.info(f"  增强特征: {n_enhanced} 条")
            logger.info(f"  GAN生成: {n_synthetic} 条")
            logger.info(f"  总计: {len(all_features)} 条")

        # 创建DataFrame
        df = pd.DataFrame(
            all_features,
            columns=['P', 'T', 'R']
        )
        df['data_source'] = source_labels
        df['is_synthetic'] = (source_labels == 'gan_synthetic').astype(int)

        # 添加标签（复制真实数据的标签）
        if strategy == 'mixed' and self.original_features is not None:
            # 混合策略：需要组合标签
            synthetic_labels = np.random.choice(
                self.enhanced_labels,
                size=len(synthetic_features)
            )
            all_labels = np.concatenate([
                np.zeros(len(self.original_features), dtype=int),  # 原始数据标签需从原始数据集获取
                self.enhanced_labels,
                synthetic_labels
            ])
        else:
            synthetic_labels = np.random.choice(
                self.enhanced_labels,
                size=len(synthetic_features)
            )
            all_labels = np.concatenate([
                self.enhanced_labels,
                synthetic_labels
            ])

        df['is_anomalous'] = all_labels

        # 保存
        output_path = self.output_dir / 'gan_augmented_features.csv'
        df.to_csv(output_path, index=False)

        logger.info(f"\n数据集已保存: {output_path}")
        logger.info(f"  文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")

        # 显示统计
        logger.info(f"\n数据集统计:")
        logger.info(f"  总样本数: {len(df)}")
        logger.info(f"  特征维度: 3 (P, T, R)")
        logger.info(f"  标签分布:")
        logger.info(f"    正常: {(df['is_anomalous']==0).sum()} ({(df['is_anomalous']==0).sum()/len(df)*100:.1f}%)")
        logger.info(f"    异常: {(df['is_anomalous']==1).sum()} ({(df['is_anomalous']==1).sum()/len(df)*100:.1f}%)")
        logger.info(f"  数据来源:")
        for source in df['data_source'].unique():
            count = (df['data_source'] == source).sum()
            pct = count / len(df) * 100
            logger.info(f"    {source}: {count} ({pct:.1f}%)")

        return output_path

    def _plot_training_history(self, history: dict):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        ax1 = axes[0]
        ax1.plot(history['g_loss'], label='Generator Loss', linewidth=2)
        ax1.plot(history['d_loss'], label='Discriminator Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('GAN Training Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 判别器准确率
        ax2 = axes[1]
        ax2.plot(history['d_real_acc'], label='Real Accuracy', linewidth=2)
        ax2.plot(history['d_fake_acc'], label='Fake Accuracy', linewidth=2)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Discriminator Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.results_dir / 'gan_training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  训练曲线已保存: {save_path}")
        plt.close()

    def _plot_distribution_comparison(self, synthetic_features: np.ndarray):
        """绘制分布对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        feature_names = ['P (Radiation Intensity)', 'T (Duration)', 'R (Frequency Ratio)']
        colors = ['#3498db', '#e74c3c']

        for i, (ax, feat_name) in enumerate(zip(axes, feature_names)):
            real_feat = self.enhanced_features[:, i]
            syn_feat = synthetic_features[:, i]

            # 绘制直方图
            ax.hist(real_feat, bins=50, alpha=0.6, label='Real (Enhanced)',
                   color=colors[0], density=True)
            ax.hist(syn_feat, bins=50, alpha=0.6, label='GAN Generated',
                   color=colors[1], density=True)

            ax.set_xlabel(feat_name, fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{feat_name} Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.results_dir / 'gan_distribution_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  分布对比图已保存: {save_path}")
        plt.close()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("对增强后特征进行GAN扩充")
    print("=" * 70)

    start_time = time.time()

    # 初始化
    augmentor = EnhancedFeatureGANAugmentor(
        enhanced_data_path='data/augmented/enhanced_dataset_valid.csv',
        original_data_path='data/feature_dataset.csv',
        seed=42
    )

    # 步骤1: 加载数据
    augmentor.load_data()

    # 步骤2: 训练GAN
    history = augmentor.train_gan(
        epochs=400,
        batch_size=32,
        latent_dim=100,
        verbose=True
    )

    # 步骤3: 生成样本
    expansion_factor = 4  # 扩充4倍
    synthetic_features = augmentor.generate_samples(expansion_factor=expansion_factor)

    # 步骤4: 验证质量
    quality_score = augmentor.validate_quality(synthetic_features)

    # 步骤5: 保存数据集
    output_path = augmentor.save_augmented_dataset(
        synthetic_features,
        strategy='enhanced_only'  # 可选: 'enhanced_only' 或 'mixed'
    )

    # 总结
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("GAN扩充完成!")
    print("=" * 70)
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"质量评分: {quality_score}/100")
    print(f"\n输出文件:")
    print(f"  数据集: {output_path}")
    print(f"  模型: {augmentor.model_dir / 'gan_enhanced_features.pth'}")
    print(f"  报告: {augmentor.results_dir / 'gan_augmentation_report.txt'}")
    print(f"  可视化: {augmentor.results_dir / 'gan_training_history.png'}")
    print(f"          {augmentor.results_dir / 'gan_distribution_comparison.png'}")

    print("\n下一步:")
    print("  使用生成的数据集训练CNN模型:")
    print("  python experiments/train_with_gan_augmented_data.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
