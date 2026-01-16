"""
实验运行脚本
用于运行完整的变轨检测实验流程
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.feature_extraction import ManeuverFeatureExtractor
from src.gan import FeatureGAN
from src.dpc_clustering import ImprovedDPC
from src.mlf_snn.network import MLFSNN
from train import Trainer, load_dataset, prepare_dataloaders


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, exp_name: str = None, config: Config = None):
        self.config = config or Config()
        self.exp_name = exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path("experiments") / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

        with open(self.exp_dir / "log.txt", "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def save_results(self):
        """保存实验结果"""
        with open(self.exp_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def run_data_analysis(self, data_path: str):
        """数据分析"""
        self.log("开始数据分析...")

        df = pd.read_csv(data_path)
        self.log(f"数据集大小: {len(df)}")

        # 基本统计
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'columns': list(df.columns)
        }

        if 'label' in df.columns or 'label_maneuver' in df.columns:
            label_col = 'label_maneuver' if 'label_maneuver' in df.columns else 'label'
            label_counts = df[label_col].value_counts().to_dict()
            stats['label_distribution'] = {str(k): v for k, v in label_counts.items()}
            self.log(f"标签分布: {label_counts}")

        self.results['data_analysis'] = stats
        return stats

    def run_gan_augmentation(self, X, y, epochs=200):
        """GAN数据扩充"""
        self.log("开始GAN数据扩充...")

        gan = FeatureGAN(self.config)
        gan.train(X, epochs=epochs, verbose=True)

        X_aug, y_aug = gan.augment_dataset(X, y)

        self.log(f"原始样本: {len(X)}, 扩充后: {len(X_aug)}")
        self.results['gan_augmentation'] = {
            'original_samples': len(X),
            'augmented_samples': len(X_aug)
        }

        return X_aug, y_aug

    def run_dpc_clustering(self, X, n_clusters=None):
        """DPC聚类分析"""
        self.log("开始DPC聚类...")

        dpc = ImprovedDPC(self.config)
        labels = dpc.fit_predict(X, n_clusters=n_clusters)

        metrics = dpc.evaluate_clustering(X, labels)
        self.results['dpc_clustering'] = metrics

        return labels, metrics

    def run_model_training(self, X, y, model_type='mlf_snn', epochs=100):
        """模型训练"""
        self.log(f"开始训练 {model_type} 模型...")

        input_dim = X.shape[1]
        train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
            X, y, batch_size=32
        )

        if model_type == 'mlf_snn':
            model = MLFSNN(
                input_dim=input_dim,
                hidden_dims=[128, 64],
                output_dim=2,
                time_steps=16
            )
        else:
            model = MLFSNN(input_dim=input_dim, output_dim=2)

        trainer = Trainer(device=self.device)
        save_path = str(self.exp_dir / f"{model_type}_model.pth")

        history = trainer.train(
            model, train_loader, val_loader,
            epochs=epochs, save_path=save_path
        )

        self.results['training'] = {
            'model_type': model_type,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1]
        }

        return model, history

    def run_full_experiment(self, data_path: str, epochs=100):
        """运行完整实验流程"""
        self.log("=" * 50)
        self.log("开始完整实验流程")
        self.log("=" * 50)

        # 1. 数据分析
        self.run_data_analysis(data_path)

        # 2. 加载数据
        X, y, _ = load_dataset(data_path, task='detection')

        # 3. DPC聚类
        self.run_dpc_clustering(X[:1000] if len(X) > 1000 else X)

        # 4. 模型训练
        self.run_model_training(X, y, epochs=epochs)

        # 5. 保存结果
        self.save_results()
        self.log(f"实验完成，结果保存至: {self.exp_dir}")

        return self.results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行变轨检测实验')

    parser.add_argument('--data_path', type=str,
                        default='data/geo_optical_dataset.csv',
                        help='数据路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')

    args = parser.parse_args()

    # 设置随机种子
    Config.set_seed(42)

    # 运行实验
    runner = ExperimentRunner(exp_name=args.exp_name)
    runner.run_full_experiment(args.data_path, epochs=args.epochs)


if __name__ == '__main__':
    main()
