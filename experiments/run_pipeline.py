"""
推力器点火检测完整流程脚本
整合特征提取、GAN扩充、模型训练、推理评估
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """运行完整流程"""

    # 步骤1: 特征提取
    if args.step in ['all', 'extract']:
        logger.info("=" * 50)
        logger.info("步骤1: 批量特征提取")
        logger.info("=" * 50)

        from src.feature_extraction_v2 import ThrusterFeatureExtractor

        extractor = ThrusterFeatureExtractor(sampling_rate=100.0)
        df = extractor.batch_extract_features(
            data_dir=args.data_dir,
            metadata_path=args.metadata,
            output_path='data/feature_dataset.csv'
        )
        logger.info(f"特征提取完成: {len(df)} 样本")

    # 步骤2: GAN数据扩充
    if args.step in ['all', 'augment']:
        logger.info("=" * 50)
        logger.info("步骤2: GAN数据扩充")
        logger.info("=" * 50)

        import pandas as pd
        import numpy as np
        from src.gan_v2 import ThrusterFeatureGAN

        df = pd.read_csv('data/feature_dataset.csv')
        features = df[['P', 'T', 'R']].values
        labels = df['is_anomalous'].values.astype(int)
        features = np.nan_to_num(features, nan=0.0)

        gan = ThrusterFeatureGAN()
        gan.train(features, labels, epochs=args.gan_epochs)

        os.makedirs('models', exist_ok=True)
        gan.save('models/thruster_gan.pth')

        aug_features, aug_labels = gan.augment_dataset(
            features, labels, expansion_factor=args.expansion
        )

        aug_df = pd.DataFrame({
            'P': aug_features[:, 0],
            'T': aug_features[:, 1],
            'R': aug_features[:, 2],
            'is_anomalous': aug_labels
        })
        aug_df.to_csv('data/augmented_dataset.csv', index=False)
        logger.info(f"扩充完成: {len(aug_df)} 样本")

    # 步骤3: 模型训练
    if args.step in ['all', 'train']:
        logger.info("=" * 50)
        logger.info("步骤3: 模型训练")
        logger.info("=" * 50)

        import pandas as pd
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from src.cnn_1d import FeatureClassifier, CNN1DTrainer

        df = pd.read_csv('data/augmented_dataset.csv')
        X = df[['P', 'T', 'R']].values
        y = df['is_anomalous'].values.astype(int)
        X = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=32
        )

        model = FeatureClassifier(input_dim=3, num_classes=2)
        trainer = CNN1DTrainer()
        trainer.train(model, train_loader, test_loader,
                     epochs=args.epochs, save_path='models/cnn1d_model.pth')
        logger.info("模型训练完成")

    # 步骤4: 推理评估
    if args.step in ['all', 'infer']:
        logger.info("=" * 50)
        logger.info("步骤4: 推理评估")
        logger.info("=" * 50)

        from src.inference import TwoStageInference

        os.makedirs('results', exist_ok=True)
        inferencer = TwoStageInference(
            classifier_model_path='models/cnn1d_model.pth'
        )
        results = inferencer.batch_infer(
            args.data_dir, args.metadata,
            output_path='results/inference_results.csv'
        )

        logger.info(f"推理完成: {len(results)} 样本")
        logger.info(f"检测异常: {results['is_anomalous'].sum()}")


def main():
    parser = argparse.ArgumentParser(description='推力器点火检测流程')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'extract', 'augment', 'train', 'infer'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--metadata', type=str, default='data/metadata.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gan_epochs', type=int, default=200)
    parser.add_argument('--expansion', type=int, default=10)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
