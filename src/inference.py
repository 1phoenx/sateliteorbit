"""
双阶段推理模块
阶段1: 特征提取 + 点火检测
阶段2: 异常分类 + 点火时刻定位
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageInference:
    """双阶段推理器"""

    def __init__(
        self,
        feature_model_path: str = None,
        classifier_model_path: str = None,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = None
        self.classifier = None
        self.scaler = None

        # 加载模型
        if feature_model_path and os.path.exists(feature_model_path):
            self._load_feature_model(feature_model_path)
        if classifier_model_path and os.path.exists(classifier_model_path):
            self._load_classifier(classifier_model_path)

    def _load_feature_model(self, path: str):
        """加载特征提取模型"""
        logger.info(f"加载特征模型: {path}")
        # 特征提取使用规则方法，无需加载模型

    def _load_classifier(self, path: str):
        """加载分类器"""
        from src.cnn_1d import FeatureClassifier

        logger.info(f"加载分类器: {path}")
        self.classifier = FeatureClassifier(input_dim=3, num_classes=2)
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()

    def stage1_feature_extraction(
        self,
        thrust: np.ndarray,
        ton: np.ndarray,
        sampling_rate: float = 100.0
    ) -> Dict:
        """
        阶段1: 特征提取

        Args:
            thrust: 推力时序数据
            ton: 推力器开关状态
            sampling_rate: 采样率

        Returns:
            特征字典 {P, T, R, ignition_time, is_valid}
        """
        from src.feature_extraction_v2 import ThrusterFeatureExtractor

        extractor = ThrusterFeatureExtractor(sampling_rate=sampling_rate)

        # 计算基线
        B_thrust, sigma = extractor.compute_baseline(thrust, ton)

        # 提取P/T/R特征
        P = extractor.extract_P(thrust, ton, B_thrust, sigma)
        T, ignition_time, true_thrust = extractor.extract_T(thrust, B_thrust, sigma)
        R = extractor.extract_R(thrust, T)

        is_valid = 1 if (P > 0 and T >= 0.1) else 0

        return {
            'P': P,
            'T': T,
            'R': R,
            'ignition_time': ignition_time,
            'true_thrust': true_thrust,
            'is_valid': is_valid
        }

    def stage2_classification(self, features: Dict) -> Dict:
        """
        阶段2: 异常分类

        Args:
            features: 阶段1提取的特征

        Returns:
            分类结果 {is_anomalous, confidence, ignition_time}
        """
        if self.classifier is None:
            logger.warning("分类器未加载，使用规则判断")
            return self._rule_based_classification(features)

        # 准备输入
        P, T, R = features['P'], features['T'], features['R']
        R = 0.0 if np.isnan(R) else R

        x = torch.FloatTensor([[P, T, R]]).to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        return {
            'is_anomalous': pred,
            'confidence': confidence,
            'ignition_time': features['ignition_time']
        }

    def _rule_based_classification(self, features: Dict) -> Dict:
        """基于规则的分类（备用方案）"""
        P, T, R = features['P'], features['T'], features['R']

        is_anomalous = 0
        if P <= 0 or T < 0.1:
            is_anomalous = 1
        elif not np.isnan(R) and (R < 0.5 or R > 10):
            is_anomalous = 1

        return {
            'is_anomalous': is_anomalous,
            'confidence': 0.8,
            'ignition_time': features['ignition_time']
        }

    def infer(
        self,
        thrust: np.ndarray,
        ton: np.ndarray,
        sampling_rate: float = 100.0
    ) -> Dict:
        """
        完整双阶段推理

        Args:
            thrust: 推力时序数据
            ton: 推力器开关状态
            sampling_rate: 采样率

        Returns:
            完整推理结果
        """
        start_time = time.time()

        # 阶段1: 特征提取
        features = self.stage1_feature_extraction(thrust, ton, sampling_rate)

        # 阶段2: 分类
        result = self.stage2_classification(features)

        # 合并结果
        result.update({
            'P': features['P'],
            'T': features['T'],
            'R': features['R'],
            'true_thrust': features['true_thrust'],
            'is_valid': features['is_valid'],
            'inference_time': time.time() - start_time
        })

        return result

    def infer_file(self, file_path: str, sampling_rate: float = 100.0) -> Dict:
        """从CSV文件推理"""
        df = pd.read_csv(file_path)

        if 'thrust' not in df.columns or 'ton' not in df.columns:
            raise ValueError(f"文件缺少必要列: {file_path}")

        thrust = df['thrust'].values
        ton = df['ton'].values

        return self.infer(thrust, ton, sampling_rate)

    def batch_infer(
        self,
        data_dir: str,
        metadata_path: str,
        output_path: str = None
    ) -> pd.DataFrame:
        """批量推理"""
        logger.info(f"批量推理: {data_dir}")

        metadata = pd.read_csv(metadata_path)
        results = []

        for _, row in metadata.iterrows():
            filename = row['filename']
            train_path = Path(data_dir) / 'train' / filename
            test_path = Path(data_dir) / 'test' / filename

            file_path = train_path if train_path.exists() else test_path
            if not file_path.exists():
                continue

            try:
                result = self.infer_file(str(file_path))
                result['uid'] = row['uid']
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                logger.error(f"推理失败 {filename}: {e}")

        df = pd.DataFrame(results)
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"结果保存至: {output_path}")

        return df


class InferenceEvaluator:
    """推理评估器"""

    @staticmethod
    def evaluate(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict:
        """评估推理结果"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        merged = predictions.merge(ground_truth, on='uid', suffixes=('_pred', '_true'))

        y_true = merged['anomalous'].values
        y_pred = merged['is_anomalous'].values

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'false_alarm_rate': ((y_pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
        }

        # 点火时刻误差
        if 'ignition_time_true' in merged.columns:
            time_errors = np.abs(
                merged['ignition_time_pred'] - merged['ignition_time_true']
            )
            metrics['ignition_time_mae'] = np.nanmean(time_errors)

        # 平均推理时间
        if 'inference_time' in merged.columns:
            metrics['avg_inference_time'] = merged['inference_time'].mean()

        return metrics


def main():
    """主函数 - 双阶段推理"""
    import argparse

    parser = argparse.ArgumentParser(description='双阶段推理')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--metadata', type=str, default='data/metadata.csv')
    parser.add_argument('--model', type=str, default='models/cnn1d_model.pth')
    parser.add_argument('--output', type=str, default='results/inference_results.csv')

    args = parser.parse_args()

    # 创建推理器
    inferencer = TwoStageInference(classifier_model_path=args.model)

    # 批量推理
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = inferencer.batch_infer(
        args.data_dir, args.metadata, args.output
    )

    # 输出统计
    print("\n" + "=" * 50)
    print("推理统计")
    print("=" * 50)
    print(f"总样本: {len(results)}")
    print(f"检测异常: {results['is_anomalous'].sum()}")
    print(f"平均推理时间: {results['inference_time'].mean():.4f}s")


if __name__ == '__main__':
    main()
