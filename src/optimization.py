"""
模型性能优化模块
包含模型集成、超参数调优、性能评估
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """集成分类器"""

    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for model in self.models:
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """加权投票预测"""
        x = x.to(self.device)
        all_probs = []

        for model, weight in zip(self.models, self.weights):
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs * weight)

        avg_probs = torch.stack(all_probs).sum(dim=0)
        preds = avg_probs.argmax(dim=1)
        confidence = avg_probs.max(dim=1)[0]

        return preds, confidence


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, model_class, param_grid: Dict):
        self.model_class = model_class
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = 0

    def grid_search(self, X, y, n_splits: int = 5) -> Dict:
        """网格搜索"""
        from sklearn.model_selection import StratifiedKFold
        from itertools import product

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        best_score = 0
        best_params = None

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            scores = []

            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, val_idx in kfold.split(X, y):
                score = self._train_and_evaluate(
                    X[train_idx], y[train_idx],
                    X[val_idx], y[val_idx],
                    params
                )
                scores.append(score)

            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        self.best_params = best_params
        self.best_score = best_score
        return {'best_params': best_params, 'best_score': best_score}

    def _train_and_evaluate(self, X_train, y_train, X_val, y_val, params) -> float:
        """训练并评估单个模型"""
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.model_class(**params).to(device)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=32, shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 快速训练
        model.train()
        for _ in range(20):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            output = model(X_val_t)
            if isinstance(output, tuple):
                output = output[0]
            preds = output.argmax(dim=1).cpu().numpy()

        return accuracy_score(y_val, preds)


class PerformanceEvaluator:
    """性能评估器"""

    @staticmethod
    def evaluate_model(y_true, y_pred, y_prob=None) -> Dict:
        """全面评估模型性能"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['miss_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # AUC
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0

        return metrics

    @staticmethod
    def check_performance_targets(metrics: Dict) -> Dict:
        """检查是否达到性能目标"""
        targets = {
            'accuracy': 0.92,
            'false_alarm_rate': 0.03,
        }

        results = {}
        for key, target in targets.items():
            if key == 'false_alarm_rate':
                passed = metrics.get(key, 1.0) <= target
            else:
                passed = metrics.get(key, 0.0) >= target
            results[key] = {'target': target, 'actual': metrics.get(key), 'passed': passed}

        return results


def main():
    """主函数 - 模型优化"""
    import argparse
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description='模型性能优化')
    parser.add_argument('--input', type=str, default='data/augmented_dataset.csv')
    parser.add_argument('--tune', action='store_true', help='执行超参数调优')

    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.input)
    X = df[['P', 'T', 'R']].values
    y = df['is_anomalous'].values.astype(int)
    X = np.nan_to_num(X, nan=0.0)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.tune:
        from src.cnn_1d import FeatureClassifier
        param_grid = {
            'input_dim': [3],
            'num_classes': [2]
        }
        tuner = HyperparameterTuner(FeatureClassifier, param_grid)
        result = tuner.grid_search(X_train, y_train)
        logger.info(f"最佳参数: {result}")

    logger.info("优化完成")


if __name__ == '__main__':
    main()
