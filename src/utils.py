"""
工具函数模块：提供辅助功能
"""
import time
import numpy as np
import torch
from functools import wraps
from typing import Dict, Any

def timing_decorator(func):
    """计时装饰器：测量函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Timing] {func.__name__} 执行时间: {elapsed_time:.4f}秒")
        return result
    return wrapper

def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    计算信噪比 (Signal-to-Noise Ratio)

    Args:
        signal: 信号数据
        noise: 噪声数据

    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def normalize_features(features: np.ndarray,
                       mean: np.ndarray = None,
                       std: np.ndarray = None) -> tuple:
    """
    特征归一化

    Args:
        features: 原始特征矩阵 (N, D)
        mean: 均值向量
        std: 标准差向量

    Returns:
        normalized_features, mean, std
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)

    # 避免除以零
    std = np.where(std == 0, 1, std)

    normalized = (features - mean) / std
    return normalized, mean, std

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    filepath: str,
                    **kwargs):
    """
    保存模型检查点

    Args:
        model: PyTorch模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 损失值
        filepath: 保存路径
        **kwargs: 其他需要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] 已保存到: {filepath}")

def load_checkpoint(filepath: str,
                     model: torch.nn.Module = None,
                     optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    加载模型检查点

    Args:
        filepath: 检查点文件路径
        model: PyTorch模型（可选）
        optimizer: 优化器（可选）

    Returns:
        checkpoint字典
    """
    checkpoint = torch.load(filepath)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"[Checkpoint] 已加载: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    return checkpoint

def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    计算分类性能指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）

    Returns:
        性能指标字典
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }

    # 计算虚警率 (False Alarm Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 如果提供概率，计算AUC
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0

    return metrics

def print_metrics(metrics: Dict[str, float]):
    """打印性能指标"""
    print("\n" + "="*50)
    print("性能指标:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*50 + "\n")
