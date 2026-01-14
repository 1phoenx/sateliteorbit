"""
配置文件：定义项目的全局配置参数
"""
import torch
from pathlib import Path

class Config:
    """全局配置类"""

    # 项目路径
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODEL_DIR = PROJECT_ROOT / 'models'
    EXPERIMENT_DIR = PROJECT_ROOT / 'experiments'

    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 特征提取配置
    FEATURE_CONFIG = {
        'peak_intensity': True,       # P: 辐射强度峰值
        'duration': True,              # T: 持续时间
        'intensity_ratio': True,       # R: 226nm/306nm强度比
        'wavelength_226nm': 226,
        'wavelength_306nm': 306,
    }

    # 数据处理配置
    DATA_CONFIG = {
        'sampling_rate': 1000,         # 采样率 Hz
        'signal_noise_ratio': 5.0,     # 最低信噪比 dB
        'min_delta_v': 0.1,            # 最小可检测变轨速度 m/s
    }

    # GAN网络配置
    GAN_CONFIG = {
        'latent_dim': 100,
        'sample_expansion_factor': 10,  # 样本扩充倍数
        'generator_lr': 0.0002,
        'discriminator_lr': 0.0002,
        'batch_size': 32,
        'epochs': 200,
        'beta1': 0.5,
        'beta2': 0.999,
    }

    # DPC聚类配置
    DPC_CONFIG = {
        'distance_metric': 'euclidean',
        'dc_percent': 0.02,            # 截断距离百分比
        'min_cluster_size': 5,
    }

    # CNN分类器配置
    CNN_CONFIG = {
        'input_channels': 3,           # 三维特征向量 (P, T, R)
        'num_filters': [64, 128, 256],
        'kernel_sizes': [3, 3, 3],
        'dropout_rate': 0.3,
        'num_classes': 2,              # 二分类：变轨/非变轨
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100,
        'early_stopping_patience': 10,
    }

    # 性能指标阈值
    PERFORMANCE_THRESHOLDS = {
        'accuracy': 0.92,              # 准确率目标 ≥92%
        'false_alarm_rate': 0.03,      # 虚警率目标 ≤3%
        'response_time': 5.0,          # 响应时间目标 ≤5秒
    }

    # 随机种子
    SEED = 42

    @classmethod
    def set_seed(cls, seed=None):
        """设置全局随机种子"""
        import random
        import numpy as np

        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.EXPERIMENT_DIR.mkdir(exist_ok=True)
        (cls.DATA_DIR / 'raw').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'processed').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'augmented').mkdir(exist_ok=True)
