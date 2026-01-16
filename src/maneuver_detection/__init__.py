"""
双阶段卫星轨道机动识别系统
Stage 1: CNN/LSTM 点火时刻检测
Stage 2: P/T/R 特征回归 Δv
"""

from .models.ignition_detector import IgnitionDetector, IgnitionCNN, IgnitionLSTM
from .models.delta_v_regressor import DeltaVRegressor, DeltaVMLP, DeltaVEnsemble
from .data.preprocessor import OrbitDataPreprocessor
from .data.dataset import ManeuverDataset
from .utils.metrics import ManeuverMetrics
from .pipeline import ManeuverDetectionPipeline

__version__ = "1.0.0"
__all__ = [
    "IgnitionDetector",
    "IgnitionCNN",
    "IgnitionLSTM",
    "DeltaVRegressor",
    "DeltaVMLP",
    "DeltaVEnsemble",
    "OrbitDataPreprocessor",
    "ManeuverDataset",
    "ManeuverMetrics",
    "ManeuverDetectionPipeline",
]
