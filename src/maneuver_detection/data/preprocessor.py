"""
轨道数据预处理模块
- 滑动窗口分割
- RTN坐标系转换
- 残差计算与标准化
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


@dataclass
class OrbitState:
    """轨道状态数据结构"""
    time: np.ndarray          # 时间戳 (MJD)
    position: np.ndarray      # 位置 (km) [N, 3]
    velocity: np.ndarray      # 速度 (km/s) [N, 3]
    residuals_rtn: Optional[np.ndarray] = None  # RTN残差 [N, 3]


class OrbitDataPreprocessor:
    """轨道数据预处理器"""

    # 地球引力常数 (km³/s²)
    MU_EARTH = 398600.4418

    def __init__(
        self,
        window_size: int = 120,      # 窗口大小 (数据点数)
        stride: int = 10,            # 滑动步长
        sampling_interval: float = 60.0,  # 采样间隔 (秒)
        normalize: bool = True
    ):
        self.window_size = window_size
        self.stride = stride
        self.sampling_interval = sampling_interval
        self.normalize = normalize

        # 标准化参数 (训练时计算)
        self.mean_rtn: Optional[np.ndarray] = None
        self.std_rtn: Optional[np.ndarray] = None

    def eci_to_rtn(
        self,
        position: np.ndarray,
        velocity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ECI坐标系转RTN坐标系
        R: 径向 (沿位置矢量方向)
        T: 沿迹 (垂直于R，在轨道面内)
        N: 法向 (垂直于轨道面)

        Returns:
            R, T, N 单位向量 [N, 3]
        """
        # 径向单位向量
        r_mag = np.linalg.norm(position, axis=1, keepdims=True)
        R_hat = position / r_mag

        # 法向单位向量 (r × v)
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h, axis=1, keepdims=True)
        N_hat = h / h_mag

        # 沿迹单位向量 (N × R)
        T_hat = np.cross(N_hat, R_hat)

        return R_hat, T_hat, N_hat

    def compute_residuals(
        self,
        observed: OrbitState,
        predicted: OrbitState
    ) -> np.ndarray:
        """
        计算轨道残差 (RTN坐标系)

        Args:
            observed: 观测轨道状态
            predicted: 预测轨道状态 (来自轨道传播器)

        Returns:
            RTN残差 [N, 3] (km)
        """
        # 位置差
        delta_pos = observed.position - predicted.position

        # 转换到RTN坐标系
        R_hat, T_hat, N_hat = self.eci_to_rtn(
            observed.position, observed.velocity
        )

        residuals_rtn = np.zeros_like(delta_pos)
        residuals_rtn[:, 0] = np.sum(delta_pos * R_hat, axis=1)  # R分量
        residuals_rtn[:, 1] = np.sum(delta_pos * T_hat, axis=1)  # T分量
        residuals_rtn[:, 2] = np.sum(delta_pos * N_hat, axis=1)  # N分量

        return residuals_rtn

    def compute_velocity_residuals(
        self,
        observed: OrbitState,
        predicted: OrbitState
    ) -> np.ndarray:
        """计算速度残差 (RTN坐标系)"""
        delta_vel = observed.velocity - predicted.velocity

        R_hat, T_hat, N_hat = self.eci_to_rtn(
            observed.position, observed.velocity
        )

        vel_residuals_rtn = np.zeros_like(delta_vel)
        vel_residuals_rtn[:, 0] = np.sum(delta_vel * R_hat, axis=1)
        vel_residuals_rtn[:, 1] = np.sum(delta_vel * T_hat, axis=1)
        vel_residuals_rtn[:, 2] = np.sum(delta_vel * N_hat, axis=1)

        return vel_residuals_rtn

    def extract_features(
        self,
        residuals: np.ndarray,
        time: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        从残差序列提取特征

        Returns:
            特征字典，包含P/T/R相关特征
        """
        features = {}

        # 基础统计特征
        features['mean_r'] = np.mean(residuals[:, 0])
        features['mean_t'] = np.mean(residuals[:, 1])
        features['mean_n'] = np.mean(residuals[:, 2])

        features['std_r'] = np.std(residuals[:, 0])
        features['std_t'] = np.std(residuals[:, 1])
        features['std_n'] = np.std(residuals[:, 2])

        # 斜率特征 (线性趋势)
        dt = time - time[0]
        if len(dt) > 1:
            features['slope_r'] = np.polyfit(dt, residuals[:, 0], 1)[0]
            features['slope_t'] = np.polyfit(dt, residuals[:, 1], 1)[0]
            features['slope_n'] = np.polyfit(dt, residuals[:, 2], 1)[0]
        else:
            features['slope_r'] = 0.0
            features['slope_t'] = 0.0
            features['slope_n'] = 0.0

        # 变化量特征 (前后差异)
        mid = len(residuals) // 2
        features['delta_r'] = np.mean(residuals[mid:, 0]) - np.mean(residuals[:mid, 0])
        features['delta_t'] = np.mean(residuals[mid:, 1]) - np.mean(residuals[:mid, 1])
        features['delta_n'] = np.mean(residuals[mid:, 2]) - np.mean(residuals[:mid, 2])

        # 累积量
        features['cumsum_r'] = np.sum(residuals[:, 0])
        features['cumsum_t'] = np.sum(residuals[:, 1])
        features['cumsum_n'] = np.sum(residuals[:, 2])

        # 最大跳变
        diff_r = np.abs(np.diff(residuals[:, 0]))
        diff_t = np.abs(np.diff(residuals[:, 1]))
        diff_n = np.abs(np.diff(residuals[:, 2]))

        features['max_jump_r'] = np.max(diff_r) if len(diff_r) > 0 else 0
        features['max_jump_t'] = np.max(diff_t) if len(diff_t) > 0 else 0
        features['max_jump_n'] = np.max(diff_n) if len(diff_n) > 0 else 0

        return features

    def create_sliding_windows(
        self,
        residuals: np.ndarray,
        time: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        创建滑动窗口样本

        Args:
            residuals: RTN残差序列 [N, 3]
            time: 时间序列 [N]
            labels: 机动标签序列 [N] (0: 无机动, 1: 机动)

        Returns:
            windows: 窗口数据 [num_windows, window_size, 3]
            window_times: 窗口中心时间 [num_windows]
            window_labels: 窗口标签 [num_windows] (如果提供labels)
        """
        n_samples = len(residuals)
        n_windows = (n_samples - self.window_size) // self.stride + 1

        windows = np.zeros((n_windows, self.window_size, 3))
        window_times = np.zeros(n_windows)
        window_labels = np.zeros(n_windows) if labels is not None else None

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            windows[i] = residuals[start_idx:end_idx]
            window_times[i] = time[start_idx + self.window_size // 2]

            if labels is not None:
                # 窗口标签：窗口内是否存在机动
                window_labels[i] = np.max(labels[start_idx:end_idx])

        return windows, window_times, window_labels

    def fit_normalize(self, residuals: np.ndarray):
        """计算标准化参数"""
        self.mean_rtn = np.mean(residuals, axis=0)
        self.std_rtn = np.std(residuals, axis=0)
        self.std_rtn[self.std_rtn < 1e-8] = 1.0  # 避免除零

    def apply_normalize(self, residuals: np.ndarray) -> np.ndarray:
        """应用标准化"""
        if self.mean_rtn is None or self.std_rtn is None:
            raise ValueError("请先调用 fit_normalize()")
        return (residuals - self.mean_rtn) / self.std_rtn

    def inverse_normalize(self, normalized: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.mean_rtn is None or self.std_rtn is None:
            raise ValueError("请先调用 fit_normalize()")
        return normalized * self.std_rtn + self.mean_rtn

    def smooth_residuals(
        self,
        residuals: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3
    ) -> np.ndarray:
        """使用Savitzky-Golay滤波器平滑残差"""
        smoothed = np.zeros_like(residuals)
        for i in range(3):
            smoothed[:, i] = savgol_filter(
                residuals[:, i], window_length, polyorder
            )
        return smoothed

    def detect_outliers(
        self,
        residuals: np.ndarray,
        threshold: float = 3.0
    ) -> np.ndarray:
        """检测异常值 (基于MAD)"""
        median = np.median(residuals, axis=0)
        mad = np.median(np.abs(residuals - median), axis=0)
        mad[mad < 1e-8] = 1e-8

        z_scores = np.abs(residuals - median) / (1.4826 * mad)
        outliers = np.any(z_scores > threshold, axis=1)

        return outliers


class FeatureExtractor:
    """
    P/T/R 三维特征提取器
    用于第二阶段 Δv 回归
    """

    def __init__(self, orbit_period: float = 5400.0):
        """
        Args:
            orbit_period: 轨道周期 (秒)
        """
        self.orbit_period = orbit_period

    def extract_maneuver_features(
        self,
        residuals_before: np.ndarray,
        residuals_after: np.ndarray,
        time_before: np.ndarray,
        time_after: np.ndarray,
        orbit_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        提取机动特征用于Δv回归

        Args:
            residuals_before: 机动前残差 [N1, 3]
            residuals_after: 机动后残差 [N2, 3]
            time_before: 机动前时间
            time_after: 机动后时间
            orbit_params: 轨道参数 (可选)

        Returns:
            特征向量 [n_features]
        """
        features = []

        # === P特征 (Position相关) ===
        # 位置残差均值变化
        mean_before = np.mean(residuals_before, axis=0)
        mean_after = np.mean(residuals_after, axis=0)
        delta_mean = mean_after - mean_before
        features.extend(delta_mean)  # 3个特征

        # 位置残差标准差变化
        std_before = np.std(residuals_before, axis=0)
        std_after = np.std(residuals_after, axis=0)
        delta_std = std_after - std_before
        features.extend(delta_std)  # 3个特征

        # === T特征 (Tangential/时间相关) ===
        # 残差斜率变化
        dt_before = time_before - time_before[0]
        dt_after = time_after - time_after[0]

        slope_before = np.zeros(3)
        slope_after = np.zeros(3)

        for i in range(3):
            if len(dt_before) > 1:
                slope_before[i] = np.polyfit(dt_before, residuals_before[:, i], 1)[0]
            if len(dt_after) > 1:
                slope_after[i] = np.polyfit(dt_after, residuals_after[:, i], 1)[0]

        delta_slope = slope_after - slope_before
        features.extend(delta_slope)  # 3个特征

        # 累积残差变化率
        cumsum_rate_before = np.sum(residuals_before, axis=0) / len(residuals_before)
        cumsum_rate_after = np.sum(residuals_after, axis=0) / len(residuals_after)
        delta_cumsum_rate = cumsum_rate_after - cumsum_rate_before
        features.extend(delta_cumsum_rate)  # 3个特征

        # === R特征 (Radial/轨道根数相关) ===
        # 近似轨道能量变化 (通过沿迹残差斜率估计)
        # Δa ≈ 2a²/μ · v · Δv_T, 体现在沿迹残差漂移率
        energy_proxy = delta_slope[1] * self.orbit_period / (2 * np.pi)
        features.append(energy_proxy)  # 1个特征

        # 残差幅度变化 (与偏心率变化相关)
        amplitude_before = np.max(residuals_before, axis=0) - np.min(residuals_before, axis=0)
        amplitude_after = np.max(residuals_after, axis=0) - np.min(residuals_after, axis=0)
        delta_amplitude = amplitude_after - amplitude_before
        features.extend(delta_amplitude)  # 3个特征

        # 法向残差周期性变化 (与轨道倾角变化相关)
        # 使用FFT估计主周期幅度
        if len(residuals_before) >= 4:
            fft_before = np.abs(np.fft.fft(residuals_before[:, 2]))[:len(residuals_before)//2]
            fft_after = np.abs(np.fft.fft(residuals_after[:, 2]))[:len(residuals_after)//2]
            period_amp_change = np.max(fft_after) - np.max(fft_before)
        else:
            period_amp_change = 0.0
        features.append(period_amp_change)  # 1个特征

        # === 辅助轨道参数特征 ===
        if orbit_params is not None:
            features.append(orbit_params.get('semi_major_axis', 7000.0))  # km
            features.append(orbit_params.get('eccentricity', 0.001))
            features.append(orbit_params.get('inclination', 45.0))  # deg
        else:
            features.extend([7000.0, 0.001, 45.0])  # 默认值

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表"""
        names = [
            # P特征
            'delta_mean_R', 'delta_mean_T', 'delta_mean_N',
            'delta_std_R', 'delta_std_T', 'delta_std_N',
            # T特征
            'delta_slope_R', 'delta_slope_T', 'delta_slope_N',
            'delta_cumsum_rate_R', 'delta_cumsum_rate_T', 'delta_cumsum_rate_N',
            # R特征
            'energy_proxy',
            'delta_amplitude_R', 'delta_amplitude_T', 'delta_amplitude_N',
            'period_amp_change',
            # 轨道参数
            'semi_major_axis', 'eccentricity', 'inclination'
        ]
        return names
