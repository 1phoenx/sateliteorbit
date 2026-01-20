"""
点火时刻精确定位模块
基于滑动窗口动态阈值(MAD) + 加权质心法实现高精度点火时刻定位
目标: 点火时刻误差 ≤ 1秒
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, Dict, List


class IgnitionDetector:
    """
    点火时刻精确检测器

    使用滑动窗口动态阈值(MAD)检测变点，结合加权质心法精确定位点火时刻
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        window_size: int = 15,
        mad_coefficient: float = 1.8,
        min_duration: float = 0.05,
        smoothing_window: int = 5
    ):
        """
        初始化点火检测器

        Args:
            sampling_rate: 采样率 (Hz)
            window_size: 滑动窗口大小 (时间步)
            mad_coefficient: MAD系数，用于动态阈值计算
            min_duration: 最小有效持续时间 (秒)
            smoothing_window: 平滑窗口大小
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.window_size = window_size
        self.mad_coefficient = mad_coefficient
        self.min_duration = min_duration
        self.smoothing_window = smoothing_window

    def _smooth_signal(self, data: np.ndarray) -> np.ndarray:
        """信号平滑处理"""
        if len(data) < self.smoothing_window:
            return data
        return uniform_filter1d(data, size=self.smoothing_window, mode='nearest')

    def _compute_mad_threshold(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算滑动窗口MAD动态阈值

        MAD (Median Absolute Deviation) 对异常值更鲁棒
        阈值 = median + mad_coefficient * MAD

        Returns:
            (中位数序列, 动态阈值序列)
        """
        n = len(data)
        half_window = self.window_size // 2

        medians = np.zeros(n)
        thresholds = np.zeros(n)

        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window_data = data[start:end]

            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))

            # MAD标准化因子 (对于正态分布约为1.4826)
            mad_normalized = 1.4826 * mad

            medians[i] = median
            thresholds[i] = median + self.mad_coefficient * mad_normalized

        return medians, thresholds

    def _detect_change_points(
        self,
        data: np.ndarray,
        thresholds: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        检测变点区间

        Returns:
            变点区间列表 [(start_idx, end_idx), ...]
        """
        above_threshold = data > thresholds

        # 找到连续超过阈值的区间
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # 处理边界情况
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(data)]])

        # 过滤太短的区间
        min_samples = int(self.min_duration * self.sampling_rate)
        intervals = []
        for s, e in zip(starts, ends):
            if e - s >= min_samples:
                intervals.append((s, e))

        return intervals

    def _weighted_centroid(
        self,
        data: np.ndarray,
        start_idx: int,
        end_idx: int,
        threshold: float
    ) -> float:
        """
        加权质心法精确定位点火时刻

        使用信号强度作为权重，计算加权质心位置

        Args:
            data: 信号数据
            start_idx: 区间起始索引
            end_idx: 区间结束索引
            threshold: 阈值

        Returns:
            精确的点火时刻索引 (浮点数)
        """
        segment = data[start_idx:end_idx]
        indices = np.arange(start_idx, end_idx)

        # 计算权重: 超过阈值的部分
        weights = np.maximum(segment - threshold, 0)

        if np.sum(weights) < 1e-10:
            return float(start_idx)

        # 加权质心
        centroid = np.sum(indices * weights) / np.sum(weights)
        return centroid

    def _find_onset_point(
        self,
        data: np.ndarray,
        start_idx: int,
        threshold: float,
        search_window: int = 20
    ) -> int:
        """
        精确查找信号起始点 (onset)

        使用一阶导数和二阶导数检测信号突变点
        """
        # 扩展搜索范围
        search_start = max(0, start_idx - search_window)
        search_end = min(len(data), start_idx + search_window)

        segment = data[search_start:search_end]

        # 计算一阶导数
        gradient = np.gradient(segment)

        # 计算二阶导数 (加速度)
        acceleration = np.gradient(gradient)

        # 找到最大加速度点 (信号突变点)
        max_acc_idx = np.argmax(acceleration)

        # 结合阈值检测
        above_threshold = segment > threshold
        if np.any(above_threshold):
            first_above = np.argmax(above_threshold)
            # 取加速度最大点和首次超阈值点的较早者
            onset_local = min(max_acc_idx, first_above)
        else:
            onset_local = max_acc_idx

        return search_start + onset_local

    def detect_ignition_time(
        self,
        thrust: np.ndarray,
        ton: Optional[np.ndarray] = None,
        return_details: bool = False
    ) -> Dict:
        """
        检测点火时刻

        Args:
            thrust: 推力时序数据
            ton: 推力器开关状态 (可选，用于计算背景)
            return_details: 是否返回详细信息

        Returns:
            {
                'ignition_time': 点火时刻 (秒),
                'ignition_index': 点火时刻索引,
                'confidence': 置信度,
                'duration': 持续时间,
                'details': 详细信息 (可选)
            }
        """
        # 信号平滑
        thrust_smooth = self._smooth_signal(thrust)

        # 计算背景基线
        if ton is not None:
            background_mask = ton == 0
            if np.sum(background_mask) > 10:
                background = thrust_smooth[background_mask]
                baseline = np.median(background)
            else:
                baseline = np.percentile(thrust_smooth, 10)
        else:
            baseline = np.percentile(thrust_smooth, 10)

        # 去除基线
        signal_detrended = thrust_smooth - baseline

        # 计算MAD动态阈值
        medians, thresholds = self._compute_mad_threshold(signal_detrended)

        # 检测变点区间
        intervals = self._detect_change_points(signal_detrended, thresholds)

        if not intervals:
            return {
                'ignition_time': np.nan,
                'ignition_index': -1,
                'confidence': 0.0,
                'duration': 0.0,
                'details': {} if return_details else None
            }

        # 选择最显著的区间 (信号强度最大)
        best_interval = None
        best_strength = 0

        for start, end in intervals:
            strength = np.max(signal_detrended[start:end])
            if strength > best_strength:
                best_strength = strength
                best_interval = (start, end)

        start_idx, end_idx = best_interval

        # 精确定位点火时刻
        # 方法1: 使用onset检测
        onset_idx = self._find_onset_point(
            signal_detrended, start_idx, thresholds[start_idx]
        )

        # 方法2: 加权质心法 (用于验证)
        centroid_idx = self._weighted_centroid(
            signal_detrended, start_idx, end_idx, thresholds[start_idx]
        )

        # 综合两种方法: onset更适合定位起始点
        ignition_index = onset_idx
        ignition_time = ignition_index * self.dt

        # 计算持续时间
        duration = (end_idx - start_idx) * self.dt

        # 计算置信度
        snr = best_strength / (np.std(signal_detrended[:start_idx]) + 1e-10)
        confidence = min(1.0, snr / 10.0)

        result = {
            'ignition_time': ignition_time,
            'ignition_index': ignition_index,
            'confidence': confidence,
            'duration': duration
        }

        if return_details:
            result['details'] = {
                'onset_idx': onset_idx,
                'centroid_idx': centroid_idx,
                'interval': best_interval,
                'snr': snr,
                'threshold_at_onset': thresholds[onset_idx],
                'signal_at_onset': signal_detrended[onset_idx]
            }

        return result

    def detect_ignition_time_multiscale(
        self,
        thrust: np.ndarray,
        ton: Optional[np.ndarray] = None,
        scales: List[int] = None
    ) -> Dict:
        """
        多尺度点火时刻检测

        在多个时间尺度上检测点火时刻，取加权平均提高精度

        Args:
            thrust: 推力时序数据
            ton: 推力器开关状态
            scales: 窗口尺度列表

        Returns:
            检测结果
        """
        if scales is None:
            scales = [5, 10, 15, 20, 30]

        detections = []
        weights = []

        original_window = self.window_size

        for scale in scales:
            self.window_size = scale
            result = self.detect_ignition_time(thrust, ton)

            if not np.isnan(result['ignition_time']):
                detections.append(result['ignition_time'])
                weights.append(result['confidence'])

        self.window_size = original_window

        if not detections:
            return {
                'ignition_time': np.nan,
                'ignition_index': -1,
                'confidence': 0.0,
                'duration': 0.0
            }

        # 加权平均
        weights = np.array(weights)
        detections = np.array(detections)

        # 去除异常值 (超过2倍标准差)
        if len(detections) > 2:
            mean_det = np.mean(detections)
            std_det = np.std(detections)
            mask = np.abs(detections - mean_det) < 2 * std_det
            if np.sum(mask) > 0:
                detections = detections[mask]
                weights = weights[mask]

        weights = weights / np.sum(weights)
        ignition_time = np.sum(detections * weights)
        ignition_index = int(ignition_time * self.sampling_rate)

        # 重新计算持续时间
        final_result = self.detect_ignition_time(thrust, ton)

        return {
            'ignition_time': ignition_time,
            'ignition_index': ignition_index,
            'confidence': np.mean(weights) * len(detections) / len(scales),
            'duration': final_result['duration']
        }


class IgnitionDetectorV2:
    """
    点火时刻检测器 V2 - 基于信号导数的精确检测

    使用信号的一阶和二阶导数检测突变点，实现亚采样周期精度
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        derivative_threshold: float = 3.0,
        smoothing_sigma: float = 2.0
    ):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.derivative_threshold = derivative_threshold
        self.smoothing_sigma = smoothing_sigma

    def _gaussian_smooth(self, data: np.ndarray) -> np.ndarray:
        """高斯平滑"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=self.smoothing_sigma)

    def _compute_derivatives(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算一阶和二阶导数"""
        # 使用Savitzky-Golay滤波器计算平滑导数
        window_length = min(11, len(data) // 2 * 2 + 1)
        if window_length < 5:
            window_length = 5

        try:
            first_derivative = signal.savgol_filter(data, window_length, 3, deriv=1)
            second_derivative = signal.savgol_filter(data, window_length, 3, deriv=2)
        except ValueError:
            first_derivative = np.gradient(data)
            second_derivative = np.gradient(first_derivative)

        return first_derivative, second_derivative

    def detect(self, thrust: np.ndarray, ton: Optional[np.ndarray] = None) -> Dict:
        """
        检测点火时刻

        使用导数方法检测信号突变点
        """
        # 平滑处理
        thrust_smooth = self._gaussian_smooth(thrust)

        # 计算背景
        if ton is not None and np.sum(ton == 0) > 10:
            background = np.median(thrust_smooth[ton == 0])
        else:
            background = np.percentile(thrust_smooth, 10)

        signal_detrended = thrust_smooth - background

        # 计算导数
        first_deriv, second_deriv = self._compute_derivatives(signal_detrended)

        # 标准化导数
        deriv_std = np.std(first_deriv)
        if deriv_std < 1e-10:
            deriv_std = 1.0
        first_deriv_norm = first_deriv / deriv_std

        # 找到导数超过阈值的点
        above_threshold = first_deriv_norm > self.derivative_threshold

        if not np.any(above_threshold):
            # 降低阈值重试
            above_threshold = first_deriv_norm > self.derivative_threshold / 2

        if not np.any(above_threshold):
            return {
                'ignition_time': np.nan,
                'ignition_index': -1,
                'confidence': 0.0,
                'duration': 0.0
            }

        # 找到第一个超过阈值的点
        first_idx = np.argmax(above_threshold)

        # 在该点附近找到二阶导数最大点 (加速度最大)
        search_start = max(0, first_idx - 10)
        search_end = min(len(second_deriv), first_idx + 10)

        local_second_deriv = second_deriv[search_start:search_end]
        max_acc_local = np.argmax(local_second_deriv)
        ignition_index = search_start + max_acc_local

        # 亚采样精度: 使用抛物线拟合
        if 1 <= max_acc_local < len(local_second_deriv) - 1:
            y0 = local_second_deriv[max_acc_local - 1]
            y1 = local_second_deriv[max_acc_local]
            y2 = local_second_deriv[max_acc_local + 1]

            # 抛物线顶点
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-10)
            delta = np.clip(delta, -0.5, 0.5)

            ignition_time = (ignition_index + delta) * self.dt
        else:
            ignition_time = ignition_index * self.dt

        # 计算持续时间
        signal_threshold = background + 3 * np.std(thrust_smooth[ton == 0] if ton is not None else thrust_smooth[:100])
        above_signal = thrust_smooth > signal_threshold
        if np.any(above_signal):
            indices = np.where(above_signal)[0]
            duration = (indices[-1] - indices[0]) * self.dt
        else:
            duration = 0.0

        # 置信度
        confidence = min(1.0, np.max(first_deriv_norm) / 10.0)

        return {
            'ignition_time': ignition_time,
            'ignition_index': ignition_index,
            'confidence': confidence,
            'duration': duration
        }


def detect_ignition_ensemble(
    thrust: np.ndarray,
    ton: Optional[np.ndarray] = None,
    sampling_rate: float = 100.0
) -> Dict:
    """
    集成点火时刻检测

    结合多种方法的检测结果，提高精度和鲁棒性
    """
    # 方法1: MAD动态阈值
    detector_mad = IgnitionDetector(
        sampling_rate=sampling_rate,
        window_size=15,
        mad_coefficient=1.8
    )
    result_mad = detector_mad.detect_ignition_time(thrust, ton)

    # 方法2: 多尺度检测
    result_multiscale = detector_mad.detect_ignition_time_multiscale(thrust, ton)

    # 方法3: 导数方法
    detector_deriv = IgnitionDetectorV2(
        sampling_rate=sampling_rate,
        derivative_threshold=3.0
    )
    result_deriv = detector_deriv.detect(thrust, ton)

    # 收集有效结果
    times = []
    confidences = []

    for result in [result_mad, result_multiscale, result_deriv]:
        if not np.isnan(result['ignition_time']):
            times.append(result['ignition_time'])
            confidences.append(result['confidence'])

    if not times:
        return {
            'ignition_time': np.nan,
            'ignition_index': -1,
            'confidence': 0.0,
            'duration': 0.0,
            'method': 'none'
        }

    # 加权平均
    times = np.array(times)
    confidences = np.array(confidences)

    # 去除异常值
    if len(times) > 1:
        median_time = np.median(times)
        mad_time = np.median(np.abs(times - median_time))
        mask = np.abs(times - median_time) < 3 * mad_time + 0.1
        if np.sum(mask) > 0:
            times = times[mask]
            confidences = confidences[mask]

    weights = confidences / np.sum(confidences)
    ignition_time = np.sum(times * weights)
    ignition_index = int(ignition_time * sampling_rate)

    # 取最大持续时间
    duration = max(result_mad['duration'], result_multiscale['duration'], result_deriv['duration'])

    return {
        'ignition_time': ignition_time,
        'ignition_index': ignition_index,
        'confidence': np.mean(confidences),
        'duration': duration,
        'method': 'ensemble'
    }
