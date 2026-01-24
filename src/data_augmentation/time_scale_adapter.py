"""
时间尺度处理模块
处理100Hz短脉冲采样与实际变轨（分钟到小时级）的时间尺度差异
"""
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from scipy import signal, interpolate
from dataclasses import dataclass


@dataclass
class TimeScaleConfig:
    """时间尺度配置"""
    original_sampling_rate: float = 100.0  # 原始采样率 (Hz)
    target_sampling_rate: float = 1.0  # 目标采样率 (Hz)
    max_stretch_factor: float = 100.0  # 最大时间拉伸因子
    min_stretch_factor: float = 0.1  # 最小时间拉伸因子


class TimeScaleAdapter:
    """
    时间尺度适配器

    处理短脉冲数据到长时间变轨的转换
    """

    def __init__(
        self,
        config: Optional[TimeScaleConfig] = None,
        seed: Optional[int] = None
    ):
        """
        初始化时间尺度适配器

        Args:
            config: 时间尺度配置
            seed: 随机种子
        """
        self.config = config or TimeScaleConfig()
        self.rng = np.random.default_rng(seed)

    def resample(
        self,
        signal_data: np.ndarray,
        original_rate: float,
        target_rate: float,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        重采样信号

        Args:
            signal_data: 原始信号
            original_rate: 原始采样率
            target_rate: 目标采样率
            method: 插值方法 ('linear', 'cubic', 'nearest')

        Returns:
            重采样后的信号
        """
        n_original = len(signal_data)
        duration = n_original / original_rate

        # 计算新的样本数
        n_target = int(duration * target_rate)

        if n_target == n_original:
            return signal_data.copy()

        # 创建时间轴
        t_original = np.linspace(0, duration, n_original)
        t_target = np.linspace(0, duration, n_target)

        # 插值
        if method == 'linear':
            interpolator = interpolate.interp1d(
                t_original, signal_data, kind='linear', fill_value='extrapolate'
            )
        elif method == 'cubic':
            interpolator = interpolate.interp1d(
                t_original, signal_data, kind='cubic', fill_value='extrapolate'
            )
        elif method == 'nearest':
            interpolator = interpolate.interp1d(
                t_original, signal_data, kind='nearest', fill_value='extrapolate'
            )
        else:
            raise ValueError(f"未知的插值方法: {method}")

        return interpolator(t_target)

    def time_stretch(
        self,
        signal_data: np.ndarray,
        stretch_factor: float,
        preserve_amplitude: bool = True
    ) -> np.ndarray:
        """
        时间拉伸/压缩

        Args:
            signal_data: 原始信号
            stretch_factor: 拉伸因子 (>1为拉伸, <1为压缩)
            preserve_amplitude: 是否保持幅度

        Returns:
            拉伸后的信号
        """
        n_original = len(signal_data)
        n_stretched = int(n_original * stretch_factor)

        # 使用相位声码器方法进行时间拉伸
        stretched = self._phase_vocoder_stretch(signal_data, stretch_factor)

        # 幅度调整
        if preserve_amplitude:
            # 保持峰值幅度
            original_max = np.max(np.abs(signal_data))
            stretched_max = np.max(np.abs(stretched))
            if stretched_max > 0:
                stretched = stretched * (original_max / stretched_max)

        return stretched

    def _phase_vocoder_stretch(
        self,
        signal_data: np.ndarray,
        stretch_factor: float
    ) -> np.ndarray:
        """
        相位声码器时间拉伸

        Args:
            signal_data: 原始信号
            stretch_factor: 拉伸因子

        Returns:
            拉伸后的信号
        """
        n_original = len(signal_data)
        n_stretched = int(n_original * stretch_factor)

        # 简化实现：使用插值
        t_original = np.arange(n_original)
        t_stretched = np.linspace(0, n_original - 1, n_stretched)

        interpolator = interpolate.interp1d(
            t_original, signal_data, kind='cubic', fill_value='extrapolate'
        )

        return interpolator(t_stretched)

    def adapt_to_maneuver_duration(
        self,
        signal_data: np.ndarray,
        original_duration: float,
        target_duration: float,
        sampling_rate: float = 100.0
    ) -> Tuple[np.ndarray, float]:
        """
        将信号适配到目标变轨持续时间

        Args:
            signal_data: 原始信号
            original_duration: 原始持续时间 (s)
            target_duration: 目标持续时间 (s)
            sampling_rate: 采样率

        Returns:
            (适配后的信号, 新的采样率)
        """
        stretch_factor = target_duration / original_duration

        # 限制拉伸因子
        stretch_factor = np.clip(
            stretch_factor,
            self.config.min_stretch_factor,
            self.config.max_stretch_factor
        )

        # 时间拉伸
        stretched = self.time_stretch(signal_data, stretch_factor)

        # 计算新的采样率
        new_sampling_rate = sampling_rate / stretch_factor

        return stretched, new_sampling_rate


class MultiScaleProcessor:
    """
    多尺度处理器

    处理不同时间尺度的变轨信号
    """

    # 变轨类型的典型时间尺度
    MANEUVER_TIME_SCALES = {
        'attitude_control': (0.1, 10.0),      # 0.1-10秒
        'station_keeping': (1.0, 60.0),       # 1-60秒
        'collision_avoidance': (5.0, 120.0),  # 5-120秒
        'phase_adjustment': (10.0, 300.0),    # 10-300秒
        'orbit_raising': (30.0, 1800.0),      # 30秒-30分钟
        'orbit_lowering': (30.0, 1800.0),     # 30秒-30分钟
        'hohmann_transfer': (60.0, 3600.0),   # 1-60分钟
        'plane_change': (120.0, 7200.0),      # 2-120分钟
        'deorbit': (300.0, 3600.0),           # 5-60分钟
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.adapter = TimeScaleAdapter(seed=seed)

    def generate_multi_scale_signal(
        self,
        base_signal: np.ndarray,
        maneuver_type: str,
        sampling_rate: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        生成多尺度变轨信号

        Args:
            base_signal: 基础信号
            maneuver_type: 变轨类型
            sampling_rate: 采样率

        Returns:
            多尺度信号字典
        """
        time_scale = self.MANEUVER_TIME_SCALES.get(
            maneuver_type.lower(),
            (1.0, 60.0)
        )

        # 原始持续时间
        original_duration = len(base_signal) / sampling_rate

        # 生成多个时间尺度的信号
        scales = {}

        # 短时间尺度
        short_duration = self.rng.uniform(time_scale[0], time_scale[0] * 2)
        short_signal, short_rate = self.adapter.adapt_to_maneuver_duration(
            base_signal, original_duration, short_duration, sampling_rate
        )
        scales['short'] = {
            'signal': short_signal,
            'sampling_rate': short_rate,
            'duration': short_duration
        }

        # 中等时间尺度
        mid_duration = self.rng.uniform(
            (time_scale[0] + time_scale[1]) / 2 * 0.8,
            (time_scale[0] + time_scale[1]) / 2 * 1.2
        )
        mid_signal, mid_rate = self.adapter.adapt_to_maneuver_duration(
            base_signal, original_duration, mid_duration, sampling_rate
        )
        scales['medium'] = {
            'signal': mid_signal,
            'sampling_rate': mid_rate,
            'duration': mid_duration
        }

        # 长时间尺度
        long_duration = self.rng.uniform(time_scale[1] * 0.8, time_scale[1])
        long_signal, long_rate = self.adapter.adapt_to_maneuver_duration(
            base_signal, original_duration, long_duration, sampling_rate
        )
        scales['long'] = {
            'signal': long_signal,
            'sampling_rate': long_rate,
            'duration': long_duration
        }

        return scales

    def downsample_for_long_duration(
        self,
        signal_data: np.ndarray,
        original_rate: float,
        target_duration: float,
        max_samples: int = 10000
    ) -> Tuple[np.ndarray, float]:
        """
        为长时间变轨降采样

        Args:
            signal_data: 原始信号
            original_rate: 原始采样率
            target_duration: 目标持续时间
            max_samples: 最大样本数

        Returns:
            (降采样信号, 新采样率)
        """
        # 计算需要的采样率
        required_samples = int(target_duration * original_rate)

        if required_samples <= max_samples:
            # 不需要降采样
            return signal_data, original_rate

        # 计算新的采样率
        new_rate = max_samples / target_duration

        # 降采样
        downsampled = self.adapter.resample(
            signal_data, original_rate, new_rate, method='linear'
        )

        return downsampled, new_rate


class TemporalFeatureExtractor:
    """
    时域特征提取器

    提取适应不同时间尺度的特征
    """

    def __init__(self):
        pass

    def extract_scale_invariant_features(
        self,
        signal_data: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """
        提取尺度不变特征

        Args:
            signal_data: 信号数据
            sampling_rate: 采样率

        Returns:
            特征字典
        """
        features = {}

        # 归一化信号
        signal_norm = signal_data / (np.max(np.abs(signal_data)) + 1e-10)

        # 峰值特征（尺度不变）
        features['peak_normalized'] = np.max(signal_norm)
        features['peak_to_mean_ratio'] = np.max(signal_data) / (np.mean(signal_data) + 1e-10)

        # 形状特征（尺度不变）
        features['skewness'] = self._calculate_skewness(signal_norm)
        features['kurtosis'] = self._calculate_kurtosis(signal_norm)

        # 能量分布特征
        features['energy_concentration'] = self._calculate_energy_concentration(signal_norm)

        # 上升/下降时间比（尺度不变）
        features['rise_fall_ratio'] = self._calculate_rise_fall_ratio(signal_norm)

        # 占空比
        threshold = np.mean(signal_data) + np.std(signal_data)
        features['duty_cycle'] = np.sum(signal_data > threshold) / len(signal_data)

        return features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_energy_concentration(self, data: np.ndarray) -> float:
        """计算能量集中度"""
        energy = data ** 2
        total_energy = np.sum(energy)
        if total_energy < 1e-10:
            return 0.0

        # 计算能量集中在中心区域的比例
        center_start = len(data) // 4
        center_end = 3 * len(data) // 4
        center_energy = np.sum(energy[center_start:center_end])

        return center_energy / total_energy

    def _calculate_rise_fall_ratio(self, data: np.ndarray) -> float:
        """计算上升/下降时间比"""
        peak_idx = np.argmax(data)

        if peak_idx == 0 or peak_idx == len(data) - 1:
            return 1.0

        rise_time = peak_idx
        fall_time = len(data) - peak_idx - 1

        return rise_time / (fall_time + 1e-10)


class LongDurationSimulator:
    """
    长时间变轨模拟器

    模拟持续数分钟到数小时的变轨过程
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def simulate_continuous_thrust(
        self,
        duration: float,
        sampling_rate: float = 1.0,
        thrust_profile: str = 'constant',
        thrust_level: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        模拟连续推力变轨

        Args:
            duration: 持续时间 (s)
            sampling_rate: 采样率 (Hz)
            thrust_profile: 推力曲线类型
            thrust_level: 推力水平

        Returns:
            模拟数据字典
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate

        if thrust_profile == 'constant':
            thrust = np.full(n_samples, thrust_level)

        elif thrust_profile == 'ramp_up':
            thrust = thrust_level * (t / duration)

        elif thrust_profile == 'ramp_down':
            thrust = thrust_level * (1 - t / duration)

        elif thrust_profile == 'bell':
            # 钟形曲线
            thrust = thrust_level * np.exp(
                -0.5 * ((t - duration/2) / (duration/4)) ** 2
            )

        elif thrust_profile == 'trapezoidal':
            # 梯形曲线
            ramp_time = duration * 0.1
            thrust = np.zeros(n_samples)
            for i, ti in enumerate(t):
                if ti < ramp_time:
                    thrust[i] = thrust_level * ti / ramp_time
                elif ti > duration - ramp_time:
                    thrust[i] = thrust_level * (duration - ti) / ramp_time
                else:
                    thrust[i] = thrust_level

        else:
            thrust = np.full(n_samples, thrust_level)

        # 添加小幅波动
        thrust += self.rng.normal(0, thrust_level * 0.02, n_samples)
        thrust = np.maximum(thrust, 0)

        return {
            'time': t,
            'thrust': thrust,
            'duration': duration,
            'profile': thrust_profile
        }

    def simulate_pulsed_thrust(
        self,
        duration: float,
        sampling_rate: float = 1.0,
        pulse_width: float = 1.0,
        pulse_interval: float = 10.0,
        thrust_level: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        模拟脉冲推力变轨

        Args:
            duration: 总持续时间 (s)
            sampling_rate: 采样率 (Hz)
            pulse_width: 脉冲宽度 (s)
            pulse_interval: 脉冲间隔 (s)
            thrust_level: 推力水平

        Returns:
            模拟数据字典
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate

        thrust = np.zeros(n_samples)

        # 生成脉冲
        current_time = 0
        pulse_count = 0
        while current_time < duration:
            pulse_start = current_time
            pulse_end = min(current_time + pulse_width, duration)

            # 找到对应的索引
            start_idx = int(pulse_start * sampling_rate)
            end_idx = int(pulse_end * sampling_rate)

            # 添加脉冲
            thrust[start_idx:end_idx] = thrust_level

            current_time += pulse_interval
            pulse_count += 1

        # 添加噪声
        thrust += self.rng.normal(0, thrust_level * 0.01, n_samples)
        thrust = np.maximum(thrust, 0)

        return {
            'time': t,
            'thrust': thrust,
            'duration': duration,
            'pulse_count': pulse_count,
            'pulse_width': pulse_width,
            'pulse_interval': pulse_interval
        }

    def extend_short_signal(
        self,
        short_signal: np.ndarray,
        short_duration: float,
        target_duration: float,
        extension_method: str = 'repeat'
    ) -> np.ndarray:
        """
        将短信号扩展到长时间

        Args:
            short_signal: 短信号
            short_duration: 短信号持续时间
            target_duration: 目标持续时间
            extension_method: 扩展方法 ('repeat', 'interpolate', 'pad')

        Returns:
            扩展后的信号
        """
        n_short = len(short_signal)
        n_target = int(n_short * target_duration / short_duration)

        if extension_method == 'repeat':
            # 重复信号
            n_repeats = int(np.ceil(n_target / n_short))
            extended = np.tile(short_signal, n_repeats)[:n_target]

        elif extension_method == 'interpolate':
            # 插值扩展
            t_short = np.linspace(0, 1, n_short)
            t_target = np.linspace(0, 1, n_target)
            interpolator = interpolate.interp1d(
                t_short, short_signal, kind='cubic', fill_value='extrapolate'
            )
            extended = interpolator(t_target)

        elif extension_method == 'pad':
            # 零填充
            extended = np.zeros(n_target)
            # 将短信号放在中间
            start_idx = (n_target - n_short) // 2
            extended[start_idx:start_idx + n_short] = short_signal

        else:
            raise ValueError(f"未知的扩展方法: {extension_method}")

        return extended
