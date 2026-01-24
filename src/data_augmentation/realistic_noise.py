"""
真实噪声模型模块
模拟实际辐射观测中的各类噪声：大气闪烁、背景杂波、探测器噪声
"""
import numpy as np
from typing import Tuple, Optional, Dict, Union
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class AtmosphericScintillation:
    """
    大气闪烁噪声模型

    大气湍流导致的辐射强度随机波动，主要特征：
    - 低频波动（0.1-10Hz）
    - 幅度与大气条件相关
    - 具有时间相关性（非白噪声）
    """

    def __init__(
        self,
        cn2: float = 1e-14,  # 大气折射率结构常数 (m^(-2/3))
        wavelength: float = 226e-9,  # 观测波长 (m)
        zenith_angle: float = 45.0,  # 天顶角 (度)
        altitude: float = 500e3,  # 目标高度 (m)
        seed: Optional[int] = None
    ):
        """
        初始化大气闪烁模型

        Args:
            cn2: 大气折射率结构常数，典型值1e-15(好)到1e-13(差)
            wavelength: 观测波长
            zenith_angle: 天顶角
            altitude: 目标高度
            seed: 随机种子
        """
        self.cn2 = cn2
        self.wavelength = wavelength
        self.zenith_angle = np.radians(zenith_angle)
        self.altitude = altitude
        self.rng = np.random.default_rng(seed)

        # 计算Rytov方差（弱湍流近似）
        self.sigma_rytov = self._calculate_rytov_variance()

    def _calculate_rytov_variance(self) -> float:
        """计算Rytov方差"""
        k = 2 * np.pi / self.wavelength  # 波数
        L = self.altitude / np.cos(self.zenith_angle)  # 传播路径长度

        # Rytov方差公式（平面波近似）
        sigma2 = 1.23 * self.cn2 * (k ** (7/6)) * (L ** (11/6))
        return sigma2

    def generate(
        self,
        n_samples: int,
        sampling_rate: float = 100.0
    ) -> np.ndarray:
        """
        生成大气闪烁噪声序列

        Args:
            n_samples: 样本数量
            sampling_rate: 采样率 (Hz)

        Returns:
            闪烁噪声序列（乘性因子，均值为1）
        """
        # 生成低频相关噪声
        # 大气闪烁的典型频率范围：0.1-10Hz
        cutoff_freq = 10.0 / (sampling_rate / 2)  # 归一化截止频率

        # 生成白噪声
        white_noise = self.rng.standard_normal(n_samples)

        # 低通滤波产生相关噪声
        if cutoff_freq < 1.0:
            b, a = signal.butter(4, cutoff_freq, btype='low')
            correlated_noise = signal.filtfilt(b, a, white_noise)
        else:
            correlated_noise = white_noise

        # 归一化并缩放到Rytov方差
        correlated_noise = correlated_noise / np.std(correlated_noise)

        # 对数正态分布的闪烁（弱湍流）
        scintillation = np.exp(
            np.sqrt(self.sigma_rytov) * correlated_noise - self.sigma_rytov / 2
        )

        return scintillation


class BackgroundClutter:
    """
    背景杂波噪声模型

    包括：
    - 天空背景辐射
    - 地球反照
    - 其他空间目标干扰
    """

    def __init__(
        self,
        background_level: float = 100.0,  # 背景辐射水平
        clutter_density: float = 0.01,  # 杂波密度（每秒出现概率）
        clutter_amplitude: float = 50.0,  # 杂波幅度
        seed: Optional[int] = None
    ):
        """
        初始化背景杂波模型

        Args:
            background_level: 平均背景辐射水平
            clutter_density: 杂波出现密度
            clutter_amplitude: 杂波幅度
            seed: 随机种子
        """
        self.background_level = background_level
        self.clutter_density = clutter_density
        self.clutter_amplitude = clutter_amplitude
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_samples: int,
        sampling_rate: float = 100.0
    ) -> np.ndarray:
        """
        生成背景杂波噪声

        Args:
            n_samples: 样本数量
            sampling_rate: 采样率 (Hz)

        Returns:
            背景杂波噪声序列（加性）
        """
        # 基础背景噪声（高斯）
        background = self.rng.normal(
            self.background_level,
            self.background_level * 0.1,
            n_samples
        )

        # 随机杂波脉冲（泊松过程）
        duration = n_samples / sampling_rate
        expected_clutters = int(self.clutter_density * duration * sampling_rate)

        if expected_clutters > 0:
            # 随机杂波位置
            clutter_positions = self.rng.choice(
                n_samples,
                size=min(expected_clutters, n_samples // 10),
                replace=False
            )

            # 添加杂波脉冲
            for pos in clutter_positions:
                # 杂波持续时间（1-10个采样点）
                clutter_duration = self.rng.integers(1, 11)
                end_pos = min(pos + clutter_duration, n_samples)

                # 杂波幅度（指数分布）
                amplitude = self.rng.exponential(self.clutter_amplitude)

                # 添加高斯形状的杂波
                t = np.arange(end_pos - pos)
                clutter_shape = amplitude * np.exp(
                    -0.5 * ((t - clutter_duration/2) / (clutter_duration/4)) ** 2
                )
                background[pos:end_pos] += clutter_shape

        return background - self.background_level  # 返回零均值噪声


class DetectorNoise:
    """
    探测器噪声模型

    包括：
    - 散粒噪声（光子噪声）
    - 暗电流噪声
    - 读出噪声
    - 量化噪声
    """

    def __init__(
        self,
        dark_current: float = 10.0,  # 暗电流 (e-/s)
        read_noise: float = 5.0,  # 读出噪声 (e-)
        quantum_efficiency: float = 0.8,  # 量子效率
        bit_depth: int = 16,  # ADC位深
        full_well: float = 100000.0,  # 满阱容量 (e-)
        seed: Optional[int] = None
    ):
        """
        初始化探测器噪声模型

        Args:
            dark_current: 暗电流
            read_noise: 读出噪声
            quantum_efficiency: 量子效率
            bit_depth: ADC位深
            full_well: 满阱容量
            seed: 随机种子
        """
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.quantum_efficiency = quantum_efficiency
        self.bit_depth = bit_depth
        self.full_well = full_well
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        signal_level: Union[float, np.ndarray],
        integration_time: float = 0.01  # 积分时间 (s)
    ) -> np.ndarray:
        """
        生成探测器噪声

        Args:
            signal_level: 信号水平（光子数或数组）
            integration_time: 积分时间

        Returns:
            探测器噪声序列
        """
        signal_level = np.atleast_1d(signal_level)
        n_samples = len(signal_level)

        # 散粒噪声（泊松分布）
        # 信号光子数
        signal_electrons = signal_level * self.quantum_efficiency
        shot_noise = np.sqrt(np.maximum(signal_electrons, 0))
        shot_noise = self.rng.normal(0, shot_noise)

        # 暗电流噪声
        dark_electrons = self.dark_current * integration_time
        dark_noise = self.rng.poisson(dark_electrons, n_samples) - dark_electrons

        # 读出噪声（高斯）
        read_noise = self.rng.normal(0, self.read_noise, n_samples)

        # 量化噪声
        lsb = self.full_well / (2 ** self.bit_depth)
        quantization_noise = self.rng.uniform(-lsb/2, lsb/2, n_samples)

        # 总噪声
        total_noise = shot_noise + dark_noise + read_noise + quantization_noise

        return total_noise


class RealisticNoiseModel:
    """
    综合真实噪声模型

    整合大气闪烁、背景杂波和探测器噪声
    """

    def __init__(
        self,
        snr_target: float = 5.0,  # 目标信噪比 (dB)
        atmospheric_params: Optional[Dict] = None,
        background_params: Optional[Dict] = None,
        detector_params: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        初始化综合噪声模型

        Args:
            snr_target: 目标信噪比
            atmospheric_params: 大气闪烁参数
            background_params: 背景杂波参数
            detector_params: 探测器噪声参数
            seed: 随机种子
        """
        self.snr_target = snr_target
        self.seed = seed

        # 初始化各噪声模型
        atm_params = atmospheric_params or {}
        bg_params = background_params or {}
        det_params = detector_params or {}

        self.atmospheric = AtmosphericScintillation(seed=seed, **atm_params)
        self.background = BackgroundClutter(seed=seed, **bg_params)
        self.detector = DetectorNoise(seed=seed, **det_params)

    def apply_noise(
        self,
        clean_signal: np.ndarray,
        sampling_rate: float = 100.0,
        noise_components: Optional[Dict[str, bool]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        对干净信号应用真实噪声

        Args:
            clean_signal: 干净信号
            sampling_rate: 采样率
            noise_components: 启用的噪声成分

        Returns:
            (带噪信号, 各噪声成分字典)
        """
        n_samples = len(clean_signal)

        # 默认启用所有噪声成分
        components = noise_components or {
            'atmospheric': True,
            'background': True,
            'detector': True
        }

        noisy_signal = clean_signal.copy()
        noise_dict = {}

        # 大气闪烁（乘性噪声）
        if components.get('atmospheric', True):
            scintillation = self.atmospheric.generate(n_samples, sampling_rate)
            noisy_signal = noisy_signal * scintillation
            noise_dict['atmospheric'] = scintillation

        # 背景杂波（加性噪声）
        if components.get('background', True):
            clutter = self.background.generate(n_samples, sampling_rate)
            noisy_signal = noisy_signal + clutter
            noise_dict['background'] = clutter

        # 探测器噪声（加性噪声）
        if components.get('detector', True):
            det_noise = self.detector.generate(noisy_signal)
            noisy_signal = noisy_signal + det_noise
            noise_dict['detector'] = det_noise

        # 调整到目标信噪比
        noisy_signal = self._adjust_snr(clean_signal, noisy_signal)

        return noisy_signal, noise_dict

    def _adjust_snr(
        self,
        clean_signal: np.ndarray,
        noisy_signal: np.ndarray
    ) -> np.ndarray:
        """
        调整信号到目标信噪比

        Args:
            clean_signal: 干净信号
            noisy_signal: 带噪信号

        Returns:
            调整后的信号
        """
        # 计算当前信噪比
        signal_power = np.mean(clean_signal ** 2)
        noise = noisy_signal - clean_signal
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return noisy_signal

        current_snr_db = 10 * np.log10(signal_power / noise_power)

        # 计算需要的噪声缩放因子
        target_noise_power = signal_power / (10 ** (self.snr_target / 10))
        scale_factor = np.sqrt(target_noise_power / noise_power)

        # 调整噪声
        adjusted_signal = clean_signal + scale_factor * noise

        return adjusted_signal

    def generate_noise_profile(
        self,
        n_samples: int,
        sampling_rate: float = 100.0,
        signal_level: float = 1000.0
    ) -> Dict[str, np.ndarray]:
        """
        生成噪声特性分析数据

        Args:
            n_samples: 样本数量
            sampling_rate: 采样率
            signal_level: 信号水平

        Returns:
            噪声特性字典
        """
        # 生成各类噪声
        scintillation = self.atmospheric.generate(n_samples, sampling_rate)
        clutter = self.background.generate(n_samples, sampling_rate)
        det_noise = self.detector.generate(
            np.full(n_samples, signal_level)
        )

        # 计算功率谱密度
        freqs = np.fft.rfftfreq(n_samples, 1/sampling_rate)

        profile = {
            'frequencies': freqs,
            'scintillation_psd': np.abs(np.fft.rfft(scintillation)) ** 2,
            'clutter_psd': np.abs(np.fft.rfft(clutter)) ** 2,
            'detector_psd': np.abs(np.fft.rfft(det_noise)) ** 2,
            'scintillation_std': np.std(scintillation),
            'clutter_std': np.std(clutter),
            'detector_std': np.std(det_noise)
        }

        return profile
