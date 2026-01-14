"""
特征提取模块：提取P, T, R三维特征向量
"""
import numpy as np
from scipy import signal
from typing import Dict, Tuple, List, Optional
from src.utils import timing_decorator
from src.config import Config

class ManeuverFeatureExtractor:
    """变轨特征提取器"""

    def __init__(self, config: Config = None):
        """
        初始化特征提取器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.wavelength_226 = self.config.FEATURE_CONFIG['wavelength_226nm']
        self.wavelength_306 = self.config.FEATURE_CONFIG['wavelength_306nm']

    def extract_peak_intensity(self, intensity: np.ndarray) -> float:
        """
        提取辐射强度峰值 (P)

        Args:
            intensity: 辐射强度时间序列

        Returns:
            峰值强度
        """
        # 找到所有峰值
        peaks, properties = signal.find_peaks(intensity, height=0)

        if len(peaks) > 0:
            # 返回最大峰值
            peak_intensity = np.max(properties['peak_heights'])
        else:
            # 如果没有明显峰值，返回最大值
            peak_intensity = np.max(intensity)

        return peak_intensity

    def extract_duration(self,
                         intensity: np.ndarray,
                         timestamp: np.ndarray,
                         threshold_ratio: float = 0.5) -> float:
        """
        提取持续时间 (T)

        Args:
            intensity: 辐射强度时间序列
            timestamp: 时间戳数组
            threshold_ratio: 阈值比例（相对于峰值）

        Returns:
            持续时间（秒）
        """
        # 找到峰值
        peak_value = self.extract_peak_intensity(intensity)
        threshold = peak_value * threshold_ratio

        # 找到超过阈值的区间
        above_threshold = intensity >= threshold

        if not np.any(above_threshold):
            return 0.0

        # 找到连续区间
        diff = np.diff(above_threshold.astype(int))
        start_indices = np.where(diff == 1)[0] + 1
        end_indices = np.where(diff == -1)[0]

        # 处理边界情况
        if above_threshold[0]:
            start_indices = np.concatenate([[0], start_indices])
        if above_threshold[-1]:
            end_indices = np.concatenate([end_indices, [len(intensity) - 1]])

        if len(start_indices) == 0 or len(end_indices) == 0:
            return 0.0

        # 找到最长的连续区间
        durations = timestamp[end_indices] - timestamp[start_indices]
        max_duration = np.max(durations)

        return max_duration

    def extract_intensity_ratio(self,
                                 intensity_226: np.ndarray,
                                 intensity_306: np.ndarray) -> float:
        """
        提取226nm/306nm强度比 (R)

        Args:
            intensity_226: 226nm波长辐射强度
            intensity_306: 306nm波长辐射强度

        Returns:
            强度比
        """
        # 计算峰值比
        peak_226 = self.extract_peak_intensity(intensity_226)
        peak_306 = self.extract_peak_intensity(intensity_306)

        if peak_306 == 0:
            # 避免除以零
            return 0.0

        ratio = peak_226 / peak_306
        return ratio

    @timing_decorator
    def extract_features(self,
                         data: Dict[str, np.ndarray],
                         window_size: Optional[int] = None) -> np.ndarray:
        """
        提取三维特征向量 (P, T, R)

        Args:
            data: 包含辐射强度数据的字典
            window_size: 滑动窗口大小（可选）

        Returns:
            特征矩阵 (N, 3) - N个样本,每个样本3个特征
        """
        intensity_226 = data[f'intensity_{self.wavelength_226}nm']
        intensity_306 = data[f'intensity_{self.wavelength_306}nm']
        timestamp = data['timestamp']

        if window_size is None:
            # 全局特征提取
            features = self._extract_global_features(
                intensity_226, intensity_306, timestamp
            )
        else:
            # 滑动窗口特征提取
            features = self._extract_windowed_features(
                intensity_226, intensity_306, timestamp, window_size
            )

        return features

    def _extract_global_features(self,
                                  intensity_226: np.ndarray,
                                  intensity_306: np.ndarray,
                                  timestamp: np.ndarray) -> np.ndarray:
        """提取全局特征（单个特征向量）"""
        P = self.extract_peak_intensity(intensity_226)
        T = self.extract_duration(intensity_226, timestamp)
        R = self.extract_intensity_ratio(intensity_226, intensity_306)

        features = np.array([[P, T, R]])
        return features

    def _extract_windowed_features(self,
                                     intensity_226: np.ndarray,
                                     intensity_306: np.ndarray,
                                     timestamp: np.ndarray,
                                     window_size: int) -> np.ndarray:
        """使用滑动窗口提取多个特征向量"""
        n_samples = len(intensity_226)
        stride = window_size // 2  # 50%重叠

        features_list = []

        for start in range(0, n_samples - window_size + 1, stride):
            end = start + window_size

            # 窗口数据
            window_226 = intensity_226[start:end]
            window_306 = intensity_306[start:end]
            window_time = timestamp[start:end]

            # 提取特征
            P = self.extract_peak_intensity(window_226)
            T = self.extract_duration(window_226, window_time)
            R = self.extract_intensity_ratio(window_226, window_306)

            features_list.append([P, T, R])

        features = np.array(features_list)
        return features

    def extract_frequency_features(self, intensity: np.ndarray) -> Dict[str, float]:
        """
        提取频域特征（可选的额外特征）

        Args:
            intensity: 辐射强度时间序列

        Returns:
            频域特征字典
        """
        # 计算FFT
        fft_values = np.fft.fft(intensity)
        fft_freq = np.fft.fftfreq(len(intensity))
        power_spectrum = np.abs(fft_values) ** 2

        # 只取正频率
        positive_freq_idx = fft_freq > 0
        power_spectrum = power_spectrum[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]

        # 提取频域特征
        features = {
            'dominant_frequency': fft_freq[np.argmax(power_spectrum)],
            'spectral_centroid': np.sum(fft_freq * power_spectrum) / np.sum(power_spectrum),
            'spectral_bandwidth': np.sqrt(np.sum(((fft_freq - features.get('spectral_centroid', 0)) ** 2) * power_spectrum) / np.sum(power_spectrum)),
            'total_power': np.sum(power_spectrum),
        }

        return features

    def extract_statistical_features(self, intensity: np.ndarray) -> Dict[str, float]:
        """
        提取统计特征（可选的额外特征）

        Args:
            intensity: 辐射强度时间序列

        Returns:
            统计特征字典
        """
        features = {
            'mean': np.mean(intensity),
            'std': np.std(intensity),
            'variance': np.var(intensity),
            'skewness': self._calculate_skewness(intensity),
            'kurtosis': self._calculate_kurtosis(intensity),
            'rms': np.sqrt(np.mean(intensity ** 2)),
        }

        return features

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def create_feature_dataset(self,
                                data_dict: Dict[str, np.ndarray],
                                labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        创建特征数据集

        Args:
            data_dict: 数据字典
            labels: 标签数组（可选）

        Returns:
            (features, labels) 元组
        """
        features = self.extract_features(data_dict, window_size=None)

        print(f"[Features] 提取特征完成")
        print(f"  特征形状: {features.shape}")
        print(f"  特征统计:")
        print(f"    P (峰值强度): {features[:, 0].mean():.4f} ± {features[:, 0].std():.4f}")
        print(f"    T (持续时间): {features[:, 1].mean():.4f} ± {features[:, 1].std():.4f}")
        print(f"    R (强度比):   {features[:, 2].mean():.4f} ± {features[:, 2].std():.4f}")

        return features, labels
