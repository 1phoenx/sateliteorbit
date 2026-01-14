"""
数据预处理模块：处理卫星辐射强度数据
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from scipy import signal
from src.utils import calculate_snr, timing_decorator
from src.config import Config

class RadiationDataProcessor:
    """辐射强度数据处理器"""

    def __init__(self, config: Config = None):
        """
        初始化数据处理器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.sampling_rate = self.config.DATA_CONFIG['sampling_rate']
        self.min_snr = self.config.DATA_CONFIG['signal_noise_ratio']

    @timing_decorator
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        加载原始辐射强度数据

        Args:
            filepath: 数据文件路径

        Returns:
            DataFrame包含时间序列和辐射强度数据
        """
        # 支持多种格式
        file_path = Path(filepath)

        if file_path.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif file_path.suffix == '.h5':
            df = pd.read_hdf(filepath)
        elif file_path.suffix in ['.npy', '.npz']:
            data = np.load(filepath)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        print(f"[Data] 加载数据: {filepath}")
        print(f"  形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")

        return df

    def denoise_signal(self, signal_data: np.ndarray,
                       method: str = 'butterworth',
                       **kwargs) -> np.ndarray:
        """
        信号去噪

        Args:
            signal_data: 原始信号
            method: 去噪方法 ('butterworth', 'wavelet', 'savgol')
            **kwargs: 方法特定参数

        Returns:
            去噪后的信号
        """
        if method == 'butterworth':
            # Butterworth低通滤波器
            order = kwargs.get('order', 4)
            cutoff = kwargs.get('cutoff', 0.3)
            b, a = signal.butter(order, cutoff, btype='low')
            filtered = signal.filtfilt(b, a, signal_data)

        elif method == 'savgol':
            # Savitzky-Golay滤波器
            window_length = kwargs.get('window_length', 11)
            polyorder = kwargs.get('polyorder', 3)
            filtered = signal.savgol_filter(signal_data, window_length, polyorder)

        elif method == 'wavelet':
            # 小波去噪（需要安装pywt）
            try:
                import pywt
                wavelet = kwargs.get('wavelet', 'db4')
                level = kwargs.get('level', 1)
                coeffs = pywt.wavedec(signal_data, wavelet, level=level)
                threshold = kwargs.get('threshold', np.std(coeffs[-1]))
                coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                filtered = pywt.waverec(coeffs, wavelet)[:len(signal_data)]
            except ImportError:
                print("[Warning] pywt未安装，使用butterworth滤波")
                return self.denoise_signal(signal_data, method='butterworth')

        else:
            raise ValueError(f"未知的去噪方法: {method}")

        return filtered

    def extract_wavelength_intensity(self,
                                      df: pd.DataFrame,
                                      wavelength: int) -> np.ndarray:
        """
        提取特定波长的辐射强度

        Args:
            df: 数据DataFrame
            wavelength: 波长 (nm)

        Returns:
            强度时间序列
        """
        # 假设列名格式为 'intensity_XXXnm'
        col_name = f'intensity_{wavelength}nm'

        if col_name in df.columns:
            return df[col_name].values
        else:
            # 尝试其他可能的列名
            possible_names = [
                f'{wavelength}nm',
                f'I_{wavelength}',
                f'wavelength_{wavelength}'
            ]
            for name in possible_names:
                if name in df.columns:
                    return df[name].values

            raise KeyError(f"未找到波长 {wavelength}nm 的数据列")

    def check_signal_quality(self,
                             signal_data: np.ndarray,
                             noise_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        检查信号质量

        Args:
            signal_data: 信号数据
            noise_data: 噪声数据（可选）

        Returns:
            质量指标字典
        """
        quality_metrics = {}

        # 信号统计
        quality_metrics['mean'] = np.mean(signal_data)
        quality_metrics['std'] = np.std(signal_data)
        quality_metrics['max'] = np.max(signal_data)
        quality_metrics['min'] = np.min(signal_data)

        # 计算信噪比
        if noise_data is not None:
            quality_metrics['snr'] = calculate_snr(signal_data, noise_data)
        else:
            # 估计噪声（假设信号的高频成分为噪声）
            b, a = signal.butter(4, 0.1, btype='high')
            estimated_noise = signal.filtfilt(b, a, signal_data)
            quality_metrics['snr_estimated'] = calculate_snr(signal_data, estimated_noise)

        return quality_metrics

    @timing_decorator
    def preprocess_pipeline(self,
                            df: pd.DataFrame,
                            denoise: bool = True,
                            normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        完整的预处理流程

        Args:
            df: 原始数据DataFrame
            denoise: 是否去噪
            normalize: 是否归一化

        Returns:
            处理后的数据字典
        """
        processed_data = {}

        # 提取226nm和306nm波长数据
        wavelengths = [
            self.config.FEATURE_CONFIG['wavelength_226nm'],
            self.config.FEATURE_CONFIG['wavelength_306nm']
        ]

        for wl in wavelengths:
            try:
                intensity = self.extract_wavelength_intensity(df, wl)

                # 去噪
                if denoise:
                    intensity = self.denoise_signal(intensity, method='butterworth')

                # 归一化
                if normalize:
                    intensity = (intensity - np.mean(intensity)) / np.std(intensity)

                processed_data[f'intensity_{wl}nm'] = intensity

                # 检查质量
                quality = self.check_signal_quality(intensity)
                print(f"[Quality] {wl}nm - SNR: {quality.get('snr_estimated', 'N/A'):.2f} dB")

            except KeyError as e:
                print(f"[Warning] {e}")
                continue

        # 提取时间信息
        if 'timestamp' in df.columns:
            processed_data['timestamp'] = df['timestamp'].values
        elif 'time' in df.columns:
            processed_data['timestamp'] = df['time'].values
        else:
            # 创建虚拟时间戳
            processed_data['timestamp'] = np.arange(len(df)) / self.sampling_rate

        # 提取标签（如果存在）
        if 'label' in df.columns:
            processed_data['labels'] = df['label'].values
        elif 'maneuver' in df.columns:
            processed_data['labels'] = df['maneuver'].values

        return processed_data

    def save_processed_data(self, data: Dict[str, np.ndarray], filepath: str):
        """
        保存处理后的数据

        Args:
            data: 数据字典
            filepath: 保存路径
        """
        np.savez(filepath, **data)
        print(f"[Data] 已保存处理后的数据: {filepath}")

    def load_processed_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        加载处理后的数据

        Args:
            filepath: 文件路径

        Returns:
            数据字典
        """
        loaded = np.load(filepath)
        data = {key: loaded[key] for key in loaded.files}
        print(f"[Data] 已加载处理后的数据: {filepath}")
        return data
