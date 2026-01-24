"""
数据增强统一接口模块
整合所有增强模块，提供统一的数据增强流水线
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

from .realistic_noise import (
    RealisticNoiseModel,
    AtmosphericScintillation,
    BackgroundClutter,
    DetectorNoise
)
from .distance_attenuation import (
    DistanceAttenuationModel,
    DistanceVariationSimulator,
    OrbitParameters
)
from .maneuver_type_mapper import (
    ManeuverTypeLabelMapper,
    ManeuverTypeAugmenter,
    ManeuverType,
    MANEUVER_CHARACTERISTICS
)
from .time_scale_adapter import (
    TimeScaleAdapter,
    MultiScaleProcessor,
    LongDurationSimulator,
    TemporalFeatureExtractor,
    TimeScaleConfig
)


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 噪声配置
    enable_noise: bool = True
    target_snr: float = 5.0  # 目标信噪比 (dB)
    atmospheric_cn2: float = 1e-14  # 大气折射率结构常数
    background_level: float = 100.0  # 背景辐射水平
    clutter_density: float = 0.01  # 杂波密度

    # 距离衰减配置
    enable_distance_attenuation: bool = True
    reference_distance: float = 500e3  # 参考距离 (m)
    min_distance: float = 400e3  # 最小距离 (m)
    max_distance: float = 2000e3  # 最大距离 (m)
    include_atmospheric_extinction: bool = True

    # 变轨类型配置
    enable_maneuver_labeling: bool = True
    augment_with_maneuver_types: bool = True
    maneuver_augmentation_factor: int = 2

    # 时间尺度配置
    enable_time_scale_adaptation: bool = True
    original_sampling_rate: float = 100.0  # 原始采样率 (Hz)
    target_sampling_rate: float = 1.0  # 目标采样率 (Hz)
    enable_multi_scale: bool = True

    # 随机种子
    seed: int = 42


class DataAugmentationPipeline:
    """
    数据增强流水线

    整合所有增强模块，提供统一的接口
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        初始化数据增强流水线

        Args:
            config: 增强配置
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # 初始化各模块
        self._init_modules()

    def _init_modules(self):
        """初始化各增强模块"""
        # 噪声模型
        if self.config.enable_noise:
            self.noise_model = RealisticNoiseModel(
                snr_target=self.config.target_snr,
                atmospheric_params={'cn2': self.config.atmospheric_cn2},
                background_params={
                    'background_level': self.config.background_level,
                    'clutter_density': self.config.clutter_density
                },
                seed=self.config.seed
            )
        else:
            self.noise_model = None

        # 距离衰减模型
        if self.config.enable_distance_attenuation:
            self.distance_model = DistanceAttenuationModel(
                reference_distance=self.config.reference_distance,
                seed=self.config.seed
            )
            self.distance_simulator = DistanceVariationSimulator(
                seed=self.config.seed
            )
        else:
            self.distance_model = None
            self.distance_simulator = None

        # 变轨类型映射器
        if self.config.enable_maneuver_labeling:
            self.maneuver_mapper = ManeuverTypeLabelMapper(seed=self.config.seed)
            self.maneuver_augmenter = ManeuverTypeAugmenter(seed=self.config.seed)
        else:
            self.maneuver_mapper = None
            self.maneuver_augmenter = None

        # 时间尺度适配器
        if self.config.enable_time_scale_adaptation:
            time_config = TimeScaleConfig(
                original_sampling_rate=self.config.original_sampling_rate,
                target_sampling_rate=self.config.target_sampling_rate
            )
            self.time_adapter = TimeScaleAdapter(config=time_config, seed=self.config.seed)
            self.multi_scale_processor = MultiScaleProcessor(seed=self.config.seed)
            self.long_duration_simulator = LongDurationSimulator(seed=self.config.seed)
            self.temporal_extractor = TemporalFeatureExtractor()
        else:
            self.time_adapter = None
            self.multi_scale_processor = None
            self.long_duration_simulator = None
            self.temporal_extractor = None

    def augment_signal(
        self,
        signal_data: np.ndarray,
        sampling_rate: float = 100.0,
        maneuver_type: Optional[str] = None,
        distance: Optional[float] = None,
        augmentation_options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        对单个信号进行增强

        Args:
            signal_data: 原始信号
            sampling_rate: 采样率
            maneuver_type: 变轨类型（可选）
            distance: 观测距离（可选）
            augmentation_options: 增强选项

        Returns:
            增强结果字典
        """
        options = augmentation_options or {}
        result = {
            'original_signal': signal_data.copy(),
            'augmented_signal': signal_data.copy(),
            'metadata': {}
        }

        current_signal = signal_data.copy()

        # 1. 距离衰减
        if self.config.enable_distance_attenuation and options.get('distance', True):
            if distance is None:
                # 随机生成距离
                distance = self.distance_simulator.simulate_pass(
                    len(current_signal),
                    sampling_rate,
                    self.config.min_distance,
                    self.config.max_distance,
                    pass_type='random'
                )

            attenuated, attn_params = self.distance_model.apply_attenuation(
                current_signal,
                distance=distance,
                include_atmospheric=self.config.include_atmospheric_extinction
            )
            current_signal = attenuated
            result['metadata']['distance_attenuation'] = attn_params

        # 2. 添加噪声
        if self.config.enable_noise and options.get('noise', True):
            noisy, noise_components = self.noise_model.apply_noise(
                current_signal,
                sampling_rate
            )
            current_signal = noisy
            result['metadata']['noise_components'] = {
                k: np.std(v) for k, v in noise_components.items()
            }

        # 3. 时间尺度适配
        if self.config.enable_time_scale_adaptation and options.get('time_scale', True):
            if maneuver_type:
                # 根据变轨类型调整时间尺度
                scales = self.multi_scale_processor.generate_multi_scale_signal(
                    current_signal,
                    maneuver_type,
                    sampling_rate
                )
                result['multi_scale_signals'] = scales

                # 使用中等尺度作为主输出
                current_signal = scales['medium']['signal']
                result['metadata']['time_scale'] = {
                    'duration': scales['medium']['duration'],
                    'sampling_rate': scales['medium']['sampling_rate']
                }

        result['augmented_signal'] = current_signal

        # 4. 提取尺度不变特征
        if self.temporal_extractor:
            result['scale_invariant_features'] = \
                self.temporal_extractor.extract_scale_invariant_features(
                    current_signal, sampling_rate
                )

        return result

    def augment_dataset(
        self,
        feature_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        signal_column: Optional[str] = None,
        augmentation_factor: int = 1
    ) -> pd.DataFrame:
        """
        对整个数据集进行增强

        Args:
            feature_df: 特征DataFrame
            metadata_df: 元数据DataFrame（可选）
            signal_column: 信号列名（如果有原始信号）
            augmentation_factor: 增强倍数

        Returns:
            增强后的DataFrame
        """
        augmented_rows = []

        for idx, row in feature_df.iterrows():
            # 保留原始数据
            original_row = row.to_dict()
            original_row['is_augmented'] = 0
            original_row['augmentation_type'] = 'original'
            augmented_rows.append(original_row)

            # 获取元数据
            if metadata_df is not None:
                uid = row.get('uid', idx)
                meta_row = metadata_df[metadata_df['uid'] == uid]
                if len(meta_row) > 0:
                    test_mode = meta_row['test_mode'].values[0]
                    test_pressure = meta_row['test_pressure'].values[0]
                else:
                    test_mode = 'unknown'
                    test_pressure = 18.0
            else:
                test_mode = 'unknown'
                test_pressure = 18.0

            # 生成增强样本
            for aug_idx in range(augmentation_factor):
                new_row = row.to_dict()
                new_row['is_augmented'] = 1

                # 随机选择增强类型
                aug_type = self.rng.choice([
                    'noise', 'distance', 'time_scale', 'combined'
                ])
                new_row['augmentation_type'] = aug_type

                # 应用增强
                if aug_type == 'noise' or aug_type == 'combined':
                    # 添加噪声扰动到特征
                    noise_factor = self.rng.uniform(0.9, 1.1)
                    new_row['P'] = row['P'] * noise_factor
                    new_row['R'] = row['R'] * self.rng.uniform(0.95, 1.05)

                if aug_type == 'distance' or aug_type == 'combined':
                    # 模拟距离变化
                    distance_factor = self.rng.uniform(0.5, 2.0)
                    new_row['P'] = new_row.get('P', row['P']) / (distance_factor ** 2)

                if aug_type == 'time_scale' or aug_type == 'combined':
                    # 时间尺度变化
                    time_factor = self.rng.uniform(0.5, 2.0)
                    new_row['T'] = row['T'] * time_factor

                # 添加变轨类型标签
                if self.config.enable_maneuver_labeling:
                    maneuver_type = self.maneuver_mapper.map_to_maneuver_type(
                        test_mode, test_pressure,
                        duration=new_row.get('T'),
                        thrust=new_row.get('true_thrust')
                    )
                    new_row['maneuver_type'] = maneuver_type.name
                    new_row['maneuver_type_id'] = maneuver_type.value

                augmented_rows.append(new_row)

        return pd.DataFrame(augmented_rows)

    def generate_maneuver_labels(
        self,
        metadata_df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        生成变轨类型标签

        Args:
            metadata_df: 元数据DataFrame
            feature_df: 特征DataFrame（可选）

        Returns:
            带有变轨类型标签的DataFrame
        """
        if not self.config.enable_maneuver_labeling:
            raise ValueError("变轨类型标签功能未启用")

        return self.maneuver_mapper.generate_labels(metadata_df, feature_df)

    def simulate_realistic_observation(
        self,
        source_intensity: float,
        duration: float,
        maneuver_type: str = 'station_keeping',
        sampling_rate: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        模拟真实观测场景

        Args:
            source_intensity: 源强度
            duration: 持续时间 (s)
            maneuver_type: 变轨类型
            sampling_rate: 采样率

        Returns:
            模拟观测数据字典
        """
        n_samples = int(duration * sampling_rate)
        t = np.arange(n_samples) / sampling_rate

        # 生成基础推力信号
        thrust_data = self.long_duration_simulator.simulate_continuous_thrust(
            duration, sampling_rate,
            thrust_profile='bell',
            thrust_level=source_intensity
        )

        # 模拟距离变化
        distance = self.distance_simulator.simulate_pass(
            n_samples, sampling_rate,
            self.config.min_distance,
            self.config.max_distance,
            pass_type='overhead'
        )

        # 应用距离衰减
        observed, attn_params = self.distance_model.apply_attenuation(
            thrust_data['thrust'],
            distance=distance,
            include_atmospheric=True
        )

        # 添加噪声
        noisy, noise_components = self.noise_model.apply_noise(
            observed, sampling_rate
        )

        return {
            'time': t,
            'source_signal': thrust_data['thrust'],
            'distance': distance,
            'attenuated_signal': observed,
            'observed_signal': noisy,
            'attenuation_factor': attn_params['attenuation_factor'],
            'noise_std': {k: np.std(v) for k, v in noise_components.items()},
            'maneuver_type': maneuver_type
        }

    def get_augmentation_summary(self) -> Dict[str, any]:
        """
        获取增强配置摘要

        Returns:
            配置摘要字典
        """
        return {
            'noise_enabled': self.config.enable_noise,
            'target_snr': self.config.target_snr,
            'distance_attenuation_enabled': self.config.enable_distance_attenuation,
            'reference_distance': self.config.reference_distance,
            'maneuver_labeling_enabled': self.config.enable_maneuver_labeling,
            'time_scale_adaptation_enabled': self.config.enable_time_scale_adaptation,
            'original_sampling_rate': self.config.original_sampling_rate,
            'target_sampling_rate': self.config.target_sampling_rate,
            'seed': self.config.seed
        }


def create_augmentation_pipeline(
    snr_db: float = 5.0,
    enable_all: bool = True,
    seed: int = 42
) -> DataAugmentationPipeline:
    """
    创建数据增强流水线的便捷函数

    Args:
        snr_db: 目标信噪比
        enable_all: 是否启用所有增强
        seed: 随机种子

    Returns:
        数据增强流水线实例
    """
    config = AugmentationConfig(
        enable_noise=enable_all,
        target_snr=snr_db,
        enable_distance_attenuation=enable_all,
        enable_maneuver_labeling=enable_all,
        enable_time_scale_adaptation=enable_all,
        seed=seed
    )

    return DataAugmentationPipeline(config)
