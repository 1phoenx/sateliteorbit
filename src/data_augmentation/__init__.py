"""
数据增强模块

解决STFT数据集的四个局限性：
1. 缺少真实噪声特性 - realistic_noise.py
2. 无距离衰减 - distance_attenuation.py
3. 无变轨类型标签 - maneuver_type_mapper.py
4. 时间尺度差异 - time_scale_adapter.py

使用方法：
    from src.data_augmentation import (
        DataAugmentationPipeline,
        AugmentationConfig,
        create_augmentation_pipeline
    )

    # 快速创建流水线
    pipeline = create_augmentation_pipeline(snr_db=5.0)

    # 增强单个信号
    result = pipeline.augment_signal(signal_data, sampling_rate=100.0)

    # 增强整个数据集
    augmented_df = pipeline.augment_dataset(feature_df, metadata_df)
"""

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
    ManeuverCharacteristics,
    MANEUVER_CHARACTERISTICS
)

from .time_scale_adapter import (
    TimeScaleAdapter,
    TimeScaleConfig,
    MultiScaleProcessor,
    LongDurationSimulator,
    TemporalFeatureExtractor
)

from .pipeline import (
    DataAugmentationPipeline,
    AugmentationConfig,
    create_augmentation_pipeline
)

__all__ = [
    # 噪声模型
    'RealisticNoiseModel',
    'AtmosphericScintillation',
    'BackgroundClutter',
    'DetectorNoise',

    # 距离衰减
    'DistanceAttenuationModel',
    'DistanceVariationSimulator',
    'OrbitParameters',

    # 变轨类型
    'ManeuverTypeLabelMapper',
    'ManeuverTypeAugmenter',
    'ManeuverType',
    'ManeuverCharacteristics',
    'MANEUVER_CHARACTERISTICS',

    # 时间尺度
    'TimeScaleAdapter',
    'TimeScaleConfig',
    'MultiScaleProcessor',
    'LongDurationSimulator',
    'TemporalFeatureExtractor',

    # 流水线
    'DataAugmentationPipeline',
    'AugmentationConfig',
    'create_augmentation_pipeline',
]
