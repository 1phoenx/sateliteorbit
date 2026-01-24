"""
变轨类型标签映射模块
基于test_mode和test_pressure组合定义变轨类型标签
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class ManeuverType(Enum):
    """变轨类型枚举"""
    HOHMANN_TRANSFER = auto()      # 霍曼转移
    PHASE_ADJUSTMENT = auto()       # 相位调整
    PLANE_CHANGE = auto()           # 轨道面变换
    STATION_KEEPING = auto()        # 位置保持
    COLLISION_AVOIDANCE = auto()    # 碰撞规避
    ORBIT_RAISING = auto()          # 升轨
    ORBIT_LOWERING = auto()         # 降轨
    DEORBIT = auto()                # 离轨
    ATTITUDE_CONTROL = auto()       # 姿态控制
    UNKNOWN = auto()                # 未知类型


@dataclass
class ManeuverCharacteristics:
    """变轨特征"""
    delta_v_range: Tuple[float, float]  # 速度增量范围 (m/s)
    duration_range: Tuple[float, float]  # 持续时间范围 (s)
    thrust_profile: str  # 推力曲线类型
    typical_frequency: float  # 典型执行频率 (次/天)


# 变轨类型特征定义
MANEUVER_CHARACTERISTICS: Dict[ManeuverType, ManeuverCharacteristics] = {
    ManeuverType.HOHMANN_TRANSFER: ManeuverCharacteristics(
        delta_v_range=(10.0, 500.0),
        duration_range=(60.0, 3600.0),
        thrust_profile='impulsive',
        typical_frequency=0.01
    ),
    ManeuverType.PHASE_ADJUSTMENT: ManeuverCharacteristics(
        delta_v_range=(0.1, 10.0),
        duration_range=(10.0, 300.0),
        thrust_profile='continuous',
        typical_frequency=0.1
    ),
    ManeuverType.PLANE_CHANGE: ManeuverCharacteristics(
        delta_v_range=(50.0, 1000.0),
        duration_range=(120.0, 7200.0),
        thrust_profile='impulsive',
        typical_frequency=0.001
    ),
    ManeuverType.STATION_KEEPING: ManeuverCharacteristics(
        delta_v_range=(0.01, 1.0),
        duration_range=(1.0, 60.0),
        thrust_profile='pulsed',
        typical_frequency=1.0
    ),
    ManeuverType.COLLISION_AVOIDANCE: ManeuverCharacteristics(
        delta_v_range=(0.1, 5.0),
        duration_range=(5.0, 120.0),
        thrust_profile='impulsive',
        typical_frequency=0.01
    ),
    ManeuverType.ORBIT_RAISING: ManeuverCharacteristics(
        delta_v_range=(5.0, 200.0),
        duration_range=(30.0, 1800.0),
        thrust_profile='continuous',
        typical_frequency=0.05
    ),
    ManeuverType.ORBIT_LOWERING: ManeuverCharacteristics(
        delta_v_range=(5.0, 200.0),
        duration_range=(30.0, 1800.0),
        thrust_profile='continuous',
        typical_frequency=0.05
    ),
    ManeuverType.DEORBIT: ManeuverCharacteristics(
        delta_v_range=(50.0, 300.0),
        duration_range=(300.0, 3600.0),
        thrust_profile='continuous',
        typical_frequency=0.0001
    ),
    ManeuverType.ATTITUDE_CONTROL: ManeuverCharacteristics(
        delta_v_range=(0.001, 0.1),
        duration_range=(0.1, 10.0),
        thrust_profile='pulsed',
        typical_frequency=10.0
    ),
}


class ManeuverTypeLabelMapper:
    """
    变轨类型标签映射器

    基于test_mode和test_pressure组合推断变轨类型
    """

    # test_mode到变轨类型的映射规则
    MODE_MAPPING: Dict[str, List[ManeuverType]] = {
        'ssf': [ManeuverType.STATION_KEEPING, ManeuverType.ATTITUDE_CONTROL],
        'health_check': [ManeuverType.STATION_KEEPING],
        'ramp1': [ManeuverType.PHASE_ADJUSTMENT, ManeuverType.ORBIT_RAISING],
        'ramp2': [ManeuverType.PHASE_ADJUSTMENT, ManeuverType.ORBIT_RAISING],
        'ramp3': [ManeuverType.HOHMANN_TRANSFER, ManeuverType.ORBIT_RAISING],
        'ramp4': [ManeuverType.HOHMANN_TRANSFER, ManeuverType.PLANE_CHANGE],
        'onmod': [ManeuverType.STATION_KEEPING, ManeuverType.COLLISION_AVOIDANCE],
        'offmod': [ManeuverType.STATION_KEEPING, ManeuverType.ATTITUDE_CONTROL],
        'random_short': [ManeuverType.ATTITUDE_CONTROL, ManeuverType.COLLISION_AVOIDANCE],
        'random_long': [ManeuverType.ORBIT_RAISING, ManeuverType.ORBIT_LOWERING],
        'random_mixed': [ManeuverType.PHASE_ADJUSTMENT, ManeuverType.STATION_KEEPING],
    }

    # 压力等级到推力等级的映射
    PRESSURE_THRUST_MAPPING = {
        24.0: 'high',
        21.0: 'medium_high',
        18.0: 'medium',
        15.0: 'medium_low',
        12.0: 'low',
        9.0: 'very_low',
    }

    def __init__(self, seed: Optional[int] = None):
        """
        初始化映射器

        Args:
            seed: 随机种子
        """
        self.rng = np.random.default_rng(seed)

    def map_to_maneuver_type(
        self,
        test_mode: str,
        test_pressure: float,
        duration: Optional[float] = None,
        thrust: Optional[float] = None
    ) -> ManeuverType:
        """
        将test_mode和test_pressure映射到变轨类型

        Args:
            test_mode: 测试模式
            test_pressure: 测试压力
            duration: 持续时间（可选，用于更精确的分类）
            thrust: 推力（可选）

        Returns:
            变轨类型
        """
        # 获取候选变轨类型
        candidates = self.MODE_MAPPING.get(
            test_mode.lower(),
            [ManeuverType.UNKNOWN]
        )

        if len(candidates) == 1:
            return candidates[0]

        # 基于压力等级选择
        thrust_level = self.PRESSURE_THRUST_MAPPING.get(test_pressure, 'medium')

        # 高推力倾向于大变轨
        if thrust_level in ['high', 'medium_high']:
            # 优先选择需要大delta_v的变轨类型
            for candidate in candidates:
                chars = MANEUVER_CHARACTERISTICS.get(candidate)
                if chars and chars.delta_v_range[1] > 50:
                    return candidate

        # 低推力倾向于小变轨
        elif thrust_level in ['low', 'very_low']:
            for candidate in candidates:
                chars = MANEUVER_CHARACTERISTICS.get(candidate)
                if chars and chars.delta_v_range[0] < 1:
                    return candidate

        # 如果有持续时间信息，进一步筛选
        if duration is not None:
            for candidate in candidates:
                chars = MANEUVER_CHARACTERISTICS.get(candidate)
                if chars:
                    if chars.duration_range[0] <= duration <= chars.duration_range[1]:
                        return candidate

        # 默认返回第一个候选
        return candidates[0]

    def estimate_delta_v(
        self,
        maneuver_type: ManeuverType,
        thrust: float,
        duration: float,
        mass: float = 500.0  # 卫星质量 (kg)
    ) -> float:
        """
        估计速度增量

        Args:
            maneuver_type: 变轨类型
            thrust: 推力 (N)
            duration: 持续时间 (s)
            mass: 卫星质量 (kg)

        Returns:
            估计的delta_v (m/s)
        """
        # 基于推力和持续时间计算
        # delta_v = (F / m) * t
        delta_v = (thrust / mass) * duration

        # 根据变轨类型特征约束
        chars = MANEUVER_CHARACTERISTICS.get(maneuver_type)
        if chars:
            delta_v = np.clip(
                delta_v,
                chars.delta_v_range[0],
                chars.delta_v_range[1]
            )

        return delta_v

    def generate_labels(
        self,
        metadata_df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        为数据集生成变轨类型标签

        Args:
            metadata_df: 元数据DataFrame
            feature_df: 特征DataFrame（可选）

        Returns:
            带有变轨类型标签的DataFrame
        """
        labels = []

        for idx, row in metadata_df.iterrows():
            test_mode = row.get('test_mode', 'unknown')
            test_pressure = row.get('test_pressure', 18.0)

            # 获取持续时间和推力（如果有特征数据）
            duration = None
            thrust = None
            if feature_df is not None:
                uid = row.get('uid', idx)
                feature_row = feature_df[feature_df['uid'] == uid]
                if len(feature_row) > 0:
                    duration = feature_row['T'].values[0]
                    thrust = feature_row.get('true_thrust', pd.Series([None])).values[0]

            # 映射变轨类型
            maneuver_type = self.map_to_maneuver_type(
                test_mode, test_pressure, duration, thrust
            )

            # 估计delta_v
            if thrust is not None and duration is not None:
                delta_v = self.estimate_delta_v(
                    maneuver_type, thrust, duration
                )
            else:
                # 使用变轨类型的典型值
                chars = MANEUVER_CHARACTERISTICS.get(maneuver_type)
                if chars:
                    delta_v = np.mean(chars.delta_v_range)
                else:
                    delta_v = 0.0

            labels.append({
                'uid': row.get('uid', idx),
                'maneuver_type': maneuver_type.name,
                'maneuver_type_id': maneuver_type.value,
                'estimated_delta_v': delta_v,
                'thrust_profile': MANEUVER_CHARACTERISTICS.get(
                    maneuver_type, ManeuverCharacteristics(
                        (0, 0), (0, 0), 'unknown', 0
                    )
                ).thrust_profile
            })

        return pd.DataFrame(labels)


class ManeuverTypeAugmenter:
    """
    变轨类型数据增强器

    基于变轨类型特征生成合成数据
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.mapper = ManeuverTypeLabelMapper(seed)

    def generate_synthetic_maneuver(
        self,
        maneuver_type: ManeuverType,
        n_samples: int = 1000,
        sampling_rate: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        生成合成变轨信号

        Args:
            maneuver_type: 变轨类型
            n_samples: 样本数量
            sampling_rate: 采样率

        Returns:
            合成信号字典
        """
        chars = MANEUVER_CHARACTERISTICS.get(maneuver_type)
        if chars is None:
            raise ValueError(f"未知的变轨类型: {maneuver_type}")

        # 随机生成参数
        delta_v = self.rng.uniform(*chars.delta_v_range)
        duration = self.rng.uniform(*chars.duration_range)

        # 生成时间数组
        t = np.arange(n_samples) / sampling_rate

        # 根据推力曲线类型生成信号
        if chars.thrust_profile == 'impulsive':
            signal = self._generate_impulsive_signal(t, delta_v, duration)
        elif chars.thrust_profile == 'continuous':
            signal = self._generate_continuous_signal(t, delta_v, duration)
        elif chars.thrust_profile == 'pulsed':
            signal = self._generate_pulsed_signal(t, delta_v, duration)
        else:
            signal = np.zeros(n_samples)

        return {
            'time': t,
            'thrust_signal': signal,
            'delta_v': delta_v,
            'duration': duration,
            'maneuver_type': maneuver_type.name
        }

    def _generate_impulsive_signal(
        self,
        t: np.ndarray,
        delta_v: float,
        duration: float
    ) -> np.ndarray:
        """生成脉冲式推力信号"""
        signal = np.zeros_like(t)

        # 点火时刻
        ignition_time = t[len(t) // 4]
        end_time = ignition_time + duration

        # 脉冲信号
        mask = (t >= ignition_time) & (t <= end_time)
        signal[mask] = delta_v / duration  # 等效推力

        # 添加上升/下降沿
        rise_time = min(duration * 0.1, 1.0)
        for i, ti in enumerate(t):
            if ignition_time <= ti < ignition_time + rise_time:
                signal[i] *= (ti - ignition_time) / rise_time
            elif end_time - rise_time < ti <= end_time:
                signal[i] *= (end_time - ti) / rise_time

        return signal

    def _generate_continuous_signal(
        self,
        t: np.ndarray,
        delta_v: float,
        duration: float
    ) -> np.ndarray:
        """生成连续推力信号"""
        signal = np.zeros_like(t)

        ignition_time = t[len(t) // 4]
        end_time = ignition_time + duration

        mask = (t >= ignition_time) & (t <= end_time)

        # 连续推力（可能有缓慢变化）
        thrust_level = delta_v / duration
        signal[mask] = thrust_level * (
            1 + 0.1 * np.sin(2 * np.pi * (t[mask] - ignition_time) / duration)
        )

        return signal

    def _generate_pulsed_signal(
        self,
        t: np.ndarray,
        delta_v: float,
        duration: float
    ) -> np.ndarray:
        """生成脉冲式推力信号"""
        signal = np.zeros_like(t)

        ignition_time = t[len(t) // 4]

        # 脉冲参数
        pulse_width = min(duration * 0.1, 0.5)
        pulse_interval = duration * 0.2
        n_pulses = max(1, int(duration / pulse_interval))

        # 每个脉冲的delta_v
        pulse_delta_v = delta_v / n_pulses

        for i in range(n_pulses):
            pulse_start = ignition_time + i * pulse_interval
            pulse_end = pulse_start + pulse_width

            mask = (t >= pulse_start) & (t <= pulse_end)
            signal[mask] = pulse_delta_v / pulse_width

        return signal

    def augment_with_maneuver_types(
        self,
        feature_df: pd.DataFrame,
        augmentation_factor: int = 2
    ) -> pd.DataFrame:
        """
        基于变轨类型增强数据集

        Args:
            feature_df: 原始特征DataFrame
            augmentation_factor: 增强倍数

        Returns:
            增强后的DataFrame
        """
        augmented_rows = []

        for _, row in feature_df.iterrows():
            # 保留原始数据
            augmented_rows.append(row.to_dict())

            # 生成增强数据
            for _ in range(augmentation_factor - 1):
                new_row = row.to_dict()

                # 随机选择变轨类型
                maneuver_type = self.rng.choice(list(ManeuverType))
                chars = MANEUVER_CHARACTERISTICS.get(maneuver_type)

                if chars:
                    # 根据变轨类型调整特征
                    new_row['P'] = row['P'] * self.rng.uniform(0.8, 1.2)
                    new_row['T'] = self.rng.uniform(*chars.duration_range)
                    new_row['R'] = row['R'] * self.rng.uniform(0.9, 1.1)
                    new_row['maneuver_type'] = maneuver_type.name
                    new_row['is_synthetic'] = 1

                    augmented_rows.append(new_row)

        return pd.DataFrame(augmented_rows)
