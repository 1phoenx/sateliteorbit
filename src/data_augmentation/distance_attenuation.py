"""
距离衰减模型模块
模拟真实场景中辐射强度随距离的平方反比衰减
"""
import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class OrbitParameters:
    """轨道参数"""
    semi_major_axis: float = 7000e3  # 半长轴 (m)，约630km高度
    eccentricity: float = 0.001  # 偏心率
    inclination: float = 98.0  # 倾角 (度)
    observer_altitude: float = 0.0  # 观测站高度 (m)
    observer_latitude: float = 40.0  # 观测站纬度 (度)


class DistanceAttenuationModel:
    """
    距离衰减模型

    实现辐射强度的平方反比定律：
    I_observed = I_source / (4 * pi * d^2)

    考虑因素：
    - 目标轨道运动导致的距离变化
    - 大气消光
    - 观测几何
    """

    # 地球半径 (m)
    EARTH_RADIUS = 6371e3

    def __init__(
        self,
        orbit_params: Optional[OrbitParameters] = None,
        atmospheric_extinction: float = 0.1,  # 大气消光系数 (mag/airmass)
        reference_distance: float = 500e3,  # 参考距离 (m)
        seed: Optional[int] = None
    ):
        """
        初始化距离衰减模型

        Args:
            orbit_params: 轨道参数
            atmospheric_extinction: 大气消光系数
            reference_distance: 参考距离（用于归一化）
            seed: 随机种子
        """
        self.orbit_params = orbit_params or OrbitParameters()
        self.atmospheric_extinction = atmospheric_extinction
        self.reference_distance = reference_distance
        self.rng = np.random.default_rng(seed)

    def calculate_distance(
        self,
        time_array: np.ndarray,
        orbital_phase: float = 0.0
    ) -> np.ndarray:
        """
        计算目标到观测站的距离随时间变化

        Args:
            time_array: 时间数组 (s)
            orbital_phase: 初始轨道相位 (rad)

        Returns:
            距离数组 (m)
        """
        # 轨道周期
        mu = 3.986e14  # 地球引力常数 (m^3/s^2)
        period = 2 * np.pi * np.sqrt(
            self.orbit_params.semi_major_axis ** 3 / mu
        )

        # 平均角速度
        n = 2 * np.pi / period

        # 真近点角（简化为圆轨道）
        true_anomaly = orbital_phase + n * time_array

        # 轨道半径（考虑偏心率）
        e = self.orbit_params.eccentricity
        a = self.orbit_params.semi_major_axis
        r_orbit = a * (1 - e ** 2) / (1 + e * np.cos(true_anomaly))

        # 卫星位置（简化的地心惯性坐标系）
        inc = np.radians(self.orbit_params.inclination)
        sat_x = r_orbit * np.cos(true_anomaly)
        sat_y = r_orbit * np.sin(true_anomaly) * np.cos(inc)
        sat_z = r_orbit * np.sin(true_anomaly) * np.sin(inc)

        # 观测站位置（地固坐标系，简化处理）
        obs_lat = np.radians(self.orbit_params.observer_latitude)
        obs_r = self.EARTH_RADIUS + self.orbit_params.observer_altitude
        obs_x = obs_r * np.cos(obs_lat)
        obs_y = 0
        obs_z = obs_r * np.sin(obs_lat)

        # 计算距离
        distance = np.sqrt(
            (sat_x - obs_x) ** 2 +
            (sat_y - obs_y) ** 2 +
            (sat_z - obs_z) ** 2
        )

        return distance

    def calculate_zenith_angle(
        self,
        distance: np.ndarray
    ) -> np.ndarray:
        """
        计算天顶角

        Args:
            distance: 距离数组

        Returns:
            天顶角数组 (rad)
        """
        # 简化计算：基于距离估算天顶角
        # 最小距离对应天顶，最大距离对应地平线
        min_dist = self.orbit_params.semi_major_axis - self.EARTH_RADIUS
        max_dist = np.sqrt(
            self.orbit_params.semi_major_axis ** 2 - self.EARTH_RADIUS ** 2
        ) + self.EARTH_RADIUS

        # 线性映射到天顶角
        zenith_angle = np.pi / 2 * (distance - min_dist) / (max_dist - min_dist)
        zenith_angle = np.clip(zenith_angle, 0, np.pi / 2 - 0.01)

        return zenith_angle

    def calculate_airmass(
        self,
        zenith_angle: np.ndarray
    ) -> np.ndarray:
        """
        计算大气质量（Kasten-Young公式）

        Args:
            zenith_angle: 天顶角 (rad)

        Returns:
            大气质量
        """
        z_deg = np.degrees(zenith_angle)
        # Kasten-Young公式
        airmass = 1 / (
            np.cos(zenith_angle) +
            0.50572 * (96.07995 - z_deg) ** (-1.6364)
        )
        return airmass

    def apply_inverse_square_law(
        self,
        intensity: Union[float, np.ndarray],
        distance: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        应用平方反比定律

        Args:
            intensity: 源强度
            distance: 距离

        Returns:
            观测强度
        """
        intensity = np.atleast_1d(intensity)
        distance = np.atleast_1d(distance)

        # 归一化到参考距离
        attenuation = (self.reference_distance / distance) ** 2

        return intensity * attenuation

    def apply_atmospheric_extinction(
        self,
        intensity: np.ndarray,
        zenith_angle: np.ndarray
    ) -> np.ndarray:
        """
        应用大气消光

        Args:
            intensity: 强度
            zenith_angle: 天顶角

        Returns:
            消光后强度
        """
        airmass = self.calculate_airmass(zenith_angle)

        # 消光公式：I = I_0 * 10^(-0.4 * k * X)
        extinction_factor = 10 ** (-0.4 * self.atmospheric_extinction * airmass)

        return intensity * extinction_factor

    def apply_attenuation(
        self,
        source_intensity: np.ndarray,
        time_array: Optional[np.ndarray] = None,
        distance: Optional[np.ndarray] = None,
        include_atmospheric: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        应用完整的距离衰减模型

        Args:
            source_intensity: 源强度
            time_array: 时间数组（如果提供，则计算动态距离）
            distance: 距离数组（如果提供，则使用固定距离）
            include_atmospheric: 是否包含大气消光

        Returns:
            (衰减后强度, 衰减参数字典)
        """
        n_samples = len(source_intensity)

        # 计算或使用提供的距离
        if distance is not None:
            dist = np.atleast_1d(distance)
            if len(dist) == 1:
                dist = np.full(n_samples, dist[0])
        elif time_array is not None:
            dist = self.calculate_distance(time_array)
        else:
            # 默认使用参考距离
            dist = np.full(n_samples, self.reference_distance)

        # 应用平方反比定律
        attenuated = self.apply_inverse_square_law(source_intensity, dist)

        # 大气消光
        if include_atmospheric:
            zenith = self.calculate_zenith_angle(dist)
            attenuated = self.apply_atmospheric_extinction(attenuated, zenith)
        else:
            zenith = np.zeros(n_samples)

        # 返回结果和参数
        params = {
            'distance': dist,
            'zenith_angle': zenith,
            'attenuation_factor': attenuated / source_intensity
        }

        return attenuated, params


class DistanceVariationSimulator:
    """
    距离变化模拟器

    模拟不同观测场景下的距离变化
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def simulate_pass(
        self,
        n_samples: int,
        sampling_rate: float = 100.0,
        min_distance: float = 400e3,
        max_distance: float = 2000e3,
        pass_type: str = 'overhead'
    ) -> np.ndarray:
        """
        模拟一次过境的距离变化

        Args:
            n_samples: 样本数量
            sampling_rate: 采样率
            min_distance: 最小距离
            max_distance: 最大距离
            pass_type: 过境类型 ('overhead', 'horizon', 'random')

        Returns:
            距离数组
        """
        t = np.linspace(0, 1, n_samples)

        if pass_type == 'overhead':
            # 天顶过境：距离先减后增
            distance = min_distance + (max_distance - min_distance) * (
                4 * (t - 0.5) ** 2
            )

        elif pass_type == 'horizon':
            # 地平线过境：距离单调变化
            distance = max_distance - (max_distance - min_distance) * t

        elif pass_type == 'random':
            # 随机过境
            phase = self.rng.uniform(0, 2 * np.pi)
            amplitude = self.rng.uniform(0.3, 0.7)
            distance = (min_distance + max_distance) / 2 + \
                       amplitude * (max_distance - min_distance) / 2 * \
                       np.sin(2 * np.pi * t + phase)

        else:
            raise ValueError(f"未知的过境类型: {pass_type}")

        return distance

    def add_distance_jitter(
        self,
        distance: np.ndarray,
        jitter_amplitude: float = 1e3,  # 抖动幅度 (m)
        jitter_frequency: float = 0.1  # 抖动频率 (Hz)
    ) -> np.ndarray:
        """
        添加距离抖动（模拟轨道扰动）

        Args:
            distance: 基础距离
            jitter_amplitude: 抖动幅度
            jitter_frequency: 抖动频率

        Returns:
            带抖动的距离
        """
        n_samples = len(distance)
        t = np.arange(n_samples)

        # 多频率抖动
        jitter = jitter_amplitude * (
            0.5 * np.sin(2 * np.pi * jitter_frequency * t / 100) +
            0.3 * np.sin(2 * np.pi * jitter_frequency * 2.3 * t / 100) +
            0.2 * self.rng.standard_normal(n_samples)
        )

        return distance + jitter
