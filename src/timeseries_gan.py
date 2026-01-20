"""
原始时序数据GAN扩充模块
生成类似00001_001_SN01_24bars_ssf.csv格式的时序数据

数据结构:
- time: 时间戳
- ton: 推力器开关状态 (0/1)
- thrust: 推力值 (N)
- mfr: 质量流率 (mg/s)
- vl: 阀门状态
- anomaly_code: 异常代码

生成策略:
1. 分段生成: 背景段 + 点火段 + 背景段
2. 物理约束: thrust与mfr正相关，点火时thrust上升
3. 噪声模拟: 添加与真实数据相似的噪声特性
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta


class TimeSeriesEncoder(nn.Module):
    """时序编码器 - 将时序数据编码为潜在向量"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # 合并双向隐藏状态
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class TimeSeriesDecoder(nn.Module):
    """时序解码器 - 从潜在向量生成时序数据"""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, output_dim: int = 3, seq_len: int = 1000):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        batch_size = z.size(0)

        # 扩展潜在向量到序列
        h = self.fc(z)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        # LSTM解码
        output, _ = self.lstm(h)

        # 输出层
        output = self.output_layer(output)

        return output


class ThrusterDataVAE(nn.Module):
    """推力器数据变分自编码器"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, latent_dim: int = 64, seq_len: int = 1000):
        super().__init__()

        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TimeSeriesDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def generate(self, num_samples: int, device: str = 'cpu'):
        """从先验分布生成样本"""
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples


class PhysicsConstrainedGenerator:
    """
    物理约束时序数据生成器

    基于物理规律生成推力器时序数据:
    1. 背景噪声: 高斯白噪声
    2. 点火信号: 阶跃响应 + 稳态值 + 噪声
    3. thrust-mfr关系: mfr ∝ thrust (比冲关系)
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        background_noise_std: float = 0.005,
        thrust_range: Tuple[float, float] = (0.8, 2.0),
        duration_range: Tuple[float, float] = (100.0, 500.0),
        mfr_coefficient: float = 1100.0  # mfr = coefficient * thrust
    ):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.background_noise_std = background_noise_std
        self.thrust_range = thrust_range
        self.duration_range = duration_range
        self.mfr_coefficient = mfr_coefficient

    def generate_background(self, n_samples: int) -> np.ndarray:
        """生成背景噪声"""
        return np.random.normal(0, self.background_noise_std, n_samples)

    def generate_firing_profile(
        self,
        duration: float,
        thrust_level: float,
        rise_time: float = 0.5,
        noise_std: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成点火信号

        Args:
            duration: 点火持续时间 (秒)
            thrust_level: 稳态推力值 (N)
            rise_time: 上升时间 (秒)
            noise_std: 信号噪声标准差

        Returns:
            (thrust_signal, mfr_signal)
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.arange(n_samples) * self.dt

        # 阶跃响应模型: 1 - exp(-t/tau)
        tau = rise_time / 3  # 时间常数
        rise_profile = 1 - np.exp(-t / tau)

        # 稳态值 + 小幅波动
        steady_noise = np.random.normal(0, noise_std * thrust_level, n_samples)

        # 组合信号
        thrust = thrust_level * rise_profile + steady_noise

        # mfr与thrust的物理关系
        mfr = self.mfr_coefficient * np.maximum(thrust, 0)
        mfr += np.random.normal(0, 10, n_samples)  # mfr噪声

        return thrust, mfr

    def generate_single_sequence(
        self,
        total_duration: float = 306.0,
        pre_firing_duration: float = 1.0,
        post_firing_duration: float = 5.0,
        thrust_level: float = None,
        firing_duration: float = None,
        is_anomalous: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        生成单个完整时序

        Args:
            total_duration: 总时长 (秒)
            pre_firing_duration: 点火前背景时长
            post_firing_duration: 点火后背景时长
            thrust_level: 推力水平，None则随机
            firing_duration: 点火持续时间，None则随机
            is_anomalous: 是否生成异常样本

        Returns:
            包含所有字段的字典
        """
        if thrust_level is None:
            thrust_level = np.random.uniform(*self.thrust_range)

        if firing_duration is None:
            firing_duration = np.random.uniform(*self.duration_range)

        # 计算各段样本数
        n_pre = int(pre_firing_duration * self.sampling_rate)
        n_firing = int(firing_duration * self.sampling_rate)
        n_post = int(post_firing_duration * self.sampling_rate)
        n_total = n_pre + n_firing + n_post

        # 生成各段信号
        # 1. 点火前背景
        thrust_pre = self.generate_background(n_pre)
        mfr_pre = np.zeros(n_pre)
        ton_pre = np.zeros(n_pre)

        # 2. 点火段
        thrust_firing, mfr_firing = self.generate_firing_profile(
            firing_duration, thrust_level
        )
        ton_firing = np.ones(n_firing)

        # 3. 点火后背景
        thrust_post = self.generate_background(n_post)
        mfr_post = np.zeros(n_post)
        ton_post = np.zeros(n_post)

        # 合并
        thrust = np.concatenate([thrust_pre, thrust_firing, thrust_post])
        mfr = np.concatenate([mfr_pre, mfr_firing, mfr_post])
        ton = np.concatenate([ton_pre, ton_firing, ton_post])

        # 生成时间戳
        time_seconds = np.arange(n_total) * self.dt

        # 异常处理
        anomaly_code = np.zeros(n_total)
        if is_anomalous:
            # 随机添加异常
            anomaly_type = np.random.choice(['spike', 'dropout', 'drift'])
            if anomaly_type == 'spike':
                # 尖峰异常
                spike_idx = np.random.randint(n_pre, n_pre + n_firing)
                thrust[spike_idx:spike_idx+10] *= 2
                anomaly_code[spike_idx:spike_idx+10] = 1
            elif anomaly_type == 'dropout':
                # 信号丢失
                dropout_start = np.random.randint(n_pre, n_pre + n_firing - 100)
                thrust[dropout_start:dropout_start+50] = 0
                anomaly_code[dropout_start:dropout_start+50] = 2
            elif anomaly_type == 'drift':
                # 漂移
                drift = np.linspace(0, 0.3, n_firing)
                thrust[n_pre:n_pre+n_firing] += drift
                anomaly_code[n_pre:n_pre+n_firing] = 3

        # vl (阀门状态)
        vl = np.zeros(n_total)
        vl[n_pre + n_firing:] = 1.0  # 点火后阀门关闭

        return {
            'time': time_seconds,
            'ton': ton.astype(int),
            'thrust': thrust,
            'mfr': mfr,
            'vl': vl,
            'anomaly_code': anomaly_code.astype(int),
            'metadata': {
                'thrust_level': thrust_level,
                'firing_duration': firing_duration,
                'is_anomalous': is_anomalous
            }
        }


class ThrusterDataAugmentor:
    """
    推力器数据扩充器

    完整流程:
    1. 加载原始数据学习分布
    2. 使用物理约束生成器生成新数据
    3. 从生成数据提取P/T/R特征
    """

    def __init__(
        self,
        data_dir: str = 'data',
        output_dir: str = 'data/augmented_csv',
        sampling_rate: float = 100.0
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sampling_rate = sampling_rate

        self.generator = PhysicsConstrainedGenerator(sampling_rate=sampling_rate)

        # 学习到的数据分布参数
        self.learned_params = None

    def learn_distribution(self, n_samples: int = 100) -> Dict:
        """从原始数据学习分布参数"""
        print("从原始数据学习分布...")

        train_dir = self.data_dir / 'train'
        csv_files = list(train_dir.glob('*.csv'))[:n_samples]

        thrust_levels = []
        firing_durations = []
        noise_stds = []

        for csv_file in tqdm(csv_files, desc="分析原始数据"):
            try:
                df = pd.read_csv(csv_file)

                # 提取点火段
                firing_mask = df['ton'] == 1
                if firing_mask.sum() > 0:
                    thrust_firing = df.loc[firing_mask, 'thrust'].values

                    # 稳态推力 (去除前10%的上升段)
                    stable_start = int(len(thrust_firing) * 0.1)
                    if stable_start < len(thrust_firing):
                        thrust_stable = thrust_firing[stable_start:]
                        thrust_levels.append(np.mean(thrust_stable))
                        noise_stds.append(np.std(thrust_stable))

                    # 点火持续时间
                    duration = firing_mask.sum() / self.sampling_rate
                    firing_durations.append(duration)

            except Exception as e:
                continue

        self.learned_params = {
            'thrust_mean': np.mean(thrust_levels),
            'thrust_std': np.std(thrust_levels),
            'thrust_min': np.min(thrust_levels),
            'thrust_max': np.max(thrust_levels),
            'duration_mean': np.mean(firing_durations),
            'duration_std': np.std(firing_durations),
            'duration_min': np.min(firing_durations),
            'duration_max': np.max(firing_durations),
            'noise_std_mean': np.mean(noise_stds),
        }

        print(f"学习到的参数:")
        print(f"  推力范围: {self.learned_params['thrust_min']:.3f} - {self.learned_params['thrust_max']:.3f} N")
        print(f"  推力均值: {self.learned_params['thrust_mean']:.3f} ± {self.learned_params['thrust_std']:.3f} N")
        print(f"  持续时间: {self.learned_params['duration_min']:.1f} - {self.learned_params['duration_max']:.1f} s")

        return self.learned_params

    def generate_csv_file(
        self,
        output_path: str,
        thrust_level: float = None,
        firing_duration: float = None,
        is_anomalous: bool = False,
        start_time: str = "2021-01-18 08:00:00.000"
    ) -> str:
        """
        生成单个CSV文件

        Args:
            output_path: 输出文件路径
            thrust_level: 推力水平
            firing_duration: 点火持续时间
            is_anomalous: 是否异常
            start_time: 起始时间戳

        Returns:
            生成的文件路径
        """
        # 使用学习到的分布采样参数
        if thrust_level is None:
            if self.learned_params:
                thrust_level = np.random.normal(
                    self.learned_params['thrust_mean'],
                    self.learned_params['thrust_std']
                )
                thrust_level = np.clip(
                    thrust_level,
                    self.learned_params['thrust_min'],
                    self.learned_params['thrust_max']
                )
            else:
                thrust_level = np.random.uniform(0.8, 2.0)

        if firing_duration is None:
            if self.learned_params:
                firing_duration = np.random.normal(
                    self.learned_params['duration_mean'],
                    self.learned_params['duration_std']
                )
                firing_duration = np.clip(
                    firing_duration,
                    self.learned_params['duration_min'],
                    self.learned_params['duration_max']
                )
            else:
                firing_duration = np.random.uniform(100, 500)

        # 生成数据
        data = self.generator.generate_single_sequence(
            thrust_level=thrust_level,
            firing_duration=firing_duration,
            is_anomalous=is_anomalous
        )

        # 生成时间戳字符串
        base_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
        time_strings = [
            (base_time + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            for t in data['time']
        ]

        # 创建DataFrame
        df = pd.DataFrame({
            'time': time_strings,
            'ton': data['ton'],
            'thrust': data['thrust'],
            'mfr': data['mfr'],
            'vl': data['vl'],
            'anomaly_code': data['anomaly_code']
        })

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        return output_path

    def augment_dataset(
        self,
        n_synthetic: int = 1000,
        anomaly_ratio: float = 0.1,
        prefix: str = 'synthetic'
    ) -> List[str]:
        """
        批量生成扩充数据

        Args:
            n_synthetic: 生成样本数量
            anomaly_ratio: 异常样本比例
            prefix: 文件名前缀

        Returns:
            生成的文件路径列表
        """
        print(f"\n生成 {n_synthetic} 个合成样本...")

        # 学习分布
        if self.learned_params is None:
            self.learn_distribution()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        n_anomalous = int(n_synthetic * anomaly_ratio)

        for i in tqdm(range(n_synthetic), desc="生成合成数据"):
            is_anomalous = i < n_anomalous

            filename = f"{prefix}_{i+1:05d}.csv"
            output_path = self.output_dir / filename

            self.generate_csv_file(
                str(output_path),
                is_anomalous=is_anomalous
            )

            generated_files.append(str(output_path))

        print(f"生成完成: {len(generated_files)} 个文件")
        print(f"  正常样本: {n_synthetic - n_anomalous}")
        print(f"  异常样本: {n_anomalous}")
        print(f"  保存目录: {self.output_dir}")

        return generated_files

    def extract_features_from_generated(
        self,
        csv_files: List[str] = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        从生成的CSV文件提取P/T/R特征

        Args:
            csv_files: CSV文件列表，None则使用output_dir中的所有文件
            output_path: 特征输出路径

        Returns:
            特征DataFrame
        """
        from src.feature_extraction_v2 import ThrusterFeatureExtractor

        if csv_files is None:
            csv_files = list(self.output_dir.glob('*.csv'))

        print(f"\n从 {len(csv_files)} 个文件提取特征...")

        extractor = ThrusterFeatureExtractor(sampling_rate=self.sampling_rate)
        results = []

        for csv_file in tqdm(csv_files, desc="提取特征"):
            try:
                df = pd.read_csv(csv_file)

                thrust = df['thrust'].values
                ton = df['ton'].values

                B_thrust, sigma = extractor.compute_baseline(thrust, ton)
                P = extractor.extract_P(thrust, ton, B_thrust, sigma)
                T, ignition_time, true_thrust = extractor.extract_T(
                    thrust, B_thrust, sigma, ton, use_precise_detection=False
                )
                R = extractor.extract_R(thrust, T)

                is_anomalous = 1 if df['anomaly_code'].sum() > 0 else 0
                is_valid = 1 if (P > 0 and T >= 0.1) else 0

                results.append({
                    'filename': Path(csv_file).name,
                    'P': P,
                    'T': T,
                    'R': R,
                    'ignition_time': ignition_time,
                    'true_thrust': true_thrust,
                    'is_anomalous': is_anomalous,
                    'is_valid': is_valid,
                    'is_synthetic': 1
                })

            except Exception as e:
                print(f"处理失败 {csv_file}: {e}")
                continue

        feature_df = pd.DataFrame(results)

        if output_path:
            feature_df.to_csv(output_path, index=False)
            print(f"特征已保存至: {output_path}")

        print(f"提取完成: {len(feature_df)} 个有效样本")

        return feature_df


def main():
    """主函数 - 演示完整的数据扩充流程"""
    print("=" * 60)
    print("原始CSV数据GAN扩充")
    print("=" * 60)

    # 创建扩充器
    augmentor = ThrusterDataAugmentor(
        data_dir='data',
        output_dir='data/augmented_csv',
        sampling_rate=100.0
    )

    # 学习原始数据分布
    augmentor.learn_distribution(n_samples=100)

    # 生成合成数据
    generated_files = augmentor.augment_dataset(
        n_synthetic=100,  # 演示用，实际可设为1000+
        anomaly_ratio=0.1,
        prefix='synthetic'
    )

    # 从生成数据提取特征
    feature_df = augmentor.extract_features_from_generated(
        output_path='data/synthetic_features.csv'
    )

    # 显示统计
    print("\n" + "=" * 60)
    print("生成数据统计")
    print("=" * 60)
    print(feature_df[['P', 'T', 'R']].describe())

    # 与原始数据对比
    original_features = pd.read_csv('data/feature_dataset.csv')
    print("\n原始数据统计:")
    print(original_features[['P', 'T', 'R']].describe())


if __name__ == '__main__':
    main()
