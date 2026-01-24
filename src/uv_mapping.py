"""
推力/质量流率 → 360 nm UV 辐射映射模型
===========================================

物理机理：
- 肼单组元推进器化学发光（NH*）
- UV 强度 ∝ 质量流率（一阶近似）
- 观测波段：360 nm

数学模型：
I_uv(t) = α × mfr(t)^β + γ × thrust(t) + I_bg + noise(t)

其中：
- α: 质量流率-UV强度转换系数
- β: 非线性指数（1.0为线性，1.2为弱非线性）
- γ: 推力贡献系数
- I_bg: 背景辐射强度
- noise(t): 观测噪声（高斯白噪声）

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class UV360nmMapper:
    """
    360 nm UV 辐射映射器

    参数:
        alpha: 质量流率转换系数 (默认: 1000.0)
        beta: 非线性指数 (默认: 1.1, 弱非线性)
        gamma: 推力贡献系数 (默认: 50.0)
        I_bg: 背景辐射强度 (默认: 10.0)
        noise_std: 噪声标准差 (默认: 2.0)
        use_mfr: 是否使用质量流率 (默认: True)
        use_thrust: 是否使用推力 (默认: True)
    """

    def __init__(
        self,
        alpha: float = 1000.0,
        beta: float = 1.1,
        gamma: float = 50.0,
        I_bg: float = 10.0,
        noise_std: float = 2.0,
        use_mfr: bool = True,
        use_thrust: bool = True,
        random_seed: int = 42
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.I_bg = I_bg
        self.noise_std = noise_std
        self.use_mfr = use_mfr
        self.use_thrust = use_thrust
        self.random_seed = random_seed

        # 设置随机种子
        np.random.seed(random_seed)

    def map_single_point(
        self,
        mfr: float,
        thrust: float,
        add_noise: bool = True
    ) -> float:
        """
        单点映射：计算单个时刻的 UV 强度

        数学公式：
        I_uv = α × mfr^β + γ × thrust + I_bg + ε

        参数:
            mfr: 质量流率 (kg/s)
            thrust: 推力 (N)
            add_noise: 是否添加噪声

        返回:
            I_uv: UV 辐射强度 (任意单位)
        """
        I_uv = self.I_bg  # 背景辐射

        # 质量流率贡献（主要项）
        if self.use_mfr and mfr > 0:
            I_uv += self.alpha * (mfr ** self.beta)

        # 推力贡献（次要项）
        if self.use_thrust and thrust > 0:
            I_uv += self.gamma * thrust

        # 添加观测噪声
        if add_noise:
            noise = np.random.normal(0, self.noise_std)
            I_uv += noise

        # 确保非负
        I_uv = max(0.0, I_uv)

        return I_uv

    def map_timeseries(
        self,
        mfr_series: np.ndarray,
        thrust_series: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        时间序列映射：批量计算 UV 强度时间序列

        参数:
            mfr_series: 质量流率时间序列
            thrust_series: 推力时间序列
            add_noise: 是否添加噪声

        返回:
            uv_series: UV 辐射强度时间序列
        """
        n_points = len(mfr_series)
        uv_series = np.zeros(n_points)

        for i in range(n_points):
            uv_series[i] = self.map_single_point(
                mfr_series[i],
                thrust_series[i],
                add_noise=add_noise
            )

        return uv_series

    def process_csv(
        self,
        csv_file: Union[str, Path],
        output_file: Union[str, Path] = None,
        add_noise: bool = True
    ) -> pd.DataFrame:
        """
        处理 CSV 文件：添加 uv_360nm 列

        参数:
            csv_file: 输入 CSV 文件路径
            output_file: 输出 CSV 文件路径（None则不保存）
            add_noise: 是否添加噪声

        返回:
            df: 添加了 uv_360nm 列的 DataFrame
        """
        # 读取 CSV
        df = pd.read_csv(csv_file)

        # 检查必要列
        if 'mfr' not in df.columns or 'thrust' not in df.columns:
            raise ValueError("CSV must contain 'mfr' and 'thrust' columns")

        # 提取时间序列
        mfr_series = df['mfr'].values
        thrust_series = df['thrust'].values

        # 映射到 UV 强度
        uv_series = self.map_timeseries(
            mfr_series,
            thrust_series,
            add_noise=add_noise
        )

        # 添加新列
        df['uv_360nm'] = uv_series

        # 保存结果
        if output_file is not None:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")

        return df

    def get_parameters(self) -> dict:
        """返回模型参数"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'I_bg': self.I_bg,
            'noise_std': self.noise_std,
            'use_mfr': self.use_mfr,
            'use_thrust': self.use_thrust,
            'random_seed': self.random_seed
        }

    def __repr__(self):
        return (
            f"UV360nmMapper(\n"
            f"  I_uv = {self.alpha:.1f} × mfr^{self.beta:.2f} + "
            f"{self.gamma:.1f} × thrust + {self.I_bg:.1f} + noise(σ={self.noise_std:.1f})\n"
            f")"
        )


def batch_process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    mapper: UV360nmMapper = None,
    add_noise: bool = True
) -> Tuple[int, int]:
    """
    批量处理目录中的所有 CSV 文件

    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        mapper: UV 映射器（None则使用默认参数）
        add_noise: 是否添加噪声

    返回:
        (成功数量, 失败数量)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建映射器
    if mapper is None:
        mapper = UV360nmMapper()

    # 获取所有 CSV 文件
    csv_files = sorted(input_path.glob('*.csv'))

    success_count = 0
    failed_count = 0

    print(f"Processing {len(csv_files)} files from {input_dir}")
    print(f"Mapper: {mapper}")
    print("-" * 70)

    for csv_file in csv_files:
        try:
            # 处理文件
            output_file = output_path / csv_file.name
            mapper.process_csv(csv_file, output_file, add_noise=add_noise)
            success_count += 1

            if success_count % 100 == 0:
                print(f"Processed {success_count}/{len(csv_files)} files...")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            failed_count += 1

    print("-" * 70)
    print(f"Completed: {success_count} success, {failed_count} failed")

    return success_count, failed_count


if __name__ == '__main__':
    """
    使用示例
    """
    print("=" * 70)
    print("第一阶段：推力 → 360 nm UV 映射")
    print("=" * 70)

    # 创建映射器（使用默认参数）
    mapper = UV360nmMapper(
        alpha=1000.0,      # 质量流率转换系数
        beta=1.1,          # 弱非线性指数
        gamma=50.0,        # 推力贡献系数
        I_bg=10.0,         # 背景辐射
        noise_std=2.0,     # 噪声标准差
        use_mfr=True,      # 使用质量流率
        use_thrust=True,   # 使用推力
        random_seed=42
    )

    print(f"\n映射模型：")
    print(mapper)
    print()

    # 批量处理训练集
    print("\n处理训练集...")
    train_success, train_failed = batch_process_directory(
        input_dir='data/train',
        output_dir='data/train_with_uv',
        mapper=mapper,
        add_noise=True
    )

    # 批量处理测试集
    print("\n处理测试集...")
    test_success, test_failed = batch_process_directory(
        input_dir='data/test',
        output_dir='data/test_with_uv',
        mapper=mapper,
        add_noise=True
    )

    print("\n" + "=" * 70)
    print("第一阶段完成！")
    print("=" * 70)
    print(f"训练集: {train_success} 个文件已添加 uv_360nm 列")
    print(f"测试集: {test_success} 个文件已添加 uv_360nm 列")
    print(f"输出目录: data/train_with_uv/ 和 data/test_with_uv/")
    print("=" * 70)
