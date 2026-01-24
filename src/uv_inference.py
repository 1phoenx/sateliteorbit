"""
完整推理流程
===========================================

端到端推理：
输入：原始 CSV（thrust, mfr）
输出：
  1. 点火时刻
  2. 是否变轨
  3. 推力大小
  4. 变轨类型

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
import sys
sys.path.append(str(Path(__file__).parent))

from uv_mapping import UV360nmMapper
from uv_feature_extraction import UVFeatureExtractor
from uv_recognition_models import (
    IgnitionDetector,
    ManeuverClassifier,
    ThrustRegressor,
    ManeuverTypeClassifier
)


class UVRecognitionPipeline:
    """
    完整的 UV 识别流水线

    流程：
    1. 推力/质量流率 → UV 映射
    2. UV 特征提取
    3. 四个识别任务
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = 'models/uv_recognition'
    ):
        """
        初始化流水线

        参数:
            models_dir: 模型目录
        """
        self.models_dir = Path(models_dir)

        # 初始化 UV 映射器
        self.uv_mapper = UV360nmMapper(
            alpha=1000.0,
            beta=1.1,
            gamma=50.0,
            I_bg=10.0,
            noise_std=2.0,
            use_mfr=True,
            use_thrust=True,
            random_seed=42
        )

        # 初始化特征提取器
        self.feature_extractor = UVFeatureExtractor(
            threshold_factor=3.0,
            min_pulse_duration=0.1,
            sampling_rate=100.0
        )

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载所有识别模型"""
        print(f"加载模型从 {self.models_dir}...")

        # 1. 点火检测器 - 直接创建新实例而不是加载
        # 因为它是基于规则的，不需要训练
        self.ignition_detector = IgnitionDetector(
            threshold_factor=3.0,
            min_rise_rate=10.0,
            sampling_rate=100.0
        )

        # 2. 变轨分类器
        maneuver_clf = ManeuverClassifier()
        maneuver_clf.load(self.models_dir / 'maneuver_classifier.pkl')
        self.maneuver_classifier = maneuver_clf.model
        self.maneuver_scaler = maneuver_clf.scaler

        # 3. 推力回归器
        thrust_reg = ThrustRegressor()
        thrust_reg.load(self.models_dir / 'thrust_regressor.pkl')
        self.thrust_regressor = thrust_reg.model
        self.thrust_scaler = thrust_reg.scaler

        # 4. 变轨类型分类器
        type_clf = ManeuverTypeClassifier()
        type_clf.load(self.models_dir / 'maneuver_type_classifier.pkl')
        self.type_classifier = type_clf.model
        self.type_scaler = type_clf.scaler
        self.class_names = type_clf.class_names

        print("所有模型加载完成！")

    def predict_single(
        self,
        csv_file: Union[str, Path],
        add_noise: bool = False
    ) -> Dict:
        """
        单个文件推理

        参数:
            csv_file: 输入 CSV 文件（包含 thrust, mfr）
            add_noise: 是否添加噪声

        返回:
            results: 识别结果字典
        """
        csv_file = Path(csv_file)

        # 步骤1: 读取原始数据
        df = pd.read_csv(csv_file)

        # 步骤2: 映射到 UV
        mfr_series = df['mfr'].values
        thrust_series = df['thrust'].values
        uv_series = self.uv_mapper.map_timeseries(
            mfr_series,
            thrust_series,
            add_noise=add_noise
        )

        # 步骤3: 提取特征
        features = self.feature_extractor.extract_features(uv_series)

        # 步骤4: 准备特征向量
        feature_vector = self._prepare_feature_vector(features)

        # 步骤5: 运行识别模型
        results = {
            'file_name': csv_file.stem,
            'uv_features': features
        }

        # 5.1 点火时刻识别
        ignition_time, confidence = self.ignition_detector.detect_ignition(uv_series)
        results['ignition_time'] = float(ignition_time)
        results['ignition_confidence'] = float(confidence)

        # 5.2 是否变轨
        X_scaled = self.maneuver_scaler.transform([feature_vector])
        is_maneuver = self.maneuver_classifier.predict(X_scaled)[0]
        maneuver_proba = self.maneuver_classifier.predict_proba(X_scaled)[0, 1]
        results['is_maneuver'] = bool(is_maneuver)
        results['maneuver_probability'] = float(maneuver_proba)

        # 5.3 推力大小
        X_scaled = self.thrust_scaler.transform([feature_vector])
        thrust_estimate = self.thrust_regressor.predict(X_scaled)[0]
        results['thrust_estimate'] = float(thrust_estimate)

        # 5.4 变轨类型
        X_scaled = self.type_scaler.transform([feature_vector])
        maneuver_type = self.type_classifier.predict(X_scaled)[0]
        type_proba = self.type_classifier.predict_proba(X_scaled)[0]
        results['maneuver_type'] = int(maneuver_type)
        results['maneuver_type_name'] = self.class_names[maneuver_type]
        results['maneuver_type_probabilities'] = {
            name: float(prob)
            for name, prob in zip(self.class_names, type_proba)
        }

        # 添加真实标签（如果有）
        if 'ton' in df.columns:
            ignition_data = df[df['ton'] == 1]
            if len(ignition_data) > 0:
                results['true_ignition_time'] = ignition_data.iloc[0]['time']
                results['true_thrust'] = float(ignition_data['thrust'].mean())
                results['true_duration'] = len(ignition_data) * 0.01
            else:
                results['true_ignition_time'] = None
                results['true_thrust'] = 0.0
                results['true_duration'] = 0.0

        return results

    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """准备特征向量"""
        feature_vector = np.array([
            features['background_mean'],
            features['background_std'],
            features['threshold'],
            features['peak_intensity'],
            features['mean_intensity'],
            features['total_energy'],
            features['num_pulses'],
            features['mean_pulse_duration'],
            features['max_pulse_duration'],
            features['mean_pulse_peak'],
            features['max_pulse_peak'],
            features['mean_pulse_energy'],
            features['max_rise_rate'],
            features['mean_rise_rate'],
            features['mean_pulse_interval'],
            features['min_pulse_interval']
        ])
        return feature_vector

    def batch_predict(
        self,
        input_dir: Union[str, Path],
        output_file: Union[str, Path],
        add_noise: bool = False
    ) -> pd.DataFrame:
        """
        批量推理

        参数:
            input_dir: 输入目录
            output_file: 输出 CSV 文件
            add_noise: 是否添加噪声

        返回:
            results_df: 结果 DataFrame
        """
        input_path = Path(input_dir)
        csv_files = sorted(input_path.glob('*.csv'))

        print(f"批量推理 {len(csv_files)} 个文件...")

        all_results = []
        for i, csv_file in enumerate(csv_files):
            try:
                result = self.predict_single(csv_file, add_noise=add_noise)
                all_results.append(result)

                if (i + 1) % 100 == 0:
                    print(f"已处理 {i+1}/{len(csv_files)} 个文件...")

            except Exception as e:
                print(f"错误处理 {csv_file.name}: {e}")

        print(f"完成！成功处理 {len(all_results)} 个文件")

        # 转换为 DataFrame
        results_df = pd.DataFrame([
            {
                'file_name': r['file_name'],
                'ignition_time': r['ignition_time'],
                'ignition_confidence': r['ignition_confidence'],
                'is_maneuver': r['is_maneuver'],
                'maneuver_probability': r['maneuver_probability'],
                'thrust_estimate': r['thrust_estimate'],
                'maneuver_type': r['maneuver_type'],
                'maneuver_type_name': r['maneuver_type_name'],
                'true_ignition_time': r.get('true_ignition_time', None),
                'true_thrust': r.get('true_thrust', 0.0),
                'true_duration': r.get('true_duration', 0.0)
            }
            for r in all_results
        ])

        # 保存
        results_df.to_csv(output_file, index=False)
        print(f"结果已保存到 {output_file}")

        return results_df

    def print_result(self, result: Dict):
        """打印单个结果"""
        print("\n" + "=" * 70)
        print(f"文件: {result['file_name']}")
        print("=" * 70)
        print(f"\n1️⃣ 点火时刻识别:")
        print(f"   检测到的点火时刻: {result['ignition_time']:.2f} 秒")
        print(f"   置信度: {result['ignition_confidence']:.2%}")
        if result.get('true_ignition_time'):
            print(f"   真实点火时刻: {result['true_ignition_time']}")

        print(f"\n2️⃣ 是否变轨:")
        print(f"   预测: {'是' if result['is_maneuver'] else '否'}")
        print(f"   概率: {result['maneuver_probability']:.2%}")

        print(f"\n3️⃣ 推力大小:")
        print(f"   估计推力: {result['thrust_estimate']:.4f} N")
        if result.get('true_thrust'):
            print(f"   真实推力: {result['true_thrust']:.4f} N")
            error = abs(result['thrust_estimate'] - result['true_thrust'])
            print(f"   误差: {error:.4f} N")

        print(f"\n4️⃣ 变轨类型:")
        print(f"   预测类型: {result['maneuver_type_name']}")
        print(f"   各类型概率:")
        for name, prob in result['maneuver_type_probabilities'].items():
            print(f"     - {name}: {prob:.2%}")

        print("=" * 70)


def main():
    """主函数：演示推理流程"""
    print("=" * 70)
    print("UV 识别系统 - 完整推理流程")
    print("=" * 70)

    # 创建流水线
    pipeline = UVRecognitionPipeline(models_dir='models/uv_recognition')

    # 测试单个文件
    print("\n测试单个文件推理...")
    test_file = Path('data/test/00001_001_SN13_24bars_ssf.csv')
    if test_file.exists():
        result = pipeline.predict_single(test_file, add_noise=False)
        pipeline.print_result(result)
    else:
        print(f"测试文件不存在: {test_file}")

    # 批量推理测试集
    print("\n\n批量推理测试集...")
    results_df = pipeline.batch_predict(
        input_dir='data/test',
        output_file='results/uv_recognition_results.csv',
        add_noise=False
    )

    # 统计分析
    print("\n" + "=" * 70)
    print("统计分析")
    print("=" * 70)
    print(f"总样本数: {len(results_df)}")
    print(f"检测到变轨: {results_df['is_maneuver'].sum()} 个")
    print(f"平均推力估计: {results_df['thrust_estimate'].mean():.4f} N")
    print(f"\n变轨类型分布:")
    print(results_df['maneuver_type_name'].value_counts())

    # 计算准确率（如果有真实标签）
    if 'true_thrust' in results_df.columns:
        mae = np.abs(results_df['thrust_estimate'] - results_df['true_thrust']).mean()
        print(f"\n推力估计 MAE: {mae:.4f} N")

    print("\n" + "=" * 70)
    print("推理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
