"""
UV 识别系统 - 单元测试
===========================================

测试所有核心功能模块

作者: Claude Code
日期: 2026-01-24
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import unittest

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.uv_mapping import UV360nmMapper
from src.uv_feature_extraction import UVFeatureExtractor
from src.uv_recognition_models import (
    IgnitionDetector,
    ManeuverClassifier,
    ThrustRegressor,
    ManeuverTypeClassifier
)


class TestUVMapping(unittest.TestCase):
    """测试 UV 映射模块"""

    def setUp(self):
        self.mapper = UV360nmMapper()

    def test_single_point_mapping(self):
        """测试单点映射"""
        mfr = 0.001
        thrust = 1.0
        uv = self.mapper.map_single(mfr, thrust, add_noise=False)

        self.assertIsInstance(uv, float)
        self.assertGreater(uv, 0)
        print(f"✓ 单点映射测试通过: mfr={mfr}, thrust={thrust}, uv={uv:.2f}")

    def test_timeseries_mapping(self):
        """测试时间序列映射"""
        mfr = np.array([0.001, 0.002, 0.003])
        thrust = np.array([1.0, 1.5, 2.0])
        uv = self.mapper.map_timeseries(mfr, thrust, add_noise=False)

        self.assertEqual(len(uv), len(mfr))
        self.assertTrue(np.all(uv > 0))
        print(f"✓ 时间序列映射测试通过: 长度={len(uv)}, 范围=[{uv.min():.2f}, {uv.max():.2f}]")

    def test_noise_addition(self):
        """测试噪声添加"""
        mfr = np.ones(100) * 0.001
        thrust = np.ones(100) * 1.0

        uv_no_noise = self.mapper.map_timeseries(mfr, thrust, add_noise=False)
        uv_with_noise = self.mapper.map_timeseries(mfr, thrust, add_noise=True)

        # 有噪声的应该有更大的标准差
        self.assertGreater(np.std(uv_with_noise), np.std(uv_no_noise))
        print(f"✓ 噪声添加测试通过: std_no_noise={np.std(uv_no_noise):.2f}, std_with_noise={np.std(uv_with_noise):.2f}")


class TestFeatureExtraction(unittest.TestCase):
    """测试特征提取模块"""

    def setUp(self):
        self.extractor = UVFeatureExtractor()

    def test_feature_extraction(self):
        """测试特征提取"""
        # 创建模拟 UV 时间序列（包含一个脉冲）
        uv_series = np.zeros(1000)
        uv_series[200:500] = 50.0  # 脉冲
        uv_series += 10.0  # 背景

        features = self.extractor.extract_features(uv_series)

        self.assertIn('num_pulses', features)
        self.assertIn('peak_intensity', features)
        self.assertIn('mean_pulse_duration', features)

        # 应该检测到1个脉冲
        self.assertEqual(features['num_pulses'], 1)
        print(f"✓ 特征提取测试通过: 检测到 {features['num_pulses']} 个脉冲")

    def test_no_pulse_detection(self):
        """测试无脉冲情况"""
        # 只有背景噪声
        uv_series = np.random.randn(1000) * 2 + 10.0

        features = self.extractor.extract_features(uv_series)

        # 应该检测不到脉冲
        self.assertEqual(features['num_pulses'], 0)
        print(f"✓ 无脉冲检测测试通过: 检测到 {features['num_pulses']} 个脉冲")

    def test_multiple_pulses(self):
        """测试多脉冲检测"""
        uv_series = np.zeros(1000)
        uv_series[100:200] = 50.0  # 脉冲1
        uv_series[400:500] = 50.0  # 脉冲2
        uv_series[700:800] = 50.0  # 脉冲3
        uv_series += 10.0  # 背景

        features = self.extractor.extract_features(uv_series)

        # 应该检测到3个脉冲
        self.assertEqual(features['num_pulses'], 3)
        print(f"✓ 多脉冲检测测试通过: 检测到 {features['num_pulses']} 个脉冲")


class TestIgnitionDetector(unittest.TestCase):
    """测试点火检测模块"""

    def setUp(self):
        self.detector = IgnitionDetector()

    def test_ignition_detection(self):
        """测试点火检测"""
        # 创建模拟 UV 时间序列（在 t=2s 点火）
        uv_series = np.zeros(1000)
        uv_series[:200] = 10.0  # 背景
        uv_series[200:] = 50.0  # 点火后

        ignition_time, confidence = self.detector.detect_ignition(uv_series)

        self.assertGreater(ignition_time, 0)
        self.assertGreater(confidence, 0)
        print(f"✓ 点火检测测试通过: 时刻={ignition_time:.2f}s, 置信度={confidence:.2%}")

    def test_no_ignition(self):
        """测试无点火情况"""
        # 只有背景
        uv_series = np.ones(1000) * 10.0

        ignition_time, confidence = self.detector.detect_ignition(uv_series)

        # 应该检测不到点火
        self.assertEqual(ignition_time, -1.0)
        self.assertEqual(confidence, 0.0)
        print(f"✓ 无点火检测测试通过: 时刻={ignition_time:.2f}s")


class TestEndToEndPipeline(unittest.TestCase):
    """测试端到端流水线"""

    def test_full_pipeline(self):
        """测试完整流水线"""
        print("\n" + "=" * 70)
        print("端到端流水线测试")
        print("=" * 70)

        # 1. 创建模拟数据
        print("\n1. 创建模拟数据...")
        time = np.linspace(0, 10, 1000)
        thrust = np.zeros_like(time)
        mfr = np.zeros_like(time)

        # 在 t=2s 到 t=5s 之间有推力
        pulse_mask = (time >= 2.0) & (time <= 5.0)
        thrust[pulse_mask] = 0.5
        mfr[pulse_mask] = 0.0005

        print(f"   ✓ 数据长度: {len(time)}")
        print(f"   ✓ 脉冲时间: 2.0 - 5.0 秒")

        # 2. UV 映射
        print("\n2. UV 映射...")
        mapper = UV360nmMapper()
        uv_series = mapper.map_timeseries(mfr, thrust, add_noise=True)
        print(f"   ✓ UV 强度范围: [{uv_series.min():.2f}, {uv_series.max():.2f}]")

        # 3. 特征提取
        print("\n3. 特征提取...")
        extractor = UVFeatureExtractor()
        features = extractor.extract_features(uv_series, time)
        print(f"   ✓ 检测到脉冲数: {features['num_pulses']}")
        print(f"   ✓ 峰值强度: {features['peak_intensity']:.2f}")

        # 4. 点火检测
        print("\n4. 点火检测...")
        detector = IgnitionDetector()
        ignition_time, confidence = detector.detect_ignition(uv_series)
        print(f"   ✓ 检测到的点火时刻: {ignition_time:.2f}s")
        print(f"   ✓ 置信度: {confidence:.2%}")

        # 验证结果
        self.assertGreater(features['num_pulses'], 0, "应该检测到至少1个脉冲")
        self.assertGreater(ignition_time, 0, "应该检测到点火时刻")

        print("\n" + "=" * 70)
        print("✓ 端到端流水线测试通过")
        print("=" * 70)


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("UV 识别系统 - 单元测试")
    print("=" * 70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestUVMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestIgnitionDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
