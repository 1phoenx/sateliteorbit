"""
简化运行脚本 - 确保代码可以正常运行
用于验证整个流程的正确性
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd


def check_dependencies():
    """检查依赖库"""
    print("检查依赖库...")
    required = ['numpy', 'pandas', 'sklearn', 'torch', 'scipy']
    missing = []

    for lib in required:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ✗ {lib} (缺失)")
            missing.append(lib)

    if missing:
        print(f"\n缺失的库: {', '.join(missing)}")
        print("请运行: pip install " + ' '.join(missing))
        return False
    return True


def check_data():
    """检查数据文件"""
    print("\n检查数据文件...")
    data_files = [
        'data/metadata.csv',
        'data/feature_dataset.csv'
    ]

    for f in data_files:
        path = os.path.join(project_root, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (不存在)")
            return False
    return True


def run_feature_extraction_test():
    """测试特征提取"""
    print("\n测试特征提取...")
    try:
        from src.feature_extraction_v2 import ThrusterFeatureExtractor

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        thrust = np.random.randn(n_samples) * 0.1
        thrust[300:500] += 1.0  # 模拟点火
        ton = np.zeros(n_samples)
        ton[300:500] = 1

        extractor = ThrusterFeatureExtractor(sampling_rate=100.0)
        B_thrust, sigma = extractor.compute_baseline(thrust, ton)
        P = extractor.extract_P(thrust, ton, B_thrust, sigma)
        T, ignition_time, true_thrust = extractor.extract_T(thrust, B_thrust, sigma, ton, use_precise_detection=False)
        R = extractor.extract_R(thrust, T)

        print(f"  P = {P:.4f}")
        print(f"  T = {T:.4f}s")
        print(f"  R = {R:.4f}" if not np.isnan(R) else "  R = NaN")
        print(f"  点火时刻 = {ignition_time:.4f}s")
        print("  ✓ 特征提取测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 特征提取测试失败: {e}")
        return False


def run_ignition_detection_test():
    """测试点火时刻检测"""
    print("\n测试点火时刻检测...")
    try:
        from src.ignition_detector import IgnitionDetector, detect_ignition_ensemble

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        thrust = np.random.randn(n_samples) * 0.1
        thrust[300:500] += 1.0  # 模拟点火
        ton = np.zeros(n_samples)
        ton[300:500] = 1

        # 测试单一检测器
        detector = IgnitionDetector(sampling_rate=100.0)
        result = detector.detect_ignition_time(thrust, ton)

        print(f"  检测到点火时刻: {result['ignition_time']:.4f}s")
        print(f"  真实点火时刻: 3.00s")
        print(f"  误差: {abs(result['ignition_time'] - 3.0):.4f}s")
        print(f"  置信度: {result['confidence']:.4f}")

        # 测试集成检测
        result_ensemble = detect_ignition_ensemble(thrust, ton, 100.0)
        print(f"  集成检测点火时刻: {result_ensemble['ignition_time']:.4f}s")

        print("  ✓ 点火时刻检测测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 点火时刻检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_test():
    """测试模型"""
    print("\n测试模型...")
    try:
        import torch
        from src.final_model import ManeuverDetectionSystem, extract_features

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 100
        P = np.random.rand(n_samples) * 2
        T = np.random.rand(n_samples) * 500
        R = np.random.rand(n_samples) * 10

        X = np.column_stack([P, T, R])
        y = (P > 1.0).astype(int)
        t = np.random.rand(n_samples) * 100

        # 测试特征提取
        features = extract_features(P, T, R)
        print(f"  增强特征维度: {features.shape}")

        # 测试模型训练 (简化版)
        system = ManeuverDetectionSystem(threshold=0.5)
        system.train(X[:80], y[:80], t[:80], use_smote=False)

        # 测试预测
        result = system.predict(X[80:])
        print(f"  预测样本数: {len(result['is_maneuver'])}")
        print(f"  预测变轨数: {sum(result['is_maneuver'])}")

        print("  ✓ 模型测试通过")
        return True
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_gan_test():
    """测试GAN"""
    print("\n测试GAN...")
    try:
        from src.gan_v3 import FeatureGANV2

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 100
        X = np.random.rand(n_samples, 3)

        # 测试GAN训练 (少量epochs)
        gan = FeatureGANV2(latent_dim=32, feature_dim=3)
        history = gan.train(X, epochs=5, batch_size=16, verbose=False)

        print(f"  训练epochs: {len(history['g_loss'])}")
        print(f"  最终G_loss: {history['g_loss'][-1]:.4f}")
        print(f"  最终D_loss: {history['d_loss'][-1]:.4f}")

        # 测试生成
        generated = gan.generate_samples(10)
        print(f"  生成样本形状: {generated.shape}")

        print("  ✓ GAN测试通过")
        return True
    except Exception as e:
        print(f"  ✗ GAN测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("代码运行测试")
    print("=" * 60)

    results = []

    # 检查依赖
    results.append(('依赖检查', check_dependencies()))

    # 检查数据
    results.append(('数据检查', check_data()))

    # 运行测试
    results.append(('特征提取', run_feature_extraction_test()))
    results.append(('点火检测', run_ignition_detection_test()))
    results.append(('模型测试', run_model_test()))
    results.append(('GAN测试', run_gan_test()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查上述错误信息")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
