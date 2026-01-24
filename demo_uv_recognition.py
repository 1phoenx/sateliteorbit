"""
UV 识别系统演示脚本
===========================================

演示完整的识别流程，包括：
1. 单个文件推理
2. 批量推理
3. 结果可视化
4. 性能分析

作者: Claude Code
日期: 2026-01-24
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.uv_inference import UVRecognitionPipeline
from src.uv_visualization import UVVisualizer
from src.uv_results_analysis import UVResultsAnalyzer


def demo_single_file_inference():
    """演示1: 单个文件推理"""
    print("\n" + "=" * 70)
    print("演示 1: 单个文件推理")
    print("=" * 70)

    # 创建流水线
    pipeline = UVRecognitionPipeline(models_dir='models/uv_recognition')

    # 选择一个测试文件
    test_files = list(Path('data/test').glob('*.csv'))
    if len(test_files) == 0:
        print("错误: 没有找到测试文件")
        return

    # 推理第一个文件
    test_file = test_files[0]
    print(f"\n推理文件: {test_file.name}")

    result = pipeline.predict_single(test_file, add_noise=False)

    # 打印结果
    pipeline.print_result(result)

    return result


def demo_batch_inference():
    """演示2: 批量推理"""
    print("\n" + "=" * 70)
    print("演示 2: 批量推理")
    print("=" * 70)

    # 创建流水线
    pipeline = UVRecognitionPipeline(models_dir='models/uv_recognition')

    # 批量推理（只处理前10个文件作为演示）
    test_files = sorted(Path('data/test').glob('*.csv'))[:10]

    print(f"\n批量推理 {len(test_files)} 个文件...")

    results = []
    for i, test_file in enumerate(test_files):
        result = pipeline.predict_single(test_file, add_noise=False)
        results.append(result)
        print(f"  [{i+1}/{len(test_files)}] {test_file.name}: "
              f"推力={result['thrust_estimate']:.4f}N, "
              f"变轨={'是' if result['is_maneuver'] else '否'}")

    # 转换为 DataFrame
    results_df = pd.DataFrame([
        {
            'file_name': r['file_name'],
            'ignition_time': r['ignition_time'],
            'is_maneuver': r['is_maneuver'],
            'thrust_estimate': r['thrust_estimate'],
            'maneuver_type_name': r['maneuver_type_name']
        }
        for r in results
    ])

    print(f"\n批量推理完成！")
    print(f"检测到变轨: {results_df['is_maneuver'].sum()} / {len(results_df)}")
    print(f"平均推力估计: {results_df['thrust_estimate'].mean():.4f} N")

    return results_df


def demo_visualization():
    """演示3: 可视化"""
    print("\n" + "=" * 70)
    print("演示 3: 可视化")
    print("=" * 70)

    visualizer = UVVisualizer(output_dir='demo_figures')

    # 1. UV 映射可视化
    print("\n生成 UV 映射可视化...")
    uv_files = list(Path('data/train_with_uv').glob('*.csv'))
    if len(uv_files) > 0:
        visualizer.plot_uv_mapping(uv_files[0], 'demo_uv_mapping.png')
        print(f"  ✓ 保存: demo_figures/demo_uv_mapping.png")

    # 2. 脉冲检测可视化
    print("\n生成脉冲检测可视化...")
    if len(uv_files) > 0:
        visualizer.plot_pulse_detection(uv_files[0], threshold_factor=3.0,
                                       save_name='demo_pulse_detection.png')
        print(f"  ✓ 保存: demo_figures/demo_pulse_detection.png")

    # 3. 识别结果可视化
    print("\n生成识别结果可视化...")
    results_file = Path('results/uv_recognition_results.csv')
    if results_file.exists():
        visualizer.plot_recognition_results(results_file, 'demo_recognition_performance.png')
        print(f"  ✓ 保存: demo_figures/demo_recognition_performance.png")

    print("\n可视化完成！")


def demo_performance_analysis():
    """演示4: 性能分析"""
    print("\n" + "=" * 70)
    print("演示 4: 性能分析")
    print("=" * 70)

    results_file = Path('results/uv_recognition_results.csv')
    if not results_file.exists():
        print("错误: 结果文件不存在，请先运行推理")
        return

    # 创建分析器
    analyzer = UVResultsAnalyzer(results_file)

    # 分析各项性能
    print("\n分析点火检测性能...")
    analyzer.analyze_ignition_detection()

    print("\n分析变轨分类性能...")
    analyzer.analyze_maneuver_classification()

    print("\n分析推力估计性能...")
    analyzer.analyze_thrust_estimation()

    print("\n分析变轨类型分布...")
    analyzer.analyze_maneuver_type()

    print("\n查找错误案例...")
    analyzer.find_error_cases(top_n=3)

    print("\n生成综合分析图表...")
    analyzer.plot_comprehensive_analysis()

    print("\n性能分析完成！")


def demo_custom_inference():
    """演示5: 自定义推理"""
    print("\n" + "=" * 70)
    print("演示 5: 自定义推理 - 模拟实时数据")
    print("=" * 70)

    from src.uv_mapping import UV360nmMapper
    from src.uv_feature_extraction import UVFeatureExtractor

    # 创建映射器和特征提取器
    mapper = UV360nmMapper()
    extractor = UVFeatureExtractor()

    # 模拟推力和质量流率数据
    print("\n生成模拟数据...")
    time = np.linspace(0, 10, 1000)  # 10秒，100Hz采样

    # 模拟一个脉冲
    thrust = np.zeros_like(time)
    mfr = np.zeros_like(time)

    # 在 t=2s 到 t=5s 之间有推力
    pulse_mask = (time >= 2.0) & (time <= 5.0)
    thrust[pulse_mask] = 0.5  # 0.5 N
    mfr[pulse_mask] = 0.0005  # 0.5 g/s

    print(f"  时间范围: {time[0]:.2f} - {time[-1]:.2f} 秒")
    print(f"  脉冲时间: 2.0 - 5.0 秒")
    print(f"  推力: {thrust[pulse_mask][0]:.4f} N")

    # 映射到 UV
    print("\n映射到 UV 360nm...")
    uv_series = mapper.map_timeseries(mfr, thrust, add_noise=True)
    print(f"  UV 强度范围: {uv_series.min():.2f} - {uv_series.max():.2f}")

    # 提取特征
    print("\n提取特征...")
    features = extractor.extract_features(uv_series, time)
    print(f"  检测到脉冲数: {features['num_pulses']}")
    print(f"  峰值强度: {features['peak_intensity']:.2f}")
    print(f"  平均脉冲持续时间: {features['mean_pulse_duration']:.2f} 秒")
    print(f"  最大上升率: {features['max_rise_rate']:.2f}")

    # 使用模型预测
    print("\n加载模型进行预测...")
    pipeline = UVRecognitionPipeline(models_dir='models/uv_recognition')

    # 准备特征向量
    feature_vector = pipeline._prepare_feature_vector(features)

    # 点火检测
    ignition_time, confidence = pipeline.ignition_detector.detect_ignition(uv_series)
    print(f"\n点火检测:")
    print(f"  检测到的点火时刻: {ignition_time:.2f} 秒")
    print(f"  置信度: {confidence:.2%}")

    # 变轨分类
    X_scaled = pipeline.maneuver_scaler.transform([feature_vector])
    is_maneuver = pipeline.maneuver_classifier.predict(X_scaled)[0]
    maneuver_proba = pipeline.maneuver_classifier.predict_proba(X_scaled)[0, 1]
    print(f"\n变轨分类:")
    print(f"  是否变轨: {'是' if is_maneuver else '否'}")
    print(f"  概率: {maneuver_proba:.2%}")

    # 推力估计
    X_scaled = pipeline.thrust_scaler.transform([feature_vector])
    thrust_estimate = pipeline.thrust_regressor.predict(X_scaled)[0]
    print(f"\n推力估计:")
    print(f"  估计推力: {thrust_estimate:.4f} N")
    print(f"  真实推力: 0.5000 N")
    print(f"  误差: {abs(thrust_estimate - 0.5):.4f} N")

    # 变轨类型
    X_scaled = pipeline.type_scaler.transform([feature_vector])
    maneuver_type = pipeline.type_classifier.predict(X_scaled)[0]
    print(f"\n变轨类型:")
    print(f"  预测类型: {pipeline.class_names[maneuver_type]}")


def main():
    """主函数"""
    print("=" * 70)
    print("UV 识别系统 - 完整演示")
    print("=" * 70)
    print("\n本演示将展示系统的各项功能：")
    print("  1. 单个文件推理")
    print("  2. 批量推理")
    print("  3. 可视化")
    print("  4. 性能分析")
    print("  5. 自定义推理")
    print("\n" + "=" * 70)

    try:
        # 演示1: 单个文件推理
        demo_single_file_inference()

        # 演示2: 批量推理
        demo_batch_inference()

        # 演示3: 可视化
        demo_visualization()

        # 演示4: 性能分析
        demo_performance_analysis()

        # 演示5: 自定义推理
        demo_custom_inference()

        print("\n" + "=" * 70)
        print("✓ 所有演示完成！")
        print("=" * 70)
        print("\n生成的文件:")
        print("  - demo_figures/demo_uv_mapping.png")
        print("  - demo_figures/demo_pulse_detection.png")
        print("  - demo_figures/demo_recognition_performance.png")
        print("  - analysis/uv_recognition/comprehensive_analysis.png")
        print("  - analysis/uv_recognition/analysis_report.txt")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
