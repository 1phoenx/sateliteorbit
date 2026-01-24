"""
完整的 UV 识别流水线执行脚本
===========================================

端到端执行：
1. 推力/质量流率 → UV 映射
2. UV 特征提取
3. 模型训练
4. 推理识别
5. 结果可视化

作者: Claude Code
日期: 2026-01-24
"""

import sys
from pathlib import Path
import argparse

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / 'src'))


def run_stage1_mapping(args):
    """阶段1: UV 映射"""
    print("\n" + "=" * 70)
    print("阶段 1: 推力/质量流率 → UV 360nm 映射")
    print("=" * 70)

    from src.uv_mapping import batch_process_csv

    # 处理训练集
    print("\n处理训练集...")
    batch_process_csv(
        input_dir=args.train_dir,
        output_dir=args.train_uv_dir,
        add_noise=True
    )

    # 处理测试集
    print("\n处理测试集...")
    batch_process_csv(
        input_dir=args.test_dir,
        output_dir=args.test_uv_dir,
        add_noise=False
    )

    print("\n✓ 阶段 1 完成")


def run_stage2_features(args):
    """阶段2: 特征提取"""
    print("\n" + "=" * 70)
    print("阶段 2: UV 特征提取")
    print("=" * 70)

    from src.uv_feature_extraction import batch_extract_features, UVFeatureExtractor

    extractor = UVFeatureExtractor(
        threshold_factor=3.0,
        min_pulse_duration=0.1,
        sampling_rate=100.0
    )

    # 提取训练集特征
    print("\n提取训练集特征...")
    batch_extract_features(
        input_dir=args.train_uv_dir,
        output_file=args.train_features_file,
        extractor=extractor
    )

    # 提取测试集特征
    print("\n提取测试集特征...")
    batch_extract_features(
        input_dir=args.test_uv_dir,
        output_file=args.test_features_file,
        extractor=extractor
    )

    print("\n✓ 阶段 2 完成")


def run_stage3_training(args):
    """阶段3: 模型训练"""
    print("\n" + "=" * 70)
    print("阶段 3: 识别模型训练")
    print("=" * 70)

    from src.uv_recognition_models import train_all_models

    train_all_models(
        train_features_file=args.train_features_file,
        test_features_file=args.test_features_file,
        output_dir=args.models_dir
    )

    print("\n✓ 阶段 3 完成")


def run_stage4_inference(args):
    """阶段4: 推理识别"""
    print("\n" + "=" * 70)
    print("阶段 4: 推理识别")
    print("=" * 70)

    from src.uv_inference import UVRecognitionPipeline

    # 创建流水线
    pipeline = UVRecognitionPipeline(models_dir=args.models_dir)

    # 批量推理
    results_df = pipeline.batch_predict(
        input_dir=args.test_dir,
        output_file=args.results_file,
        add_noise=False
    )

    # 打印统计
    print("\n" + "=" * 70)
    print("推理统计")
    print("=" * 70)
    print(f"总样本数: {len(results_df)}")
    print(f"检测到变轨: {results_df['is_maneuver'].sum()} 个")
    print(f"平均推力估计: {results_df['thrust_estimate'].mean():.4f} N")
    print(f"\n变轨类型分布:")
    print(results_df['maneuver_type_name'].value_counts())

    if 'true_thrust' in results_df.columns:
        import numpy as np
        mae = np.abs(results_df['thrust_estimate'] - results_df['true_thrust']).mean()
        rmse = np.sqrt(((results_df['thrust_estimate'] - results_df['true_thrust']) ** 2).mean())
        print(f"\n推力估计性能:")
        print(f"  MAE: {mae:.4f} N")
        print(f"  RMSE: {rmse:.4f} N")

    print("\n✓ 阶段 4 完成")


def run_stage5_visualization(args):
    """阶段5: 结果可视化"""
    print("\n" + "=" * 70)
    print("阶段 5: 结果可视化")
    print("=" * 70)

    from src.uv_visualization import UVVisualizer

    visualizer = UVVisualizer(output_dir=args.figures_dir)

    # 1. UV 映射可视化
    print("\n生成 UV 映射可视化...")
    uv_files = list(Path(args.train_uv_dir).glob('*.csv'))
    if len(uv_files) > 0:
        visualizer.plot_uv_mapping(uv_files[0], 'uv_mapping_example.png')

    # 2. 脉冲检测可视化
    print("生成脉冲检测可视化...")
    if len(uv_files) > 0:
        visualizer.plot_pulse_detection(uv_files[0], threshold_factor=3.0,
                                       save_name='pulse_detection_example.png')

    # 3. 识别结果可视化
    print("生成识别结果可视化...")
    if Path(args.results_file).exists():
        visualizer.plot_recognition_results(args.results_file, 'recognition_performance.png')

    # 4. 特征重要性可视化
    print("生成特征重要性可视化...")
    model_files = [
        (f'{args.models_dir}/maneuver_classifier.pkl', 'feature_importance_maneuver.png'),
        (f'{args.models_dir}/thrust_regressor.pkl', 'feature_importance_thrust.png'),
        (f'{args.models_dir}/maneuver_type_classifier.pkl', 'feature_importance_type.png')
    ]

    for model_file, save_name in model_files:
        if Path(model_file).exists():
            visualizer.plot_feature_importance(model_file, save_name)

    print(f"\n所有图表已保存到: {args.figures_dir}")
    print("\n✓ 阶段 5 完成")


def main():
    parser = argparse.ArgumentParser(description='UV 识别系统完整流水线')

    # 数据路径
    parser.add_argument('--train-dir', type=str, default='data/train',
                       help='训练集原始数据目录')
    parser.add_argument('--test-dir', type=str, default='data/test',
                       help='测试集原始数据目录')
    parser.add_argument('--train-uv-dir', type=str, default='data/train_with_uv',
                       help='训练集 UV 数据输出目录')
    parser.add_argument('--test-uv-dir', type=str, default='data/test_with_uv',
                       help='测试集 UV 数据输出目录')

    # 特征文件
    parser.add_argument('--train-features-file', type=str, default='data/uv_features_train.csv',
                       help='训练集特征文件')
    parser.add_argument('--test-features-file', type=str, default='data/uv_features_test.csv',
                       help='测试集特征文件')

    # 模型和结果
    parser.add_argument('--models-dir', type=str, default='models/uv_recognition',
                       help='模型保存目录')
    parser.add_argument('--results-file', type=str, default='results/uv_recognition_results.csv',
                       help='推理结果文件')
    parser.add_argument('--figures-dir', type=str, default='figures/uv_recognition',
                       help='可视化图表目录')

    # 执行阶段
    parser.add_argument('--stages', type=str, default='all',
                       help='执行阶段: all, 1, 2, 3, 4, 5 或组合如 1,2,3')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的输出文件')

    args = parser.parse_args()

    # 创建必要的目录
    Path(args.train_uv_dir).mkdir(parents=True, exist_ok=True)
    Path(args.test_uv_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    # 解析执行阶段
    if args.stages == 'all':
        stages = [1, 2, 3, 4, 5]
    else:
        stages = [int(s) for s in args.stages.split(',')]

    print("=" * 70)
    print("UV 识别系统 - 完整流水线")
    print("=" * 70)
    print(f"执行阶段: {stages}")
    print("=" * 70)

    # 执行各阶段
    if 1 in stages:
        run_stage1_mapping(args)

    if 2 in stages:
        run_stage2_features(args)

    if 3 in stages:
        run_stage3_training(args)

    if 4 in stages:
        run_stage4_inference(args)

    if 5 in stages:
        run_stage5_visualization(args)

    print("\n" + "=" * 70)
    print("✓ 所有阶段完成！")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - 模型: {args.models_dir}")
    print(f"  - 推理结果: {args.results_file}")
    print(f"  - 可视化图表: {args.figures_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
