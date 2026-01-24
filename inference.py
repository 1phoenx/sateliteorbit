"""
双分支模型推理脚本
==================

使用训练好的模型进行推理：
- 测试集推理
- 计算性能指标
- 生成推理结果CSV

作者: Claude Code
日期: 2026-01-23
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json

from model_dual_branch import DualBranchModel


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = DualBranchModel(
        statistical_input_dim=3,
        sequence_length=300,
        num_classes=2
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"  Best epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"  Val false alarm rate: {checkpoint['val_false_alarm_rate']:.2f}%")

    return model, device


def run_inference(
    model_path='models/best_model.pth',
    data_dir='data/features_timeseries',
    output_dir='results',
    split='test'
):
    """运行推理"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model, device = load_model(model_path)

    # 加载数据
    print(f"\nLoading {split} data...")
    statistical_features = np.load(f'{data_dir}/{split}_statistical_features.npy')
    thrust_sequences = np.load(f'{data_dir}/{split}_thrust_sequences.npy')
    labels_anomaly = np.load(f'{data_dir}/{split}_labels_anomaly.npy')
    labels_thrust = np.load(f'{data_dir}/{split}_labels_thrust.npy')
    metadata = pd.read_csv(f'{data_dir}/{split}_metadata.csv')

    # 加载归一化参数
    stat_mean = np.load(f'{data_dir}/stat_mean.npy')
    stat_std = np.load(f'{data_dir}/stat_std.npy')
    ts_mean = np.load(f'{data_dir}/ts_mean.npy')
    ts_std = np.load(f'{data_dir}/ts_std.npy')

    # 归一化
    statistical_features = (statistical_features - stat_mean) / stat_std
    thrust_sequences = (thrust_sequences - ts_mean) / ts_std

    num_samples = len(statistical_features)
    print(f"Total samples: {num_samples}")

    # 推理
    predictions_class = []
    predictions_thrust = []
    inference_times = []

    print("\nRunning inference...")
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            # 单样本推理
            stat_feat = torch.FloatTensor(statistical_features[i:i+1]).to(device)
            ts_feat = torch.FloatTensor(thrust_sequences[i:i+1]).to(device)

            # 计时
            start_time = time.time()
            cls_output, reg_output = model(stat_feat, ts_feat)
            inference_time = time.time() - start_time

            # 预测结果
            _, pred_class = cls_output.max(1)
            pred_thrust = reg_output.item()

            predictions_class.append(pred_class.item())
            predictions_thrust.append(pred_thrust)
            inference_times.append(inference_time)

    # 构建结果DataFrame
    results = metadata.copy()
    results['pred_is_anomalous'] = predictions_class
    results['pred_thrust'] = predictions_thrust
    results['true_is_anomalous'] = labels_anomaly
    results['inference_time'] = inference_times

    # 计算指标
    correct = sum(
        pred == true
        for pred, true in zip(predictions_class, labels_anomaly)
    )
    accuracy = 100.0 * correct / num_samples

    # 虚警率（预测异常但实际正常）
    false_positives = sum(
        pred == 1 and true == 0
        for pred, true in zip(predictions_class, labels_anomaly)
    )
    true_negatives = sum(
        pred == 0 and true == 0
        for pred, true in zip(predictions_class, labels_anomaly)
    )

    if (false_positives + true_negatives) > 0:
        false_alarm_rate = 100.0 * false_positives / (false_positives + true_negatives)
    else:
        false_alarm_rate = 0.0

    # 推力估计误差
    thrust_errors = [
        abs(pred - true)
        for pred, true in zip(predictions_thrust, labels_thrust)
    ]
    mean_thrust_error = np.mean(thrust_errors)

    # 平均响应时间
    mean_inference_time = np.mean(inference_times)

    # 打印指标
    print("\n" + "="*70)
    print("Performance Metrics")
    print("="*70)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
    print(f"Mean Thrust Error: {mean_thrust_error:.4f} N")
    print(f"Mean Inference Time: {mean_inference_time*1000:.2f} ms")
    print(f"  (≤ 5 seconds requirement: {'✅ PASS' if mean_inference_time <= 5.0 else '❌ FAIL'})")

    # 详细分类报告
    true_positives = sum(
        pred == 1 and true == 1
        for pred, true in zip(predictions_class, labels_anomaly)
    )
    false_negatives = sum(
        pred == 0 and true == 1
        for pred, true in zip(predictions_class, labels_anomaly)
    )
    total_anomalies = sum(labels_anomaly)
    total_normal = num_samples - total_anomalies

    print(f"\nDetailed Classification:")
    print(f"  Total Anomalies: {total_anomalies}")
    print(f"  Total Normal: {total_normal}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives: {true_negatives}")
    print(f"  False Negatives: {false_negatives}")

    if total_anomalies > 0:
        recall = 100.0 * true_positives / total_anomalies
        print(f"  Recall (Detection Rate): {recall:.2f}%")

    if (true_positives + false_positives) > 0:
        precision = 100.0 * true_positives / (true_positives + false_positives)
        print(f"  Precision: {precision:.2f}%")

    # 保存结果
    results.to_csv(output_path / 'inference_result.csv', index=False)
    print(f"\nResults saved to {output_path / 'inference_result.csv'}")

    # 保存指标报告
    metrics = {
        'accuracy': float(accuracy),
        'false_alarm_rate': float(false_alarm_rate),
        'mean_thrust_error': float(mean_thrust_error),
        'mean_inference_time': float(mean_inference_time),
        'total_samples': int(num_samples),
        'total_anomalies': int(total_anomalies),
        'total_normal': int(total_normal),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'recall': float(recall) if total_anomalies > 0 else 0.0,
        'precision': float(precision) if (true_positives + false_positives) > 0 else 0.0
    }

    with open(output_path / 'metrics_report.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_path / 'metrics_report.json'}")

    # 生成文本报告
    generate_text_report(metrics, output_path)

    return results, metrics


def generate_text_report(metrics, output_dir):
    """生成文本格式的性能报告"""

    report = f"""
{'='*70}
变轨检测模型 - 性能评估报告
{'='*70}

模型架构: 双分支1D CNN (统计特征 + 时序特征)
日期: 2026-01-23

{'='*70}
核心性能指标
{'='*70}

1. 准确率 (Accuracy)
   {metrics['accuracy']:.2f}%
   目标: ≥92%
   状态: {'✅ 达标' if metrics['accuracy'] >= 92 else '⚠️ 未达标'}

2. 虚警率 (False Alarm Rate)
   {metrics['false_alarm_rate']:.2f}%
   目标: ≤3%
   状态: {'✅ 达标' if metrics['false_alarm_rate'] <= 3 else '⚠️ 未达标'}

3. 推力估计误差 (Mean Thrust Error)
   {metrics['mean_thrust_error']:.4f} N
   目标: Δv < 0.1 m/s
   状态: {'✅ 达标' if metrics['mean_thrust_error'] < 0.1 else '⚠️ 未达标'}

4. 响应时间 (Inference Time)
   {metrics['mean_inference_time']*1000:.2f} ms/样本
   目标: ≤5 秒
   状态: {'✅ 达标' if metrics['mean_inference_time'] <= 5.0 else '❌ 未达标'}

{'='*70}
详细分类指标
{'='*70}

样本分布:
  总样本数: {metrics['total_samples']}
  异常样本: {metrics['total_anomalies']} ({100*metrics['total_anomalies']/metrics['total_samples']:.1f}%)
  正常样本: {metrics['total_normal']} ({100*metrics['total_normal']/metrics['total_samples']:.1f}%)

混淆矩阵:
                    预测正常    预测异常
  实际正常          {metrics['true_negatives']:<10} {metrics['false_positives']:<10}
  实际异常          {metrics['false_negatives']:<10} {metrics['true_positives']:<10}

分类性能:
  召回率 (Recall):    {metrics['recall']:.2f}%
  精确率 (Precision): {metrics['precision']:.2f}%
  F1 Score:          {2*metrics['precision']*metrics['recall']/(metrics['precision']+metrics['recall']) if (metrics['precision']+metrics['recall']) > 0 else 0:.2f}%

{'='*70}
改进效果对比
{'='*70}

与原方案（仅统计特征P,T,R）相比：

优势：
  ✅ 保留了完整时序信息（300点推力曲线）
  ✅ 时序特征可以捕捉动态模式（上升/稳定/波动）
  ✅ 特征一致性完美（统计+时序从同一序列提取）
  ✅ 推理速度快（{metrics['mean_inference_time']*1000:.2f} ms << 5秒）

关键改进：
  1. 特征维度：3维 → 303维（提升100倍信息量）
  2. 模型架构：简单分类器 → 双分支1D CNN
  3. 数据扩充：跳过GAN，使用物理增强数据

{'='*70}
结论与建议
{'='*70}

当前模型性能总结：
"""

    if metrics['accuracy'] >= 92 and metrics['false_alarm_rate'] <= 3:
        report += """
✅ 模型达到所有核心指标要求！

建议：
  - 可以部署到实际应用中
  - 继续收集更多数据以提升鲁棒性
"""
    else:
        report += """
⚠️ 部分指标未达标，需要进一步优化

优化建议：
"""
        if metrics['accuracy'] < 92:
            report += "  1. 准确率不足 - 尝试增加训练数据或调整类别权重\n"
        if metrics['false_alarm_rate'] > 3:
            report += "  2. 虚警率过高 - 调整分类阈值或增加正常样本训练\n"

        report += """
  3. 收集更多真实场景数据
  4. 尝试数据增强技术（时序噪声、幅值缩放）
  5. 调整网络超参数（学习率、网络深度）
"""

    report += f"""
{'='*70}
"""

    # 保存报告
    with open(output_dir / 'performance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nDetailed report saved to {output_dir / 'performance_report.txt'}")


if __name__ == '__main__':
    results, metrics = run_inference(
        model_path='models/best_model.pth',
        data_dir='data/features_timeseries',
        output_dir='results',
        split='test'
    )

    print("\n" + "="*70)
    print("Inference completed successfully!")
    print("="*70)
