"""
召回率优化实验 - 提高变轨样本的召回率
策略：
1. 针对少数类(变轨样本)的GAN数据增强
2. 调整分类阈值
3. 使用更强的类别权重
4. SMOTE过采样
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_features(P, T, R):
    """提取20维增强特征"""
    features = np.column_stack([
        P, T, R,
        P * T, P / (R + 1e-6), T / (R + 1e-6), P * T / (R + 1e-6),
        P ** 2, T ** 2, R ** 2,
        np.sqrt(np.abs(P) + 1e-6), np.sqrt(np.abs(T) + 1e-6),
        np.log1p(np.abs(P)), np.log1p(np.abs(T)), np.log1p(np.abs(R)),
        T * R, P / (T + 1e-6), P * T * R,
        np.exp(-np.clip(T, 0, 10)), np.sqrt(np.abs(T) + 1e-6) * P
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_and_analyze_data():
    """加载数据并分析类别分布"""
    print("=" * 80)
    print("数据加载与类别分布分析")
    print("=" * 80)

    # 加载原始数据
    df_orig = pd.read_csv("data/feature_dataset.csv")
    df_orig = df_orig[df_orig['is_valid'] == 1].copy()

    # 加载扩充数据
    df_aug = pd.read_csv("data/augmented_dataset.csv")

    # 分析类别分布
    print("\n[原始数据集类别分布]")
    orig_counts = df_orig['is_anomalous'].value_counts()
    print(f"  正常样本 (0): {orig_counts.get(False, 0)}")
    print(f"  变轨样本 (1): {orig_counts.get(True, 0)}")
    print(f"  变轨样本比例: {orig_counts.get(True, 0) / len(df_orig) * 100:.2f}%")

    print("\n[扩充数据集类别分布]")
    aug_counts = df_aug['is_anomalous'].value_counts()
    print(f"  正常样本 (0): {aug_counts.get(0, 0)}")
    print(f"  变轨样本 (1): {aug_counts.get(1, 0)}")
    print(f"  变轨样本比例: {aug_counts.get(1, 0) / len(df_aug) * 100:.2f}%")

    return df_orig, df_aug


def smote_oversample(X, y, target_ratio=0.5):
    """SMOTE过采样 - 增加少数类样本"""
    from collections import Counter

    # 统计类别
    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    majority_class = max(counter, key=counter.get)

    n_minority = counter[minority_class]
    n_majority = counter[majority_class]

    # 计算需要生成的样本数
    target_minority = int(n_majority * target_ratio)
    n_synthetic = target_minority - n_minority

    if n_synthetic <= 0:
        return X, y

    print(f"\n[SMOTE过采样]")
    print(f"  少数类原始样本: {n_minority}")
    print(f"  需要生成样本: {n_synthetic}")

    # 获取少数类样本
    minority_idx = np.where(y == minority_class)[0]
    X_minority = X[minority_idx]

    # 生成合成样本
    synthetic_samples = []
    for _ in range(n_synthetic):
        # 随机选择两个少数类样本
        idx1, idx2 = np.random.choice(len(X_minority), 2, replace=False)
        # 线性插值
        alpha = np.random.random()
        synthetic = X_minority[idx1] + alpha * (X_minority[idx2] - X_minority[idx1])
        # 添加小噪声
        noise = np.random.normal(0, 0.05, synthetic.shape)
        synthetic = synthetic + noise
        synthetic_samples.append(synthetic)

    synthetic_samples = np.array(synthetic_samples)
    synthetic_labels = np.full(n_synthetic, minority_class)

    # 合并数据
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.concatenate([y, synthetic_labels])

    print(f"  过采样后少数类样本: {target_minority}")
    print(f"  过采样后总样本: {len(y_resampled)}")

    return X_resampled, y_resampled


def evaluate_with_recall(clf, X_test, y_test, threshold=0.5):
    """评估分类器，包含召回率指标"""
    # 获取预测概率
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 使用阈值进行预测
    y_pred = (y_proba >= threshold).astype(int)

    # 计算各项指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # 变轨类别的召回率
    recall_maneuver = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision_maneuver = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

    # 虚警率
    normal_mask = y_test == 0
    far = ((y_pred == 1) & normal_mask).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'recall_maneuver': recall_maneuver,
        'precision_maneuver': precision_maneuver,
        'far': far,
        'confusion_matrix': cm,
        'threshold': threshold
    }


def print_results(name, results):
    """打印评估结果"""
    print(f"\n[{name}]")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  宏平均召回率: {results['recall']:.4f}")
    print(f"  变轨样本召回率: {results['recall_maneuver']:.4f}")
    print(f"  变轨样本精确率: {results['precision_maneuver']:.4f}")
    print(f"  虚警率: {results['far']:.4f}")
    print(f"  阈值: {results['threshold']}")
    print(f"  混淆矩阵:")
    cm = results['confusion_matrix']
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")


def run_baseline_experiment(X_train, y_train, X_test, y_test):
    """基线实验 - 当前方法"""
    print("\n" + "=" * 80)
    print("实验1: 基线方法 (class_weight='balanced')")
    print("=" * 80)

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    results = evaluate_with_recall(clf, X_test, y_test, threshold=0.5)
    print_results("基线方法", results)

    return clf, results


def run_custom_weight_experiment(X_train, y_train, X_test, y_test):
    """自定义类别权重实验"""
    print("\n" + "=" * 80)
    print("实验2: 自定义类别权重 (增加变轨类权重)")
    print("=" * 80)

    # 计算类别比例
    n_normal = (y_train == 0).sum()
    n_maneuver = (y_train == 1).sum()
    ratio = n_normal / n_maneuver if n_maneuver > 0 else 10

    # 使用更强的权重偏向变轨类
    custom_weights = {0: 1.0, 1: ratio * 2}  # 变轨类权重翻倍
    print(f"  自定义权重: {custom_weights}")

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        class_weight=custom_weights,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    results = evaluate_with_recall(clf, X_test, y_test, threshold=0.5)
    print_results("自定义权重", results)

    return clf, results


def run_threshold_tuning_experiment(clf, X_test, y_test):
    """阈值调整实验 - 降低阈值提高召回率"""
    print("\n" + "=" * 80)
    print("实验3: 阈值调整 (降低阈值提高召回率)")
    print("=" * 80)

    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    best_result = None
    best_threshold = 0.5

    print(f"\n{'阈值':>8} {'准确率':>10} {'召回率':>10} {'虚警率':>10} {'F1':>10}")
    print("-" * 50)

    for thresh in thresholds:
        results = evaluate_with_recall(clf, X_test, y_test, threshold=thresh)
        print(f"{thresh:>8.2f} {results['accuracy']:>10.4f} "
              f"{results['recall_maneuver']:>10.4f} {results['far']:>10.4f} "
              f"{results['f1']:>10.4f}")

        # 选择召回率最高且虚警率<3%的阈值
        if results['far'] <= 0.03:
            if best_result is None or results['recall_maneuver'] > best_result['recall_maneuver']:
                best_result = results
                best_threshold = thresh

    print(f"\n最优阈值: {best_threshold} (虚警率≤3%约束下)")
    if best_result:
        print_results(f"阈值={best_threshold}", best_result)

    return best_threshold, best_result


def run_smote_experiment(X_train, y_train, X_test, y_test):
    """SMOTE过采样实验"""
    print("\n" + "=" * 80)
    print("实验4: SMOTE过采样 (增加变轨样本)")
    print("=" * 80)

    # SMOTE过采样
    X_resampled, y_resampled = smote_oversample(X_train, y_train, target_ratio=0.5)

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_resampled, y_resampled)

    results = evaluate_with_recall(clf, X_test, y_test, threshold=0.5)
    print_results("SMOTE过采样", results)

    return clf, results


def run_combined_optimization(X_train, y_train, X_test, y_test):
    """综合优化实验 - SMOTE + 自定义权重 + 阈值调整"""
    print("\n" + "=" * 80)
    print("实验5: 综合优化 (SMOTE + 自定义权重 + 阈值调整)")
    print("=" * 80)

    # SMOTE过采样
    X_resampled, y_resampled = smote_oversample(X_train, y_train, target_ratio=0.5)

    # 计算类别比例
    n_normal = (y_resampled == 0).sum()
    n_maneuver = (y_resampled == 1).sum()
    ratio = n_normal / n_maneuver if n_maneuver > 0 else 10

    # 自定义权重
    custom_weights = {0: 1.0, 1: ratio * 1.5}
    print(f"  自定义权重: {custom_weights}")

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        class_weight=custom_weights,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_resampled, y_resampled)

    # 阈值调整
    best_threshold, best_result = run_threshold_tuning_experiment(clf, X_test, y_test)

    return clf, best_threshold, best_result


def main():
    """主函数"""
    print("=" * 80)
    print("变轨样本召回率优化实验")
    print("=" * 80)

    # 加载数据
    df_orig, df_aug = load_and_analyze_data()

    # 提取特征
    X_aug = extract_features(
        df_aug['P'].values,
        df_aug['T'].values,
        df_aug['R'].values
    )
    y_aug = df_aug['is_anomalous'].values.astype(int)

    X_orig = extract_features(
        df_orig['P'].values,
        df_orig['T'].values,
        df_orig['R'].values
    )
    y_orig = df_orig['is_anomalous'].values.astype(int)

    # 标准化
    scaler = RobustScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)
    X_orig_scaled = scaler.transform(X_orig)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug_scaled, y_aug,
        test_size=0.2,
        random_state=42,
        stratify=y_aug
    )

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"测试集变轨样本: {(y_test == 1).sum()}")
    print(f"测试集正常样本: {(y_test == 0).sum()}")

    # 运行实验
    all_results = {}

    # 实验1: 基线
    clf_baseline, res_baseline = run_baseline_experiment(
        X_train, y_train, X_test, y_test
    )
    all_results['基线方法'] = res_baseline

    # 实验2: 自定义权重
    clf_weight, res_weight = run_custom_weight_experiment(
        X_train, y_train, X_test, y_test
    )
    all_results['自定义权重'] = res_weight

    # 实验3: 阈值调整
    best_thresh, res_thresh = run_threshold_tuning_experiment(
        clf_baseline, X_test, y_test
    )
    if res_thresh:
        all_results['阈值调整'] = res_thresh

    # 实验4: SMOTE
    clf_smote, res_smote = run_smote_experiment(
        X_train, y_train, X_test, y_test
    )
    all_results['SMOTE过采样'] = res_smote

    # 实验5: 综合优化
    clf_combined, thresh_combined, res_combined = run_combined_optimization(
        X_train, y_train, X_test, y_test
    )
    if res_combined:
        all_results['综合优化'] = res_combined

    # 汇总结果
    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(f"{'方法':<20} {'准确率':>10} {'召回率':>10} {'虚警率':>10} {'F1':>10}")
    print("-" * 60)

    for name, res in all_results.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} "
              f"{res['recall_maneuver']:>10.4f} "
              f"{res['far']:>10.4f} {res['f1']:>10.4f}")

    # 保存最优模型
    best_method = max(
        all_results.items(),
        key=lambda x: x[1]['recall_maneuver'] if x[1]['far'] <= 0.03 else 0
    )
    print(f"\n最优方法: {best_method[0]}")
    print(f"  召回率: {best_method[1]['recall_maneuver']:.4f}")
    print(f"  虚警率: {best_method[1]['far']:.4f}")

    return all_results


if __name__ == '__main__':
    main()
