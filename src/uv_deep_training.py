"""
UV 识别系统 - 基于GAN增强数据的深度训练
===========================================

使用GAN增强后的数据进行深度训练，增加epoch数，生成论文级别的结果

作者: Claude Code
日期: 2026-01-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DeepManeuverClassifier(nn.Module):
    """深度变轨分类器"""

    def __init__(self, input_dim=16, hidden_dims=[128, 256, 128], dropout=0.3):
        super(DeepManeuverClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepThrustRegressor(nn.Module):
    """深度推力回归器"""

    def __init__(self, input_dim=16, hidden_dims=[128, 256, 256, 128], dropout=0.3):
        super(DeepThrustRegressor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepModelTrainer:
    """深度模型训练器"""

    def __init__(self, model, device='cuda', lr=0.001):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        print(f"使用设备: {self.device}")

    def train_classifier(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=64,
        early_stopping_patience=20
    ):
        """训练分类器"""
        print(f"\n开始训练分类器...")
        print(f"  训练样本: {len(X_train)}")
        print(f"  验证样本: {len(X_val)}")
        print(f"  训练轮数: {epochs}")

        # 准备数据
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_preds.extend((outputs > 0.5).cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)

            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_preds = (val_outputs > 0.5).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_acc)
            self.val_metrics.append(val_acc)

            # 学习率调整
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        print(f"\n✓ 训练完成！最佳验证损失: {best_val_loss:.4f}")

    def train_regressor(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=64,
        early_stopping_patience=20
    ):
        """训练回归器"""
        print(f"\n开始训练回归器...")
        print(f"  训练样本: {len(X_train)}")
        print(f"  验证样本: {len(X_val)}")
        print(f"  训练轮数: {epochs}")

        # 准备数据
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                # 计算R²
                val_preds = val_outputs.cpu().numpy().flatten()
                val_r2 = r2_score(y_val, val_preds)

            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_r2)

            # 学习率调整
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        print(f"\n✓ 训练完成！最佳验证损失: {best_val_loss:.4f}")

    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()

    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, filepath)
        print(f"模型已保存: {filepath}")


def train_deep_models_with_gan_data():
    """使用GAN增强数据训练深度模型"""
    print("=" * 70)
    print("基于GAN增强数据的深度模型训练")
    print("=" * 70)

    # 1. 加载GAN增强后的数据
    print("\n[1/6] 加载GAN增强后的数据...")
    df_train = pd.read_csv('data/uv_features_train_gan_augmented.csv')
    df_test = pd.read_csv('data/uv_features_test.csv')

    print(f"  训练集样本: {len(df_train)} (包含GAN生成)")
    print(f"  测试集样本: {len(df_test)}")

    # 选择特征
    feature_cols = [
        'background_mean', 'background_std', 'threshold',
        'peak_intensity', 'mean_intensity', 'total_energy',
        'num_pulses',
        'mean_pulse_duration', 'max_pulse_duration',
        'mean_pulse_peak', 'max_pulse_peak',
        'mean_pulse_energy',
        'max_rise_rate', 'mean_rise_rate',
        'mean_pulse_interval', 'min_pulse_interval'
    ]

    X_train_full = df_train[feature_cols].fillna(0).values
    X_test = df_test[feature_cols].fillna(0).values

    # 标准化
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # 2. 准备标签
    print("\n[2/6] 准备标签...")

    # 变轨标签
    y_maneuver_train = (df_train['num_pulses'] > 0).astype(int).values
    y_maneuver_test = (df_test['num_pulses'] > 0).astype(int).values

    # 推力标签
    if 'true_thrust' in df_train.columns:
        y_thrust_train = df_train['true_thrust'].fillna(0).values
        y_thrust_test = df_test['true_thrust'].fillna(0).values
    else:
        y_thrust_train = df_train['peak_intensity'].values * 0.01
        y_thrust_test = df_test['peak_intensity'].values * 0.01

    print(f"  变轨样本: {y_maneuver_train.sum()}/{len(y_maneuver_train)}")
    print(f"  推力范围: [{y_thrust_train.min():.4f}, {y_thrust_train.max():.4f}]")

    # 3. 训练变轨分类器
    print("\n[3/6] 训练深度变轨分类器...")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_maneuver_train,
        test_size=0.2, random_state=42, stratify=y_maneuver_train
    )

    # 创建模型
    maneuver_model = DeepManeuverClassifier(
        input_dim=len(feature_cols),
        hidden_dims=[128, 256, 256, 128],
        dropout=0.3
    )

    # 训练
    maneuver_trainer = DeepModelTrainer(maneuver_model, lr=0.001)
    maneuver_trainer.train_classifier(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=64,
        early_stopping_patience=20
    )

    # 测试
    print("\n测试集评估:")
    y_pred_proba = maneuver_trainer.predict(X_test_scaled).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_maneuver_test, y_pred)
    prec = precision_score(y_maneuver_test, y_pred, zero_division=0)
    rec = recall_score(y_maneuver_test, y_pred, zero_division=0)
    f1 = f1_score(y_maneuver_test, y_pred, zero_division=0)

    print(f"  准确率: {acc:.4f}")
    print(f"  精确率: {prec:.4f}")
    print(f"  召回率: {rec:.4f}")
    print(f"  F1分数: {f1:.4f}")

    # 保存模型
    model_dir = Path('models/deep_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    maneuver_trainer.save(model_dir / 'deep_maneuver_classifier.pth')

    # 4. 训练推力回归器
    print("\n[4/6] 训练深度推力回归器...")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_thrust_train,
        test_size=0.2, random_state=42
    )

    # 创建模型
    thrust_model = DeepThrustRegressor(
        input_dim=len(feature_cols),
        hidden_dims=[128, 256, 256, 128],
        dropout=0.3
    )

    # 训练
    thrust_trainer = DeepModelTrainer(thrust_model, lr=0.001)
    thrust_trainer.train_regressor(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=64,
        early_stopping_patience=20
    )

    # 测试
    print("\n测试集评估:")
    y_pred = thrust_trainer.predict(X_test_scaled).flatten()

    mae = mean_absolute_error(y_thrust_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_thrust_test, y_pred))
    r2 = r2_score(y_thrust_test, y_pred)

    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

    # 保存模型
    thrust_trainer.save(model_dir / 'deep_thrust_regressor.pth')

    # 5. 生成训练曲线
    print("\n[5/6] 生成训练曲线...")
    plot_training_curves(maneuver_trainer, thrust_trainer, model_dir)

    # 6. 生成混淆矩阵
    print("\n[6/6] 生成混淆矩阵...")
    # 使用分类器的预测结果
    y_pred_maneuver = (y_pred_proba > 0.5).astype(int)
    plot_confusion_matrix(y_maneuver_test, y_pred_maneuver, model_dir)

    print("\n" + "=" * 70)
    print("✓ 深度模型训练完成！")
    print("=" * 70)

    return {
        'maneuver_classifier': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        },
        'thrust_regressor': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    }


def plot_training_curves(maneuver_trainer, thrust_trainer, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 变轨分类器 - 损失
    ax1 = axes[0, 0]
    ax1.plot(maneuver_trainer.train_losses, label='Train Loss', linewidth=2)
    ax1.plot(maneuver_trainer.val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Maneuver Classifier - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 变轨分类器 - 准确率
    ax2 = axes[0, 1]
    ax2.plot(maneuver_trainer.train_metrics, label='Train Acc', linewidth=2)
    ax2.plot(maneuver_trainer.val_metrics, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Maneuver Classifier - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 推力回归器 - 损失
    ax3 = axes[1, 0]
    ax3.plot(thrust_trainer.train_losses, label='Train Loss', linewidth=2)
    ax3.plot(thrust_trainer.val_losses, label='Val Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.set_title('Thrust Regressor - Loss', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 推力回归器 - R²
    ax4 = axes[1, 1]
    ax4.plot(thrust_trainer.val_metrics, label='Val R²', linewidth=2, color='green')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('R²', fontsize=12)
    ax4.set_title('Thrust Regressor - R²', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'deep_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Maneuver', 'Maneuver'],
                yticklabels=['No Maneuver', 'Maneuver'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Maneuver Classification', fontsize=14, fontweight='bold')

    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  混淆矩阵已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    results = train_deep_models_with_gan_data()

    print("\n" + "=" * 70)
    print("最终结果总结")
    print("=" * 70)
    print("\n变轨分类器:")
    for key, value in results['maneuver_classifier'].items():
        print(f"  {key}: {value:.4f}")

    print("\n推力回归器:")
    for key, value in results['thrust_regressor'].items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
