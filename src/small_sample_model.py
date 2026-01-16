"""
小样本鲁棒变轨识别模型
创新点：
1. 对比学习预训练 - 自监督特征学习
2. Mixup数据增强 - 特征空间插值
3. 多头自注意力 - 特征交互建模
4. MC Dropout不确定性估计 - 贝叶斯近似
5. 多任务学习 - 分类+回归联合优化
6. 标签平滑 - 防止过拟合
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ContrastiveLoss(nn.Module):
    """对比学习损失 (InfoNCE Loss) - 数值稳定版本"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        # L2归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 正样本相似度
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature

        # 负样本相似度 (同一batch内的其他样本)
        neg_sim_i = torch.mm(z_i, z_j.t()) / self.temperature
        neg_sim_j = torch.mm(z_j, z_i.t()) / self.temperature

        # 移除对角线(正样本)
        mask = torch.eye(batch_size, device=z_i.device).bool()
        neg_sim_i = neg_sim_i.masked_fill(mask, -1e4)
        neg_sim_j = neg_sim_j.masked_fill(mask, -1e4)

        # InfoNCE损失
        logits_i = torch.cat([pos_sim.unsqueeze(1), neg_sim_i], dim=1)
        logits_j = torch.cat([pos_sim.unsqueeze(1), neg_sim_j], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)

        loss_i = F.cross_entropy(logits_i, labels)
        loss_j = F.cross_entropy(logits_j, labels)

        return (loss_i + loss_j) / 2


class FeatureAugmentor(nn.Module):
    """特征增强模块 - Mixup + 噪声注入"""
    def __init__(self, noise_std=0.1, mixup_alpha=0.2):
        super().__init__()
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha

    def forward(self, x, training=True):
        if not training:
            return x, None, None

        # 高斯噪声注入
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise

        # Mixup增强
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            x_mixed = lam * x_noisy + (1 - lam) * x_noisy[index]
            return x_mixed, index, lam

        return x_noisy, None, None


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.size(0)
        residual = x

        # 多头注意力
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(out)

        # 残差连接 + LayerNorm
        out = self.layer_norm(out + residual)
        return out.squeeze(1)


class MCDropout(nn.Module):
    """MC Dropout - 训练和推理时都启用dropout实现贝叶斯近似"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)  # 始终启用


class FeatureEncoder(nn.Module):
    """特征编码器 - 用于对比学习预训练"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 投影头 (对比学习专用)
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, return_projection=False):
        z = self.encoder(x)
        if return_projection:
            return self.projector(z)
        return z


class SmallSampleRobustModel(nn.Module):
    """
    小样本鲁棒变轨识别模型

    创新点：
    1. 对比学习预训练特征编码器
    2. Mixup数据增强
    3. 多头自注意力特征交互
    4. MC Dropout不确定性估计
    5. 多任务学习(分类+回归)
    6. 标签平滑正则化
    """

    def __init__(self, input_dim=20, hidden_dim=128, num_heads=4, mc_samples=10):
        super().__init__()
        self.mc_samples = mc_samples

        # 特征增强
        self.augmentor = FeatureAugmentor(noise_std=0.05, mixup_alpha=0.2)

        # 特征编码器 (可预训练)
        self.encoder = FeatureEncoder(input_dim, hidden_dim, hidden_dim)

        # 多头自注意力
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout=0.1)

        # MC Dropout层
        self.mc_dropout = MCDropout(p=0.2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            MCDropout(0.2),
            nn.Linear(64, 2)
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            MCDropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, return_uncertainty=False):
        # 特征编码
        z = self.encoder(x)

        # 自注意力 (需要序列维度)
        z = z.unsqueeze(1)
        z = self.attention(z)

        # MC Dropout
        z = self.mc_dropout(z)

        # 多任务输出
        cls_logits = self.classifier(z)
        reg_output = self.regressor(z)

        if return_uncertainty:
            return cls_logits, reg_output, z
        return cls_logits, reg_output

    def predict_with_uncertainty(self, x, n_samples=None):
        """MC Dropout不确定性估计"""
        if n_samples is None:
            n_samples = self.mc_samples

        self.train()  # 启用dropout
        cls_preds = []
        reg_preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                cls_logits, reg_output = self(x)
                cls_preds.append(F.softmax(cls_logits, dim=1))
                reg_preds.append(reg_output)

        cls_preds = torch.stack(cls_preds)
        reg_preds = torch.stack(reg_preds)

        # 分类：均值和熵
        cls_mean = cls_preds.mean(dim=0)
        cls_entropy = -torch.sum(cls_mean * torch.log(cls_mean + 1e-8), dim=1)

        # 回归：均值和标准差
        reg_mean = reg_preds.mean(dim=0)
        reg_std = reg_preds.std(dim=0)

        return {
            'cls_prob': cls_mean,
            'cls_pred': cls_mean.argmax(dim=1),
            'cls_uncertainty': cls_entropy,
            'reg_pred': reg_mean.squeeze(),
            'reg_uncertainty': reg_std.squeeze()
        }


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, num_classes=2, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * confidence + (1 - one_hot) * smooth_value

        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_prob).sum(dim=1).mean()
        return loss


class SmallSampleTrainer:
    """小样本模型训练器"""

    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device

        # 损失函数
        self.cls_criterion = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
        self.reg_criterion = nn.SmoothL1Loss()
        self.contrastive_criterion = ContrastiveLoss(temperature=0.5)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def pretrain_contrastive(self, X_train, epochs=100, batch_size=32):
        """对比学习预训练"""
        print("对比学习预训练...")
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            total_loss = 0
            for (x,) in loader:
                x = x.to(self.device)

                # 两个增强视图
                x_i, _, _ = self.model.augmentor(x, training=True)
                x_j, _, _ = self.model.augmentor(x, training=True)

                # 对比学习
                z_i = self.model.encoder(x_i, return_projection=True)
                z_j = self.model.encoder(x_j, return_projection=True)

                loss = self.contrastive_criterion(z_i, z_j)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Pretrain Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    def train(self, X_train, y_train, t_train, X_val, y_val, t_val,
              epochs=300, batch_size=32, cls_weight=1.0, reg_weight=0.5):
        """多任务训练"""
        print("多任务训练...")

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
            torch.FloatTensor(t_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

        best_val_loss = float('inf')
        best_state = None
        patience = 0

        for epoch in range(epochs):
            self.model.train()
            total_cls_loss = 0
            total_reg_loss = 0

            for x, y, t in train_loader:
                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

                # Mixup增强
                x_aug, index, lam = self.model.augmentor(x, training=True)

                # 前向传播
                cls_logits, reg_output = self.model(x_aug)

                # 分类损失 (Mixup)
                if index is not None and lam is not None:
                    cls_loss = lam * self.cls_criterion(cls_logits, y) + \
                               (1 - lam) * self.cls_criterion(cls_logits, y[index])
                else:
                    cls_loss = self.cls_criterion(cls_logits, y)

                # 回归损失
                reg_loss = self.reg_criterion(reg_output.squeeze(), t)

                # 总损失
                loss = cls_weight * cls_loss + reg_weight * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()

            scheduler.step()

            # 验证
            val_loss, val_acc, val_mae = self.evaluate(X_val, y_val, t_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= 50:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: cls_loss={total_cls_loss/len(train_loader):.4f}, "
                      f"reg_loss={total_reg_loss/len(train_loader):.4f}, "
                      f"val_acc={val_acc:.4f}, val_mae={val_mae:.2f}s")

        self.model.load_state_dict(best_state)

    def evaluate(self, X, y, t):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            t_tensor = torch.FloatTensor(t).to(self.device)

            cls_logits, reg_output = self.model(x_tensor)

            cls_loss = self.cls_criterion(cls_logits, y_tensor)
            reg_loss = self.reg_criterion(reg_output.squeeze(), t_tensor)
            total_loss = cls_loss + 0.5 * reg_loss

            y_pred = cls_logits.argmax(dim=1).cpu().numpy()
            t_pred = reg_output.squeeze().cpu().numpy()

            acc = accuracy_score(y, y_pred)
            mae = mean_absolute_error(t, t_pred)

        return total_loss.item(), acc, mae

    def predict(self, X, with_uncertainty=True):
        """预测 (带不确定性估计)"""
        x_tensor = torch.FloatTensor(X).to(self.device)

        if with_uncertainty:
            return self.model.predict_with_uncertainty(x_tensor)
        else:
            self.model.eval()
            with torch.no_grad():
                cls_logits, reg_output = self.model(x_tensor)
                return {
                    'cls_pred': cls_logits.argmax(dim=1).cpu().numpy(),
                    'reg_pred': reg_output.squeeze().cpu().numpy()
                }


def extract_features_v2(df):
    """提取增强特征 (20维)"""
    P = df['P'].values
    T = df['T'].values
    R = df['R'].values

    features = np.column_stack([
        P, T, R,
        P * T, P / (R + 1e-6), T / (R + 1e-6), P * T / (R + 1e-6),
        P ** 2, T ** 2, R ** 2,
        np.sqrt(P + 1e-6), np.sqrt(T + 1e-6),
        np.log1p(P), np.log1p(T), np.log1p(R),
        T * R, P / (T + 1e-6), P * T * R,
        np.exp(-T), T ** 0.5 * P
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
