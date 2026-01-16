"""
第二阶段：Δv 回归模型
基于 P/T/R 三维特征估计速度增量
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class DeltaVMLP(nn.Module):
    """
    MLP Δv 回归模型
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 3,  # RTN三分量
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 输出头: RTN三分量
        self.output_head = nn.Linear(prev_dim, output_dim)

        # 不确定性估计头 (可选)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(prev_dim, output_dim),
            nn.Softplus()  # 确保正数
        )

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入特征 [batch, input_dim]
            return_uncertainty: 是否返回不确定性估计

        Returns:
            delta_v: 预测Δv [batch, 3]
            uncertainty: 不确定性 [batch, 3] (可选)
        """
        features = self.backbone(x)
        delta_v = self.output_head(features)

        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
            return delta_v, uncertainty

        return delta_v, None


class DeltaVResNet(nn.Module):
    """
    带残差连接的MLP
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        output_dim: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 残差块
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.output_head = nn.Linear(hidden_dim, output_dim)

    def _make_block(self, dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)

        return self.output_head(x), None


class GradientBoostingDeltaV:
    """
    基于梯度提升的Δv回归器
    支持XGBoost和LightGBM
    """

    def __init__(
        self,
        backend: str = 'xgboost',  # 'xgboost', 'lightgbm', 'sklearn'
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        self.backend = backend
        self.scaler = StandardScaler()
        self.models = []  # RTN三分量各一个模型

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            **kwargs
        }

        # 根据后端创建模型
        for _ in range(3):  # R, T, N 三分量
            if backend == 'xgboost' and HAS_XGBOOST:
                model = xgb.XGBRegressor(**params)
            elif backend == 'lightgbm' and HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(**params)
            else:
                model = GradientBoostingRegressor(**params)
            self.models.append(model)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        训练模型

        Args:
            X: 特征 [N, n_features]
            y: 标签 [N, 3] (RTN分量)
            eval_set: 验证集 (X_val, y_val)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 训练每个分量的模型
        for i, model in enumerate(self.models):
            if eval_set is not None:
                X_val_scaled = self.scaler.transform(eval_set[0])
                if self.backend in ['xgboost', 'lightgbm']:
                    model.fit(
                        X_scaled, y[:, i],
                        eval_set=[(X_val_scaled, eval_set[1][:, i])],
                        verbose=False
                    )
                else:
                    model.fit(X_scaled, y[:, i])
            else:
                model.fit(X_scaled, y[:, i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测Δv"""
        X_scaled = self.scaler.transform(X)

        predictions = np.zeros((len(X), 3))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_scaled)

        return predictions

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """获取特征重要性"""
        importances = {}
        for i, (comp, model) in enumerate(zip(['R', 'T', 'N'], self.models)):
            if hasattr(model, 'feature_importances_'):
                importances[comp] = model.feature_importances_
        return importances

    def save(self, path: str):
        """保存模型"""
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'backend': self.backend
        }, path)

    def load(self, path: str):
        """加载模型"""
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.backend = data['backend']


class DeltaVEnsemble:
    """
    集成模型：结合MLP和梯度提升
    """

    def __init__(
        self,
        mlp_weight: float = 0.4,
        gbm_weight: float = 0.6,
        device: str = 'cuda'
    ):
        self.mlp_weight = mlp_weight
        self.gbm_weight = gbm_weight
        self.device = device

        self.mlp_model: Optional[DeltaVMLP] = None
        self.gbm_model: Optional[GradientBoostingDeltaV] = None
        self.scaler = StandardScaler()

    def build_models(
        self,
        input_dim: int,
        mlp_config: Optional[Dict] = None,
        gbm_config: Optional[Dict] = None
    ):
        """构建模型"""
        # MLP
        mlp_config = mlp_config or {}
        self.mlp_model = DeltaVMLP(input_dim=input_dim, **mlp_config)
        self.mlp_model = self.mlp_model.to(self.device)

        # GBM
        gbm_config = gbm_config or {}
        self.gbm_model = GradientBoostingDeltaV(**gbm_config)

    def fit_mlp(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """训练MLP"""
        optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.mlp_model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred, _ = self.mlp_model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证
            self.mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred, _ = self.mlp_model(X)
                    val_loss += criterion(pred, y).item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def fit_gbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """训练GBM"""
        eval_set = (X_val, y_val) if X_val is not None else None
        self.gbm_model.fit(X_train, y_train, eval_set=eval_set)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测

        Args:
            X: 特征 [N, n_features]

        Returns:
            预测Δv [N, 3]
        """
        # MLP预测
        self.mlp_model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        mlp_pred, _ = self.mlp_model(X_tensor)
        mlp_pred = mlp_pred.cpu().numpy()

        # GBM预测
        gbm_pred = self.gbm_model.predict(X)

        # 加权平均
        ensemble_pred = self.mlp_weight * mlp_pred + self.gbm_weight * gbm_pred

        return ensemble_pred

    def save(self, path: str):
        """保存集成模型"""
        torch.save({
            'mlp_state_dict': self.mlp_model.state_dict(),
            'mlp_weight': self.mlp_weight,
            'gbm_weight': self.gbm_weight
        }, f"{path}_mlp.pt")

        self.gbm_model.save(f"{path}_gbm.pkl")

    def load(self, path: str):
        """加载集成模型"""
        mlp_checkpoint = torch.load(f"{path}_mlp.pt", map_location=self.device)
        self.mlp_model.load_state_dict(mlp_checkpoint['mlp_state_dict'])
        self.mlp_weight = mlp_checkpoint['mlp_weight']
        self.gbm_weight = mlp_checkpoint['gbm_weight']

        self.gbm_model.load(f"{path}_gbm.pkl")


class DeltaVRegressor:
    """
    Δv回归器封装类
    """

    def __init__(
        self,
        model_type: str = 'ensemble',  # 'mlp', 'gbm', 'ensemble'
        device: str = 'cuda',
        **kwargs
    ):
        self.model_type = model_type
        self.device = device

        if model_type == 'mlp':
            input_dim = kwargs.get('input_dim', 20)
            self.model = DeltaVMLP(input_dim=input_dim)
            self.model = self.model.to(device)
        elif model_type == 'gbm':
            self.model = GradientBoostingDeltaV(**kwargs)
        elif model_type == 'ensemble':
            self.model = DeltaVEnsemble(device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测Δv"""
        if self.model_type == 'mlp':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                pred, _ = self.model(X_tensor)
                return pred.cpu().numpy()
        else:
            return self.model.predict(X)

    def save(self, path: str):
        """保存模型"""
        if self.model_type == 'mlp':
            torch.save(self.model.state_dict(), path)
        else:
            self.model.save(path)

    def load(self, path: str):
        """加载模型"""
        if self.model_type == 'mlp':
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
        else:
            self.model.load(path)
