"""
变轨检测与点火时刻估计模型
- 变轨检测: RandomForest, DNN
- 点火时刻估计: LSTM, Transformer
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple


# ==================== 变轨检测模型 ====================

class DNNClassifier(nn.Module):
    """DNN变轨检测模型"""

    def __init__(self, input_dim: int = 3, hidden_dims: list = [64, 128, 64], num_classes: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================== 点火时刻估计模型 ====================

class LSTMRegressor(nn.Module):
    """LSTM点火时刻估计模型"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features) for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TransformerRegressor(nn.Module):
    """Transformer点火时刻估计模型"""

    def __init__(self, input_dim: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out


# ==================== 组合模型包装器 ====================

class ModelCombination:
    """模型组合包装器"""

    def __init__(self, classifier_type: str, regressor_type: str, device: str = 'cpu'):
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type
        self.device = device
        self.classifier = None
        self.regressor = None
        self.scaler = None

    def build_classifier(self, input_dim: int = 3):
        if self.classifier_type == 'RF':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.classifier_type == 'DNN':
            self.classifier = DNNClassifier(input_dim=input_dim).to(self.device)

    def build_regressor(self, input_dim: int = 3):
        if self.regressor_type == 'LSTM':
            self.regressor = LSTMRegressor(input_dim=input_dim).to(self.device)
        elif self.regressor_type == 'Transformer':
            self.regressor = TransformerRegressor(input_dim=input_dim).to(self.device)

    def train_classifier(self, X_train, y_train, X_val, y_val, epochs: int = 100, lr: float = 1e-3):
        if self.classifier_type == 'RF':
            self.classifier.fit(X_train, y_train)
        else:
            self._train_nn(self.classifier, X_train, y_train, X_val, y_val,
                          nn.CrossEntropyLoss(), epochs, lr, is_classifier=True)

    def train_regressor(self, X_train, y_train, X_val, y_val, epochs: int = 100, lr: float = 1e-3):
        self._train_nn(self.regressor, X_train, y_train, X_val, y_val,
                      nn.MSELoss(), epochs, lr, is_classifier=False)

    def _train_nn(self, model, X_train, y_train, X_val, y_val, criterion, epochs, lr, is_classifier):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device) if is_classifier else torch.FloatTensor(y_train).to(self.device)

        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_t)
            if not is_classifier:
                out = out.squeeze()
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                X_v = torch.FloatTensor(X_val).to(self.device)
                y_v = torch.LongTensor(y_val).to(self.device) if is_classifier else torch.FloatTensor(y_val).to(self.device)
                val_out = model(X_v)
                if not is_classifier:
                    val_out = val_out.squeeze()
                val_loss = criterion(val_out, y_v).item()

            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break

    def predict_class(self, X):
        if self.classifier_type == 'RF':
            return self.classifier.predict(X)
        else:
            self.classifier.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                out = self.classifier(X_t)
                return out.argmax(dim=1).cpu().numpy()

    def predict_regression(self, X):
        self.regressor.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            out = self.regressor(X_t)
            return out.squeeze().cpu().numpy()
