"""
改进的密度峰值聚类(DPC)算法
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional
from src.config import Config
from src.utils import timing_decorator

class ImprovedDPC:
    """改进的密度峰值聚类算法"""

    def __init__(self, config: Config = None):
        """
        初始化DPC聚类器

        Args:
            config: 配置对象
        """
        self.config = config or Config()
        dpc_config = self.config.DPC_CONFIG

        self.distance_metric = dpc_config['distance_metric']
        self.dc_percent = dpc_config['dc_percent']
        self.min_cluster_size = dpc_config['min_cluster_size']

        self.distance_matrix = None
        self.dc = None
        self.rho = None  # 局部密度
        self.delta = None  # 距离
        self.cluster_centers = None
        self.labels = None

    @timing_decorator
    def fit(self, X: np.ndarray, dc: Optional[float] = None) -> 'ImprovedDPC':
        """
        拟合DPC模型

        Args:
            X: 特征矩阵 (N, D)
            dc: 截断距离（可选）

        Returns:
            self
        """
        n_samples = X.shape[0]

        # 计算距离矩阵
        print(f"[DPC] 计算距离矩阵...")
        self.distance_matrix = squareform(pdist(X, metric=self.distance_metric))

        # 确定截断距离dc
        if dc is None:
            self.dc = self._compute_dc()
        else:
            self.dc = dc

        print(f"[DPC] 截断距离 dc = {self.dc:.4f}")

        # 计算局部密度
        print(f"[DPC] 计算局部密度...")
        self.rho = self._compute_local_density()

        # 计算delta（到更高密度点的最小距离）
        print(f"[DPC] 计算delta...")
        self.delta = self._compute_delta()

        return self

    def _compute_dc(self) -> float:
        """计算截断距离dc"""
        distances = self.distance_matrix[np.triu_indices_from(self.distance_matrix, k=1)]
        dc = np.percentile(distances, self.dc_percent * 100)
        return dc

    def _compute_local_density(self) -> np.ndarray:
        """
        计算局部密度 rho
        使用高斯核函数
        """
        n_samples = self.distance_matrix.shape[0]
        rho = np.zeros(n_samples)

        for i in range(n_samples):
            # 高斯核密度估计
            rho[i] = np.sum(np.exp(-(self.distance_matrix[i] / self.dc) ** 2)) - 1

        return rho

    def _compute_delta(self) -> np.ndarray:
        """
        计算delta: 到更高密度点的最小距离
        """
        n_samples = self.distance_matrix.shape[0]
        delta = np.zeros(n_samples)

        # 按密度降序排序
        sorted_indices = np.argsort(-self.rho)

        for i, idx in enumerate(sorted_indices):
            if i == 0:
                # 密度最高的点，delta为最大距离
                delta[idx] = np.max(self.distance_matrix[idx])
            else:
                # 到所有更高密度点的最小距离
                higher_density_indices = sorted_indices[:i]
                delta[idx] = np.min(self.distance_matrix[idx, higher_density_indices])

        return delta

    def select_cluster_centers(self,
                                 method: str = 'auto',
                                 n_clusters: Optional[int] = None) -> np.ndarray:
        """
        选择聚类中心

        Args:
            method: 选择方法 ('auto', 'manual', 'threshold')
            n_clusters: 聚类数量（method='auto'时使用）

        Returns:
            聚类中心索引数组
        """
        if method == 'auto':
            # 自动选择：gamma = rho * delta
            gamma = self.rho * self.delta

            if n_clusters is None:
                # 自动确定聚类数量
                # 使用gamma的拐点
                sorted_gamma = np.sort(gamma)[::-1]
                diff = np.diff(sorted_gamma)
                n_clusters = np.argmax(diff) + 1

                # 限制最小和最大聚类数
                n_clusters = max(2, min(n_clusters, 10))

            # 选择gamma最大的n_clusters个点作为聚类中心
            self.cluster_centers = np.argsort(-gamma)[:n_clusters]

        elif method == 'threshold':
            # 阈值方法
            rho_threshold = np.percentile(self.rho, 90)
            delta_threshold = np.percentile(self.delta, 90)

            self.cluster_centers = np.where(
                (self.rho > rho_threshold) & (self.delta > delta_threshold)
            )[0]

        else:
            raise ValueError(f"未知的方法: {method}")

        print(f"[DPC] 选择了 {len(self.cluster_centers)} 个聚类中心")
        return self.cluster_centers

    def assign_labels(self) -> np.ndarray:
        """
        为所有样本分配标签

        Returns:
            标签数组
        """
        if self.cluster_centers is None:
            raise ValueError("请先调用 select_cluster_centers()")

        n_samples = self.distance_matrix.shape[0]
        n_clusters = len(self.cluster_centers)
        self.labels = -np.ones(n_samples, dtype=int)

        # 为聚类中心分配标签
        for i, center_idx in enumerate(self.cluster_centers):
            self.labels[center_idx] = i

        # 按密度降序分配标签
        sorted_indices = np.argsort(-self.rho)

        for idx in sorted_indices:
            if self.labels[idx] == -1:
                # 找到距离最近的已分配标签的点
                distances_to_labeled = self.distance_matrix[idx]
                labeled_mask = self.labels != -1
                labeled_indices = np.where(labeled_mask)[0]

                if len(labeled_indices) > 0:
                    nearest_labeled = labeled_indices[np.argmin(distances_to_labeled[labeled_indices])]
                    self.labels[idx] = self.labels[nearest_labeled]

        # 处理孤立点（距离所有聚类中心都很远的点）
        outlier_label = n_clusters
        for i in range(n_samples):
            if self.labels[i] == -1:
                self.labels[i] = outlier_label

        print(f"[DPC] 聚类完成")
        print(f"  聚类数量: {n_clusters}")
        print(f"  标签分布: {np.bincount(self.labels)}")

        return self.labels

    def fit_predict(self,
                     X: np.ndarray,
                     n_clusters: Optional[int] = None) -> np.ndarray:
        """
        拟合并预测

        Args:
            X: 特征矩阵 (N, D)
            n_clusters: 聚类数量（可选）

        Returns:
            标签数组
        """
        self.fit(X)
        self.select_cluster_centers(method='auto', n_clusters=n_clusters)
        self.assign_labels()
        return self.labels

    def get_decision_graph_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取决策图数据

        Returns:
            (rho, delta) 元组
        """
        if self.rho is None or self.delta is None:
            raise ValueError("请先调用 fit() 方法")

        return self.rho, self.delta

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> dict:
        """
        评估聚类质量

        Args:
            X: 特征矩阵
            labels: 聚类标签

        Returns:
            评估指标字典
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # 过滤掉孤立点
        valid_mask = labels < len(self.cluster_centers)
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        metrics = {}

        try:
            metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
        except:
            metrics['silhouette_score'] = -1

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_valid, labels_valid)
        except:
            metrics['davies_bouldin_score'] = -1

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)
        except:
            metrics['calinski_harabasz_score'] = -1

        print(f"[DPC] 聚类评估:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        return metrics
