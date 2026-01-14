# 小推力变轨检测系统 - 技术流程文档

---

## 步骤1：数据结构与特征提取清单

### 1️⃣ 统一数据表结构

| 字段名 | 类型 | 含义 | 数据源 |
|--------|------|------|--------|
| `timestamp` | float | Unix时间戳(s) | 通用 |
| `target_id` | str | 目标编号 | 通用 |
| `orbit_type` | enum | 轨道类型: `GEO` / `LEO` | 通用 |
| `res_RA` | float | 赤经残差(arcsec) | GEO光学 |
| `res_DEC` | float | 赤纬残差(arcsec) | GEO光学 |
| `res_AZ` | float | 方位角残差(arcsec) | LEO雷达 |
| `res_EL` | float | 俯仰角残差(arcsec) | LEO雷达 |
| `res_Ra` | float | 斜距残差(m) | LEO雷达 |
| `res_RR` | float | 距离变化率残差(m/s) | LEO雷达 |
| `rad_226` | float | 226nm辐射强度(W/sr) | 模拟辐射 |
| `rad_306` | float | 306nm辐射强度(W/sr) | 模拟辐射 |
| `label_maneuver` | int | 变轨标签: 0/1 | 标注 |
| `label_ignition_t` | float | 点火时刻(s), NaN表示无 | 标注 |
| `label_thrust` | float | 推力大小(mN), NaN表示无 | 标注 |

---

### 2️⃣ 10类残差特征定义

设滑动窗口内残差序列为 $\{r_i\}_{i=1}^{N}$，窗口长度 $N$。

| 编号 | 特征名 | 数学定义 | 类型 |
|------|--------|----------|------|
| F1 | 均值 | $\mu = \frac{1}{N}\sum_{i=1}^{N} r_i$ | 时域 |
| F2 | 标准差 | $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(r_i - \mu)^2}$ | 时域 |
| F3 | 峰峰值 | $PP = \max(r) - \min(r)$ | 时域 |
| F4 | 偏度 | $S = \frac{1}{N}\sum\left(\frac{r_i-\mu}{\sigma}\right)^3$ | 时域 |
| F5 | 峰度 | $K = \frac{1}{N}\sum\left(\frac{r_i-\mu}{\sigma}\right)^4 - 3$ | 时域 |
| F6 | 一阶差分均值 | $\Delta_1 = \frac{1}{N-1}\sum_{i=1}^{N-1}(r_{i+1}-r_i)$ | 时域 |
| F7 | 一阶差分标准差 | $\sigma_{\Delta_1} = \text{std}(r_{i+1}-r_i)$ | 时域 |
| F8 | 二阶差分均值 | $\Delta_2 = \frac{1}{N-2}\sum_{i=1}^{N-2}(r_{i+2}-2r_{i+1}+r_i)$ | 时域 |
| F9 | 线性趋势斜率 | $k = \text{slope of } \text{OLS}(t, r)$ | 时域 |
| F10 | 主频能量占比 | $E_{dom} = \frac{|FFT(r)|^2_{max}}{\sum|FFT(r)|^2}$ | 频域 |

---

### 3️⃣ 三维辐射特征 P / T / R 定义

设 $I_{226}$, $I_{306}$ 为两波段辐射强度。

| 特征 | 名称 | 数学定义 | 物理含义 |
|------|------|----------|----------|
| **P** | 辐射功率 | $P = \sqrt{I_{226}^2 + I_{306}^2}$ | 总辐射强度，反映推力大小 |
| **T** | 等效温度 | $T = \frac{hc}{\lambda_2 k_B} \cdot \left[\ln\left(\frac{I_{226}}{I_{306}} \cdot \left(\frac{\lambda_{226}}{\lambda_{306}}\right)^5\right)\right]^{-1}$ | 双波段比值反演黑体温度 |
| **R** | 波段比值 | $R = \frac{I_{226}}{I_{306}}$ | 推进剂类型/燃烧状态指示 |

简化工程公式（假设普朗克近似）：
- $T \approx \frac{C}{\ln(R \cdot \alpha)}$，其中 $C \approx 2898 \mu m \cdot K$，$\alpha$ 为波长修正系数

---

### 4️⃣ 特征用途分配表

| 特征 | 变轨检测 | 点火时刻定位 | 推力大小回归 |
|------|:--------:|:------------:|:------------:|
| **残差特征** | | | |
| F1 均值 | ✓ | | |
| F2 标准差 | ✓ | | ✓ |
| F3 峰峰值 | ✓ | | ✓ |
| F4 偏度 | ✓ | ✓ | |
| F5 峰度 | ✓ | ✓ | |
| F6 一阶差分均值 | ✓ | ✓ | |
| F7 一阶差分标准差 | ✓ | ✓ | ✓ |
| F8 二阶差分均值 | | ✓ | |
| F9 线性趋势斜率 | ✓ | | ✓ |
| F10 主频能量占比 | ✓ | | |
| **辐射特征** | | | |
| P 辐射功率 | ✓ | | ✓ |
| T 等效温度 | | | ✓ |
| R 波段比值 | ✓ | ✓ | ✓ |

---

### 特征用途说明

| 任务 | 核心特征 | 原理 |
|------|----------|------|
| **变轨检测** | F2, F3, F6, F7, P, R | 残差突变 + 辐射出现 → 二分类 |
| **点火时刻定位** | F4, F5, F6, F8, R | 高阶统计量对变点敏感 → 时刻回归 |
| **推力大小回归** | F2, F3, F7, F9, P, T, R | 残差幅度 + 辐射强度/温度 → 连续值回归 |

---

## 步骤2：226nm/306nm 尾焰辐射 Monte-Carlo 模拟方法

### 1️⃣ P、T、R 与 Δv 的物理关系

#### 基本假设
- 小推力发动机推力 $F$ 范围：1~100 mN
- 比冲 $I_{sp}$：1000~3000 s（电推进典型值）
- 速度增量：$\Delta v = \frac{F \cdot \Delta t}{m}$

#### 辐射强度与推力关系

| 特征 | 与 Δv 关系 | 数学模型 | 参数说明 |
|------|-----------|----------|----------|
| **P (辐射功率)** | 正相关 | $P = \eta \cdot F^{\beta} + P_0$ | $\eta$: 辐射效率系数, $\beta \approx 0.8$~$1.2$, $P_0$: 背景辐射 |
| **T (等效温度)** | 弱相关 | $T = T_{base} + \gamma \cdot \ln(1 + F/F_{ref})$ | $T_{base} \approx 1500K$, $\gamma \approx 200$~$400$, $F_{ref}=10mN$ |
| **R (波段比值)** | 推进剂相关 | $R = R_0 \cdot (1 + \delta \cdot F/F_{max})$ | $R_0$: 基准比值(推进剂决定), $\delta \approx 0.05$~$0.15$ |

#### 双波段辐射强度计算

$$I_{226} = P \cdot \frac{B_{226}(T)}{B_{226}(T) + B_{306}(T)}$$

$$I_{306} = P \cdot \frac{B_{306}(T)}{B_{226}(T) + B_{306}(T)}$$

其中 $B_{\lambda}(T)$ 为普朗克函数：
$$B_{\lambda}(T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc/(\lambda k_B T)} - 1}$$

---

### 2️⃣ 变轨 / 非变轨样本生成逻辑

#### 生成流程

```
┌─────────────────────────────────────────────────────────┐
│                    Monte-Carlo 采样                      │
├─────────────────────────────────────────────────────────┤
│  1. 采样 label ∈ {0, 1}，按先验概率 p_maneuver          │
│  2. if label == 1 (变轨):                               │
│       - 采样推力 F ~ U(F_min, F_max)                    │
│       - 采样点火时刻 t_ign ~ U(t_start, t_end)          │
│       - 采样持续时间 Δt ~ U(Δt_min, Δt_max)             │
│       - 计算 P, T, R → I_226, I_306                     │
│  3. if label == 0 (非变轨):                             │
│       - I_226 = I_306 = 0 (或背景噪声)                  │
│  4. 叠加噪声                                            │
└─────────────────────────────────────────────────────────┘
```

#### 参数采样分布

| 参数 | 分布 | 范围 | 说明 |
|------|------|------|------|
| `label` | Bernoulli | p=0.3 | 变轨先验概率 |
| `F` (推力) | Uniform | [1, 100] mN | 小推力范围 |
| `t_ign` (点火时刻) | Uniform | [0.2T, 0.8T] | T为观测窗口长度 |
| `Δt` (持续时间) | Uniform | [60, 3600] s | 1分钟~1小时 |
| `T_base` | Normal | μ=1800K, σ=200K | 燃烧温度基准 |
| `R_0` | Uniform | [0.8, 1.5] | 推进剂类型差异 |

#### 时序辐射信号生成

变轨样本的辐射时序：
$$I_{\lambda}(t) = \begin{cases}
0 & t < t_{ign} \\
I_{\lambda,peak} \cdot \phi(t - t_{ign}) & t_{ign} \leq t \leq t_{ign} + \Delta t \\
0 & t > t_{ign} + \Delta t
\end{cases}$$

其中 $\phi(t)$ 为点火包络函数：
$$\phi(t) = (1 - e^{-t/\tau_{rise}}) \cdot e^{-t/\tau_{decay}}$$

典型参数：$\tau_{rise} \approx 5s$，$\tau_{decay} \approx 1000s$

---

### 3️⃣ 噪声叠加方式 (SNR = 3~10 dB)

#### SNR 定义

$$SNR_{dB} = 10 \cdot \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)$$

反解噪声功率：
$$\sigma_{noise} = \frac{I_{signal}}{\sqrt{10^{SNR_{dB}/10}}}$$

#### 噪声模型

| 噪声类型 | 数学模型 | 物理来源 |
|----------|----------|----------|
| 高斯白噪声 | $n_g \sim \mathcal{N}(0, \sigma_g^2)$ | 探测器热噪声 |
| 泊松噪声 | $n_p \sim \sqrt{I} \cdot \mathcal{N}(0,1)$ | 光子计数统计涨落 |
| 背景辐射 | $n_b \sim \mathcal{N}(\mu_b, \sigma_b^2)$ | 地球/大气背景 |

#### 综合噪声叠加公式

$$I_{\lambda,obs}(t) = I_{\lambda,true}(t) + n_g + n_p + n_b$$

其中：
- $\sigma_g = \frac{\bar{I}_{signal}}{10^{SNR_{dB}/20}}$ （主噪声项）
- $n_p = \sqrt{I_{\lambda,true}(t)} \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0, 0.1^2)$
- $n_b \sim \mathcal{N}(0.01 \cdot \bar{I}_{signal}, (0.005 \cdot \bar{I}_{signal})^2)$

#### SNR 采样策略

```python
# 伪代码
def sample_snr():
    """SNR在3-10dB范围内均匀采样"""
    return np.random.uniform(3, 10)

def add_noise(I_true, snr_db):
    """叠加综合噪声"""
    I_mean = np.mean(I_true[I_true > 0]) if np.any(I_true > 0) else 1e-10

    # 主高斯噪声
    sigma_g = I_mean / (10 ** (snr_db / 20))
    n_g = np.random.normal(0, sigma_g, len(I_true))

    # 泊松噪声
    n_p = np.sqrt(np.abs(I_true)) * np.random.normal(0, 0.1, len(I_true))

    # 背景噪声
    n_b = np.random.normal(0.01 * I_mean, 0.005 * I_mean, len(I_true))

    I_obs = I_true + n_g + n_p + n_b
    return np.maximum(I_obs, 0)  # 辐射强度非负
```

---

### 4️⃣ 模拟验证指标

| 指标 | 公式 | 目标值 |
|------|------|--------|
| 信号可检测率 | $P(I_{obs} > 3\sigma_{noise} | label=1)$ | > 90% @ SNR=10dB |
| 虚警率 | $P(I_{obs} > 3\sigma_{noise} | label=0)$ | < 5% |
| P-F 相关性 | $\rho(P_{sim}, F_{true})$ | > 0.85 |
| T 反演误差 | $RMSE(T_{inv}, T_{true})$ | < 100K |

---

## 步骤3：特征提取 + GAN 小样本扩充

### 1️⃣ GAN 输入特征向量定义（13维）

| 维度 | 特征名 | 来源 | 归一化方式 |
|------|--------|------|------------|
| x1 | F2_std | 残差标准差 | Z-score |
| x2 | F3_pp | 残差峰峰值 | Min-Max |
| x3 | F4_skew | 残差偏度 | Z-score |
| x4 | F5_kurt | 残差峰度 | Z-score |
| x5 | F6_diff1_mean | 一阶差分均值 | Z-score |
| x6 | F7_diff1_std | 一阶差分标准差 | Min-Max |
| x7 | F8_diff2_mean | 二阶差分均值 | Z-score |
| x8 | F9_slope | 线性趋势斜率 | Z-score |
| x9 | F10_freq_ratio | 主频能量占比 | Min-Max [0,1] |
| x10 | P | 辐射功率 | Log + Min-Max |
| x11 | T | 等效温度 | Min-Max |
| x12 | R | 波段比值 | Min-Max |
| x13 | label | 变轨标签 (条件) | One-hot / 直接 |

#### 特征向量表示

$$\mathbf{x} = [x_1, x_2, ..., x_{12}]^T \in \mathbb{R}^{12}$$
$$c = label \in \{0, 1\}$$

---

### 2️⃣ Conditional GAN (cGAN) 架构设计

#### 网络结构

```
┌─────────────────────────────────────────────────────────────┐
│                    Conditional GAN                          │
├─────────────────────────────────────────────────────────────┤
│  Generator G(z, c):                                         │
│    Input:  z ∈ R^32 (噪声) + c ∈ {0,1} (条件标签)           │
│    Output: x_fake ∈ R^12 (生成特征)                         │
│                                                             │
│  Discriminator D(x, c):                                     │
│    Input:  x ∈ R^12 (特征) + c ∈ {0,1} (条件标签)           │
│    Output: p ∈ [0,1] (真实概率)                             │
└─────────────────────────────────────────────────────────────┘
```

#### 维度配置表

| 组件 | 层 | 输入维度 | 输出维度 | 激活函数 |
|------|-----|---------|---------|----------|
| **Generator** | | | | |
| | Input | 32+1=33 | 33 | - |
| | FC1 | 33 | 64 | LeakyReLU(0.2) |
| | FC2 | 64 | 128 | LeakyReLU(0.2) + BN |
| | FC3 | 128 | 64 | LeakyReLU(0.2) + BN |
| | Output | 64 | 12 | Tanh |
| **Discriminator** | | | | |
| | Input | 12+1=13 | 13 | - |
| | FC1 | 13 | 64 | LeakyReLU(0.2) |
| | FC2 | 64 | 128 | LeakyReLU(0.2) + Dropout(0.3) |
| | FC3 | 128 | 64 | LeakyReLU(0.2) + Dropout(0.3) |
| | Output | 64 | 1 | Sigmoid |

#### 损失函数

$$\mathcal{L}_D = -\mathbb{E}[\log D(x_{real}, c)] - \mathbb{E}[\log(1 - D(G(z,c), c))]$$

$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z,c), c)]$$

---

### 3️⃣ 样本扩充比例设计（10×）

#### 扩充策略

| 原始样本类型 | 原始数量 | 扩充倍数 | 扩充后数量 | 说明 |
|-------------|---------|---------|-----------|------|
| 变轨样本 (label=1) | N_pos | 10× | 10×N_pos | 少数类重点扩充 |
| 非变轨样本 (label=0) | N_neg | 3× | 3×N_neg | 多数类适度扩充 |

#### 扩充流程

```
原始数据集: D_orig = {(x_i, c_i)}_{i=1}^{N}
    ↓
训练 cGAN (epochs=500, batch_size=32)
    ↓
生成扩充样本:
  - 变轨: G(z, c=1) × 10×N_pos
  - 非变轨: G(z, c=0) × 3×N_neg
    ↓
合并数据集: D_aug = D_orig ∪ D_gen
```

#### 数量示例

| 阶段 | 变轨样本 | 非变轨样本 | 总计 | 正负比 |
|------|---------|-----------|------|--------|
| 原始 | 100 | 900 | 1000 | 1:9 |
| 扩充后 | 1000 | 2700 | 3700 | 1:2.7 |

---

### 4️⃣ 扩充有效性验证方式

#### 验证方法总览

| 方法 | 目的 | 通过标准 |
|------|------|----------|
| t-SNE 可视化 | 分布一致性 | 生成样本与真实样本重叠 |
| 统计检验 | 特征分布匹配 | KS检验 p > 0.05 |
| 分类性能 | 下游任务提升 | F1提升 > 5% |
| 特征相关性 | 保持物理约束 | 相关矩阵差异 < 0.1 |

#### 方法1: t-SNE 可视化

```python
# 伪代码
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def validate_tsne(X_real, X_gen, labels_real, labels_gen):
    """t-SNE可视化验证"""
    X_all = np.vstack([X_real, X_gen])
    source = ['real']*len(X_real) + ['gen']*len(X_gen)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X_all)

    # 绘制: 真实vs生成, 按类别着色
    # 期望: 同类别的真实/生成样本应重叠
```

**判定标准**: 生成样本点云与真实样本点云在同类别内高度重叠，无明显分离

#### 方法2: KS 统计检验

```python
# 伪代码
from scipy.stats import ks_2samp

def validate_ks(X_real, X_gen):
    """逐特征KS检验"""
    results = []
    for i in range(X_real.shape[1]):
        stat, p_value = ks_2samp(X_real[:, i], X_gen[:, i])
        results.append({
            'feature': i,
            'ks_stat': stat,
            'p_value': p_value,
            'pass': p_value > 0.05
        })
    return results
```

**判定标准**: 所有特征 p_value > 0.05，表示无法拒绝"同分布"假设

#### 方法3: 下游分类性能对比

| 实验组 | 训练数据 | 测试数据 | 对比指标 |
|--------|---------|---------|----------|
| Baseline | D_orig | D_test | F1_base |
| Augmented | D_aug | D_test | F1_aug |

**判定标准**: $\Delta F1 = F1_{aug} - F1_{base} > 5\%$

#### 方法4: 特征相关性矩阵对比

```python
# 伪代码
def validate_correlation(X_real, X_gen):
    """相关矩阵一致性检验"""
    corr_real = np.corrcoef(X_real.T)
    corr_gen = np.corrcoef(X_gen.T)

    diff = np.abs(corr_real - corr_gen)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    return max_diff < 0.15 and mean_diff < 0.1
```

**判定标准**: 相关矩阵元素差异均值 < 0.1，最大差异 < 0.15

---

### 5️⃣ GAN 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| z_dim | 32 | 噪声向量维度 |
| lr_G | 2e-4 | Generator学习率 |
| lr_D | 1e-4 | Discriminator学习率 |
| beta1 | 0.5 | Adam优化器参数 |
| batch_size | 32 | 批大小 |
| epochs | 500 | 训练轮数 |
| D_steps | 1 | 每轮D更新次数 |
| G_steps | 1 | 每轮G更新次数 |

---

## 步骤4：RF 与 DNN 变轨检测模型设计

### 1️⃣ 两种模型的输入特征、输出、训练流程

#### 输入特征定义

| 特征组 | 特征 | RF (LEO雷达) | DNN (GEO光学) |
|--------|------|:------------:|:-------------:|
| 残差特征 | F2_std | ✓ | ✓ |
| | F3_pp | ✓ | ✓ |
| | F4_skew | ✓ | ✓ |
| | F5_kurt | ✓ | ✓ |
| | F6_diff1_mean | ✓ | ✓ |
| | F7_diff1_std | ✓ | ✓ |
| | F8_diff2_mean | ✓ | ✓ |
| | F9_slope | ✓ | ✓ |
| | F10_freq_ratio | | ✓ |
| 辐射特征 | P | ✓ | ✓ |
| | T | | ✓ |
| | R | ✓ | ✓ |
| **总维度** | | **10** | **12** |

#### 输出定义

| 模型 | 输出 | 类型 | 说明 |
|------|------|------|------|
| RF | $\hat{y} \in \{0, 1\}$ | 硬分类 | 0=非变轨, 1=变轨 |
| RF | $p(y=1|x)$ | 概率 | 变轨置信度 |
| DNN | $\hat{y} \in [0, 1]$ | Sigmoid输出 | 变轨概率 |
| DNN | $\hat{y}_{class} = \mathbb{1}[\hat{y} > \tau]$ | 阈值分类 | $\tau$ 可调

#### RF 训练流程

```
┌─────────────────────────────────────────────────────────┐
│                 Random Forest 训练流程                   │
├─────────────────────────────────────────────────────────┤
│  1. 数据准备                                            │
│     - 加载 LEO 雷达残差特征 (10维)                      │
│     - 划分 train/val/test = 70/15/15                    │
│                                                         │
│  2. 超参数搜索 (GridSearchCV)                           │
│     - n_estimators: [50, 100, 200]                      │
│     - max_depth: [5, 10, 15, None]                      │
│     - min_samples_split: [2, 5, 10]                     │
│     - class_weight: ['balanced', None]                  │
│                                                         │
│  3. 训练最优模型                                        │
│     - 5-fold CV 选择最优参数                            │
│     - 在全量训练集上重新训练                            │
│                                                         │
│  4. 评估                                                │
│     - 在 test 集计算 Precision/Recall/F1               │
└─────────────────────────────────────────────────────────┘
```

#### DNN 训练流程 (含 GSCV)

```
┌─────────────────────────────────────────────────────────┐
│              DNN + GridSearchCV 训练流程                 │
├─────────────────────────────────────────────────────────┤
│  1. 数据准备                                            │
│     - 加载 GEO 光学残差特征 (12维)                      │
│     - Z-score / Min-Max 归一化                          │
│     - 划分 train/val/test = 70/15/15                    │
│                                                         │
│  2. GSCV 超参数搜索                                     │
│     - hidden_layers: [(64,32), (128,64), (128,64,32)]   │
│     - batch_size: [16, 32, 64]                          │
│     - learning_rate: [1e-3, 5e-4, 1e-4]                 │
│     - dropout: [0.2, 0.3, 0.5]                          │
│                                                         │
│  3. 训练 (Early Stopping)                               │
│     - patience = 20, monitor = val_loss                 │
│     - max_epochs = 200                                  │
│                                                         │
│  4. 阈值优化                                            │
│     - 在 val 集搜索最优 τ (最大化 F1)                   │
│                                                         │
│  5. 评估                                                │
│     - 在 test 集计算最终指标                            │
└─────────────────────────────────────────────────────────┘
```

---

### 2️⃣ DNN 结构建议

#### 网络架构

| 层 | 输入维度 | 输出维度 | 激活函数 | 正则化 |
|----|---------|---------|----------|--------|
| Input | 12 | 12 | - | - |
| FC1 | 12 | 128 | ReLU | Dropout(0.3) |
| FC2 | 128 | 64 | ReLU | Dropout(0.3) |
| FC3 | 64 | 32 | ReLU | Dropout(0.2) |
| Output | 32 | 1 | Sigmoid | - |

#### GSCV 搜索空间

| 超参数 | 搜索范围 | 推荐值 |
|--------|---------|--------|
| hidden_layers | [(64,32), (128,64), (128,64,32)] | (128,64,32) |
| batch_size | [16, 32, 64] | 32 |
| learning_rate | [1e-3, 5e-4, 1e-4] | 5e-4 |
| dropout | [0.2, 0.3, 0.5] | 0.3 |
| optimizer | [Adam, AdamW] | AdamW |
| weight_decay | [1e-4, 1e-5] | 1e-4 |

#### 损失函数

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[w_1 \cdot y_i \log(\hat{y}_i) + w_0 \cdot (1-y_i)\log(1-\hat{y}_i)\right]$$

其中 $w_1 / w_0$ 为类别权重，用于处理样本不平衡。

---

### 3️⃣ RF vs DNN 适用场景分析

#### 为什么 RF 更适合 LEO 雷达数据

| 因素 | LEO 雷达特点 | RF 优势 |
|------|-------------|---------|
| **采样率高** | AZ/EL/Ra/RR 多维同步采样 | 树模型天然处理多维特征交互 |
| **特征可解释** | 残差物理意义明确 | 特征重要性可直接输出 |
| **噪声鲁棒** | 雷达测量噪声较大 | 集成学习对噪声不敏感 |
| **样本量有限** | LEO 目标观测弧段短 | 小样本下不易过拟合 |
| **实时性要求** | 需快速响应 | 推理速度快 (无矩阵运算) |

#### 为什么 DNN 更适合 GEO 光学数据

| 因素 | GEO 光学特点 | DNN 优势 |
|------|-------------|----------|
| **采样率低** | 30s~1min 间隔 | 可学习时序隐含模式 |
| **特征耦合** | RA/DEC + 辐射 P/T/R 耦合 | 非线性拟合能力强 |
| **信噪比低** | 光学观测受大气影响 | 深层网络可提取弱信号 |
| **样本可扩充** | GAN 扩充后样本充足 | 大样本下性能更优 |
| **精度优先** | GEO 变轨检测容忍延迟 | 可牺牲速度换精度 |

#### 对比总结

| 维度 | RF (LEO雷达) | DNN (GEO光学) |
|------|-------------|---------------|
| 样本需求 | 小 (100+) | 大 (1000+) |
| 训练速度 | 快 | 慢 |
| 推理速度 | 极快 (<1ms) | 较快 (~5ms) |
| 可解释性 | 高 | 低 |
| 非线性能力 | 中 | 高 |
| 过拟合风险 | 低 | 中 |

---

### 4️⃣ 检测率 / 虚警率 / 推理时间 对比方法

#### 评估指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| 检测率 (Recall) | $P_d = \frac{TP}{TP + FN}$ | 真实变轨被正确检出的比例 |
| 虚警率 (FAR) | $P_{fa} = \frac{FP}{FP + TN}$ | 非变轨被误判为变轨的比例 |
| 精确率 (Precision) | $P = \frac{TP}{TP + FP}$ | 检出中真正变轨的比例 |
| F1-Score | $F1 = \frac{2 \cdot P \cdot P_d}{P + P_d}$ | 综合指标 |
| 推理时间 | $T_{inf}$ (ms/sample) | 单样本推理耗时 |

#### ROC 曲线对比

```python
# 伪代码
from sklearn.metrics import roc_curve, auc

def plot_roc_comparison(y_true, y_prob_rf, y_prob_dnn):
    """绘制RF与DNN的ROC曲线对比"""
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_prob_rf)
    fpr_dnn, tpr_dnn, _ = roc_curve(y_true, y_prob_dnn)

    auc_rf = auc(fpr_rf, tpr_rf)
    auc_dnn = auc(fpr_dnn, tpr_dnn)

    # 绘制对比图
    # X轴: FAR (虚警率), Y轴: Pd (检测率)
```

#### 推理时间测量

```python
# 伪代码
import time

def measure_inference_time(model, X_test, n_runs=100):
    """测量推理时间"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        end = time.perf_counter()
        times.append((end - start) / len(X_test) * 1000)  # ms/sample

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'p95': np.percentile(times, 95)
    }
```

#### 综合对比表模板

| 模型 | AUC | Pd@FAR=1% | Pd@FAR=5% | F1 | T_inf (ms) |
|------|-----|-----------|-----------|-----|------------|
| RF (LEO) | - | - | - | - | - |
| DNN (GEO) | - | - | - | - | - |

#### 目标性能指标

| 指标 | LEO雷达 (RF) | GEO光学 (DNN) |
|------|-------------|---------------|
| 检测率 Pd | > 90% | > 85% |
| 虚警率 FAR | < 5% | < 3% |
| F1-Score | > 0.85 | > 0.80 |
| 推理时间 | < 1ms | < 10ms |
| AUC | > 0.92 | > 0.90 |

---

## 步骤5：点火时刻定位模型设计

### 1️⃣ LSTM vs Transformer 输入形式设计

#### 输入数据结构

| 数据类型 | 维度 | 说明 |
|----------|------|------|
| 残差时序 | $(T, d_r)$ | $T$=序列长度, $d_r$=残差特征数 |
| 辐射时序 | $(T, d_{rad})$ | $d_{rad}$=3 (P, T, R) |
| 检测置信度 | $(1,)$ | 变轨检测模型输出 $p_{maneuver}$ |
| **融合输入** | $(T, d_{total})$ | $d_{total} = d_r + d_{rad} + 1$ |

#### 具体维度配置

| 场景 | T (序列长度) | $d_r$ | $d_{rad}$ | $d_{total}$ |
|------|-------------|-------|-----------|-------------|
| GEO光学 | 60 (30min@30s) | 4 | 3 | 8 |
| LEO雷达 | 120 (10min@5s) | 6 | 3 | 10 |

#### LSTM 输入形式

```
输入张量: X ∈ R^(B × T × d_total)
    │
    ▼
┌─────────────────────────────────────┐
│  时间步 t=1: [res_1, rad_1, p_det]  │
│  时间步 t=2: [res_2, rad_2, p_det]  │
│  ...                                │
│  时间步 t=T: [res_T, rad_T, p_det]  │
└─────────────────────────────────────┘
    │
    ▼
LSTM(hidden_size=64, num_layers=2, bidirectional=True)
    │
    ▼
输出: h_T ∈ R^(B × 128) 或 全序列 H ∈ R^(B × T × 128)
```

#### Transformer 输入形式

```
输入张量: X ∈ R^(B × T × d_total)
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. 线性投影: X → X' ∈ R^(B × T × d_model)  │
│     d_model = 64                            │
│                                             │
│  2. 位置编码: X' + PE(T, d_model)           │
│     PE: 正弦位置编码 或 可学习位置编码       │
│                                             │
│  3. Transformer Encoder                     │
│     - num_layers = 2                        │
│     - num_heads = 4                         │
│     - dim_feedforward = 128                 │
└─────────────────────────────────────────────┘
    │
    ▼
输出: H ∈ R^(B × T × d_model) 或 CLS token
```

---

### 2️⃣ LSTM vs Transformer: 小样本 + 实时性对比

| 维度 | LSTM | Transformer | 胜出 |
|------|------|-------------|------|
| **小样本适应** | 参数少，不易过拟合 | 参数多，需大量数据 | **LSTM** |
| **训练数据需求** | 500+ 样本可收敛 | 2000+ 样本才稳定 | **LSTM** |
| **推理延迟** | 顺序计算，O(T) | 并行计算，O(1) | Transformer |
| **实时增量推理** | 天然支持流式输入 | 需重新计算全序列 | **LSTM** |
| **长序列建模** | 梯度消失，T<200 | 全局注意力，T可更长 | Transformer |
| **位置敏感性** | 隐式编码时序 | 依赖位置编码质量 | **LSTM** |
| **工程复杂度** | 简单，成熟 | 较复杂，调参多 | **LSTM** |

#### 结论：LSTM 更适合本任务

**理由**：
1. 小样本场景 (GAN扩充后仍<5000)，LSTM参数效率更高
2. 实时性要求支持流式推理，LSTM可逐步更新隐状态
3. 序列长度适中 (T=60~120)，LSTM足以建模
4. 工程落地简单，调试成本低

---

### 3️⃣ 点火时刻：回归 vs 序列分类

#### 两种建模方式对比

| 方式 | 输出形式 | 损失函数 | 优点 | 缺点 |
|------|---------|---------|------|------|
| **回归** | $\hat{t}_{ign} \in [0, T]$ | MSE / Huber | 直接输出时刻，精度高 | 对异常值敏感 |
| **序列分类** | $\hat{y}_t \in \{0,1\}^T$ | BCE | 可输出置信度分布 | 需后处理提取时刻 |

#### 推荐方案：序列分类 + 加权质心回归

**理由**：

1. **不确定性量化**：序列分类输出每个时刻的点火概率 $p_t$，可评估定位置信度
2. **多峰鲁棒**：若存在多次点火，可检测多个峰值
3. **标签构造简单**：将点火时刻附近标记为1，其余为0
4. **后处理灵活**：可用加权质心精确定位

#### 标签构造方式

$$y_t = \begin{cases}
1 & |t - t_{ign}^{true}| \leq \delta \\
0 & \text{otherwise}
\end{cases}$$

其中 $\delta$ 为容忍窗口，建议 $\delta = 2$~$3$ 个采样间隔。

#### 点火时刻提取（加权质心法）

$$\hat{t}_{ign} = \frac{\sum_{t=1}^{T} t \cdot p_t \cdot \mathbb{1}[p_t > \tau]}{\sum_{t=1}^{T} p_t \cdot \mathbb{1}[p_t > \tau]}$$

其中 $\tau = 0.5$ 为概率阈值。

---

### 4️⃣ 时间戳误差评估方式

#### 评估指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| MAE | $\frac{1}{N}\sum_{i=1}^{N}|\hat{t}_i - t_i^{true}|$ | 平均绝对误差 (秒) |
| RMSE | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{t}_i - t_i^{true})^2}$ | 均方根误差 (秒) |
| 命中率@k | $\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[|\hat{t}_i - t_i^{true}| \leq k \cdot \Delta t]$ | k个采样间隔内命中比例 |
| 90%分位误差 | $P_{90}(|\hat{t} - t^{true}|)$ | 90%样本的误差上界 |

#### 评估伪代码

```python
def evaluate_ignition_timing(t_pred, t_true, dt):
    """评估点火时刻定位精度"""
    errors = np.abs(t_pred - t_true)

    return {
        'MAE': np.mean(errors),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'Hit@1': np.mean(errors <= 1 * dt),
        'Hit@2': np.mean(errors <= 2 * dt),
        'Hit@3': np.mean(errors <= 3 * dt),
        'P90': np.percentile(errors, 90)
    }
```

#### 目标性能指标

| 场景 | MAE | Hit@1 | Hit@3 | P90 |
|------|-----|-------|-------|-----|
| GEO光学 (Δt=30s) | < 45s | > 70% | > 95% | < 90s |
| LEO雷达 (Δt=5s) | < 10s | > 75% | > 95% | < 20s |

---

### 5️⃣ LSTM 点火定位模型结构

```
输入: X ∈ R^(B × T × d)
    │
    ▼
┌─────────────────────────────┐
│  Bi-LSTM Layer 1            │
│  hidden=64, bidirectional   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Bi-LSTM Layer 2            │
│  hidden=64, bidirectional   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  FC: 128 → 1                │
│  Sigmoid (逐时刻)           │
└─────────────────────────────┘
    │
    ▼
输出: p ∈ R^(B × T × 1)  点火概率序列
```

---

## 步骤6：MLF-SNN 多阈值脉冲神经网络

### 1️⃣ 三阈值神经元设计动机

#### 传统 LIF 神经元 vs 多阈值神经元

| 类型 | 阈值数 | 输出 | 信息容量 |
|------|--------|------|----------|
| 标准 LIF | 1 (θ=1.0) | 0/1 二值脉冲 | 1 bit/spike |
| **MLF (Multi-Level Firing)** | 3 (0.6/1.6/2.6) | 0/1/2/3 多级脉冲 | 2 bit/spike |

#### 阈值设计原理

```
膜电位 u(t)
    │
    │                         ┌─── θ₃ = 2.6 → 发放 3 级脉冲
    │                    ─────┤
    │               ─────     └─── θ₂ = 1.6 → 发放 2 级脉冲
    │          ─────
    │     ─────               ┌─── θ₁ = 0.6 → 发放 1 级脉冲
    │─────                ────┤
    └─────────────────────────┴─── u < 0.6 → 不发放 (0)
    0                              时间 t
```

#### 阈值 0.6 / 1.6 / 2.6 的设计动机

| 阈值 | 间隔 | 设计动机 |
|------|------|----------|
| θ₁ = 0.6 | - | 低于标准阈值1.0，提高弱信号敏感度 |
| θ₂ = 1.6 | Δ=1.0 | 等间隔设计，线性量化膜电位 |
| θ₃ = 2.6 | Δ=1.0 | 捕获强激励，避免信息饱和 |

**核心优势**：
1. **信息效率**：单次发放携带 2bit 信息，减少时间步数
2. **弱信号检测**：θ₁=0.6 低阈值对小推力微弱残差敏感
3. **动态范围**：3级量化覆盖更大输入范围，适应 SNR 3~10dB

#### MLF 神经元数学模型

**膜电位更新**：
$$u^{t+1} = \beta \cdot u^t + \sum_j w_j \cdot s_j^t - s_{out}^t \cdot \theta_{fire}$$

**多阈值发放函数**：
$$s_{out}^t = \begin{cases}
3 & u^t \geq \theta_3 = 2.6 \\
2 & \theta_2 \leq u^t < \theta_3, \quad \theta_2 = 1.6 \\
1 & \theta_1 \leq u^t < \theta_2, \quad \theta_1 = 0.6 \\
0 & u^t < \theta_1
\end{cases}$$

其中 $\beta = 0.9$ 为膜电位衰减系数。

---

### 2️⃣ 输入特征转脉冲序列

#### 编码方式：速率编码 + 时间编码混合

```
原始特征 x ∈ R^d
    │
    ▼
┌─────────────────────────────┐
│  1. 归一化: x' = (x-μ)/σ    │
│  2. 映射到 [0, 1]: x'' = σ(x') │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  速率编码 (Rate Coding)      │
│  P(spike=1) = x''           │
│  生成 T 步脉冲序列           │
└─────────────────────────────┘
    │
    ▼
输出: S ∈ {0,1}^(T × d)
```

#### 编码伪代码

```python
def encode_to_spikes(x, T=16):
    """将特征编码为脉冲序列"""
    # 归一化到 [0, 1]
    x_norm = torch.sigmoid(x)

    # 速率编码: 按概率生成脉冲
    spikes = torch.zeros(T, x.shape[-1])
    for t in range(T):
        spikes[t] = (torch.rand_like(x_norm) < x_norm).float()

    return spikes  # (T, d)
```

#### 直接编码（推荐）

对于时序残差数据，可直接作为电流输入：

```python
def direct_encoding(x_seq):
    """直接将时序特征作为输入电流"""
    # x_seq: (T, d) 原始时序
    # 归一化后直接输入 SNN
    return (x_seq - x_seq.mean()) / (x_seq.std() + 1e-8)
```

---

### 3️⃣ Surrogate Gradient 训练方法

#### 问题：脉冲函数不可微

$$s = H(u - \theta) = \begin{cases} 1 & u \geq \theta \\ 0 & u < \theta \end{cases}$$

Heaviside 阶跃函数导数为 Dirac δ 函数，无法反向传播。

#### 解决方案：Surrogate Gradient

**前向传播**：使用真实阶跃函数
**反向传播**：用平滑函数近似梯度

| 替代函数 | 公式 | 特点 |
|----------|------|------|
| Sigmoid | $\sigma'(u) = \alpha \cdot \sigma(\alpha u)(1-\sigma(\alpha u))$ | 平滑，计算简单 |
| **Fast Sigmoid** | $\frac{\alpha}{2(1+\alpha|u-\theta|)^2}$ | 推荐，收敛快 |
| Arctan | $\frac{\alpha}{\pi(1+(\alpha(u-\theta))^2)}$ | 梯度衰减慢 |

#### MLF-SNN 多阈值 Surrogate Gradient

```python
class MultiThresholdSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, thresholds=[0.6, 1.6, 2.6]):
        # 多阈值发放
        s = torch.zeros_like(u)
        s[u >= thresholds[0]] = 1
        s[u >= thresholds[1]] = 2
        s[u >= thresholds[2]] = 3
        ctx.save_for_backward(u)
        ctx.thresholds = thresholds
        return s

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        alpha = 4.0
        # 多阈值梯度叠加
        grad = torch.zeros_like(u)
        for theta in ctx.thresholds:
            grad += alpha / (2 * (1 + alpha * torch.abs(u - theta))**2)
        return grad_output * grad, None
```

#### 训练流程

```
┌─────────────────────────────────────┐
│  1. 前向传播 (真实脉冲)              │
│     u → H(u-θ) → s ∈ {0,1,2,3}      │
├─────────────────────────────────────┤
│  2. 计算损失                         │
│     L = CrossEntropy(output, label) │
├─────────────────────────────────────┤
│  3. 反向传播 (替代梯度)              │
│     ∂L/∂u ≈ ∂L/∂s · surrogate'(u)  │
├─────────────────────────────────────┤
│  4. 更新权重                         │
│     w = w - lr · ∂L/∂w              │
└─────────────────────────────────────┘
```

---

### 4️⃣ MLF-SNN vs DNN 计算量与实时性对比

#### 计算量对比

| 指标 | DNN | MLF-SNN | 优势 |
|------|-----|---------|------|
| 基本运算 | 浮点乘加 (MAC) | 加法 (AC) | **SNN** |
| 单层计算 | $O(n_{in} \times n_{out})$ MAC | $O(n_{in} \times n_{out})$ AC | **SNN** |
| 能耗比 | 1× (基准) | 0.1×~0.01× | **SNN** |
| 稀疏性 | 无 | 高 (仅脉冲时计算) | **SNN** |

#### 能耗分析

| 运算类型 | 45nm CMOS 能耗 | 说明 |
|----------|---------------|------|
| 32-bit 浮点乘法 | 3.7 pJ | DNN 主要运算 |
| 32-bit 浮点加法 | 0.9 pJ | DNN 辅助运算 |
| 32-bit 整数加法 | 0.1 pJ | SNN 主要运算 |

**SNN 能效优势**：$\frac{3.7 + 0.9}{0.1} \approx 46\times$

#### 实时性对比

| 指标 | DNN | MLF-SNN | 说明 |
|------|-----|---------|------|
| GPU 推理 | ~5ms | ~3ms | SNN 稀疏计算优势 |
| CPU 推理 | ~20ms | ~15ms | SNN 无乘法优势 |
| 神经形态芯片 | N/A | **<1ms** | SNN 专用硬件 |
| 边缘部署 | 需 GPU | 可纯 CPU | **SNN** |

#### 综合对比表

| 维度 | DNN | MLF-SNN | 本任务推荐 |
|------|-----|---------|-----------|
| 精度 | 高 | 中高 | DNN |
| 能耗 | 高 | **极低** | MLF-SNN |
| 实时性 | 中 | **高** | MLF-SNN |
| 训练难度 | 低 | 中 | DNN |
| 硬件要求 | GPU | CPU/神经形态 | MLF-SNN |
| 小样本 | 中 | **好** | MLF-SNN |

#### 本任务推荐策略

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 地面站处理 | DNN | 算力充足，精度优先 |
| 星载实时检测 | **MLF-SNN** | 低功耗、低延迟 |
| 边缘预警 | **MLF-SNN** | 无需 GPU，快速响应 |

---