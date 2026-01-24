# UV 识别系统 - 快速启动指南

## 5分钟快速开始

### 1. 运行完整演示

```bash
python demo_uv_recognition.py
```

这将执行：
- ✅ 单个文件推理示例
- ✅ 批量推理示例
- ✅ 可视化生成
- ✅ 性能分析
- ✅ 自定义推理示例

### 2. 查看结果

演示完成后，查看生成的文件：

```bash
# 查看可视化图表
ls demo_figures/
# demo_uv_mapping.png - UV映射效果
# demo_pulse_detection.png - 脉冲检测
# demo_recognition_performance.png - 识别性能

# 查看分析报告
cat analysis/uv_recognition/analysis_report.txt
```

---

## 常用命令

### 推理单个文件

```python
from src.uv_inference import UVRecognitionPipeline

pipeline = UVRecognitionPipeline()
result = pipeline.predict_single('data/test/example.csv')

print(f"点火时刻: {result['ignition_time']:.2f}s")
print(f"推力估计: {result['thrust_estimate']:.4f}N")
print(f"变轨类型: {result['maneuver_type_name']}")
```

### 批量推理

```bash
python src/uv_inference.py
```

输出: `results/uv_recognition_results.csv`

### 生成可视化

```bash
python src/uv_visualization.py
```

输出: `figures/uv_recognition/*.png`

### 性能分析

```bash
python src/uv_results_analysis.py
```

输出: `analysis/uv_recognition/analysis_report.txt`

---

## 完整流水线

如果需要从头开始训练模型：

```bash
# 执行所有阶段（映射→特征→训练→推理→可视化）
python run_uv_pipeline.py --stages all

# 只执行推理和可视化
python run_uv_pipeline.py --stages 4,5
```

---

## 当前性能指标

基于测试集 1344 个样本：

| 任务 | 指标 | 值 |
|------|------|-----|
| 变轨分类 | 准确率 | **99.26%** |
| 变轨分类 | F1分数 | **98.51%** |
| 推力估计 | MAE | **0.0297 N** |
| 推力估计 | R² | **0.9698** |
| 点火检测 | 检测率 | 25.37% * |

\* 注：点火检测率较低是因为大部分测试样本是稳态燃烧数据(ssf)，没有明显的点火上升沿。对于脉冲数据(health_check)，检测率接近100%。

---

## 文件结构速查

```
src/
├── uv_mapping.py              # 推力→UV映射
├── uv_feature_extraction.py   # 特征提取
├── uv_recognition_models.py   # 模型定义
├── uv_inference.py            # 推理流程
├── uv_visualization.py        # 可视化
└── uv_results_analysis.py     # 结果分析

models/uv_recognition/         # 训练好的模型
├── ignition_detector.pkl
├── maneuver_classifier.pkl
├── thrust_regressor.pkl
└── maneuver_type_classifier.pkl

results/
└── uv_recognition_results.csv # 推理结果

figures/uv_recognition/        # 可视化图表
└── *.png

analysis/uv_recognition/       # 分析报告
├── analysis_report.txt
└── comprehensive_analysis.png
```

---

## 自定义参数

### UV 映射参数

```python
from src.uv_mapping import UV360nmMapper

mapper = UV360nmMapper(
    alpha=1000.0,      # 缩放因子
    beta=1.1,          # 质量流率权重
    gamma=50.0,        # 非线性指数
    I_bg=10.0,         # 背景辐射
    noise_std=2.0      # 噪声标准差
)
```

### 特征提取参数

```python
from src.uv_feature_extraction import UVFeatureExtractor

extractor = UVFeatureExtractor(
    threshold_factor=3.0,      # 阈值因子
    min_pulse_duration=0.1,    # 最小脉冲持续时间(秒)
    sampling_rate=100.0        # 采样率(Hz)
)
```

---

## 故障排查

### 问题1: 模型文件不存在

```bash
# 重新训练模型
python src/uv_recognition_models.py
```

### 问题2: 数据文件不存在

```bash
# 检查数据目录
ls data/train/ data/test/

# 如果需要生成UV数据
python src/uv_mapping.py
```

### 问题3: 推理结果不准确

可能原因：
1. UV映射参数需要调整
2. 特征提取阈值需要调整
3. 需要更多训练数据

解决方案：
```python
# 调整UV映射参数
mapper = UV360nmMapper(alpha=1500.0, beta=1.2, gamma=60.0)

# 调整特征提取阈值
extractor = UVFeatureExtractor(threshold_factor=2.5)
```

---

## 下一步

1. **查看完整文档**: `UV_RECOGNITION_README.md`
2. **运行演示**: `python demo_uv_recognition.py`
3. **查看可视化**: 打开 `figures/uv_recognition/` 中的图表
4. **阅读分析报告**: `analysis/uv_recognition/analysis_report.txt`

---

## 技术支持

如有问题，请检查：
1. Python版本 >= 3.7
2. 依赖包已安装: numpy, pandas, scikit-learn, torch, matplotlib
3. 数据文件格式正确（包含 time, thrust, mfr 列）

---

**最后更新**: 2026-01-24
