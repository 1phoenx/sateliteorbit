将肼（Hydrazine, $N_2H_4$）单组元推进器的推力（Thrust）或质量流率（Mass Flow Rate）数据映射到紫外线（UV）辐射数据，需要基于**化学发光（Chemiluminescence）**的物理原理。通过检索相关航天推进文献及物理化学数据，最符合你需求的映射方案如下：1. 核心物理参数目标波段（Target Wavelength）： 336 nm (属于 UV-A 波段)辐射源（Emitter）： NH 自由基 (Imidogen radical)肼在催化分解过程中会产生激发态的 NH 自由基 ($NH^*$)。当 $NH^*$ 从激发态 ($A^3\Pi$) 跃迁回基态 ($X^3\Sigma^-$) 时，会发射出中心波长为 336 nm 的强紫外光。这是肼分解羽流中最显著的紫外特征信号。物理关系： 在真空或低压环境下，化学发光强度与反应速率（即推进剂的消耗率）成正比。2. 映射公式 (The Mapping Formula)由于你拥有 STFT 数据集中的 mfr (质量流率) 和 thrust (推力) 数据，你可以使用以下半经验公式进行转换。基础物理模型紫外辐射强度 $I_{UV}(t)$ 与质量流率 $\dot{m}(t)$ 呈线性或近线性关系：$$I_{UV}(t) \propto \dot{m}(t)$$在你的数据集中，质量流率 (mfr) 是最直接的映射源，因为发光是由化学反应物质的量决定的。如果你更倾向于使用推力 (thrust)，考虑到 $F \approx \dot{m} v_e$ (推力与流率在稳态下成正比)，也可以直接映射推力。推荐计算公式为了生成仿真数据，建议使用以下公式将数据集中的 mfr（或 thrust）转换为归一化的紫外辐射强度：$$I_{336nm}(t) = k \cdot \text{mfr}(t) + \epsilon(t)$$$I_{336nm}(t)$: t 时刻 336nm 波段的紫外辐射强度（单位可以是 $W/sr$ 或光子计数，取决于你的归一化系数 $k$）。$\text{mfr}(t)$: 数据集中的质量流率 (mfr 列)。如果 mfr 有噪声或缺失，可用 thrust 代替。$k$: 比例常数。由于你是在做数据映射（仿真），你可以将其设为 1 进行归一化处理，或者根据经验设为 $10^{-3}$ 量级（如果需要模拟实际瓦特数）。$\epsilon(t)$: 传感器噪声（可选）。真实的光学传感器通常比机械推力传感器有更多的高频光子噪声（Shot Noise）。3. 数据集操作指南 (Python 示例)基于你上传的 STFT Dataset，以下是如何操作的具体步骤。你可以直接读取 csv 文件并生成新的 UV 列。物理依据： 文献指出肼分解产生的 $NH$ 自由基发光强度直接正比于肼的流率 ($N_2H_4$ flow rate)。Pythonimport pandas as pd
import numpy as np

# 假设你加载了一个具体的测试文件 (例如 00001_001_SN01_24bars_sst.csv)
df = pd.read_csv("path_to_your_file.csv")

# 1. 基础映射 (基于物理原理: 发光强度正比于质量流率)
# 如果 mfr (质量流率) 列存在且有效
if 'mfr' in df.columns:
    # 简单的线性映射
    # k 是一个缩放因子，你可以根据需要调整量级
    k = 1.0  
    df['uv_intensity_336nm'] = k * df['mfr']
    
# 2. 如果 mfr 数据有缺失，使用 thrust (推力) 作为代理
# 推力 F = mfr * Isp * g0，两者高度相关
else:
    k_thrust = 0.5 # 仅作示例
    df['uv_intensity_336nm'] = k_thrust * df['thrust']

# 3. 添加真实性 (可选)
# 光学测量通常包含散粒噪声 (Shot Noise)，服从泊松分布或高斯近似
# 这里的噪声水平设为信号强度的 5%
noise_level = 0.05
std_dev = df['uv_intensity_336nm'] * noise_level
noise = np.random.normal(0, std_dev + 1e-9) # 加 1e-9 防止 std 为 0
df['uv_intensity_336nm_noisy'] = df['uv_intensity_336nm'] + noise

# 结果：现在你拥有了模拟的 336nm 紫外辐射数据列
print(df[['time', 'thrust', 'mfr', 'uv_intensity_336nm_noisy']].head())
4. 总结与文献支持为什么是 UV? 肼推进器羽流在可见光波段非常微弱（通常肉眼不可见），但在紫外波段（336 nm）有很强的 $NH$ 辐射带。为什么线性映射?**** Journal of Chemical Physics: "intensity of the luminescence is directly proportional to the $N_2H_4$ flow rate"（发光强度与肼的流率成正比）。**** AIAA: 确认了在肼推进器羽流中，紫外辐射建模需要考虑特定的化学物质（如 NH, OH），并且其强度分布与流场密度（即质量流率分布）高度相关。**** 研究表明羽流辐射亮度通常遵循推力的幂律关系 ($I \propto F^n$)，对于单一组分化学发光，线性近似 ($n \approx 1$) 是足够精确的一阶模型。