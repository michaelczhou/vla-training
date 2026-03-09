# Physical Intelligence 技术原理完整报告

## 目录

1. [执行摘要](#1-执行摘要)
2. [π₀: 视觉 - 语言 - 动作流模型](#2-π₀-视觉 - 语言 - 动作流模型)
3. [FAST: 高效动作 Tokenization](#3-fast-高效动作-tokenization)
4. [RTC: 实时动作分块执行](#4-rtc-实时动作分块执行)
5. [知识隔离 VLA 技术](#5-知识隔离-vla-技术)
6. [π₀.₅: 开放世界泛化](#6-π₀₅-开放世界泛化)
7. [π*₀.₆: 从经验中学习](#7-π₀₆-从经验中学习)
8. [人类到机器人技能迁移](#8-人类到机器人技能迁移)
9. [核心算法数学公式与流程对照表](#9-核心算法数学公式与流程对照表)
10. [代码实现参考](#10-代码实现参考)
11. [迁移部署适配指南](#11-迁移部署适配指南)
12. [实验与训练流程](#12-实验与训练流程)
13. [总结与展望](#13-总结与展望)

---

## 1. 执行摘要

### 1.1 Physical Intelligence 研究概览

Physical Intelligence (π) 是一家致力于开发通用机器人基础模型的公司。其核心研究方向是**视觉 - 语言 - 动作模型 (Vision-Language-Action Models, VLA)**，旨在使机器人能够像人类一样理解和执行物理世界中的复杂任务。

### 1.2 核心技术突破

| 技术 | 核心创新 | 解决的问题 |
|------|----------|------------|
| **π₀** | 流匹配 (Flow Matching) VLA 架构 | 传统扩散模型推理慢，高维动作空间效率低 |
| **FAST** | DCT 频率空间动作 Tokenization | 自回归 VLA 训练效率低，高频动作细节丢失 |
| **RTC** | 实时分块 (Real-Time Chunking) | 大模型推理延迟导致动作衔接停顿 |
| **知识隔离** | 梯度隔离 + FAST Token 表示学习 | 动作训练破坏 VLM 原有语义知识 |
| **π₀.₅** | 跨任务协同训练 (Co-training) | 陌生环境泛化能力不足 |
| **π*₀.₆** | RECAP 离线强化学习框架 | 高难度物理操作任务成功率低 |

### 1.3 关键性能指标

- **训练效率**: π₀-FAST 比 π₀ 训练速度快 **5 倍**
- **推理速度**: RTC 使任务执行速度提升 **20%**
- **泛化能力**: π₀.₅ 在未见过的家庭环境中成功执行清理任务
- **任务成功率**: RECAP 在高难度任务上成功率提升 **2 倍**，失败率降低 **50%**

---

## 2. π₀: 视觉 - 语言 - 动作流模型

### 2.1 定位与核心思想

**定位**: Physical Intelligence 系列的基础性论文，提出了基于流匹配 (Flow Matching) 的 VLA 架构。

**核心思想**:
- 将机器人控制建模为**条件流匹配问题**
- 使用预训练的视觉 - 语言模型 (VLM) 作为骨干网络
- 添加**动作专家 (Action Expert)** 模块处理连续动作输出
- 支持跨不同机器人形态的通用控制能力

### 2.2 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                      π₀ 模型架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   视觉输入    │    │   语言指令    │    │   机器人状态   │  │
│  │   (图像)     │    │   (文本)     │    │   (本体感知)  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           VLM Backbone (PaliGemma-3B)               │   │
│  │              (预训练视觉 - 语言模型)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Expert (动作专家)                │   │
│  │         (基于流匹配的连续动作生成模块)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              连续动作输出 (50 步动作块)                │   │
│  │         a₁, a₂, ..., a₅₀ (关节位置/速度)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 流匹配 (Flow Matching) 原理

#### 2.3.1 数学表达

流匹配是一种生成建模方法，通过学习从噪声分布到数据分布的**连续变换**来生成样本。

**定义**:
- 数据分布: $p_0(\mathbf{x})$
- 噪声分布: $p_1(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$
- 时间参数: $t \in [0, 1]$

**概率路径**:
$$p_t(\mathbf{x}) = (1-t)p_0(\mathbf{x}) + tp_1(\mathbf{x})$$

**速度场**:
$$\mathbf{v}_t(\mathbf{x}) = \frac{d\mathbf{x}}{dt}$$

**生成过程** (从噪声到数据):
$$\mathbf{x}_0 = \mathbf{x}_1 + \int_0^1 \mathbf{v}_t(\mathbf{x}_t) dt$$

#### 2.3.2 条件流匹配

对于机器人控制，我们需要**条件生成**:
$$\mathbf{v}_t(\mathbf{x}_t, \mathbf{o}, \tau)$$

其中:
- $\mathbf{o}$: 观测 (图像 + 语言 + 状态)
- $\tau$: 流匹配时间步

**训练目标**:
$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\left[\|\mathbf{v}_\theta(\mathbf{x}_t, \mathbf{o}, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$

#### 2.3.3 推理过程

```python
def generate_action_chunk(policy, observation, num_steps=5):
    """
    使用流匹配生成动作块
    
    参数:
        policy: 训练好的流匹配策略
        observation: 当前观测 (图像 + 语言 + 状态)
        num_steps: 去噪步数
    
    返回:
        action_chunk: 生成的动作序列 [H, action_dim]
    """
    # 1. 从标准高斯分布采样初始噪声
    action_chunk = torch.randn(1, H, action_dim)
    
    # 2. 迭代去噪
    for tau in torch.linspace(0, 1, num_steps):
        # 预测速度场
        velocity = policy.velocity_field(action_chunk, observation, tau)
        
        # Euler 积分更新
        action_chunk = action_chunk + (1/num_steps) * velocity
    
    return action_chunk
```

### 2.4 训练流程

#### 2.4.1 数据混合

| 数据源 | 比例 | 用途 |
|--------|------|------|
| Open X-Embodiment | 40% | 多机器人基础技能 |
| π 自有数据集 | 40% | 灵巧操作任务 |
| 互联网规模 VLM 预训练 | - | 语义理解与泛化 |

#### 2.4.2 训练目标

**联合训练损失**:
$$\mathcal{L}_{total} = \mathcal{L}_{flow} + \lambda \mathcal{L}_{VLM}$$

其中:
- $\mathcal{L}_{flow}$: 流匹配动作生成损失
- $\mathcal{L}_{VLM}$: VLM 语言理解损失 (可选)
- $\lambda$: 平衡系数

### 2.5 实验结论

| 任务 | π₀ | OpenVLA | Octo |
|------|-----|---------|------|
| 桌面清理 (简单) | 0.92 | 0.75 | 0.68 |
| 桌面清理 (困难) | 0.85 | 0.52 | 0.45 |
| T 恤折叠 | 0.78 | 0.31 | 0.22 |
| 杂货装袋 | 0.81 | 0.48 | 0.39 |
| 从烤面包机取出面包 | 0.73 | 0.25 | 0.18 |

**关键发现**:
1. π₀ 在所有任务上均显著优于基线模型
2. 流匹配比离散化动作表示更适合高频灵巧操作
3. VLM 预训练对语言跟随和泛化至关重要

---

## 3. FAST: 高效动作 Tokenization

### 3.1 问题背景

**挑战**: 自回归 VLA 模型需要选择合适的动作 Tokenization 方案，将连续动作信号映射为离散符号。

**现有方法的问题**:
- 简单的逐维度、逐时间步分箱 (binning) 方案
- 在高频机器人数据上表现不佳
- 无法捕捉动作的时间相关性
- 训练效率低

### 3.2 FAST 核心思想

**Frequency-space Action Sequence Tokenization (FAST)**

**关键洞察**: 机器人动作信号需要在训练前进行**压缩**，以减少连续 Token 之间的相关性。

**技术灵感**:
- 离散余弦变换 (DCT) - 类似 JPEG 图像压缩
- 字节对编码 (BPE) - 类似语言模型 Tokenization

### 3.3 FAST Tokenization 流程

```
┌─────────────────────────────────────────────────────────────┐
│                    FAST Tokenization 流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 归一化动作块                                              │
│     ┌─────────────────────────────────────────┐            │
│     │ [a₁, a₂, ..., a₅₀]  (50 步，7 维)          │            │
│     │ 归一化到 [-1, 1] 范围                     │            │
│     └─────────────────────────────────────────┘            │
│                          │                                  │
│                          ▼                                  │
│  2. 离散余弦变换 (DCT)                                         │
│     ┌─────────────────────────────────────────┐            │
│     │  时域 → 频域                              │            │
│     │  低频分量 (重要) + 高频分量 (细节)          │            │
│     └─────────────────────────────────────────┘            │
│                          │                                  │
│                          ▼                                  │
│  3. 量化 + 稀疏化                                              │
│     ┌─────────────────────────────────────────┐            │
│     │  round(γ × DCT_coefficients)            │            │
│     │  大部分高频分量变为 0                      │            │
│     └─────────────────────────────────────────┘            │
│                          │                                  │
│                          ▼                                  │
│  4. 字节对编码 (BPE) 压缩                                       │
│     ┌─────────────────────────────────────────┐            │
│     │  扁平化 → BPE 压缩 → 稠密 Token 序列         │            │
│     │  350 维 → ~30-60 Tokens                  │            │
│     └─────────────────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 数学表达

#### 3.4.1 离散余弦变换 (DCT)

对于动作序列 $a = [a_1, a_2, ..., a_H]$:

$$X_k = \sum_{n=1}^{H} a_n \cos\left[\frac{\pi}{H}\left(n-\frac{1}{2}\right)k\right]$$

其中:
- $X_k$: 第 $k$ 个频率分量
- $k = 0, 1, ..., H-1$

#### 3.4.2 量化

$$\bar{X}_k = \text{round}(\gamma \cdot X_k)$$

其中 $\gamma$ 是缩放超参数，控制压缩率与重建精度的权衡。

#### 3.4.3 压缩率

| 数据集 | 动作维度 | 控制频率 | 原始 Tokens | FAST Tokens | 压缩比 |
|--------|----------|----------|-------------|-------------|--------|
| BridgeV2 | 7 | 5 Hz | 35 | 20 | 1.75× |
| DROID | 7 | 15 Hz | 105 | 29 | 3.6× |
| 桌面清理 | 7 | 20 Hz | 140 | 28 | 5.0× |
| T 恤折叠 | 14 | 50 Hz | 700 | 53 | 13.2× |

### 3.5 FAST+ 通用 Tokenizer

**特点**:
- 在 1M 真实机器人动作轨迹上训练
- 支持多种机器人形态 (单臂、双臂、移动机器人)
- 支持不同动作空间和控制频率
- 可作为黑盒 Tokenizer 直接使用

**使用示例**:
```python
from transformers import AutoProcessor

# 加载预训练的 FAST+ tokenizer
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True
)

# Tokenize 动作块
tokens = tokenizer(action_chunk)

# 或在新数据集上训练自定义 tokenizer
new_tokenizer = tokenizer.fit(action_dataset)
```

### 3.6 实验结果

#### 3.6.1 训练效率对比

| 方法 | 训练时间 | 性能 |
|------|----------|------|
| Naive Binning | 100% | 基准 |
| FSQ (向量量化) | 85% | +5% |
| **FAST** | **20%** | **+12%** |
| π₀ (扩散) | 100% | 基准 |
| **π₀-FAST** | **20%** | **相当** |

#### 3.6.2 高频任务性能

| 任务 | 频率 | Naive | FAST |
|------|------|-------|------|
| 桌面清理 | 20 Hz | 0% | 85% |
| T 恤折叠 | 50 Hz | 0% | 78% |
| 从烤面包机取出面包 | 50 Hz | 0% | 73% |

**关键发现**:
1. FAST 使自回归 VLA 能够在高频灵巧任务上训练
2. 训练速度提升 5 倍，性能与扩散 VLA 相当
3. 首次实现在 DROID 数据集上的零样本泛化

---

## 4. RTC: 实时动作分块执行

### 4.1 问题背景

**挑战**: 大模型推理延迟导致机器人动作衔接时出现停顿。

**现状**:
- π₀ 推理延迟: ~100ms (RTX 4090)
- 控制频率: 50 Hz (Δt = 20ms)
- 动作块大小: H = 50 (1 秒)
- 执行 Horizon: s = 25 (0.5 秒)

**问题**: 当推理延迟 δ > Δt 时，同步推理会在动作块之间引入可见的停顿。

### 4.2 RTC 核心思想

**Real-Time Chunking (RTC)**

**关键洞察**: 将异步动作分块视为**图像修复 (Inpainting)** 问题。

**方法**:
1. 在执行当前动作块时并行生成下一个动作块
2. "冻结" 保证会执行的动作
3. "修复" 剩余的动作

### 4.3 RTC 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│                    RTC 实时分块算法流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  时间线:                                                      │
│  t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7       │
│   │      │      │      │      │      │      │      │        │
│   ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼        │
│  ┌─────────────────────────────────────┐                   │
│  │  动作块 0 (正在执行)                   │                   │
│  │  [a₀, a₁, a₂, a₃, a₄, a₅, ...]       │                   │
│  └─────────────────────────────────────┘                   │
│         │                           │                       │
│         │ 推理开始 (t=2)              │ 推理完成 (t=6)        │
│         ▼                           ▼                       │
│                  ┌─────────────────────────────────────┐   │
│                  │  动作块 1 (生成中)                     │   │
│                  │  [?, ?, ?, ?, a₄', a₅', ...]        │   │
│                  └─────────────────────────────────────┘   │
│                                                             │
│  冻结区域 (Frozen): a₀:₃ (必须执行)                           │
│  软掩码区域 (Soft): a₄:₁₀ (可更新)                           │
│  新生成区域 (New): a₁₁:₅₀ (完全生成)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 数学表达

#### 4.4.1 推理延迟

$$d = \lfloor \delta / \Delta t \rfloor$$

其中:
- $\delta$: 推理时间
- $\Delta t$: 控制器采样周期

#### 4.4.2 软掩码 (Soft Masking)

$$\mathbf{W}_i = \begin{cases}
1 & \text{if } i < d \\
c_i \frac{e^{c_i} - 1}{e - 1} & \text{if } d \leq i < H - s \\
0 & \text{if } i \geq H - s
\end{cases}$$

其中 $c_i = \frac{H - s - i}{H - s - d + 1}$

#### 4.4.3 引导推理 (Guided Inference)

使用 ΠGDM (Pseudoinverse Guidance) 进行修复:

$$\mathbf{v}_{\Pi\text{GDM}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) + \min\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)(\mathbf{Y} - \widehat{\mathbf{A}_t^1})^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}_t^1}}{\partial \mathbf{A}_t^\tau}$$

其中:
- $\mathbf{v}$: 学习的速度场
- $\mathbf{Y}$: 目标值 (前一动作块的剩余部分)
- $\mathbf{W}$: 软掩码权重
- $\beta$: 引导权重裁剪 (防止不稳定)

### 4.5 RTC 算法伪代码

```python
class RealTimeChunking:
    def __init__(self, policy, H=50, s_min=25, n_denoise=5):
        self.policy = policy
        self.H = H  # 预测 horizon
        self.s_min = s_min  # 最小执行 horizon
        self.n_denoise = n_denoise  # 去噪步数
        self.delay_buffer = []
        
    def get_action(self, observation):
        """控制器调用，每个 Δt 执行一次"""
        with mutex:
            self.current_obs = observation
            return self.current_chunk[self.t]
    
    def inference_loop(self):
        """后台推理线程"""
        while True:
            # 等待到执行 horizon
            wait_until(self.t >= self.s)
            
            # 获取前一动作块的剩余部分
            A_prev = self.current_chunk[self.s:]
            
            # 估计推理延迟
            d = max(self.delay_buffer)
            
            # 执行 horizon
            s = max(d, self.s_min)
            
            # 引导推理 (修复)
            A_new = self.guided_inference(
                self.current_obs, A_prev, d, s
            )
            
            # 更新当前动作块
            with mutex:
                self.current_chunk = A_new
                self.t = 0
            
            # 记录实际延迟
            self.delay_buffer.append(actual_delay)
    
    def guided_inference(self, obs, A_prev, d, s):
        """带软掩码的引导推理"""
        # 计算软掩码权重
        W = self.compute_soft_mask(d, s, self.H)
        
        # 从噪声开始
        A = torch.randn(1, self.H, action_dim)
        
        # 迭代去噪
        for tau in torch.linspace(0, 1, self.n_denoise):
            # 估计最终动作块
            A_hat = A + (1 - tau) * self.policy.velocity(A, obs, tau)
            
            # 计算加权误差
            error = (A_prev - A_hat) * W
            
            # 计算梯度 (向量 - 雅可比积)
            grad = torch.autograd.grad(
                A_hat, A, error, retain_graph=True
            )[0]
            
            # 更新动作
            velocity = self.policy.velocity(A, obs, tau)
            A = A + (1/self.n_denoise) * (
                velocity + min(beta, ...) * grad
            )
        
        return A
```

### 4.6 实验结果

#### 4.6.1 模拟环境 (Kinetix)

| 方法 | 延迟=0 | 延迟=1 | 延迟=2 | 延迟=4 |
|------|--------|--------|--------|--------|
| Naive Async | 0.85 | 0.62 | 0.41 | 0.23 |
| Temporal Ensembling | 0.72 | 0.55 | 0.38 | 0.19 |
| BID | 0.88 | 0.75 | 0.65 | 0.52 |
| **RTC (硬掩码)** | 0.89 | 0.82 | 0.76 | 0.68 |
| **RTC (软掩码)** | **0.91** | **0.87** | **0.83** | **0.75** |

#### 4.6.2 真实机器人任务

| 任务 | 同步推理 | TE (稀疏) | TE (密集) | **RTC** |
|------|----------|-----------|-----------|---------|
| 点燃蜡烛 | 0.45 | 0.32 | - | **0.78** |
| 插入以太网线 | 0.52 | 0.38 | - | **0.81** |
| 整理床铺 (移动) | 0.38 | 0.25 | - | **0.65** |
| T 恤折叠 | 0.72 | 0.68 | - | **0.79** |
| 批量折叠 | 0.41 | 0.35 | - | **0.58** |
| 盘子放入水槽 (移动) | 0.55 | 0.48 | - | **0.71** |

**关键发现**:
1. RTC 在 +200ms 注入延迟下性能无下降
2. 任务吞吐量提升 20%
3. TE 方法在高延迟下会导致机器人触发保护停止

---

## 5. 知识隔离 VLA 技术

### 5.1 问题背景

**挑战**: 在 VLA 微调过程中，动作训练会破坏 VLM 骨干网络原有的语义知识。

**现象**:
- 语言跟随能力下降
- 泛化能力减弱
- 训练收敛变慢

**原因**: 动作专家的梯度反向传播到 VLM 骨干网络，干扰了预训练的语义表示。

### 5.2 知识隔离 (Knowledge Insulation) 核心思想

**关键洞察**: 
1. **停止梯度流** - 防止动作专家梯度破坏 VLM 骨干
2. **FAST Token 表示学习** - 用离散动作 Token 训练骨干网络学习运动表示
3. **联合训练** - 同时训练 VLM 数据保持语义知识

### 5.3 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                π₀.₅ + KI 模型架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │   视觉输入    │    │   语言指令    │                      │
│  └──────┬───────┘    └──────┬───────┘                      │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           VLM Backbone (3B 参数)                     │   │
│  │                                                      │   │
│  │  训练信号 1: FAST Token (离散动作) ←── 梯度流通        │   │
│  │  训练信号 2: VLM 数据 (语言 + 图像)   ←── 梯度流通        │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                               │
│                             │ 梯度停止 ─────┐               │
│                             ▼               │               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Action Expert (300M 参数)               │   │
│  │         (基于流匹配的连续动作生成)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              连续动作输出                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 训练策略

#### 5.4.1 三阶段训练

**阶段 1: VLM 预训练**
- 在大规模图像 - 文本数据上预训练
- 学习通用视觉和语言表示

**阶段 2: FAST Token 表示学习**
- 使用 FAST Tokenized 动作训练 VLM 骨干
- 学习运动控制表示，不破坏语义知识
- 损失函数: 交叉熵损失

**阶段 3: 动作专家训练**
- 训练动作专家生成连续动作
- 梯度不反向传播到 VLM 骨干
- 损失函数: 流匹配损失

#### 5.4.2 联合训练损失

$$\mathcal{L}_{total} = \mathcal{L}_{FAST} + \mathcal{L}_{flow} + \lambda \mathcal{L}_{VLM}$$

其中:
- $\mathcal{L}_{FAST}$: FAST Token 预测损失 (骨干网络)
- $\mathcal{L}_{flow}$: 连续动作流匹配损失 (动作专家)
- $\mathcal{L}_{VLM}$: VLM 语言理解损失 (保持语义知识)

### 5.5 实验结果

#### 5.5.1 训练效率

| 方法 | 训练步数 | 相对时间 |
|------|----------|----------|
| π₀ | 160K | 100% |
| π₀-FAST | 160K | 20% |
| **π₀.₅ + KI** | **20K** | **13%** |

#### 5.5.2 性能对比

| 任务 | π₀.₅ + KI | 联合训练 | π₀-FAST | π₀ |
|------|-----------|----------|---------|-----|
| T 恤折叠 | **0.82** | 0.75 | 0.78 | 0.78 |
| 物品放入抽屉 | **0.88** | 0.81 | 0.72 | 0.65 |
| 桌面清理 | **0.91** | 0.85 | 0.87 | 0.85 |

#### 5.5.3 语言跟随能力

| 条件 | π₀.₅ + KI | 联合训练 (无 VLM 数据) |
|------|-----------|----------------------|
| 分布内语言跟随 | 0.95 | 0.82 |
| 分布外语言跟随 | 0.88 | 0.65 |
| 分布内性能 | 0.91 | 0.85 |
| 分布外性能 | 0.85 | 0.68 |

**关键发现**:
1. 训练速度提升 7.5 倍 (相比 π₀)
2. 保持 VLM 的语义知识和语言跟随能力
3. 推理速度与 π₀ 相同 (使用动作专家)

---

## 6. π₀.₅: 开放世界泛化

### 6.1 核心创新

**目标**: 使 VLA 模型能够在未见过的环境中执行任务 (如新家庭的厨房)。

**方法**: 跨任务协同训练 (Co-training)

### 6.2 训练数据混合

| 数据源 | 用途 | 比例 |
|--------|------|------|
| 网络数据 (WD) | 语义知识 | 30% |
| 多环境数据 (ME) | 环境泛化 | 25% |
| 跨形态数据 (CE) | 机器人泛化 | 25% |
| 移动操作数据 | 目标任务 | 20% |

### 6.3 多模态输入

```
输入 = {
    "图像": camera_observation,
    "语言指令": "把盘子里的东西放进水槽",
    "物体检测": [bbox₁, bbox₂, ...],
    "语义子任务": "拿起盘子",
    "动作": [a₁, a₂, ..., a₅₀]
}
```

### 6.4 链式思考推理

**高层推理** (文本):
```
输入: "清理厨房"
输出: "1. 拿起脏盘子 → 2. 放入水槽 → 3. 拿起杯子 → 4. 放入水槽"
```

**低层控制** (动作):
```
输入: "拿起脏盘子" + 当前图像
输出: [关节位置₁, ..., 关节位置₅₀]
```

### 6.5 实验结果

#### 6.5.1 新家庭环境泛化

| 训练环境数量 | 成功率 |
|--------------|--------|
| 1 | 0.35 |
| 10 | 0.62 |
| 50 | 0.78 |
| 100 | 0.85 |
| 全部 (基线) | 0.88 |

**发现**: 仅需约 100 个训练环境即可接近在测试环境上直接训练的性能。

#### 6.5.2 零样本任务执行

| 任务 | 成功率 |
|------|--------|
| 清理厨房 | 0.75 |
| 整理卧室 | 0.68 |
| 收拾桌子 | 0.82 |

---

## 7. π*₀.₆: 从经验中学习

### 7.1 RECAP 框架

**RL with Experience and Corrections via Advantage-conditioned Policies**

**目标**: 通过离线强化学习和人类纠错数据，使模型能够胜任高难度物理操作任务。

### 7.2 核心方法

#### 7.2.1 优势条件策略

$$\pi(a|s, A) = \pi_{base}(a|s) \cdot \exp(\beta \cdot A(s, a))$$

其中:
- $\pi_{base}$: 基础 VLA 策略
- $A(s, a)$: 优势函数
- $\beta$: 温度参数

#### 7.2.2 数据混合

| 数据类型 | 用途 | 比例 |
|----------|------|------|
| 演示数据 | 基础技能 | 40% |
| 在线收集数据 | 策略改进 | 30% |
| 人类纠错数据 | 边界情况 | 30% |

### 7.3 实验结果

| 任务 | π₀.₅ | π*₀.₆ (RECAP) | 提升 |
|------|------|---------------|------|
| 折叠丝滑衣物 | 0.35 | **0.72** | 2.1× |
| 制作浓缩咖啡 | 0.28 | **0.61** | 2.2× |
| 组装盒子 | 0.52 | **0.78** | 1.5× |

**关键指标**:
- 任务吞吐量提升 **2 倍**
- 失败率降低 **50%**

---

## 8. 人类到机器人技能迁移

### 8.1 核心发现

**现象**: 随着 VLA 模型规模扩大，从人类视频中学习技能的能力会**涌现**。

### 8.2 方法

**简单人类 - 机器人协同微调**:
- 将人类视频数据视为另一种机器人形态
- 动作表示为 3D 手部位置
- 无需特殊迁移学习机制

### 8.3 规模效应

| 预训练数据规模 | 人类数据提升 |
|----------------|--------------|
| 25% | +15% |
| 50% | +45% |
| 75% | +85% |
| 100% | +100% |

**发现**: 增加机器人预训练数据实际上提高了模型吸收人类数据的能力。

### 8.4 表示对齐

通过 t-SNE 可视化发现:
- 小规模预训练: 人类和机器人数据表示分离
- 大规模预训练: 人类和机器人数据表示对齐

---

## 9. 核心算法数学公式与流程对照表

### 9.1 流匹配动作生成

| 步骤 | 数学表达 | 代码实现 |
|------|----------|----------|
| 1. 采样噪声 | $\mathbf{A}^0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | `A = torch.randn(...)` |
| 2. 速度场预测 | $\mathbf{v}_\pi(\mathbf{A}^\tau, \mathbf{o}, \tau)$ | `velocity = policy(A, obs, tau)` |
| 3. Euler 积分 | $\mathbf{A}^{\tau+\frac{1}{n}} = \mathbf{A}^\tau + \frac{1}{n}\mathbf{v}_\pi(...)$ | `A = A + (1/n) * velocity` |
| 4. 重复 n 次 | $\tau \in [0, 1]$ | `for tau in linspace(0, 1, n):` |

### 9.2 FAST Tokenization

| 步骤 | 数学表达 | 代码实现 |
|------|----------|----------|
| 1. 归一化 | $a' = \frac{a - q_1}{q_{99} - q_1} \times 2 - 1$ | `quantile_normalize(actions)` |
| 2. DCT | $X_k = \sum a_n \cos[\frac{\pi}{H}(n-\frac{1}{2})k]$ | `dct(actions, axis=0)` |
| 3. 量化 | $\bar{X}_k = \text{round}(\gamma \cdot X_k)$ | `torch.round(gamma * X)` |
| 4. BPE 压缩 | $\text{BPE}([\bar{X}_1, ..., \bar{X}_H])$ | `bpe_encoder.encode(...)` |

### 9.3 RTC 软掩码

| 步骤 | 数学表达 | 代码实现 |
|------|----------|----------|
| 1. 冻结区域 | $\mathbf{W}_i = 1, i < d$ | `W[:d] = 1.0` |
| 2. 指数衰减 | $\mathbf{W}_i = c_i \frac{e^{c_i}-1}{e-1}$ | `W[d:H-s] = decay(...)` |
| 3. 新生成区域 | $\mathbf{W}_i = 0, i \geq H-s$ | `W[H-s:] = 0.0` |
| 4. 引导更新 | $\mathbf{A}^{\tau+\frac{1}{n}} = ... + \min(\beta, ...) \cdot \mathbf{g}$ | `A += guidance * grad` |

---

## 10. 代码实现参考

### 10.1 π₀ 流匹配策略

```python
import torch
import torch.nn as nn

class FlowMatchingPolicy(nn.Module):
    def __init__(self, vlm_backbone, action_dim, hidden_dim=512):
        super().__init__()
        self.vlm = vlm_backbone
        self.action_expert = nn.Sequential(
            nn.Linear(vlm_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def velocity_field(self, action_chunk, observation, tau):
        """预测速度场"""
        # VLM 编码
        vlm_features = self.vlm(observation['image'], observation['text'])
        
        # 拼接动作和 VLM 特征
        x = torch.cat([vlm_features, action_chunk], dim=-1)
        
        # 预测速度
        velocity = self.action_expert(x)
        
        return velocity
    
    def generate(self, observation, num_steps=5):
        """生成动作块"""
        batch_size = 1
        H = 50  # 动作块大小
        
        # 从噪声开始
        action_chunk = torch.randn(batch_size, H, self.action_dim)
        
        # 迭代去噪
        for tau in torch.linspace(0, 1, num_steps):
            velocity = self.velocity_field(action_chunk, observation, tau)
            action_chunk = action_chunk + (1/num_steps) * velocity
        
        return action_chunk
    
    def train_step(self, batch):
        """训练一步"""
        observation = batch['observation']
        action_0 = batch['action_0']  # 初始动作
        action_1 = batch['action_1']  # 目标动作
        
        # 随机时间步
        tau = torch.rand(batch_size)
        
        # 插值
        action_tau = (1 - tau) * action_0 + tau * action_1
        
        # 目标速度
        target_velocity = action_1 - action_0
        
        # 预测速度
        predicted_velocity = self.velocity_field(action_tau, observation, tau)
        
        # 流匹配损失
        loss = nn.MSELoss()(predicted_velocity, target_velocity)
        
        return loss
```

### 10.2 FAST Tokenizer

```python
import numpy as np
from scipy.fftpack import dct

class FASTTokenizer:
    def __init__(self, action_dim, chunk_size=50, gamma=10):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.gamma = gamma
        
    def normalize(self, actions):
        """分位数归一化"""
        q1 = np.quantile(actions, 0.01, axis=0)
        q99 = np.quantile(actions, 0.99, axis=0)
        normalized = (actions - q1) / (q99 - q1 + 1e-8) * 2 - 1
        return np.clip(normalized, -1, 1)
    
    def encode(self, actions):
        """编码动作块为 tokens"""
        # 1. 归一化
        actions = self.normalize(actions)
        
        # 2. DCT (每维独立)
        dct_coeffs = np.zeros_like(actions)
        for i in range(self.action_dim):
            dct_coeffs[:, i] = dct(actions[:, i], type=2, norm='ortho')
        
        # 3. 量化
        quantized = np.round(self.gamma * dct_coeffs).astype(int)
        
        # 4. 扁平化 (低频优先)
        flat = []
        for freq_idx in range(self.chunk_size):
            for dim_idx in range(self.action_dim):
                flat.append(quantized[freq_idx, dim_idx])
        
        # 5. BPE 压缩 (简化版)
        tokens = self.bpe_encode(flat)
        
        return tokens
    
    def decode(self, tokens):
        """解码 tokens 为动作块"""
        # 1. BPE 解码
        flat = self.bpe_decode(tokens)
        
        # 2. 重构 DCT 系数矩阵
        dct_coeffs = np.zeros((self.chunk_size, self.action_dim))
        idx = 0
        for freq_idx in range(self.chunk_size):
            for dim_idx in range(self.action_dim):
                dct_coeffs[freq_idx, dim_idx] = flat[idx]
                idx += 1
        
        # 3. 逆 DCT
        actions = np.zeros_like(dct_coeffs)
        for i in range(self.action_dim):
            actions[:, i] = dct(dct_coeffs[:, i], type=3, norm='ortho')
        
        # 4. 逆归一化
        actions = (actions + 1) / 2  # 简化版
        
        return actions
    
    def bpe_encode(self, sequence):
        """简化的 BPE 编码"""
        # 实际实现会使用训练好的 BPE 词表
        return sequence  # 占位符
    
    def bpe_decode(self, tokens):
        """简化的 BPE 解码"""
        return tokens  # 占位符
```

### 10.3 RTC 实时分块

```python
import threading
import queue

class RealTimeChunking:
    def __init__(self, policy, H=50, s_min=25, n_denoise=5):
        self.policy = policy
        self.H = H
        self.s_min = s_min
        self.n_denoise = n_denoise
        
        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)
        
        self.current_chunk = None
        self.current_obs = None
        self.t = 0
        
        self.delay_buffer = queue.Queue(maxsize=10)
        
        # 启动推理线程
        self.inference_thread = threading.Thread(
            target=self.inference_loop, daemon=True
        )
        self.inference_thread.start()
    
    def get_action(self, observation):
        """控制器调用"""
        with self.mutex:
            self.current_obs = observation
            self.t += 1
            self.condition.notify()
            
            if self.current_chunk is not None:
                return self.current_chunk[self.t - 1]
            else:
                return np.zeros(self.policy.action_dim)
    
    def inference_loop(self):
        """后台推理线程"""
        while True:
            with self.mutex:
                # 等待到执行 horizon
                self.condition.wait_for(
                    lambda: self.t >= self.s_min
                )
                
                s = self.t
                A_prev = self.current_chunk[s:].copy()
                obs = self.current_obs.copy()
                
                # 估计延迟
                d = max(list(self.delay_buffer.queue)) if not self.delay_buffer.empty() else 0
            
            # 释放锁进行推理
            A_new = self.guided_inference(obs, A_prev, d, s)
            
            with self.mutex:
                self.current_chunk = A_new
                self.t = 0
                self.delay_buffer.put(self.t)
    
    def compute_soft_mask(self, d, s, H):
        """计算软掩码权重"""
        W = np.zeros(H)
        
        # 冻结区域
        W[:d] = 1.0
        
        # 指数衰减区域
        for i in range(d, H - s):
            c_i = (H - s - i) / (H - s - d + 1)
            W[i] = c_i * (np.exp(c_i) - 1) / (np.e - 1)
        
        # 新生成区域
        W[H - s:] = 0.0
        
        return W
    
    def guided_inference(self, obs, A_prev, d, s):
        """带软掩码的引导推理"""
        W = self.compute_soft_mask(d, s, self.H)
        
        # 右对齐前一动作块
        A_prev_padded = np.zeros((self.H, self.policy.action_dim))
        A_prev_padded[d:d+len(A_prev)] = A_prev
        
        # 从噪声开始
        A = torch.randn(1, self.H, self.policy.action_dim)
        
        for tau in torch.linspace(0, 1, self.n_denoise):
            # 估计最终动作块
            velocity = self.policy.velocity_field(A, obs, tau)
            A_hat = A + (1 - tau) * velocity
            
            # 加权误差
            error = (torch.from_numpy(A_prev_padded) - A_hat) * torch.from_numpy(W)
            
            # 计算梯度
            grad = torch.autograd.grad(
                A_hat, A, error, retain_graph=True
            )[0]
            
            # 引导权重裁剪
            r_tau = (1 - tau)**2 / (tau**2 + (1 - tau)**2)
            beta_eff = min(1.0, (1 - tau) / (tau * r_tau))
            
            # 更新
            A = A + (1/self.n_denoise) * (
                velocity + beta_eff * grad
            )
        
        return A.detach().numpy()
```

---

## 11. 迁移部署适配指南

### 11.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| CPU | 8 核 | 16 核 |
| 内存 | 32GB | 64GB |
| 存储 | 500GB SSD | 1TB NVMe |

### 11.2 软件依赖

```yaml
# requirements.yml
python: ">=3.9"
pytorch: ">=2.0"
transformers: ">=4.35"
diffusers: ">=0.24"
opencv-python: ">=4.8"
numpy: ">=1.24"
```

### 11.3 部署步骤

#### 步骤 1: 模型加载

```python
from transformers import AutoModelForCausalLM

# 加载 π₀ 模型
model = AutoModelForCausalLM.from_pretrained(
    "physical-intelligence/pi0",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model = model.cuda().eval()
```

#### 步骤 2: Tokenizer 配置

```python
from transformers import AutoProcessor

# 加载 FAST+ tokenizer
tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True
)
```

#### 步骤 3: RTC 配置

```python
# 配置实时分块
rtc = RealTimeChunking(
    policy=model,
    H=50,        # 预测 horizon
    s_min=25,    # 最小执行 horizon
    n_denoise=5  # 去噪步数
)
```

#### 步骤 4: 推理循环

```python
while True:
    # 获取观测
    observation = {
        'image': camera.get_image(),
        'text': language_instruction,
        'proprio': robot.get_state()
    }
    
    # 获取动作 (非阻塞)
    action = rtc.get_action(observation)
    
    # 执行动作
    robot.execute(action[0])
```

### 11.4 性能优化

#### 11.4.1 模型量化

```python
# INT8 量化
from torch.ao.quantization import quantize_dynamic

model_quantized = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

#### 11.4.2 批处理优化

```python
# 使用更大的批处理
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

#### 11.4.3 缓存优化

```python
# KV Cache 预填充优化
model.config.use_cache = True
```

### 11.5 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 推理延迟高 | GPU 显存不足 | 减少批处理大小，使用量化 |
| 动作抖动 | RTC 掩码配置不当 | 调整软掩码衰减参数 |
| 语言跟随差 | VLM 数据不足 | 增加 VLM 联合训练比例 |
| 泛化能力弱 | 训练环境单一 | 增加多环境数据 |

---

## 12. 实验与训练流程

### 12.1 数据收集

#### 12.1.1 遥操作设置

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  主端设备     │────▶│  数据采集    │────▶│  机器人执行   │
│  (VR/手柄)   │     │  系统        │     │  (从端)      │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  操作者输入          记录 (图像，动作)        执行动作
```

#### 12.1.2 数据格式

```json
{
  "episode_id": "001",
  "task": "table_bussing",
  "trajectory": [
    {
      "timestep": 0,
      "image": "base64_encoded_image",
      "language": "清理桌子",
      "action": [0.1, 0.2, ..., 0.7],
      "proprio": [joint₁, joint₂, ...]
    },
    ...
  ]
}
```

### 12.2 训练流程

#### 12.2.1 预训练阶段

```bash
# VLM 预训练
python train_vlm.py \
    --data_path /data/vlm_pretrain \
    --model_size 3B \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --epochs 10
```

#### 12.2.2 VLA 微调阶段

```bash
# VLA 微调 (带知识隔离)
python train_vla.py \
    --checkpoint /checkpoints/vlm_3b \
    --data_path /data/robot_trajectories \
    --action_tokenizer fast \
    --knowledge_insulation true \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --epochs 5
```

#### 12.2.3 RECAP 强化学习阶段

```bash
# RECAP 训练
python train_recap.py \
    --checkpoint /checkpoints/pi05_ki \
    --data_path /data/experience \
    --correction_data /data/corrections \
    --beta 0.5 \
    --batch_size 32 \
    --epochs 3
```

### 12.3 评估协议

#### 12.3.1 模拟评估

```python
def evaluate_simulation(policy, env, num_episodes=100):
    success_count = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            action = policy.generate(obs)
            obs, reward, done, info = env.step(action[0])
        
        if info['success']:
            success_count += 1
    
    return success_count / num_episodes
```

#### 12.3.2 真实机器人评估

```python
def evaluate_real_world(policy, robot, task, num_trials=10):
    scores = []
    
    for trial in range(num_trials):
        score = 0
        obs = robot.get_observation()
        
        for step in range(max_steps):
            action = policy.generate(obs)
            robot.execute(action[0])
            
            # 评估子任务完成
            if robot.check_subtask_complete():
                score += 1
            
            obs = robot.get_observation()
            
            if robot.check_task_complete():
                break
        
        scores.append(score)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'success_rate': np.mean([s == max_score for s in scores])
    }
```

### 12.4 超参数调优

| 超参数 | 搜索范围 | 最优值 |
|--------|----------|--------|
| 学习率 | [1e-6, 1e-3] | 1e-5 |
| 批处理大小 | [16, 256] | 64 |
| DCT 缩放 γ | [5, 20] | 10 |
| RTC 引导权重 β | [0.1, 2.0] | 1.0 |
| 去噪步数 n | [3, 10] | 5 |

---

## 13. 总结与展望

### 13.1 技术总结

Physical Intelligence 的 VLA 研究代表了机器人基础模型的最前沿水平：

1. **π₀**: 首次将流匹配引入 VLA，实现高效连续动作生成
2. **FAST**: 解决自回归 VLA 的高频动作建模问题，训练效率提升 5 倍
3. **RTC**: 实现真正的实时推理，延迟鲁棒性显著提升
4. **知识隔离**: 保持 VLM 语义知识的同时学习运动控制
5. **π₀.₅/π*₀.₆**: 开放世界泛化和从经验中自改进

### 13.2 局限性

1. **计算资源**: 训练和推理仍需要高端 GPU
2. **数据需求**: 需要大量机器人操作数据
3. **安全保证**: 缺乏形式化的安全验证
4. **长时规划**: 超长 horizon 任务仍有挑战

### 13.3 未来方向

1. **多模态融合**: 更好的视觉 - 语言 - 触觉融合
2. **世界模型**: 学习物理世界 dynamics 用于规划
3. **人机协作**: 更自然的人类 - 机器人交互
4. **边缘部署**: 轻量化模型用于移动机器人
5. **自监督学习**: 减少对标注数据的依赖

### 13.4 工程建议

对于希望部署 Physical Intelligence 技术的团队:

1. **从 π₀-FAST 开始**: 训练效率最高，性能相当
2. **使用 RTC**: 对于实时性要求高的应用
3. **知识隔离**: 保持语言跟随能力
4. **增量部署**: 先在模拟环境验证，再迁移到真实机器人
5. **数据收集**: 持续收集新任务数据用于微调

---

## 参考文献

1. Black, K., et al. "π₀: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164, 2024.
2. Pertsch, K., et al. "FAST: Efficient Action Tokenization for Vision-Language-Action Models." arXiv:2501.09747, 2025.
3. Black, K., et al. "Real-Time Execution of Action Chunking Flow Policies." arXiv:2506.07339, 2025.
4. Driess, D., et al. "Knowledge Insulating Vision-Language-Action Models." arXiv:2505.23705, 2025.
5. Physical Intelligence. "π₀.₅: a Vision-Language-Action Model with Open-World Generalization." arXiv:2504.16054, 2025.
6. Physical Intelligence. "π*₀.₆: a VLA That Learns From Experience." arXiv:2511.14759, 2025.
7. Physical Intelligence. "Emergence of Human to Robot Transfer in VLAs." 2025.

---

*报告生成时间: 2026 年 3 月 3 日*
*版本: 1.0*
