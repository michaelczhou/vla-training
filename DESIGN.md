# VLA Training Framework - 设计文档

**版本**: v2.0  
**最后更新**: 2026-03-09  
**状态**: 生产中

---

## 目录

1. [概述](#1-概述)
2. [代码设计](#2-代码设计)
3. [模型架构](#3-模型架构)
4. [算法步骤和原理](#4-算法步骤和原理)
5. [数学原理](#5-数学原理)
6. [训练流程](#6-训练流程)
7. [推理部署](#7-推理部署)
8. [性能优化](#8-性能优化)

---

## 1. 概述

### 1.1 项目定位

VLA Training Framework 是一个生产级的视觉 - 语言 - 动作 (Vision-Language-Action) 模型训练框架，支持多种架构 (扩散、流匹配、自回归)，旨在为机器人学习提供统一、高效、可扩展的解决方案。

### 1.2 设计目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| **模块化** | 清晰的组件分离，易于扩展 | P0 |
| **高性能** | 支持分布式训练，推理优化 | P0 |
| **多架构支持** | 扩散、流匹配、自回归 | P0 |
| **易用性** | 简洁的 API，完善的文档 | P1 |
| **可复现** | 确定性训练，版本控制 | P1 |

### 1.3 核心特性

- ✅ **多动作头支持**: 流匹配 (推荐)、扩散策略、MLP 回归
- ✅ **混合精度训练**: AMP (Automatic Mixed Precision)
- ✅ **梯度累积**: 模拟大批次训练
- ✅ **学习率调度**: Cosine、Linear、Constant
- ✅ **检查点管理**: 自动保存、恢复训练
- ✅ **多数据集支持**: RLDS、LeRobot 格式

---

## 2. 代码设计

### 2.1 项目结构

```
vla-training/
├── src/                        # 源代码
│   ├── models/                 # 模型定义
│   │   ├── vla_model.py        # VLA 主模型
│   │   ├── pi0_model.py        # π₀模型实现
│   │   ├── rdt2_model.py       # RDT2 模型
│   │   ├── vision_encoder.py   # 视觉编码器
│   │   ├── language_model.py   # 语言模型
│   │   ├── fusion_module.py    # 特征融合
│   │   ├── action_head.py      # 动作头 (扩散/流匹配/MLP)
│   │   └── fast_tokenizer.py   # FAST 分词器
│   │
│   ├── data/                   # 数据模块
│   │   ├── dataset.py          # 数据集类 (RLDS/LeRobot)
│   │   ├── dataloader.py       # 数据加载器
│   │   └── transforms.py       # 数据增强
│   │
│   ├── training/               # 训练模块
│   │   ├── trainer.py          # 训练器
│   │   ├── optimizer.py        # 优化器构建
│   │   ├── losses.py           # 损失函数
│   │   └── scheduler.py        # 学习率调度
│   │
│   ├── inference/              # 推理模块
│   │   ├── policy.py           # 策略接口
│   │   └── deploy.py           # 部署工具
│   │
│   └── utils/                  # 工具函数
│       ├── config.py           # 配置管理
│       ├── checkpoint.py       # 检查点管理
│       └── logger.py           # 日志工具
│
├── configs/                    # 配置文件
│   ├── model/                  # 模型配置
│   ├── training/               # 训练配置
│   └── data/                   # 数据配置
│
├── scripts/                    # 脚本入口
│   ├── train.py                # 训练脚本
│   └── inference.py            # 推理脚本
│
├── tests/                      # 单元测试
└── examples/                   # 使用示例
```

### 2.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                      模块依赖图                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  train.py                                                   │
│    │                                                        │
│    ▼                                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Trainer   │───►│   Model     │───►│   Dataset   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Optimizer  │    │ Action Head │    │ Transforms  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                                │
│         ▼                  ▼                                │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  Scheduler  │    │   Losses    │                        │
│  └─────────────┘    └─────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 核心类设计

**VLAModel** - 主模型类:
```python
class VLAModel(nn.Module):
    """
    VLA 主模型
    
    组件:
    - VisionEncoder: 视觉编码 (ViT/SigLIP/ResNet)
    - LanguageModel: 语言编码 (Gemma/Qwen/LLaMA)
    - FusionModule: 特征融合 (Cross-Attn/Concat)
    - ActionHead: 动作生成 (Flow/Diffusion/MLP)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 初始化各组件
        pass
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Returns:
            loss: 训练损失 (如果提供 actions)
            predictions: 动作预测
        """
        pass
```

**VLATrainer** - 训练器类:
```python
class VLATrainer:
    """
    VLA 训练器
    
    功能:
    - 混合精度训练
    - 梯度累积
    - 梯度裁剪
    - 学习率调度
    - 检查点管理
    """
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        pass
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        pass
    
    def train(self) -> Dict[str, Any]:
        """完整训练循环"""
        pass
```

---

## 3. 模型架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     VLA 架构 v2.0                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Images     │    │   Text       │    │   Proprio    │  │
│  │  (多视角)     │    │   Prompt     │    │    State     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   │          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐  │
│  │   Vision     │    │   Language   │    │   State      │  │
│  │   Encoder    │    │   Encoder    │    │   Encoder    │  │
│  │  (ViT/SigLIP)│    │ (Gemma/Qwen) │    │   (MLP)      │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └────────┬──────────┘                   │          │
│                  ▼                              │          │
│         ┌────────────────┐                      │          │
│         │   Fusion       │◄─────────────────────┘          │
│         │   (Cross-Attn) │                                 │
│         └────────┬───────┘                                 │
│                  │                                         │
│                  ▼                                         │
│         ┌────────────────┐                                 │
│         │   Action Head  │                                 │
│         │ (Flow/Diff/MLP)│                                 │
│         └────────┬───────┘                                 │
│                  │                                         │
│                  ▼                                         │
│         ┌────────────────┐                                 │
│         │   Actions      │                                 │
│         └────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 视觉编码器

**支持的架构**:

| 模型 | 参数量 | 输入分辨率 | 输出维度 | 推荐场景 |
|------|--------|-----------|---------|---------|
| **ViT-Base** | 86M | 224×224 | 768 | 通用 |
| **ViT-Large** | 307M | 224×224 | 1024 | 高精度 |
| **SigLIP** | 428M | 384×384 | 768 | SOTA |
| **ResNet-50** | 25M | 224×224 | 2048 | 边缘部署 |

**实现**:
```python
class VisionEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_type = config.get('type', 'vit_base')
        
        if model_type == 'vit_base':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.output_dim = 768
        elif model_type == 'siglip':
            self.model = SigLIPModel.from_pretrained('siglip-so400m-patch14-384')
            self.output_dim = 768
        elif model_type == 'resnet50':
            self.model = resnet50(pretrained=True)
            self.output_dim = 2048
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 编码图像
        features = self.model(images)
        return features
```

### 3.3 语言编码器

**支持的模型**:

| 模型 | 参数量 | 上下文长度 | 推荐场景 |
|------|--------|-----------|---------|
| **Gemma-2B** | 2B | 8K | 边缘部署 |
| **Gemma-7B** | 7B | 8K | 通用 |
| **Qwen-7B** | 7B | 32K | 中文任务 |
| **LLaMA-2-7B** | 7B | 4K | 通用 |

### 3.4 融合模块

**Cross-Attention 融合** (推荐):
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        # 投影
        vision_proj = self.vision_proj(vision_features)
        language_proj = self.language_proj(language_features)
        
        # Cross-Attention (语言作为 Query, 视觉作为 Key/Value)
        fused, _ = self.cross_attn(
            query=language_proj,
            key=vision_proj,
            value=vision_proj
        )
        
        return fused
```

### 3.5 动作头

#### 3.5.1 流匹配头 (推荐)

**优势**:
- 推理速度快 (10 步 vs 100 步)
- 训练稳定
- 理论优雅

**数学原理**:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]$$

其中:
- $x_0 \sim \mathcal{N}(0, I)$ (噪声)
- $x_1 \sim p_{\text{data}}$ (数据)
- $x_t = (1-t)x_0 + tx_1$ (插值)
- $v_\theta$: 预测的速度场

**实现**:
```python
class FlowMatchingHead(ActionHead):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 速度预测网络
        self.velocity_net = nn.Sequential(
            nn.Linear(input_dim + action_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        # 预测速度场
        inputs = torch.cat([x, action, self.time_embed(t)], dim=-1)
        velocity = self.velocity_net(inputs)
        return velocity
    
    def compute_loss(
        self,
        x: torch.Tensor,
        action_0: torch.Tensor,
        action_1: torch.Tensor
    ) -> torch.Tensor:
        # 采样时间
        t = torch.rand(action_0.shape[0], device=x.device)
        
        # 插值
        action_t = t * action_1 + (1 - t) * action_0
        
        # 目标速度
        target_velocity = action_1 - action_0
        
        # 预测速度
        predicted_velocity = self(x, t, action_t)
        
        # MSE 损失
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        # Euler 积分
        dt = 1.0 / num_steps
        action = torch.randn(...)
        
        for i in range(num_steps):
            t = torch.ones(...) * (i * dt)
            velocity = self(x, t, action)
            action = action + velocity * dt
        
        return action
```

#### 3.5.2 扩散头

**数学原理**:
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

**实现**: 参见 `src/models/action_head.py`

#### 3.5.3 MLP 头

**最简单的回归头**:
```python
class MLPHead(ActionHead):
    def __init__(self, config: Dict[str, Any]):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=1))
```

---

## 4. 算法步骤和原理

### 4.1 训练算法

**算法 1: VLA 训练流程**

```
输入：训练数据集 D, 模型 M, 配置 C
输出：训练好的模型 M*

1. 初始化:
   - 构建数据加载器 L
   - 初始化优化器 O
   - 初始化学习率调度器 S

2. for epoch = 1 to C.num_epochs do:
   
   3. for batch in L do:
      
      4. 前向传播:
         - 编码图像：v = VisionEncoder(images)
         - 编码语言：l = LanguageModel(text)
         - 融合特征：f = Fusion(v, l)
         - 预测动作：a_pred = ActionHead(f)
      
      5. 计算损失:
         - L = Loss(a_pred, a_true)
      
      6. 反向传播:
         - 计算梯度：∇L
         - 梯度裁剪：clip(∇L)
         - 更新参数：O.step(∇L)
      
      7. 更新学习率:
         - S.step()
   
   8. 验证 (可选):
      - val_loss = Validate(M, D_val)
   
   9. 保存检查点:
      - if epoch % C.save_every == 0:
        - SaveCheckpoint(M, epoch)

10. 返回 M*
```

### 4.2 流匹配算法

**算法 2: 条件流匹配训练**

```
输入：条件特征 x, 目标动作 a_1
输出：训练好的速度场 v_θ

1. 采样噪声: a_0 ~ N(0, I)
2. 采样时间: t ~ U(0, 1)
3. 计算插值：a_t = (1-t)*a_0 + t*a_1
4. 计算目标速度：u = a_1 - a_0
5. 预测速度：v_θ = VelocityNet(x, a_t, t)
6. 计算损失：L = ||v_θ - u||²
7. 反向传播，更新参数
```

**算法 3: 流匹配推理 (Euler 积分)**

```
输入：条件特征 x, 步数 N
输出：生成的动作 a

1. 初始化：a ~ N(0, I)
2. for i = 0 to N-1 do:
   3. t = i / N
   4. v = VelocityNet(x, a, t)
   5. a = a + v * (1/N)
6. 返回 a
```

### 4.3 扩散算法

**算法 4: 扩散训练**

```
输入：条件特征 x, 目标动作 a_0
输出：训练好的噪声预测器 ε_θ

1. 采样时间步：t ~ Uniform(0, T)
2. 采样噪声：ε ~ N(0, I)
3. 计算噪声动作：a_t = √ᾱ_t*a_0 + √(1-ᾱ_t)*ε
4. 预测噪声：ε_θ = NoiseNet(x, a_t, t)
5. 计算损失：L = ||ε_θ - ε||²
6. 反向传播，更新参数
```

---

## 5. 数学原理

### 5.1 Transformer 注意力

**Scaled Dot-Product Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 5.2 流匹配理论

**最优传输问题**:
$$\min_v \mathbb{E}\left[\int_0^1 \|v_t(X_t)\|^2 dt\right]$$
$$\text{s.t. } dX_t = v_t(X_t)dt, \quad X_0 \sim p_0, \quad X_1 \sim p_1$$

**条件流匹配**:
定义概率路径：
$$p_t(x|x_0, x_1) = \mathcal{N}(x; (1-t)x_0 + tx_1, \sigma^2 I)$$

速度场：
$$u_t(x|x_0, x_1) = x_1 - x_0$$

训练目标：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, p_1(x_1), p_0(x_0)}\left[\|v_\theta(X_t, t) - (x_1 - x_0)\|^2\right]$$

**生成 ODE**:
$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0$$

**Euler 积分**:
$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

### 5.3 扩散模型

**前向过程**:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

**累积噪声**:
$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$

**任意时刻**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

**反向过程**:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_t; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**训练目标**:
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

**DDIM 采样**:
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

### 5.4 损失函数

**流匹配损失**:
$$\mathcal{L}_{\text{FM}} = \frac{1}{B} \sum_{i=1}^{B} \|v_\theta(x_t^{(i)}, t^{(i)}) - (x_1^{(i)} - x_0^{(i)})\|^2$$

**扩散损失**:
$$\mathcal{L}_{\text{diff}} = \frac{1}{B} \sum_{i=1}^{B} \|\epsilon^{(i)} - \epsilon_\theta(x_t^{(i)}, t^{(i)})\|^2$$

**MLP 回归损失**:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^{B} \|a_{\text{pred}}^{(i)} - a_{\text{true}}^{(i)}\|^2$$

**L1 损失** (更鲁棒):
$$\mathcal{L}_{\text{L1}} = \frac{1}{B} \sum_{i=1}^{B} |a_{\text{pred}}^{(i)} - a_{\text{true}}^{(i)}|$$

### 5.5 动作分块 (Action Chunking)

**时序建模**:
$$\mathbf{a}_{t:t+H} = f_\theta(o_t, \text{text})$$

其中：
- $H$: 动作块大小 (horizon)
- $\mathbf{a}_{t:t+H} \in \mathbb{R}^{H \times D}$: 预测的动作序列

**执行策略**:
$$a_{\text{executed}} = \mathbf{a}_{t:t+H}[0]$$

**优势**:
- 减少推理频率
- 保证动作平滑性
- 隐式建模未来状态

---

## 6. 训练流程

### 6.1 数据准备

**步骤 1: 数据收集**
- 使用遥操作设备收集 (图像，状态，动作，语言) 四元组
- 支持多种机器人平台

**步骤 2: 转换为标准格式**
```python
# 转换为 LeRobot 格式
from lerobot.datasets import convert_to_lerobot

convert_to_lerobot(
    raw_data_dir="/path/to/raw",
    output_dir="/path/to/lerobot"
)
```

**步骤 3: 计算归一化统计**
```python
# 计算动作和状态的归一化统计
from vla_training.data import compute_norm_stats

norm_stats = compute_norm_stats(
    dataset_path="/path/to/lerobot"
)

# 保存到 JSON
import json
with open("norm_stats.json", "w") as f:
    json.dump(norm_stats, f, indent=2)
```

### 6.2 训练配置

**示例配置** (`configs/training/default.yaml`):
```yaml
# 模型配置
model:
  vision:
    type: siglip
    pretrained: siglip-so400m-patch14-384
    freeze: false
  
  language:
    type: gemma
    pretrained: gemma-2b
    freeze: true
  
  fusion:
    type: cross_attention
    hidden_dim: 768
  
  action_head:
    type: flow_matching
    action_dim: 7
    chunk_size: 10
    hidden_dim: 512
    num_steps: 50

# 训练配置
training:
  batch_size: 32
  num_epochs: 100
  grad_accum_steps: 4
  
  optimizer:
    type: AdamW
    lr: 3e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
  
  scheduler:
    type: cosine
    warmup_steps: 1000
    min_lr: 1e-5
  
  # 混合精度
  use_amp: true
  
  # 梯度裁剪
  grad_clip: 1.0
  
  # 检查点
  checkpoint:
    save_every: 1000
    keep_last: 5
  
  # 日志
  logging:
    log_every: 100
    use_wandb: true
    project: vla-training
```

### 6.3 启动训练

**单 GPU 训练**:
```bash
python scripts/train.py \
  --config configs/model/default.yaml \
  --data configs/data/my_dataset.yaml \
  --training configs/training/default.yaml \
  --exp-name my_first_vla
```

**多 GPU 训练** (DDP):
```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/model/default.yaml \
  --data configs/data/my_dataset.yaml \
  --training configs/training/default.yaml \
  --exp-name my_first_vla_ddp
```

**恢复训练**:
```bash
python scripts/train.py \
  --config configs/model/default.yaml \
  --resume checkpoints/my_first_vla/latest.pt
```

### 6.4 监控训练

**TensorBoard**:
```bash
tensorboard --logdir logs/my_first_vla
```

**WandB**:
```python
# 配置中启用
training:
  logging:
    use_wandb: true
    project: vla-training
    name: my_first_vla
```

---

## 7. 推理部署

### 7.1 基本推理

```python
from vla_training.inference import VLAPolicy

# 加载策略
policy = VLAPolicy.from_checkpoint(
    checkpoint_path="checkpoints/my_first_vla/best.pt"
)

# 运行推理
image = load_image("scene.jpg")
prompt = "pick up the red block"

actions = policy.predict(
    image=image,
    text=prompt,
    num_steps=10  # 流匹配步数
)

# 执行动作
robot.execute(actions[0])
```

### 7.2 实时控制

```python
policy = VLAPolicy.from_checkpoint("checkpoints/my_first_vla/best.pt")

# 控制循环 (10Hz)
while True:
    image = camera.get_image()
    actions = policy.predict(image, current_prompt)
    robot.execute(actions[0])  # 执行动作块的第一步
    time.sleep(0.1)
```

### 7.3 模型导出

**ONNX 导出**:
```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/my_first_vla/best.pt \
  --output model.onnx \
  --opset 17
```

**TensorRT 优化**:
```bash
trtexec --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16 \
  --workspace=4096
```

### 7.4 部署优化

**延迟优化**:
1. 使用 torch.compile
2. 减少流匹配步数 (10 → 5)
3. 缓存语言特征
4. 量化到 INT8

**内存优化**:
1. 冻结语言模型
2. 使用更小的视觉编码器
3. 梯度检查点 (训练时)

---

## 8. 性能优化

### 8.1 训练优化

**混合精度训练**:
```python
# 启用 AMP
scaler = torch.cuda.amp.GradScaler()

with autocast():
    loss = model.forward(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**梯度累积**:
```python
# 模拟大批次
for i, batch in enumerate(dataloader):
    loss = model.forward(batch) / grad_accum_steps
    loss.backward()
    
    if (i + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**学习率预热**:
```python
# Cosine 调度 + 预热
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=min_lr
)

# 预热阶段
for step in range(warmup_steps):
    lr = base_lr * (step + 1) / warmup_steps
    set_lr(optimizer, lr)
```

### 8.2 推理优化

**torch.compile**:
```python
# 编译模型
model = torch.compile(model)
```

**特征缓存**:
```python
class CachedPolicy:
    def __init__(self, model):
        self.model = model
        self.cached_lang_features = None
        self.cached_prompt = None
    
    def predict(self, image, prompt):
        # 缓存语言特征
        if prompt != self.cached_prompt:
            self.cached_lang_features = self.model.encode_language(prompt)
            self.cached_prompt = prompt
        
        # 使用缓存的特征
        actions = self.model.decode_actions(
            image=image,
            lang_features=self.cached_lang_features
        )
        
        return actions
```

**量化**:
```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 8.3 性能基准

**训练速度** (A100 GPU):
| 配置 | 批次大小 | 步/秒 | 显存 |
|------|---------|-------|------|
| ViT-B + Gemma-2B + Flow | 32 | 2.5 | 40GB |
| ViT-B + Gemma-2B + Diff | 32 | 2.0 | 42GB |
| SigLIP + Gemma-7B + Flow | 16 | 1.2 | 65GB |

**推理延迟** (RTX 4090):
| 配置 | 流匹配步数 | 延迟 | 频率 |
|------|-----------|------|------|
| ViT-B + Gemma-2B | 10 | 45ms | 22Hz |
| ViT-B + Gemma-2B | 5 | 25ms | 40Hz |
| ResNet-50 + Gemma-2B | 10 | 30ms | 33Hz |

---

## 附录

### A. 配置参数说明

**模型配置**:
- `vision.type`: 视觉编码器类型 (vit_base, vit_large, siglip, resnet50)
- `vision.freeze`: 是否冻结视觉编码器
- `language.type`: 语言模型类型 (gemma, qwen, llama)
- `language.freeze`: 是否冻结语言模型
- `fusion.type`: 融合类型 (cross_attention, concat, film)
- `action_head.type`: 动作头类型 (flow_matching, diffusion, mlp)

**训练配置**:
- `batch_size`: 每 GPU 批次大小
- `grad_accum_steps`: 梯度累积步数
- `grad_clip`: 梯度裁剪阈值
- `use_amp`: 是否使用混合精度
- `num_epochs`: 训练轮数
- `optimizer.lr`: 学习率
- `scheduler.type`: 调度器类型 (cosine, linear, constant)

### B. 常见问题

**CUDA Out of Memory**:
1. 减小批次大小
2. 启用梯度检查点
3. 冻结语言模型
4. 使用更小的视觉编码器

**训练不收敛**:
1. 检查学习率 (尝试 1e-4 或 1e-3)
2. 检查数据归一化
3. 增加梯度裁剪
4. 增加预热步数

**推理速度慢**:
1. 使用 torch.compile
2. 减少流匹配步数
3. 缓存语言特征
4. 量化模型

### C. 参考文献

1. Flow Matching: https://arxiv.org/abs/2210.02747
2. Diffusion Policy: https://arxiv.org/abs/2303.04137
3. π₀: https://www.physicalintelligence.company/blog/pi0
4. GR00T: https://research.nvidia.com/labs/gear/gr00t-n1_6/
5. LeRobot: https://github.com/huggingface/lerobot

---

*文档版本*: v2.0  
*最后更新*: 2026-03-09  
*维护者*: VLA Training Team
