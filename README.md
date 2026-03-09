# VLA Training Framework

A comprehensive, production-ready Vision-Language-Action (VLA) model training framework with support for multiple architectures (Diffusion, Flow Matching, Autoregressive).

## Table of Contents

- [Part 1: VLA Theory & Principles](#part-1-vla-theory--principles)
- [Part 2: Quick Start](#part-2-quick-start)
- [Part 3: Project Structure](#part-3-project-structure)
- [Part 4: Configuration](#part-4-configuration)
- [Part 5: Training Guide](#part-5-training-guide)
- [Part 6: Inference & Deployment](#part-6-inference--deployment)
- [Part 7: Troubleshooting](#part-7-troubleshooting)

---

# Part 1: VLA Theory & Principles

## 1. VLA 基础概念

### 什么是 Vision-Language-Action 模型

Vision-Language-Action (VLA) 模型是一种多模态深度学习架构，它将**视觉感知**、**语言理解**和**动作生成**统一到一个端到端的系统中。VLA 模型直接接收图像和文本指令，输出机器人可执行的动作序列。

**核心思想**：将机器人控制问题转化为多模态序列建模问题，通过大规模数据训练获得通用的机器人策略。

### VLA 与 VLM、LLM 的区别

| 特性 | LLM | VLM | VLA |
|------|-----|-----|-----|
| 输入 | 文本 | 图像 + 文本 | 图像 + 文本 |
| 输出 | 文本 | 文本 | **动作序列** |
| 应用 | NLP 任务 | 视觉问答 | **机器人控制** |
| 时序性 | 可选 | 可选 | **必需** |
| 实时性 | 低 | 中 | **高** |

- **LLM (Large Language Model)**: 仅处理文本，擅长语言理解和生成
- **VLM (Vision-Language Model)**: 理解图像内容并用语言描述，但不产生物理动作
- **VLA (Vision-Language-Action)**: 理解场景和指令，直接输出可执行的动作

### VLA 的核心应用场景

1. **家庭服务机器人**: 整理物品、清洁、烹饪辅助
2. **工业操作**: 装配、分拣、质量控制
3. **医疗辅助**: 手术辅助、康复训练
4. **仓储物流**: 货物搬运、包装、分拣
5. **科研探索**: 实验室自动化、野外考察

## 2. VLA 架构组成

```
┌─────────────────────────────────────────────────────────────┐
│                      VLA 架构                                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Image      │    │   Text       │    │   Action     │  │
│  │   Input      │    │   Prompt     │    │   Output     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────▲───────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   │          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐  │
│  │   Vision     │    │  Language    │    │    Action    │  │
│  │   Encoder    │    │   Encoder    │    │     Head     │  │
│  │  (ViT/SigLIP)│    │ (LLaMA/Qwen) │    │(Diff/Flow/AR)│  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘  │
│         │                   │                                │
│         └────────┬──────────┘                                │
│                  ▼                                           │
│         ┌────────────────┐                                   │
│         │    Fusion      │                                   │
│         │    Module      │                                   │
│         │ (Cross-Attn)   │                                   │
│         └────────┬───────┘                                   │
│                  │                                           │
│                  └───────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## 3. 核心组件详解

### 视觉编码器 (Vision Encoder)

负责将输入图像转换为特征向量。

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| **ViT-Base** | 86M | 标准 Transformer, 平衡性能与速度 | 通用场景 |
| **ViT-Large** | 307M | 更高精度，需要更多计算 | 高精度需求 |
| **SigLIP** | 428M | Google 最新，对比学习预训练 | SOTA 性能 |
| **ResNet-50** | 25M | CNN 架构，推理快速 | 边缘部署 |

**推荐**: SigLIP 或 ViT-Large 用于高性能场景，ResNet-50 用于实时/边缘部署。

### 语言编码器 (Language Encoder)

处理文本指令，理解任务语义。

| 模型 | 参数量 | 特点 |
|------|--------|------|
| **LLaMA-2-7B** | 7B | Meta 开源，生态完善 |
| **Qwen-7B** | 7B | 阿里通义，中文优化 |
| **Gemma-7B** | 7B | Google 轻量级 |
| **Phi-2** | 2.7B | 微软小模型，高效 |

**推荐**: 根据显存选择，7B 模型需要约 14GB 显存 (FP16)。

### 融合方式 (Fusion Module)

将视觉和语言特征进行融合。

#### Cross-Attention (推荐)
```python
# 视觉特征作为 Key/Value, 语言特征作为 Query
attn_output = CrossAttention(
    query=lang_features,
    key=vision_features,
    value=vision_features
)
```

#### Concatenation
```python
# 简单拼接后通过 MLP 融合
fused = MLP(torch.cat([vision_features, lang_features], dim=-1))
```

#### FiLM (Feature-wise Linear Modulation)
```python
# 用语言特征调制视觉特征
gamma, beta = LanguageNet(lang_features)
modulated = gamma * vision_features + beta
```

### 动作头 (Action Head)

#### 1. Diffusion Policy (扩散策略)

基于扩散模型生成动作序列。

**原理**:
- 前向过程：逐渐向动作添加噪声
- 反向过程：学习从噪声恢复动作

**优点**: 多模态输出，处理不确定性
**缺点**: 推理慢 (需要多步去噪)

```python
# 扩散损失
loss = ||ε - ε_θ(x_t, t, condition)||²
```

#### 2. Flow Matching (流匹配) ⭐推荐

基于常微分方程 (ODE) 的生成方法。

**原理**:
- 学习从噪声分布到数据分布的速度场
- 通过积分 ODE 生成样本

**优点**: 
- 推理速度快 (少步数)
- 训练稳定
- 理论优雅

**数学推导**:

定义概率路径: `p_t = (1-t)p_0 + t*p_1`

速度场: `v_t(x) = E[v_t(x) | x_t = x]`

训练目标: `L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]`

生成过程: `dx/dt = v_θ(x, t)`, 用数值积分求解

#### 3. Autoregressive (自回归/FAST)

将动作离散化为 token，用语言模型生成。

**FAST Tokenizer**:
1. 动作分块 (Action Chunking)
2. DCT 变换到频率域
3. 量化 + BPE 编码

**优点**: 与 LLM 无缝集成
**缺点**: 离散化损失精度

#### 4. Discrete Binning

将连续动作空间离散化为 bins。

```python
# 每个动作维度分为 N 个 bin
action_bin = floor((action - min) / (max - min) * N)
```

## 4. 数学原理

### Transformer 注意力机制

```python
# Scaled Dot-Product Attention
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Multi-Head Attention
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 扩散模型原理

**前向扩散过程**:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
```

**反向去噪过程**:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**训练目标**:
```
L_simple = E_t,x_0,ε[||ε - ε_θ(x_t, t)||²]
```

### 流匹配 (Flow Matching) 推导

**最优传输问题**:
```
min_E[∫₀¹ ||v_t(X_t)||² dt]
s.t. dX_t = v_t(X_t)dt, X_0 ~ p_0, X_1 ~ p_1
```

**条件流匹配**:
```
L_CFM = E_t,p_1(x_1),p_0(x_0)[||v_θ(X_t, t) - (x_1 - x_0)||²]
```

**生成过程** (Euler 积分):
```
x_{t+Δt} = x_t + v_θ(x_t, t) * Δt
```

### 动作分块 (Action Chunking)

将动作序列分块处理，提高效率和时序一致性。

```python
# 假设动作维度为 7 (机械臂关节)
# 分块大小 = 10 步
action_chunk = actions[t:t+10]  # shape: (10, 7)

# 模型一次性预测整个 chunk
predicted_chunk = model(image, text)  # shape: (10, 7)

# 执行时只取第一步，滑动窗口推进
execute(predicted_chunk[0])
```

**优点**:
- 减少推理频率
- 保证动作平滑性
- 提高时序一致性

## 5. 训练流程

### 数据格式 (RLDS/LeRobot)

**RLDS (Robot Learning Dataset)**:
```python
{
    "observation": {
        "image": tf.Tensor,      # (H, W, 3)
        "depth": tf.Tensor,      # (H, W)
        "proprio": tf.Tensor,    # (action_dim,)
    },
    "action": tf.Tensor,         # (chunk_size, action_dim)
    "language_instruction": str,
    "task": str,
}
```

**LeRobot 格式**:
```python
{
    "observation.images.top": torch.Tensor,
    "observation.state": torch.Tensor,
    "action": torch.Tensor,
    "language": str,
}
```

### 损失函数设计

#### Flow Matching Loss
```python
def flow_matching_loss(model, x0, x1, t):
    xt = t * x1 + (1 - t) * x0
    target_velocity = x1 - x0
    predicted_velocity = model(xt, t)
    return mse_loss(predicted_velocity, target_velocity)
```

#### Diffusion Loss
```python
def diffusion_loss(model, x0, t, noise):
    xt = sqrt_alphas[t] * x0 + sqrt_one_minus_alphas[t] * noise
    predicted_noise = model(xt, t)
    return mse_loss(predicted_noise, noise)
```

#### Action Chunking Loss
```python
def action_chunk_loss(model, images, text, actions):
    predicted = model(images, text)  # (B, chunk_size, action_dim)
    return mse_loss(predicted, actions)
```

### 优化器配置

```yaml
optimizer:
  type: AdamW
  lr: 3e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  
scheduler:
  type: cosine
  warmup_steps: 1000
  min_lr: 1e-5
```

### 训练技巧

1. **梯度裁剪**: `clip_grad_norm_(model.parameters(), 1.0)`
2. **混合精度训练**: `torch.cuda.amp.autocast()`
3. **梯度累积**: 模拟更大 batch size
4. **学习率预热**: 前 1000 步线性增加
5. **数据增强**: 颜色抖动、随机裁剪、翻转

## 6. 推理部署

### 实时推理优化

1. **模型融合**: 合并 BatchNorm 到 Conv
2. **算子优化**: 使用 TensorRT / ONNX Runtime
3. **缓存机制**: 缓存语言特征 (文本不变时)
4. **异步执行**: 推理与执行并行

### 模型量化

```python
# 动态量化 (推理时)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 静态量化 (需要校准)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared_model = torch.quantization.prepare(model)
# ... 校准 ...
quantized_model = torch.quantization.convert(prepared_model)
```

### 边缘部署

**Jetson Orin**:
```bash
# 导出 ONNX
python -m torch.onnx.export model.onnx

# TensorRT 转换
trtexec --onnx=model.onnx --saveEngine=model.engine
```

**推理延迟目标**:
- 云端： < 100ms
- 边缘： < 50ms
- 实时控制： < 20ms

---

# Part 2: Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/vla-training.git
cd vla-training

# Install dependencies
pip install -e .

# Verify installation
python -c "import src.models; print('VLA Framework Ready!')"
```

## Training

```bash
# Train on DROID dataset
python scripts/train.py \
  --config configs/model/pi0_base.yaml \
  --data configs/data/droid.yaml \
  --training configs/training/droid_finetune.yaml \
  --exp-name my_first_vla

# Resume training
python scripts/train.py \
  --config configs/model/pi0_base.yaml \
  --data configs/data/droid.yaml \
  --resume checkpoints/my_first_vla/latest.pt
```

## Inference

```bash
# Run inference with checkpoint
python scripts/inference.py \
  --checkpoint checkpoints/my_first_vla/latest.pt \
  --image path/to/image.jpg \
  --prompt "pick up the red block"

# Interactive mode
python scripts/inference.py \
  --checkpoint checkpoints/my_first_vla/latest.pt \
  --interactive
```

---

# Part 3: Project Structure

```
vla-training/
├── configs/                 # 配置文件
│   ├── model/              # 模型架构配置
│   ├── training/           # 训练超参数
│   └── data/               # 数据集配置
├── src/                    # 源代码
│   ├── models/             # 模型定义
│   ├── data/               # 数据加载
│   ├── training/           # 训练逻辑
│   ├── inference/          # 推理逻辑
│   └── utils/              # 工具函数
├── scripts/                # 脚本入口
├── examples/               # 使用示例
├── tests/                  # 单元测试
└── README.md               # 本文档
```

---

# Part 4: Configuration

## Model Configuration

```yaml
# configs/model/pi0_base.yaml
model:
  name: "pi0_base"
  
  vision:
    type: "siglip"
    pretrained: "siglip-so400m-patch14-384"
    freeze: false
    
  language:
    type: "gemma"
    pretrained: "gemma-2b"
    freeze: true
    
  fusion:
    type: "cross_attention"
    num_heads: 8
    hidden_dim: 768
    
  action_head:
    type: "flow_matching"
    action_dim: 7
    chunk_size: 10
    hidden_dim: 512
    num_steps: 50
```

## Training Configuration

```yaml
# configs/training/droid_finetune.yaml
training:
  batch_size: 32
  num_epochs: 100
  grad_accum_steps: 4
  
  optimizer:
    type: "AdamW"
    lr: 3e-4
    weight_decay: 0.1
    
  scheduler:
    type: "cosine"
    warmup_steps: 1000
    
  checkpoint:
    save_every: 1000
    keep_last: 5
    
  logging:
    log_every: 100
    wandb_project: "vla-training"
```

## Data Configuration

```yaml
# configs/data/droid.yaml
data:
  name: "droid"
  data_dir: "/path/to/droid"
  
  image:
    height: 224
    width: 224
    normalize: true
    
  action:
    dim: 7
    chunk_size: 10
    normalize: true
    
  augmentation:
    color_jitter: true
    random_crop: true
    flip: false
```

---

# Part 5: Training Guide

## Step 1: Prepare Data

```bash
# Download DROID dataset
python scripts/download_data.py --dataset droid --output data/droid

# Or use LeRobot format
python scripts/convert_to_lerobot.py \
  --input data/raw \
  --output data/le_robot
```

## Step 2: Configure Training

Edit `configs/training/droid_finetune.yaml` to set:
- Batch size (根据显存调整)
- Learning rate
- Number of epochs

## Step 3: Start Training

```bash
# Single GPU
python scripts/train.py --config configs/model/pi0_base.yaml

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/model/pi0_base.yaml
```

## Step 4: Monitor Training

```bash
# View training logs
tail -f logs/my_first_vla/train.log

# Or use TensorBoard
tensorboard --logdir logs/my_first_vla
```

## Step 5: Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/my_first_vla/best.pt \
  --data configs/data/droid.yaml
```

---

# Part 6: Inference & Deployment

## Basic Inference

```python
from src.inference.policy import VLAPolicy

# Load model
policy = VLAPolicy.from_checkpoint("checkpoints/my_first_vla/latest.pt")

# Run inference
image = load_image("scene.jpg")
prompt = "pick up the red block"
actions = policy.predict(image, prompt)

# Execute first action
robot.execute(actions[0])
```

## Real-time Control

```python
policy = VLAPolicy.from_checkpoint("checkpoints/my_first_vla/latest.pt")

while True:
    image = camera.get_image()
    actions = policy.predict(image, current_prompt)
    robot.execute(actions[0])  # Execute first action of chunk
    time.sleep(0.1)  # 10Hz control loop
```

## Export for Deployment

```bash
# Export to ONNX
python scripts/export_onnx.py \
  --checkpoint checkpoints/my_first_vla/latest.pt \
  --output model.onnx

# Optimize with TensorRT
trtexec --onnx=model.onnx --saveEngine=model.engine
```

---

# Part 7: Troubleshooting

## CUDA Out of Memory

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Freeze language model
5. Use smaller vision encoder

```yaml
# configs/model/custom_vla.yaml
model:
  vision:
    type: "resnet50"  # Smaller than ViT
  language:
    freeze: true      # Don't train LLM
  training:
    gradient_checkpointing: true
```

## Training Not Converging

**Check**:
1. Learning rate (try 1e-4 or 1e-3)
2. Data normalization
3. Action normalization
4. Gradient clipping

```yaml
training:
  optimizer:
    lr: 1e-4  # Try different values
  grad_clip: 1.0
```

## Slow Inference

**Solutions**:
1. Use TensorRT / ONNX Runtime
2. Quantize model to INT8
3. Cache language features
4. Reduce action chunk size

```python
# Cache language features
cached_lang_features = None
cached_prompt = None

def predict(image, prompt):
    global cached_lang_features, cached_prompt
    if prompt != cached_prompt:
        cached_lang_features = encode_language(prompt)
        cached_prompt = prompt
    # Use cached features...
```

---

## References

- [OpenPi](https://github.com/Physical-Intelligence/openpi)
- [OpenVLA](https://github.com/openvla/openvla)
- [Octo](https://github.com/octo-models/octo)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Flow Matching](https://arxiv.org/abs/2210.02747)

## License

MIT License
