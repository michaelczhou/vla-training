# VLA 学习教程

> 从零开始学习 Vision-Language-Action 模型

---

## 目录

1. [VLA 是什么？](#1-vla-是什么)
2. [核心组件](#2-核心组件)
3. [快速上手](#3-快速上手)
4. [训练你的第一个 VLA](#4-训练你的第一个-vla)
5. [常见问题](#5-常见问题)

---

## 1. VLA 是什么？

### 1.1 一句话解释

**VLA = 看懂图像 + 理解语言 + 执行动作**

就像你看到一个苹果，听到"拿起苹果"的指令，然后伸手去拿。VLA 模型做的就是这件事。

### 1.2 为什么需要 VLA？

传统机器人控制需要：
- 人工编写复杂的规则
- 针对每个任务单独编程
- 难以适应新环境

VLA 模型：
- 从数据中学习
- 一个模型处理多种任务
- 自然语言指令控制

### 1.3 应用场景

| 场景 | 示例指令 |
|------|----------|
| 家庭服务 | "把桌上的杯子放到洗碗机" |
| 工业操作 | "将这个零件安装到位置 A" |
| 医疗辅助 | "递给我手术剪刀" |
| 仓储物流 | "把这箱货物搬到货架第三层" |

---

## 2. 核心组件

### 2.1 架构图解

```
图像输入 → [视觉编码器] → 视觉特征
                              ↓
文本指令 → [语言模型] → 语言特征 → [融合模块] → [动作头] → 动作输出
```

### 2.2 各组件详解

#### 视觉编码器 (Vision Encoder)

**作用**: 把图像变成机器能理解的数字

**类比**: 就像人眼看东西，大脑提取特征（颜色、形状、位置）

**常见选择**:
- **ViT** (Vision Transformer): 标准选择，平衡性能和速度
- **SigLIP**: 最新 SOTA，精度更高
- **ResNet**: 速度快，适合边缘设备

```python
# 示例: 使用 ViT 编码图像
from src.models.vision_encoder import build_vision_encoder

vision_encoder = build_vision_encoder({
    'type': 'vit',
    'model_name': 'vit_base_patch16_224'
})

# 输入图像 (B, C, H, W)
images = torch.randn(2, 3, 224, 224)
features = vision_encoder(images)
# 输出: (B, num_patches, hidden_dim)
```

#### 语言模型 (Language Model)

**作用**: 理解文本指令的含义

**类比**: 理解"拿起红色的方块" = 识别动作(拿起) + 目标(红色方块)

**常见选择**:
- **Gemma** (Google): 轻量级，2B 参数
- **Qwen** (阿里): 中文优化
- **LLaMA** (Meta): 生态完善

```python
# 示例: 编码文本指令
from src.models.language_model import build_language_model

language_model = build_language_model({
    'type': 'gemma',
    'model_name': 'gemma-2b'
})

# 输入 token IDs
tokens = tokenizer("pick up the red block", return_tensors="pt")
features = language_model(**tokens)
# 输出: (B, seq_len, hidden_dim)
```

#### 融合模块 (Fusion Module)

**作用**: 把视觉和语言信息结合起来

**类比**: 大脑同时处理"看到的画面"和"听到的指令"

**三种融合方式**:

1. **Cross-Attention** (推荐)
   ```python
   # 用语言特征查询视觉特征
   attn_output = CrossAttention(
       query=language_features,
       key=vision_features,
       value=vision_features
   )
   ```

2. **Concatenation** (简单)
   ```python
   # 直接拼接
   fused = torch.cat([vision_features, language_features], dim=-1)
   ```

3. **FiLM** (调制)
   ```python
   # 用语言特征调制视觉特征
   gamma, beta = language_net(language_features)
   fused = gamma * vision_features + beta
   ```

#### 动作头 (Action Head)

**作用**: 根据融合后的特征生成具体动作

**类比**: 大脑发出指令，手执行动作

**三种类型**:

##### a) Flow Matching (推荐 ⭐)

**原理**: 学习从噪声到动作的"流动"

**优点**:
- 推理快 (10-50 步)
- 训练稳定
- 数学优雅

**适用**: 大多数场景

```python
# 训练
loss = ||v_θ(x_t, t) - (action_target - action_noise)||²

# 推理 (Euler 积分)
for t in range(num_steps):
    velocity = model(x, t)
    x = x + velocity * dt
```

##### b) Diffusion Policy

**原理**: 逐步去噪生成动作

**优点**:
- 多模态输出 (一个指令多种执行方式)
- 处理不确定性

**缺点**:
- 推理慢 (需要 100+ 步)

**适用**: 需要多样性的场景

##### c) Autoregressive (FAST)

**原理**: 像生成文本一样生成动作 token

**优点**:
- 与 LLM 无缝集成
- 可以用现有语言模型技术

**缺点**:
- 离散化损失精度

**适用**: 与语言模型联合训练

### 2.3 动作块 (Action Chunking)

**问题**: 每步都推理太慢了！

**解决方案**: 一次性预测未来 N 步的动作

```python
# 传统方式 (慢)
for t in range(T):
    action = model(image_t, text)  # 每次都要推理
    robot.execute(action)

# 动作块方式 (快)
actions = model(image_t, text)  # 一次性预测 10 步
for i in range(10):
    robot.execute(actions[i])  # 依次执行
```

**参数**: `chunk_size`
- 太小 (5): 推理频繁，不够平滑
- 适中 (10-50): 推荐
- 太大 (100+): 累积误差大

---

## 3. 快速上手

### 3.1 安装

```bash
# 克隆仓库
git clone https://github.com/michaelczhou/vla-training.git
cd vla-training

# 安装依赖
pip install -e .
```

### 3.2 运行示例

```bash
# 查看快速入门示例
python examples/quickstart.py
```

### 3.3 项目结构

```
vla-training/
├── configs/          # 配置文件
│   ├── model/        # 模型配置
│   ├── training/     # 训练配置
│   └── data/         # 数据配置
├── src/              # 源代码
│   ├── models/       # 模型定义
│   ├── training/     # 训练逻辑
│   └── inference/    # 推理逻辑
├── scripts/          # 脚本
│   ├── train.py      # 训练脚本
│   └── inference.py  # 推理脚本
├── examples/         # 示例代码
└── tests/            # 单元测试
```

---

## 4. 训练你的第一个 VLA

### 4.1 准备数据

VLA 训练需要机器人操作数据，格式如下：

```python
{
    "observation": {
        "image": (H, W, 3),      # 摄像头图像
        "state": (7,),            # 机器人状态 (关节角度等)
    },
    "action": (10, 7),          # 未来 10 步的动作
    "language": "pick up the red block"  # 语言指令
}
```

**公开数据集**:
- **DROID**: 大规模机器人操作数据
- **Aloha**: 双臂操作数据
- **Open X-Embodiment**: 多机器人数据集

### 4.2 配置模型

编辑 `configs/model/my_first_vla.yaml`:

```yaml
model:
  name: "my_first_vla"
  
  vision:
    type: "vit"
    model_name: "vit_base_patch16_224"
    freeze: false  # 训练视觉编码器
    
  language:
    type: "gemma"
    model_name: "gemma-2b"
    freeze: true   # 冻结语言模型 (省显存)
    
  fusion:
    type: "cross_attention"
    hidden_dim: 768
    num_heads: 8
    
  action_head:
    type: "flow_matching"  # 推荐
    action_dim: 7          # 机械臂关节数
    chunk_size: 10         # 动作块大小
    hidden_dim: 512
    num_steps: 50          # 推理步数
```

### 4.3 配置训练

编辑 `configs/training/my_first_vla.yaml`:

```yaml
training:
  batch_size: 32
  num_epochs: 100
  grad_accum_steps: 4  # 梯度累积，模拟大 batch
  
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
```

### 4.4 开始训练

```bash
python scripts/train.py \
  --config configs/model/my_first_vla.yaml \
  --data configs/data/droid.yaml \
  --training configs/training/my_first_vla.yaml \
  --exp-name my_first_vla
```

### 4.5 监控训练

```bash
# 查看日志
tail -f logs/my_first_vla/train.log

# TensorBoard
tensorboard --logdir logs/my_first_vla
```

### 4.6 运行推理

```bash
python scripts/inference.py \
  --checkpoint checkpoints/my_first_vla/latest.pt \
  --image path/to/image.jpg \
  --prompt "pick up the red block"
```

---

## 5. 常见问题

### Q1: 显存不够怎么办？

**解决方案**:
1. 减小 batch_size
2. 冻结语言模型 (`freeze: true`)
3. 使用更小的视觉编码器 (ResNet 替代 ViT)
4. 启用梯度检查点
5. 使用混合精度训练

```yaml
model:
  vision:
    type: "resnet50"  # 更小的编码器
  language:
    freeze: true      # 冻结 LLM
    
training:
  batch_size: 8       # 减小 batch
  gradient_checkpointing: true
  mixed_precision: true
```

### Q2: 训练不收敛？

**检查清单**:
- [ ] 学习率是否合适 (尝试 1e-4 ~ 1e-3)
- [ ] 数据是否正确归一化
- [ ] 动作是否正确归一化
- [ ] 是否使用了梯度裁剪

### Q3: 推理速度慢？

**优化方案**:
1. 减少推理步数 (`num_steps: 10`)
2. 使用 TensorRT / ONNX Runtime
3. 缓存语言特征 (文本不变时)
4. 模型量化 (INT8)

### Q4: 动作不平滑？

**解决方案**:
1. 增大动作块大小
2. 使用 Diffusion Policy (更平滑)
3. 添加动作平滑损失

### Q5: 如何选择动作头？

| 动作头 | 速度 | 平滑度 | 多样性 | 推荐场景 |
|--------|------|--------|--------|----------|
| Flow Matching | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 通用场景 |
| Diffusion | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 需要多样性 |
| Autoregressive | ⭐⭐ | ⭐⭐ | ⭐⭐ | 与 LLM 联合 |

---

## 下一步

1. **阅读代码**: `src/models/` 下的各个模块
2. **运行测试**: `python -m pytest tests/`
3. **查看示例**: `examples/` 目录
4. **阅读论文**: `reference/research/` 中的论文解读

---

## 参考资源

- [OpenPi](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence 开源实现
- [OpenVLA](https://github.com/openvla/openvla) - 开源 VLA 模型
- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face 机器人学习库
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - 扩散策略论文

---

**祝你学习愉快！有任何问题随时提问。** 🤖
