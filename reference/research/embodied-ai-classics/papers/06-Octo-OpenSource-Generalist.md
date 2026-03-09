# Octo: An Open-Source Generalist Robot Policy

**Berkeley, 2024** | arXiv:2308.04131

---

## 0. 核心观点与结论

### 研究问题
- 如何构建真正通用的开源机器人策略？
- 单一模型能否处理多种机器人形态和任务？
- 如何实现高效的零样本迁移？

### 核心贡献
1. **通用策略**：单一模型适配多种机器人
2. **开源完整**：代码、数据、模型全公开
3. **模块化设计**：易于扩展和定制
4. **强泛化**：零样本迁移到新机器人

### 主要结论
- 通用策略在多数任务上接近专用模型
- 数据多样性是关键
- 开源促进社区发展
- 模块化设计便于研究

### 领域启示
- 推动通用机器人策略研究
- 建立开源生态系统
- 降低研究门槛

---

## 1. 创新点详解

### 核心创新

#### 1.1 通用 Tokenization
```python
class OctoTokenizer:
    def __init__(self):
        self.image_tokenizer = ViTTokenizer()
        self.action_tokenizer = ActionTokenizer()
        self.task_tokenizer = TaskTokenizer()
    
    def encode(self, images, actions, task):
        # 统一编码为 token 序列
        img_tokens = self.image_tokenizer(images)
        act_tokens = self.action_tokenizer(actions)
        task_tokens = self.task_tokenizer(task)
        return torch.cat([task_tokens, img_tokens, act_tokens], dim=1)
```

#### 1.2 机器人无关架构
- 动作空间归一化
- 观测空间抽象
- 任务语言统一

#### 1.3 模块化设计
```
Octo
├── Tokenizer (可替换)
├── Transformer Backbone (可替换)
├── Action Head (可替换)
└── Training Pipeline (可配置)
```

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                      Octo 架构流程                           │
├─────────────────────────────────────────────────────────────┤
│  输入标准化：                                                │
│  - 图像 → 统一分辨率 (256x256)                              │
│  - 动作 → 归一化到 [-1, 1]                                  │
│  - 任务 → 自然语言                                         │
├─────────────────────────────────────────────────────────────┤
│  Token 化：                                                  │
│  - 图像 → ViT → 256 token                                   │
│  - 动作 → 离散化 → 64 token                                 │
│  - 任务 → SentencePiece → 32 token                          │
├─────────────────────────────────────────────────────────────┤
│  Transformer 处理：                                          │
│  - 16 层自注意力                                            │
│  - 因果掩码 (动作自回归)                                     │
│  - d_model = 768                                           │
├─────────────────────────────────────────────────────────────┤
│  输出：                                                      │
│  - 动作序列 (预测未来 8 步)                                  │
│  - 置信度分数                                                │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 动作分块
$$\mathbf{A}_{t:t+H} = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}]$$

#### 条件策略
$$\pi(\mathbf{A}_{t:t+H} | \mathbf{O}_{\leq t}, \text{task}) = \prod_{h=0}^{H-1} \pi(\mathbf{a}_{t+h} | \mathbf{O}_{\leq t}, \text{task}, \mathbf{A}_{t:t+h})$$

### 实验结论

#### 跨机器人迁移
| 源机器人 | 目标机器人 | 零样本成功率 |
|----------|------------|--------------|
| WidowX | Franka | 58% |
| Franka | WidowX | 55% |
| 混合 | 新机器人 | 52% |

#### 训练效率
- 数据量：100 万轨迹
- 训练时间：3 天 (8xV100)
- 推理速度：25 Hz

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 图像编码 | $\mathbf{V} = \text{ViT}(\mathbf{I})$ | `vis_tokens = vit(images)` | ViT-Base |
| 2. 动作编码 | $\mathbf{A} = \text{Discretize}(\mathbf{a})$ | `act_tokens = discretize(actions)` | 64 bins |
| 3. 任务编码 | $\mathbf{T} = \text{SP}(task)$ | `task_tokens = sp.encode(task)` | SentencePiece |
| 4. 拼接 | $\mathbf{X} = [\mathbf{T}; \mathbf{V}; \mathbf{A}_{<t}]$ | `x = torch.cat([task, vis, act_hist])` | 统一序列 |
| 5. Transformer | $\mathbf{H} = \text{Transformer}(\mathbf{X})$ | `output = transformer(x)` | 16 层 |
| 6. 动作预测 | $\hat{\mathbf{A}} = \text{Head}(\mathbf{H})$ | `actions = head(output)` | 未来 H 步 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
pip install octo-rl
pip install tensorflow_datasets  # 数据加载
```

### 快速开始
```python
from octo.model import OctoModel
from octo.data import load_dataset

# 加载预训练模型
model = OctoModel.from_pretrained('octo-base')

# 加载数据
dataset = load_dataset('open_x_embodiment')

# 推理
action = model.sample_actions(
    observations={'image': img, 'proprio': proprio},
    task='pick up the cup',
    rng=jax.random.PRNGKey(0)
)
```

### 微调示例
```python
# 加载预训练模型
model = OctoModel.from_pretrained('octo-base')

# 冻结部分参数
model.freeze_backbone()

# 微调 action head
trainer = OctoTrainer(model, my_dataset)
trainer.train(steps=10000)
```

---

## 参考资源

- **论文**: https://arxiv.org/abs/2308.04131
- **代码**: https://github.com/octo-models/octo
- **文档**: https://octo-models.github.io/

---

*最后更新：2026-03-03*
