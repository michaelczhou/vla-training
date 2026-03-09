# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

**Google, 2023** | arXiv:2307.15818

---

## 0. 核心观点与结论

### 研究问题
- 如何将互联网规模的知识迁移到机器人控制？
- VLA 模型能否理解并执行需要世界知识的任务？
- 如何统一视觉 - 语言预训练与机器人学习？

### 核心贡献
1. **VLA 模型**：将视觉 - 语言模型直接输出机器人动作
2. **知识迁移**：从网络数据学习语义，迁移到机器人任务
3. **端到端训练**：联合优化视觉、语言、动作预测
4. **新能力涌现**：符号理解、数值推理、人类意图识别

### 主要结论
- VLA 模型成功迁移网络知识到机器人
- 在需要语义理解的任务上显著优于 RT-1
- 保持 RT-1 的基础操作能力
- 展示零样本泛化到新概念的能力

### 领域启示
- 确立了 VLA 作为具身智能主流范式
- 证明大模型知识可直接用于机器人控制
- 开启了"基础模型 + 机器人"新纪元

---

## 1. 创新点详解

### 核心创新

#### 1.1 VLA 架构
```python
class RT2(nn.Module):
    def __init__(self):
        # 复用 PaLI/PaLM-E 架构
        self.vision_encoder = SigLIP()  # 视觉
        self.llm = PaLM()               # 语言
        self.action_head = nn.Linear(d_model, action_dim * 256)
    
    def forward(self, images, text):
        # 视觉 - 语言联合编码
        tokens = self.vision_encoder(images)
        text_tokens = self.llm.embed(text)
        combined = torch.cat([text_tokens, tokens], dim=1)
        
        # LLM 处理
        output = self.llm.transformer(combined)
        
        # 动作预测 (复用语言 head)
        actions = self.action_head(output[:, -1:])
        return actions
```

#### 1.2 动作作为文本
- 动作离散化为 token
- 与文本文本共享词表
- 统一生成范式

#### 1.3 知识迁移机制
```
网络数据 (图像 - 文本对)  → 学习语义概念
         ↓
机器人数据 (图像 - 文本 - 动作) → 关联概念与动作
         ↓
新任务 (需要世界知识) → 零样本执行
```

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RT-2 架构流程                           │
├─────────────────────────────────────────────────────────────┤
│  训练数据混合：                                              │
│  - 50% 机器人数据 (RT-1 数据)                               │
│  - 50% 网络数据 (图像 - 文本)                               │
├─────────────────────────────────────────────────────────────┤
│  统一处理：                                                  │
│  - 图像 → ViT → token                                       │
│  - 文本 → Embedding → token                                 │
│  - 动作 → 离散化 → token (视为特殊文本)                      │
├─────────────────────────────────────────────────────────────┤
│  联合训练：                                                  │
│  - 机器人任务：预测动作 token                               │
│  - 网络任务：预测文本文本                                    │
│  - 共享所有参数                                              │
├─────────────────────────────────────────────────────────────┤
│  涌现能力：                                                  │
│  - 符号理解："星巴克杯子" → 定位特定物体                     │
│  - 数值推理："拿 2 个苹果" → 计数动作                        │
│  - 意图理解："我渴了" → 递饮料                              │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 统一生成
$$p(\mathbf{y} | \mathbf{x}) = \prod_{t} p(y_t | \mathbf{y}_{<t}, \mathbf{x})$$

其中 $\mathbf{y}$ 可以是文本或动作 token。

#### 知识迁移
$$\mathcal{L} = \mathbb{E}_{(\mathbf{I}, \mathbf{T}) \sim \text{web}}[-\log p(\mathbf{T}|\mathbf{I})] + \mathbb{E}_{(\mathbf{I}, \mathbf{T}, \mathbf{A}) \sim \text{robot}}[-\log p(\mathbf{A}|\mathbf{I}, \mathbf{T})]$$

### 实验结论

#### 关键结果
| 能力 | RT-1 | RT-2 | 提升 |
|------|------|------|------|
| 基础操作 | 62% | 62% | - |
| 符号理解 | 15% | 58% | +43% |
| 数值推理 | 10% | 52% | +42% |
| 意图理解 | 8% | 45% | +37% |

#### 涌现能力示例
- "拿星巴克杯子" → 识别品牌并抓取
- "给穿红衣服的人" → 识别人类属性
- "拿 2 个苹果" → 计数并抓取

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 视觉编码 | $\mathbf{V} = \text{ViT}(\mathbf{I})$ | `vis_tokens = vit(images)` | SigLIP |
| 2. 文本编码 | $\mathbf{T} = \text{Emb}(text)$ | `text_tokens = embed(text)` | PaLM |
| 3. 拼接 | $\mathbf{X} = [\mathbf{T}; \mathbf{V}]$ | `x = torch.cat([text, vis], dim=1)` | 统一输入 |
| 4. Transformer | $\mathbf{H} = \text{Transformer}(\mathbf{X})$ | `output = transformer(x)` | 共享层 |
| 5. 动作预测 | $\hat{\mathbf{a}} = \text{Head}(\mathbf{H})$ | `actions = head(output[:, -1])` | 动作 head |
| 6. 文本预测 | $\hat{\mathbf{t}} = \text{LMHead}(\mathbf{H})$ | `text = lm_head(output)` | 语言 head |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.9
torch >= 2.0
transformers >= 4.30
```

### 数据混合策略
```python
# 数据混合配置
data_mix = {
    'robotics': 0.5,      # RT-1 数据
    'web_vqa': 0.25,      # 视觉问答
    'web_caption': 0.25   # 图像描述
}
```

### 模型配置
```yaml
model:
  vision_encoder: siglip-400m
  llm: palm-2
  d_model: 2048
  action_bins: 256
  
training:
  batch_size: 128
  learning_rate: 1e-4
  data_mix:
    robotics: 0.5
    web: 0.5
```

### 推理示例
```python
# 零样本符号理解
instruction = "pick up the starbucks cup"
action = rt2.predict(image, instruction)
# 模型理解"starbucks"概念并定位对应物体
```

---

## 参考资源

- **论文**: https://arxiv.org/abs/2307.15818
- **项目页**: https://robotics-transformer2.github.io/

---

*最后更新：2026-03-03*
