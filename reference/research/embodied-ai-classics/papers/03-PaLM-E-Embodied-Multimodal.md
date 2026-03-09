# PaLM-E: An Embodied Multimodal Language Model

**Google, 2023** | arXiv:2303.03378

---

## 0. 核心观点与结论

### 研究问题
- 如何将真实世界的连续传感器模态融入语言模型？
- 如何建立词语与感知之间的连接 (grounding)？
- 多模态联合训练是否能带来正迁移？

### 核心贡献
1. **具身语言模型**：将视觉、状态估计直接注入 LLM
2. **端到端训练**：视觉编码与语言模型联合优化
3. **正迁移证明**：多领域联合训练提升各方面能力
4. **大规模具身模型**：PaLM-E-562B 达到 SOTA

### 主要结论
- 具身化提升机器人任务性能
- 视觉 - 语言预训练知识可迁移到机器人
- 模型保持通用语言和视觉能力
- 规模扩展带来持续收益

### 领域启示
- 确立了 VLA (Vision-Language-Action) 范式
- 证明了大语言模型可作为机器人"大脑"
- 开启了具身 AI 与大模型结合的研究浪潮

---

## 1. 创新点详解

### 解决的问题
**之前方法不行原因：**
- 传统机器人模型缺乏语言理解
- 语言模型缺乏真实世界 grounding
- 模块化系统误差累积

### 核心创新

#### 1.1 具身化注入
```python
# 将连续传感器注入 LLM
class PaLME(nn.Module):
    def __init__(self, llm, vision_encoder):
        self.llm = llm
        self.vision_encoder = vision_encoder
        self.state_encoder = nn.Linear(state_dim, llm.d_model)
    
    def forward(self, images, states, text):
        # 视觉编码
        vis_tokens = self.vision_encoder(images)
        vis_tokens = vis_tokens.reshape(B, -1, d_model)
        
        # 状态编码
        state_tokens = self.state_encoder(states)
        state_tokens = state_tokens.unsqueeze(1)
        
        # 文本编码
        text_tokens = self.llm.embed(text)
        
        # 拼接并输入 LLM
        emb = torch.cat([text_tokens, vis_tokens, state_tokens], dim=1)
        output = self.llm.generate(emb)
        return output
```

#### 1.2 多任务训练
- 机器人任务：规划、VQA、描述
- 视觉任务：OK-VQA、图像描述
- 语言任务：对话、推理

#### 1.3 连续值注入
$$\mathbf{e}_{\text{state}} = \text{MLP}(\mathbf{s}) \in \mathbb{R}^{d_{\text{model}}}$$

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                     PaLM-E 架构流程                          │
├─────────────────────────────────────────────────────────────┤
│  多模态输入：                                                │
│  - 图像 (多视角) → ViT → 视觉 token                         │
│  - 状态 (关节角等) → MLP → 状态 token                       │
│  - 文本指令 → Embedding → 文本 token                        │
├─────────────────────────────────────────────────────────────┤
│  具身化 LLM：                                                │
│  - PaLM 作为骨干                                            │
│  - 所有 token 拼接输入                                       │
│  - 自回归生成                                                │
├─────────────────────────────────────────────────────────────┤
│  输出：                                                      │
│  - 文本响应 (规划、描述、答案)                               │
│  - 可解析为动作序列                                          │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 多模态融合
$$\mathbf{E} = [\mathbf{E}_{\text{text}}; \mathbf{E}_{\text{vis}}; \mathbf{E}_{\text{state}}]$$

#### 自回归生成
$$p(y_t | y_{<t}, \mathbf{E}) = \text{softmax}(W \cdot \text{PaLM}(\mathbf{E}, y_{<t}))$$

#### 联合训练损失
$$\mathcal{L} = \mathcal{L}_{\text{robotics}} + \mathcal{L}_{\text{vision}} + \mathcal{L}_{\text{language}}$$

### 实验结论

#### 模型规模
| 模型 | 参数 | 机器人任务 | OK-VQA |
|------|------|------------|--------|
| PaLM-E-8B | 8B | 52% | 45% |
| PaLM-E-562B | 562B | 66% | 54% |

#### 关键结果
1. **正迁移**：联合训练比单独训练 +15%
2. **零样本泛化**：新物体 70% 成功率
3. **多步推理**：复杂指令 60% 完成率
4. **保持通用能力**：语言测试与原始 PaLM 相当

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 视觉编码 | $\mathbf{V} = \text{ViT}(\mathbf{I})$ | `vis_tokens = vit(images)` | ViT-22B |
| 2. 状态编码 | $\mathbf{s}' = \text{MLP}(\mathbf{s})$ | `state_tokens = mlp(states)` | 连续值注入 |
| 3. 文本编码 | $\mathbf{T} = \text{Emb}(text)$ | `text_tokens = embed(text)` | LLM 嵌入 |
| 4. 融合 | $\mathbf{E} = [\mathbf{T}; \mathbf{V}; \mathbf{s}']$ | `emb = torch.cat([...], dim=1)` | 拼接 |
| 5. LLM 处理 | $\mathbf{H} = \text{PaLM}(\mathbf{E})$ | `output = palm(emb)` | 自回归 |
| 6. 生成 | $y_t \sim p(\cdot | \mathbf{H}, y_{<t})$ | `response = palm.generate()` | 文本输出 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.9
torch >= 2.0
transformers >= 4.28
flax >= 0.6  # PaLM 实现
```

### 数据准备
```python
# 具身多模态数据
sample = {
    'images': torch.FloatTensor,      # (B, n_views, 3, 224, 224)
    'state': torch.FloatTensor,       # (B, state_dim)
    'instruction': str,                # 文本指令
    'response': str                    # 期望输出 (规划/动作)
}
```

### 模型配置
```yaml
model:
  llm: palm-540b
  vision_encoder: vit-22b
  state_dim: 32
  d_model: 4096
  
training:
  batch_size: 32
  learning_rate: 5e-5
  warmup_ratio: 0.1
```

### 推理部署
```python
class PaLMEDeploy:
    def __init__(self, checkpoint):
        self.model = PaLME.load(checkpoint)
    
    def plan(self, images, state, instruction):
        prompt = f"Instruction: {instruction}\nPlan:"
        response = self.model.generate(
            images=images,
            state=state,
            text=prompt,
            max_length=512
        )
        return self.parse_plan(response)
    
    def parse_plan(self, text):
        # 解析文本为动作序列
        actions = []
        for line in text.split('\n'):
            if '→' in line:
                actions.append(parse_action(line))
        return actions
```

### 常见问题

#### Q1: 显存不足
**解决：** 使用模型并行，量化到 INT8

#### Q2: 推理延迟高
**解决：** 使用更小变体 (PaLM-E-8B)，启用 KV 缓存

---

## 参考资源

- **论文**: https://arxiv.org/abs/2303.03378
- **项目页**: https://palm-e.github.io/

---

*最后更新：2026-03-03*
