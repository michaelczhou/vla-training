# GATO: A Generalist Agent

**DeepMind, 2022** | arXiv:2205.06173

---

## 0. 核心观点与结论

### 研究问题
- 能否构建单一模型处理多种模态和任务？
- 通用智能体是否能超越任务特定模型？
- 如何统一表示不同领域的序列数据？

### 核心贡献
1. **单一 Transformer 架构**：处理 600+ 不同任务
2. **多模态序列建模**：统一处理文本、图像、动作、按钮按压
3. **跨领域迁移**：从游戏到机器人到对话的零样本能力
4. **证明通用性可行**：单一权重处理多样化任务

### 主要结论
- 通用架构可以处理高度多样化任务
- 任务性能与专用模型相当或略低
- 数据质量和多样性至关重要
- 序列建模是统一不同任务的关键

### 领域启示
- 开启了"通用智能体"研究方向
- 证明了 Transformer 作为通用架构的潜力
- 为后续多模态基础模型提供思路

---

## 1. 创新点详解

### 解决的问题
**之前方法不行原因：**
- 每个任务需要独立模型
- 跨任务知识无法共享
- 无法利用相关任务的协同效应

### 核心创新

#### 1.1 统一序列表示
```
所有任务 → 离散 token 序列 → Transformer → 输出 token → 任务特定解码
```

#### 1.2 多模态 Tokenization
```python
# 文本：SentencePiece 分词
text_tokens = sentencepiece.encode(text)  # [1, 32000)

# 图像：ViT patch 嵌入
image_tokens = vit.patch_embed(image)     # 256 tokens

# 动作：离散化
action_tokens = discretize(action, bins=1024)

# 按钮：直接编码
button_tokens = one_hot(button_state)
```

#### 1.3 因果注意力掩码
- 所有任务使用相同因果掩码
- 确保自回归生成
- 支持不同长度的输入输出

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                      GATO 架构流程                           │
├─────────────────────────────────────────────────────────────┤
│  输入编码 (所有任务统一)：                                    │
│  - 文本 → SentencePiece → token                             │
│  - 图像 → ViT Patch → 256 token                             │
│  - 动作 → 离散化 → token                                    │
│  - 按钮 → One-hot → token                                   │
├─────────────────────────────────────────────────────────────┤
│  Transformer 编码器 - 解码器：                                │
│  - 12 层 Transformer                                        │
│  - 因果注意力掩码                                            │
│  - d_model = 2048                                           │
├─────────────────────────────────────────────────────────────┤
│  输出解码 (任务特定)：                                        │
│  - 文本任务 → softmax → 词表                                │
│  - 图像任务 → 解码器 → 图像                                 │
│  - 动作任务 → softmax → 动作分布                            │
│  - 游戏任务 → softmax → 按钮分布                            │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 统一序列建模
$$p(\mathbf{x}_{1:T}) = \prod_{t=1}^T p(x_t | \mathbf{x}_{<t})$$

#### 多任务损失
$$\mathcal{L} = \sum_{k \in \text{tasks}} w_k \cdot \mathcal{L}_k$$

#### 条件生成
$$p(y | x, \text{task\_id}) = \text{Transformer}(x, \text{task\_id})$$

### 实验结论

#### 任务覆盖
| 任务类型 | 数量 | 示例 |
|----------|------|------|
| 游戏 (Atari) | 300+ | Pong, Breakout |
| 视觉问答 | 50+ | VQA, OK-VQA |
| 对话 | 100+ | Chat, QA |
| 机器人控制 | 50+ | 抓取，导航 |
| 其他 | 100+ | 图像描述，翻译 |

#### 关键结果
1. **游戏性能**：达到人类水平的 75%
2. **VQA 准确率**：60%+ (零样本)
3. **对话质量**：与专用模型相当
4. **机器人控制**：成功执行简单指令

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 文本编码 | $t_i = \text{SP}(text)_i$ | `text_tokens = sp.encode(text)` | SentencePiece |
| 2. 图像编码 | $v_i = \text{ViT}(I)_i$ | `image_tokens = vit(image)` | ViT patch |
| 3. 动作编码 | $a_i = \lfloor a \cdot 1023 \rfloor$ | `action_tokens = discretize(action)` | 离散化 |
| 4. 序列拼接 | $x = [t; v; a; ...]$ | `sequence = torch.cat(tokens)` | 统一序列 |
| 5. Transformer | $h = \text{Transformer}(x)$ | `output = transformer(sequence)` | 12 层 |
| 6. 输出预测 | $\hat{y} = \text{softmax}(Wh)$ | `logits = head(output)` | 任务特定 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.8
torch >= 1.10
transformers >= 4.15
gym >= 0.21  # 游戏环境
```

### 数据准备
```python
# 统一数据格式
sample = {
    'sequence': torch.LongTensor,  # 拼接的 token 序列
    'mask': torch.BoolTensor,      # 注意力掩码
    'targets': torch.LongTensor,   # 预测目标
    'task_id': int                 # 任务标识
}
```

### 模型配置
```yaml
model:
  d_model: 2048
  n_heads: 16
  n_layers: 12
  vocab_size: 32000
  
training:
  batch_size: 256
  learning_rate: 1e-4
  sequence_length: 1024
```

### 常见问题

#### Q1: 多任务干扰
**解决：** 调整任务权重，使用梯度手术

#### Q2: 推理速度慢
**解决：** 使用更小的模型变体，启用缓存

---

## 参考资源

- **论文**: https://arxiv.org/abs/2205.06173
- **项目页**: https://deepmind.google/discover/blog/gato-a-generalist-agent/

---

*最后更新：2026-03-03*
