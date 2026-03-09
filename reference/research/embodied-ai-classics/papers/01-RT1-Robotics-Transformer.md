# RT-1: Robotics Transformer for Real-World Control at Scale

**Google, 2022** | arXiv:2212.06817

---

## 0. 核心观点与结论

### 研究问题
- 如何将大规模、多样化、任务无关的数据集知识迁移到特定机器人任务？
- 如何构建能够吸收多样化机器人数据的高容量架构？
- 机器人模型如何随数据量、模型大小和数据多样性进行扩展？

### 核心贡献
1. **提出 Robotics Transformer (RT-1)**：专为机器人控制设计的 Transformer 架构
2. **大规模真实机器人数据收集**：13 个月、700+ 任务、17 万+ 轨迹
3. **证明可扩展性**：模型性能随数据量、模型大小、数据多样性提升
4. **零样本与少样本迁移**：展示跨任务、跨场景的泛化能力

### 主要结论
- 大规模任务无关训练是成功的关键
- Transformer 架构展现出良好的可扩展性
- 数据多样性比单纯的数据量更重要
- 模型能够泛化到新物体、新位置、新干扰

### 领域启示
- 开启了"基础模型 + 机器人"的研究范式
- 证明了大规模真实世界数据收集的价值
- 为后续 VLA (Vision-Language-Action) 模型奠定基础

---

## 1. 创新点详解

### 解决的问题
**之前方法不行原因：**
- 传统模仿学习：需要任务特定数据，泛化能力差
- 强化学习：样本效率低，难以在真实机器人上训练
- 早期 Transformer 应用：未针对机器人时序控制优化

### 核心创新

#### 1.1 高效 Tokenization
```
图像 → EfficientNet-B3 → 特征图 → 空间展平 → 视觉 token
语言指令 → SentencePiece → 文本 token
动作 → 离散化 (256 bins) → 动作 token
```

#### 1.2 条件化动作生成
```python
# 伪代码示例
class RT1(nn.Module):
    def __init__(self):
        self.vision_encoder = EfficientNetB3()
        self.transformer = TransformerDecoder(
            d_model=512,
            n_heads=8,
            n_layers=12
        )
        self.action_head = nn.Linear(512, action_dim * 256)
    
    def forward(self, images, text_instruction, history):
        # 视觉编码
        vis_tokens = self.vision_encoder(images)
        vis_tokens = vis_tokens.flatten(2).permute(0, 2, 1)
        
        # 文本编码
        text_tokens = self.text_encoder(text_instruction)
        
        # Transformer 解码
        tokens = torch.cat([text_tokens, vis_tokens, history], dim=1)
        output = self.transformer(tokens)
        
        # 动作预测
        actions = self.action_head(output[:, -1:])
        return actions
```

#### 1.3 动作离散化
- 每个动作维度离散化为 256 个 bin
- 将回归问题转化为分类问题
- 提高训练稳定性和泛化能力

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RT-1 架构流程                           │
├─────────────────────────────────────────────────────────────┤
│  输入：                                                      │
│  - 多相机图像 (3x224x224)                                   │
│  - 自然语言指令                                             │
│  - 动作历史 (可选)                                           │
├─────────────────────────────────────────────────────────────┤
│  1. 视觉编码                                                 │
│     图像 → EfficientNet-B3 → 32x32 特征图 → 1024 token      │
├─────────────────────────────────────────────────────────────┤
│  2. 文本编码                                                 │
│     指令 → SentencePiece → 最多 32 token                    │
├─────────────────────────────────────────────────────────────┤
│  3. Transformer 解码                                         │
│     所有 token 拼接 → 12 层 Transformer → 隐藏状态           │
├─────────────────────────────────────────────────────────────┤
│  4. 动作输出                                                 │
│     最后隐藏状态 → Linear → 11 维动作 (每个 256 类)          │
│     - 基座 x, y, yaw                                         │
│     - 手臂 x, y, z, roll, pitch, yaw                         │
│     - 夹爪开合                                               │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 视觉 Tokenization
$$\mathbf{v}_i = \text{EffNet}(\mathbf{I})_i \in \mathbb{R}^{d_v}, \quad i=1,\ldots,N_v$$

#### 动作离散化
$$a_k^{\text{disc}} = \left\lfloor \frac{a_k - a_k^{\min}}{a_k^{\max} - a_k^{\min}} \times 255 \right\rfloor$$

#### 条件化策略
$$\pi(a_t | s_t, \text{text}) = \prod_{k=1}^{K} \text{Categorical}(a_{t,k} | f_\theta(s_t, \text{text}))$$

### 实验结论

#### 数据集统计
| 指标 | 数值 |
|------|------|
| 收集时间 | 13 个月 |
| 任务数量 | 700+ |
| 轨迹数量 | 170,000+ |
| 机器人数量 | 13 台 |
| 场景数量 | 3 个厨房 + 办公室 |

#### 关键结果
1. **规模扩展**：
   - 数据量增加 10 倍 → 成功率 +25%
   - 模型大小增加 4 倍 → 成功率 +15%

2. **泛化能力**：
   - 新物体：85% 成功率
   - 新位置：78% 成功率
   - 新干扰：72% 成功率

3. **零样本迁移**：
   - 跨场景：65% 成功率
   - 组合任务：58% 成功率

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 视觉编码 | $\mathbf{V} = f_{\text{CNN}}(\mathbf{I})$ | `vis_tokens = vision_encoder(images)` | EfficientNet-B3 提取特征 |
| 2. 特征展平 | $\mathbf{v}_i = \text{Flatten}(\mathbf{V})$ | `vis_tokens.flatten(2).permute(0,2,1)` | 空间维度转为序列 |
| 3. 文本编码 | $\mathbf{T} = \text{Embed}(\text{text})$ | `text_tokens = text_encoder(instruction)` | SentencePiece 分词 |
| 4. Token 拼接 | $\mathbf{X} = [\mathbf{T}; \mathbf{V}; \mathbf{A}_{<t}]$ | `tokens = torch.cat([text, vis, history], dim=1)` | 拼接所有输入 |
| 5. Transformer | $\mathbf{H} = \text{Transformer}(\mathbf{X})$ | `output = transformer(tokens)` | 12 层自注意力 |
| 6. 动作预测 | $\hat{a}_k = \text{argmax}(\text{softmax}(W_k \mathbf{h}))$ | `actions = action_head(output[:, -1:])` | 每维独立分类 |
| 7. 动作去离散 | $a_k = a_k^{\min} + \frac{\hat{a}_k}{255}(a_k^{\max} - a_k^{\min})$ | `actions = discretizer.decode(actions)` | 转回连续值 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
# 基础环境
python >= 3.8
torch >= 1.10
tensorflow >= 2.8  # 用于 EfficientNet

# 机器人接口
rospy >= 1.15
numpy >= 1.20
opencv-python >= 4.5

# 可选：训练依赖
transformers >= 4.15
wandb >= 0.12  # 实验追踪
```

### 数据准备

#### 1. 数据格式
```python
# 单条轨迹数据结构
trajectory = {
    'steps': [
        {
            'image': np.array,      # (224, 224, 3)
            'instruction': str,      # "pick up the apple"
            'action': np.array,      # (11,) 连续值
            'termination': bool      # 是否结束
        },
        # ...
    ]
}
```

#### 2. 数据增强
```python
transforms = [
    RandomRotation(15°),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomCrop(224),
]
```

### 模型配置

#### 基础配置
```yaml
model:
  vision_encoder: efficientnet-b3
  d_model: 512
  n_heads: 8
  n_layers: 12
  dropout: 0.1
  
training:
  batch_size: 64
  learning_rate: 1e-4
  warmup_steps: 1000
  max_steps: 100000
  
action:
  discretization_bins: 256
  action_horizon: 1
```

### 训练流程

```python
# 训练循环示例
for batch in dataloader:
    # 前向传播
    actions_pred = model(
        images=batch['images'],
        text=batch['instructions'],
        history=batch['action_history']
    )
    
    # 计算损失 (交叉熵)
    loss = 0
    for i in range(action_dim):
        loss += nn.CrossEntropyLoss()(
            actions_pred[:, i],
            batch['actions_disc'][:, i]
        )
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### 推理部署

#### 实时推理
```python
class RT1Deploy:
    def __init__(self, checkpoint_path):
        self.model = RT1.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.action_history = []
    
    def step(self, image, instruction):
        with torch.no_grad():
            action = self.model(
                image.unsqueeze(0),
                [instruction],
                self.action_history
            )
            action = action.squeeze(0).cpu().numpy()
        
        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        return action
```

#### 部署优化
1. **模型量化**：FP16/INT8 量化，2-4x 加速
2. **TensorRT**：NVIDIA GPU 优化
3. **ONNX**：跨平台部署

### 常见问题

#### Q1: 推理延迟过高
**解决：**
- 使用 EfficientNet-B0 替代 B3
- 减少 Transformer 层数至 6 层
- 启用 TensorRT 优化

#### Q2: 泛化能力差
**解决：**
- 增加训练数据多样性
- 加强数据增强
- 使用领域自适应技术

#### Q3: 动作不平滑
**解决：**
- 增加动作历史长度
- 添加动作平滑滤波器
- 使用动作分块 (参考 ACT)

---

## 参考资源

- **项目主页**: https://robotics-transformer1.github.io
- **论文**: https://arxiv.org/abs/2212.06817
- **代码**: https://github.com/google-research/robotics_transformer
- **数据集**: https://robotics-transformer1.github.io/rt-1-dataset

---

*最后更新：2026-03-03*
