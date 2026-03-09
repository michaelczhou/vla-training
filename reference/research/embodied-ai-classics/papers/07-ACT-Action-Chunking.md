# ACT: Action Chunking with Transformers

**Stanford, 2023** | arXiv:2304.13705

---

## 0. 核心观点与结论

### 研究问题
- 如何改善 Transformer 动作预测的时序一致性？
- 如何减少累积误差？
- 如何实现平滑、自然的机器人动作？

### 核心贡献
1. **动作分块**：一次预测多步动作序列
2. **时序集成**：重叠预测提高平滑度
3. **VAE 潜变量**：建模多模态动作分布
4. **高成功率**：复杂任务 90%+ 成功率

### 主要结论
- 动作分块显著改善时序一致性
- VAE 潜变量捕捉多模态分布
- 时序集成减少累积误差
- 在双手机器人上表现优异

### 领域启示
- 动作分块成为 VLA 标准组件
- 多模态动作建模重要
- 为复杂操作任务提供解决方案

---

## 1. 创新点详解

### 核心创新

#### 1.1 动作分块 (Action Chunking)
```python
# 传统：一次预测一步
action_t = model(obs_t, task)

# ACT: 一次预测 H 步
actions_t:t+H = model(obs_t, task)  # H=100

# 执行：只执行第一步，然后重新预测
execute(actions_t)
obs_t+1 = get_observation()
actions_t+1:t+1+H = model(obs_t+1, task)
```

#### 1.2 时序集成 (Temporal Ensemble)
```python
# 多个时间步的预测重叠
prediction_from_t_0: [a_0, a_1, a_2, ..., a_H]
prediction_from_t_1: [    a_1, a_2, ..., a_H, a_H+1]
prediction_from_t_2: [        a_2, ..., a_H, a_H+1, a_H+2]

# 加权平均
final_a_t = w_0 * pred_t_0[a_t] + w_1 * pred_t_1[a_t] + ...
```

#### 1.3 VAE 潜变量
```python
class ACT(nn.Module):
    def __init__(self):
        self.encoder = VAEEncoder()  # 编码动作序列
        self.decoder = TransformerDecoder()
        self.latent_dim = 32
    
    def forward(self, images, task, actions=None):
        # 训练时：从后验采样
        if actions is not None:
            mu, logvar = self.encoder(actions)
            z = mu + exp(0.5 * logvar) * epsilon
        # 推理时：从先验采样
        else:
            z = torch.randn(B, self.latent_dim)
        
        # 条件解码
        action_chunks = self.decoder(images, task, z)
        return action_chunks
```

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                       ACT 训练流程                           │
├─────────────────────────────────────────────────────────────┤
│  1. 数据准备：                                               │
│     - 采集演示轨迹 (图像 + 动作)                              │
│     - 动作分块：将长序列切分为重叠块                           │
├─────────────────────────────────────────────────────────────┤
│  2. VAE 编码：                                               │
│     - 输入：动作块 A_t:t+H                                   │
│     - 输出：潜变量 z ~ q(z | A)                              │
├─────────────────────────────────────────────────────────────┤
│  3. Transformer 解码：                                       │
│     - 输入：图像、任务、潜变量 z                              │
│     - 输出：预测动作块 Â_t:t+H                               │
├─────────────────────────────────────────────────────────────┤
│  4. 损失函数：                                               │
│     - 重建损失：||A - Â||²                                   │
│     - KL 散度：KL(q(z|A) || p(z))                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      ACT 推理流程                            │
├─────────────────────────────────────────────────────────────┤
│  1. 采样潜变量 z ~ N(0, I)                                  │
│  2. 预测动作块 Â_t:t+H = Decoder(obs_t, task, z)            │
│  3. 执行第一步 a_t                                          │
│  4. 获取新观测 obs_t+1                                      │
│  5. 重复步骤 2-4                                            │
│  6. 时序集成：对重叠预测加权平均                              │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### VAE 目标
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{A})} \left[ \log p_\theta(\mathbf{A}|\mathbf{z}, \mathbf{O}, \text{task}) \right] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{A}) || p(\mathbf{z}))$$

#### 动作分块
$$\mathbf{A}_{t:t+H} = [a_t, a_{t+1}, \ldots, a_{t+H-1}] \in \mathbb{R}^{H \times d_a}$$

#### 时序集成
$$a_t^{\text{final}} = \frac{\sum_{\tau=0}^{H-1} w_\tau \cdot \hat{a}_{t|t-\tau}}{\sum_{\tau=0}^{H-1} w_\tau}$$

### 实验结论

#### 关键结果
| 任务 | 无分块 | 动作分块 | ACT (完整) |
|------|--------|----------|------------|
| 方块抓取 | 45% | 70% | 95% |
| 插孔 | 20% | 55% | 90% |
| 双手协作 | 10% | 40% | 85% |

#### 消融研究
- 动作分块：+25% 成功率
- 时序集成：+10% 成功率
- VAE 潜变量：+15% 成功率

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. VAE 编码 | $\mu, \sigma^2 = f_{\text{enc}}(\mathbf{A})$ | `mu, logvar = encoder(actions)` | 动作→潜变量 |
| 2. 重参数化 | $\mathbf{z} = \mu + \sigma \odot \epsilon$ | `z = mu + torch.exp(0.5*logvar) * eps` | 可微采样 |
| 3. Transformer | $\mathbf{H} = \text{Transformer}(\mathbf{O}, \text{task}, \mathbf{z})$ | `output = transformer(obs, task, z)` | 条件解码 |
| 4. 动作预测 | $\hat{\mathbf{A}} = f_{\text{dec}}(\mathbf{H})$ | `action_chunk = decoder(output)` | 预测 H 步 |
| 5. 时序集成 | $a_t = \sum w_\tau \hat{a}_{t|t-\tau}$ | `action = temporal_ensemble(predictions)` | 平滑 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.8
torch >= 1.10
transformers >= 4.15
```

### 数据准备
```python
# 动作分块参数
chunk_size = 100      # 每块 100 步
chunk_overlap = 50    # 重叠 50 步

# 创建分块数据集
def create_chunks(trajectory):
    chunks = []
    for i in range(0, len(traj) - chunk_size, chunk_size - chunk_overlap):
        chunk = trajectory[i:i+chunk_size]
        chunks.append(chunk)
    return chunks
```

### 模型配置
```yaml
model:
  chunk_size: 100
  latent_dim: 32
  transformer_layers: 8
  d_model: 512
  n_heads: 8

training:
  batch_size: 64
  learning_rate: 1e-4
  kl_weight: 10.0
```

### 推理部署
```python
class ACTDeploy:
    def __init__(self, checkpoint):
        self.model = ACT.load(checkpoint)
        self.prediction_buffer = deque(maxlen=100)
    
    def step(self, observation, task):
        # 采样潜变量
        z = torch.randn(1, self.model.latent_dim)
        
        # 预测动作块
        action_chunk = self.model.predict(observation, task, z)
        
        # 加入缓冲
        self.prediction_buffer.append(action_chunk)
        
        # 时序集成
        action = self.temporal_ensemble()
        
        return action[0]  # 执行第一步
    
    def temporal_ensemble(self):
        # 加权平均重叠预测
        weights = np.linspace(1, 0, len(self.prediction_buffer))
        # ... 集成逻辑
        return ensemble_action
```

### 常见问题

#### Q1: 动作抖动
**解决：** 增加 chunk_size，调整时序集成权重

#### Q2: 多模态崩溃
**解决：** 增加 KL weight，使用更强大的 VAE

---

## 参考资源

- **论文**: https://arxiv.org/abs/2304.13705
- **项目页**: https://action-chunking.github.io/
- **代码**: https://github.com/tonyzhaozh/act

---

*最后更新：2026-03-03*
