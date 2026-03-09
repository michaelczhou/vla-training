# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**Columbia, 2023** | arXiv:2303.04137

---

## 0. 核心观点与结论

### 研究问题
- 如何建模复杂的多模态动作分布？
- 如何生成平滑、自然的机器人动作？
- 如何结合视觉观测与动作生成？

### 核心贡献
1. **动作扩散**：将扩散模型应用于机器人动作生成
2. **视觉条件**：以图像为条件生成动作序列
3. **多模态建模**：自然处理多模态动作分布
4. **SOTA 性能**：在多个基准上超越 GAN/VAE 方法

### 主要结论
- 扩散模型优于传统生成模型
- 动作序列扩散比单步扩散更好
- 视觉条件有效传递语义信息
- 在复杂操作任务上表现优异

### 领域启示
- 扩散模型成为动作生成新范式
- 多模态建模对复杂任务关键
- 为精细操作提供新方案

---

## 1. 创新点详解

### 核心创新

#### 1.1 动作扩散过程
```python
class DiffusionPolicy(nn.Module):
    def __init__(self):
        self.unet = ConditionalUNet()
        self.noise_scheduler = DDPMScheduler()
    
    def forward(self, images, actions):
        # 前向：加噪
        t = torch.randint(0, T, (B,))
        noisy_actions = self.noise_scheduler.add_noise(actions, t)
        
        # 预测噪声
        noise_pred = self.unet(noisy_actions, t, images)
        
        # 损失
        loss = nn.MSELoss()(noise_pred, noise)
        return loss
    
    def sample(self, images, n_steps=100):
        # 反向：去噪生成
        actions = torch.randn(B, action_horizon, action_dim)
        for t in reversed(range(n_steps)):
            noise_pred = self.unet(actions, t, images)
            actions = self.noise_scheduler.step(noise_pred, t, actions)
        return actions
```

#### 1.2 视觉条件化
```python
# 视觉编码
vis_features = CNN(images)

# 条件注入 (FiLM)
gamma, beta = MLP(vis_features)
h = (1 + gamma) * unet_h + beta
```

#### 1.3 动作序列扩散
- 扩散整个动作序列 (而非单步)
- 捕捉时序依赖
- 生成更平滑轨迹

### 方法流程

```
┌─────────────────────────────────────────────────────────────┐
│                   扩散策略训练流程                           │
├─────────────────────────────────────────────────────────────┤
│  1. 数据准备：                                               │
│     - 采集演示 (图像 + 动作序列)                              │
│     - 动作归一化                                            │
├─────────────────────────────────────────────────────────────┤
│  2. 前向扩散：                                               │
│     - 采样时间步 t ~ Uniform(0, T)                          │
│     - 加噪：A^t = √ᾱ_t · A + √(1-ᾱ_t) · ε                 │
├─────────────────────────────────────────────────────────────┤
│  3. 噪声预测：                                               │
│     - 输入：噪声动作 A^t, 时间 t, 图像 I                     │
│     - 输出：预测噪声 ε̂                                      │
├─────────────────────────────────────────────────────────────┤
│  4. 损失计算：                                               │
│     - L = ||ε - ε̂||²                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   扩散策略推理流程                           │
├─────────────────────────────────────────────────────────────┤
│  1. 初始化：A^T ~ N(0, I)                                   │
│  2. 迭代去噪 (t = T → 0)：                                   │
│     - ε̂ = UNet(A^t, t, I)                                  │
│     - A^{t-1} = 去噪步 (A^t, ε̂, t)                         │
│  3. 输出：A^0 (干净动作序列)                                 │
│  4. 执行：执行第一步，重复整个过程                            │
└─────────────────────────────────────────────────────────────┘
```

### 数学表达

#### 前向扩散
$$q(\mathbf{A}^t | \mathbf{A}^0) = \mathcal{N}(\mathbf{A}^t; \sqrt{\bar{\alpha}_t} \mathbf{A}^0, (1-\bar{\alpha}_t)\mathbf{I})$$

#### 反向去噪
$$p_\theta(\mathbf{A}^{t-1} | \mathbf{A}^t, \mathbf{I}) = \mathcal{N}(\mathbf{A}^{t-1}; \mu_\theta(\mathbf{A}^t, t, \mathbf{I}), \Sigma_\theta(\mathbf{A}^t, t, \mathbf{I}))$$

#### 训练目标
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{A}^0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(\mathbf{A}^t, t, \mathbf{I})||^2 \right]$$

### 实验结论

#### 性能对比
| 方法 | 方块抓取 | 插孔 | 咖啡倒水 |
|------|----------|------|----------|
| BC-GMM | 75% | 45% | 30% |
| BC-VAE | 80% | 55% | 40% |
| Diffusion Policy | 95% | 85% | 75% |

#### 消融研究
- 动作序列扩散 vs 单步：+20%
- 视觉条件：+15%
- 更多扩散步数：+5% (收益递减)

---

## 2. 公式与流程对照表

| 步骤 | 数学公式 | 代码实现 | 说明 |
|------|----------|----------|------|
| 1. 采样时间 | $t \sim \text{Uniform}(0, T)$ | `t = torch.randint(0, T, (B,))` | 随机时间步 |
| 2. 加噪 | $\mathbf{A}^t = \sqrt{\bar{\alpha}_t}\mathbf{A}^0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | `noisy = scheduler.add_noise(actions, t, noise)` | 前向扩散 |
| 3. 视觉编码 | $\mathbf{v} = f_{\text{CNN}}(\mathbf{I})$ | `vis_feat = cnn(images)` | 图像特征 |
| 4. 噪声预测 | $\hat{\epsilon} = \text{UNet}(\mathbf{A}^t, t, \mathbf{v})$ | `noise_pred = unet(noisy, t, vis_feat)` | 条件 UNet |
| 5. 损失 | $\mathcal{L} = ||\epsilon - \hat{\epsilon}||^2$ | `loss = mse_loss(noise_pred, noise)` | MSE 损失 |
| 6. 采样 | $\mathbf{A}^{t-1} = \text{Step}(\mathbf{A}^t, \hat{\epsilon}, t)$ | `actions = scheduler.step(noise_pred, t, noisy)` | 反向去噪 |

---

## 3. 迁移部署指南

### 环境依赖
```bash
python >= 3.8
torch >= 1.10
diffusers >= 0.15  # HuggingFace 扩散库
```

### 数据准备
```python
# 动作序列格式
sample = {
    'images': torch.FloatTensor,      # (B, T, 3, 256, 256)
    'actions': torch.FloatTensor,     # (B, T, action_dim)
}

# 归一化
action_mean = dataset.actions.mean()
action_std = dataset.actions.std()
actions = (actions - action_mean) / action_std
```

### 模型配置
```yaml
model:
  action_horizon: 16
  action_dim: 14
  diffusion_steps: 100
  unet_channels: [256, 512, 1024]
  
training:
  batch_size: 128
  learning_rate: 1e-4
  beta_schedule: linear
```

### 推理部署
```python
class DiffusionPolicyDeploy:
    def __init__(self, checkpoint):
        self.model = DiffusionUNet.load(checkpoint)
        self.scheduler = DDPMScheduler(num_train_timesteps=100)
    
    def sample_actions(self, image, n_samples=1):
        # 初始化噪声
        actions = torch.randn(n_samples, 16, 14)
        
        # 迭代去噪
        for t in reversed(range(100)):
            noise_pred = self.model(actions, t, image)
            actions = self.scheduler.step(noise_pred, t, actions).prev_sample
        
        # 去归一化
        actions = actions * action_std + action_mean
        return actions
    
    def step(self, image):
        actions = self.sample_actions(image)
        return actions[0, 0]  # 执行第一步
```

### 优化技巧

#### 加速采样
```python
# 使用 DDIM 加速 (100 步 → 20 步)
scheduler = DDIMScheduler.from_config(ddpm_config)
actions = scheduler.sample(model, image, num_inference_steps=20)
```

#### 蒸馏
```python
# 蒸馏到单步模型
teacher = DiffusionPolicy()
student = MLP()

# 用 teacher 生成数据训练 student
pseudo_actions = teacher.sample(images)
student.train(images, pseudo_actions)
```

### 常见问题

#### Q1: 推理速度慢
**解决：** 使用 DDIM 加速，蒸馏到单步模型

#### Q2: 训练不稳定
**解决：** 调整学习率，使用 warmup，梯度裁剪

---

## 参考资源

- **论文**: https://arxiv.org/abs/2303.04137
- **项目页**: https://diffusion-policy.github.io/
- **代码**: https://github.com/columbia-ai-robotics/diffusion_policy

---

*最后更新：2026-03-03*
