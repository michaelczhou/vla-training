# 具身智能代码示例库

本目录包含 10 篇经典论文的核心代码实现示例。

---

## 目录结构

```
code/
├── README.md                      # 本文件
├── 01-rt1-example.py             # RT-1 实现示例
├── 02-act-example.py             # ACT 实现示例
├── 03-diffusion-policy-example.py # Diffusion Policy 示例
├── 04-vla-example.py             # VLA (RT-2/OpenVLA) 示例
└── utils/
    ├── data_loader.py            # 数据加载工具
    ├── tokenizer.py              # Token 化工具
    └── robot_interface.py        # 机器人接口
```

---

## 示例代码

### 1. RT-1 风格 VLA

```python
# 01-rt1-example.py
import torch
import torch.nn as nn
from transformers import EfficientNetModel

class RT1StyleVLA(nn.Module):
    """RT-1 风格的视觉 - 语言 - 动作模型"""
    
    def __init__(self, action_dim=11, action_bins=256):
        super().__init__()
        # 视觉编码
        self.vision_encoder = EfficientNetModel.from_pretrained('efficientnet-b3')
        self.vision_proj = nn.Linear(1536, 512)
        
        # 文本编码
        self.text_embed = nn.Embedding(32000, 512)
        
        # Transformer
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=12)
        
        # 动作 head
        self.action_head = nn.Linear(512, action_dim * action_bins)
        
        self.action_dim = action_dim
        self.action_bins = action_bins
    
    def forward(self, images, text_tokens, action_history=None):
        """
        Args:
            images: (B, 3, 224, 224)
            text_tokens: (B, T_text)
            action_history: (B, T_history, action_dim) 可选
        """
        B = images.shape[0]
        
        # 视觉编码
        vis_features = self.vision_encoder(images).last_hidden_state
        vis_tokens = self.vision_proj(vis_features)  # (B, N_vis, 512)
        vis_tokens = vis_tokens.flatten(1, 2)  # 展平为序列
        
        # 文本编码
        text_embeds = self.text_embed(text_tokens)  # (B, T_text, 512)
        
        # 拼接 token
        tokens = torch.cat([text_embeds, vis_tokens], dim=1)  # (B, T_total, 512)
        tokens = tokens.permute(1, 0, 2)  # (T_total, B, 512)
        
        # Transformer (无 target，仅编码)
        output = self.transformer(tokens, tokens)
        output = output.permute(1, 0, 2)  # (B, T_total, 512)
        
        # 取最后一个 token 预测动作
        last_output = output[:, -1, :]  # (B, 512)
        action_logits = self.action_head(last_output)  # (B, action_dim * bins)
        
        # 重塑为 (B, action_dim, bins)
        action_logits = action_logits.view(B, self.action_dim, self.action_bins)
        
        return action_logits
    
    def predict(self, images, text_tokens):
        """推理：预测动作"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, text_tokens)
            # 每维取 argmax
            actions_disc = logits.argmax(dim=-1)  # (B, action_dim)
            # 去离散化
            actions = (actions_disc.float() / (self.action_bins - 1))
            actions = actions * 2 - 1  # 归一化到 [-1, 1]
        return actions
```

---

### 2. ACT 实现

```python
# 02-act-example.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """条件 VAE 用于动作编码"""
    
    def __init__(self, action_dim, action_horizon, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim * action_horizon, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * action_horizon),
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, actions):
        """编码动作序列为潜变量"""
        B = actions.shape[0]
        actions_flat = actions.view(B, -1)
        h = self.encoder(actions_flat)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """从潜变量解码动作"""
        actions_flat = self.decoder(z)
        return actions_flat
    
    def forward(self, actions):
        """训练时前向传播"""
        mu, logvar = self.encode(actions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, n_samples=1):
        """推理时采样"""
        z = torch.randn(n_samples, self.latent_dim)
        actions = self.decode(z)
        return actions


class ACT(nn.Module):
    """Action Chunking with Transformers"""
    
    def __init__(self, action_dim=14, action_horizon=100, latent_dim=32):
        super().__init__()
        # VAE
        self.vae = CVAE(action_dim, action_horizon, latent_dim)
        
        # 视觉编码
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # Transformer
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=4)
        
        # 查询 token
        self.action_query = nn.Parameter(torch.randn(1, 1, 512))
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim
    
    def forward(self, images, actions=None):
        """
        Args:
            images: (B, 3, H, W)
            actions: (B, action_horizon, action_dim) 训练时提供
        """
        B = images.shape[0]
        
        # 视觉编码
        vis_features = self.vision_encoder(images)  # (B, 512)
        vis_tokens = vis_features.unsqueeze(0)  # (1, B, 512)
        
        # VAE 编码/采样
        if actions is not None:
            # 训练时：从后验采样
            mu, logvar = self.vae.encode(actions)
            z = self.vae.reparameterize(mu, logvar)
        else:
            # 推理时：从先验采样
            z = torch.randn(B, self.latent_dim)
        
        # 将潜变量投影到 transformer 维度
        z_tokens = self.vae.decoder[:2](z).unsqueeze(0)  # (1, B, 512)
        
        # 拼接 token
        tokens = torch.cat([vis_tokens, z_tokens], dim=0)  # (2, B, 512)
        
        # 动作查询
        action_query = self.action_query.expand(-1, B, -1)  # (1, B, 512)
        
        # Transformer 解码
        output = self.transformer(action_query, tokens)  # (1, B, 512)
        
        # 预测动作
        action_flat = self.vae.decoder[2:](output.squeeze(0))
        actions_pred = action_flat.view(B, self.action_horizon, self.action_dim)
        
        return actions_pred
    
    def loss_fn(self, actions_pred, actions_gt, mu, logvar, kl_weight=10.0):
        """ACT 损失函数"""
        # 重建损失
        recon_loss = F.mse_loss(actions_pred, actions_gt)
        
        # KL 散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        loss = recon_loss + kl_weight * kl_loss
        return loss, recon_loss, kl_loss
```

---

### 3. Diffusion Policy

```python
# 03-diffusion-policy-example.py
import torch
import torch.nn as nn
import math

class ConditionalUNet(nn.Module):
    """条件 UNet 用于动作扩散"""
    
    def __init__(self, action_dim, action_horizon, time_dim=256):
        super().__init__()
        
        # 时间编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 视觉条件
        self.vis_cond = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, time_dim),
        )
        
        # U-Net 主体
        self.unet = nn.Sequential(
            nn.Linear(action_dim * action_horizon + time_dim * 2, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, action_dim * action_horizon),
        )
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
    
    def forward(self, noisy_actions, timestep, vis_features):
        """
        Args:
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) 时间步
            vis_features: (B, 3, H, W) 视觉输入
        """
        B = noisy_actions.shape[0]
        
        # 时间编码
        t_emb = self.time_embed(timestep.float().unsqueeze(-1))  # (B, time_dim)
        
        # 视觉条件
        v_emb = self.vis_cond(vis_features)  # (B, time_dim)
        
        # 展平动作
        actions_flat = noisy_actions.view(B, -1)  # (B, action_horizon * action_dim)
        
        # 拼接
        x = torch.cat([actions_flat, t_emb, v_emb], dim=-1)
        
        # U-Net 预测
        noise_pred_flat = self.unet(x)
        noise_pred = noise_pred_flat.view(B, self.action_horizon, self.action_dim)
        
        return noise_pred


class DiffusionScheduler:
    """DDPM 调度器"""
    
    def __init__(self, num_train_timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # 线性 beta 调度
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, actions, t, noise=None):
        """前向扩散：加噪"""
        if noise is None:
            noise = torch.randn_like(actions)
        
        B = actions.shape[0]
        # 获取对应时间步的 alpha
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(B, 1, 1)
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas_cumprod[t]).view(B, 1, 1)
        
        # 加噪
        noisy_actions = sqrt_alphas_cumprod * actions + sqrt_one_minus_alphas * noise
        return noisy_actions, noise
    
    def step(self, noise_pred, t, noisy_actions):
        """反向步：去噪"""
        B = noisy_actions.shape[0]
        
        # 获取系数
        alpha = self.alphas[t].view(B, 1, 1)
        alpha_cumprod = self.alphas_cumprod[t].view(B, 1, 1)
        
        # 预测干净动作
        pred_original = (noisy_actions - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
        
        # 计算上一步的分布
        mean = (torch.sqrt(alpha) * (1 - self.alphas_cumprod[t-1 if t > 0 else 0]) / (1 - alpha_cumprod) * noisy_actions +
                torch.sqrt(self.alphas[t-1 if t > 0 else 0]) * (1 - alpha) / (1 - alpha_cumprod) * pred_original)
        
        # 采样
        variance = (1 - self.alphas_cumprod[t-1 if t > 0 else 0]) / (1 - alpha_cumprod) * (1 - alpha)
        noise = torch.randn_like(noisy_actions) if t > 0 else 0
        
        prev_sample = mean + torch.sqrt(variance) * noise
        return prev_sample


class DiffusionPolicy(nn.Module):
    """完整的 Diffusion Policy"""
    
    def __init__(self, action_dim=14, action_horizon=16):
        super().__init__()
        self.unet = ConditionalUNet(action_dim, action_horizon)
        self.scheduler = DiffusionScheduler(num_train_timesteps=100)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
    
    def forward(self, images, actions):
        """训练：预测噪声"""
        B = actions.shape[0]
        
        # 采样时间步
        t = torch.randint(0, 100, (B,), device=actions.device)
        
        # 加噪
        noisy_actions, noise = self.scheduler.add_noise(actions, t)
        
        # 预测噪声
        noise_pred = self.unet(noisy_actions, t, images)
        
        # 损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, images, n_steps=100):
        """推理：采样动作"""
        B = images.shape[0]
        
        # 初始化噪声
        actions = torch.randn(B, self.action_horizon, self.action_dim, device=images.device)
        
        # 迭代去噪
        for t in reversed(range(n_steps)):
            t_batch = torch.full((B,), t, device=images.device)
            noise_pred = self.unet(actions, t_batch, images)
            actions = self.scheduler.step(noise_pred, t, actions)
        
        return actions
```

---

## 使用示例

### 训练 RT-1 风格模型

```python
# train_rt1.py
from rt1_example import RT1StyleVLA
from utils.data_loader import RobotDataset

# 准备数据
dataset = RobotDataset('path/to/data')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

# 初始化模型
model = RT1StyleVLA().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练
for epoch in range(100):
    for batch in dataloader:
        images = batch['images'].cuda()
        text_tokens = batch['text_tokens'].cuda()
        actions = batch['actions'].cuda()
        
        # 前向
        logits = model(images, text_tokens)
        
        # 损失 (交叉熵)
        actions_disc = ((actions + 1) / 2 * 255).long()
        loss = nn.functional.cross_entropy(
            logits.view(-1, 256),
            actions_disc.view(-1)
        )
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 参考资源

- 完整实现参考各论文官方代码仓库
- 本示例用于学习理解，生产环境请使用官方实现
- 更多示例见各论文目录下的详细文档

---

*最后更新：2026-03-03*
