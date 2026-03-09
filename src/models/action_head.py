"""
Action Head Module
==================
Implements action generation heads:
- Flow Matching (recommended, fast inference)
- Diffusion Policy (multi-modal, slower)
- Autoregressive (FAST tokenizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class FlowMatchingBlock(nn.Module):
    """
    流匹配 Transformer 块 (参考 π₀和 GR00T N1.6)
    
    结构:
    - LayerNorm + MultiheadAttention
    - LayerNorm + MLP (GELU 激活)
    
    优势:
    - 自注意力捕获长程依赖
    - MLP 提供非线性变换
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 16):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状 (B, hidden_dim)
            
        Returns:
            输出特征，形状 (B, hidden_dim)
        """
        # 自注意力 (添加序列维度)
        x_ = x.unsqueeze(1)  # (B, 1, hidden_dim)
        normed = self.norm1(x_)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out.squeeze(1)  # 残差连接
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class ActionHead(nn.Module):
    """Base class for action heads"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.action_dim = config.get('action_dim', 7)
        self.chunk_size = config.get('chunk_size', 10)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate actions from fused features
        
        Args:
            x: Fused features of shape (B, N, D)
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        raise NotImplementedError
    
    def sample(self, x: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        Sample actions (for generative models)
        
        Args:
            x: Fused features of shape (B, N, D)
            num_steps: Number of sampling steps
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        raise NotImplementedError


class FlowMatchingHead(ActionHead):
    """
    Flow Matching Action Head (优化版)
    
    基于条件流匹配 (Conditional Flow Matching, CFM)
    
    核心优势 (相比扩散模型):
    - 推理速度快：10 步 vs 100 步 (参考 openpi π₀)
    - 训练更稳定：直线路径，梯度更平滑
    - 理论优雅：基于最优传输理论
    
    数学原理:
    - 概率路径：p_t(x) = (1-t)p_0 + tp_1
    - 速度场：v_t(x) = E[v_t(x) | x_t = x]
    - 训练目标：L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
    - 生成 ODE: dx/dt = v_θ(x, t)
    
    参考实现:
    - Physical Intelligence openpi: https://github.com/Physical-Intelligence/openpi
    - LingBot-VLA: https://github.com/Robbyant/lingbot-vla
    
    Reference: https://arxiv.org/abs/2210.02747
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        input_dim = config.get('input_dim', 768)
        hidden_dim = config.get('hidden_dim', 1024)  # 增加隐藏维度 (参考 π₀)
        num_steps = config.get('num_steps', 10)  # 默认 10 步 (优化后足够)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Total action dimension (chunk_size * action_dim)
        self.total_action_dim = self.chunk_size * self.action_dim
        
        # Time embedding (正弦位置编码 + MLP)
        # 使用正弦编码提供更好的时间表示
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action embedding (将动作映射到隐藏空间)
        self.action_embed = nn.Linear(self.total_action_dim, hidden_dim)
        
        # Condition projection (将 VLM 特征映射到隐藏空间)
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        
        # Velocity prediction network (改进的 Transformer 风格)
        # 参考 GR00T N1.6 的 DiT 架构和 π₀的流匹配块
        self.velocity_blocks = nn.ModuleList([
            FlowMatchingBlock(hidden_dim, num_heads=16)
            for _ in range(8)  # 8 层 (平衡性能和速度)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.total_action_dim)
        )
        
        # 初始化：预测零速度 (稳定训练初期)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
    
    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time using sinusoidal positional encoding
        
        Args:
            t: Time values of shape (B,)
            
        Returns:
            time_embed: Tensor of shape (B, hidden_dim)
        """
        # Sinusoidal embedding
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        # Project through network
        time_embed = self.time_embed(emb)
        
        return time_embed
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity field
        
        Args:
            x: Fused features of shape (B, N, D)
            t: Time values of shape (B,)
            action: Action tensor of shape (B, chunk_size, action_dim) or None
            
        Returns:
            velocity: Predicted velocity of shape (B, total_action_dim)
        """
        # Pool features
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # Get time embedding
        time_embed = self._embed_time(t)  # (B, hidden_dim)
        
        # Prepare action input
        if action is None:
            # Sample random action during training
            action = torch.randn(x.shape[0], self.total_action_dim, device=x.device)
        else:
            action = action.reshape(x.shape[0], -1)  # (B, total_action_dim)
        
        # Concatenate inputs
        inputs = torch.cat([x_pooled, action, time_embed], dim=-1)  # (B, D + total_action_dim + hidden_dim)
        
        # Predict velocity
        velocity = self.velocity_net(inputs)  # (B, total_action_dim)
        
        return velocity
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        action_0: torch.Tensor, 
        action_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute flow matching loss
        
        Args:
            x: Fused features of shape (B, N, D)
            action_0: Source actions (noise) of shape (B, chunk_size, action_dim)
            action_1: Target actions of shape (B, chunk_size, action_dim)
            
        Returns:
            loss: Scalar loss value
            t: Time values used
        """
        B = x.shape[0]
        
        # Sample random time
        t = torch.rand(B, device=x.device)  # (B,)
        
        # Reshape actions
        a0 = action_0.reshape(B, -1)  # (B, total_action_dim)
        a1 = action_1.reshape(B, -1)  # (B, total_action_dim)
        
        # Interpolate action
        a_t = t.unsqueeze(-1) * a1 + (1 - t.unsqueeze(-1)) * a0  # (B, total_action_dim)
        
        # Target velocity (straight line)
        target_velocity = a1 - a0  # (B, total_action_dim)
        
        # Predict velocity
        predicted_velocity = self.forward(x, t, a_t)  # (B, total_action_dim)
        
        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss, t
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Sample actions using Euler integration
        
        Args:
            x: Fused features of shape (B, N, D)
            num_steps: Number of integration steps
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        B = x.shape[0]
        device = x.device
        
        # Start from noise
        a_t = torch.randn(B, self.total_action_dim, device=device)
        
        # Euler integration
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            
            # Predict velocity
            v_t = self.forward(x, t, a_t)  # (B, total_action_dim)
            
            # Euler step
            a_t = a_t + v_t * dt
        
        # Reshape to action sequence
        actions = a_t.reshape(B, self.chunk_size, self.action_dim)
        
        return actions


class DiffusionHead(ActionHead):
    """
    Diffusion Policy Action Head
    
    Based on Denoising Diffusion Probabilistic Models (DDPM)
    Multi-modal output, handles uncertainty well
    
    Reference: https://arxiv.org/abs/2006.11239
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        input_dim = config.get('input_dim', 768)
        hidden_dim = config.get('hidden_dim', 512)
        num_steps = config.get('num_steps', 100)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Total action dimension
        self.total_action_dim = self.chunk_size * self.action_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Noise prediction network (UNet-style)
        self.noise_net = nn.Sequential(
            nn.Linear(input_dim + self.total_action_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.total_action_dim)
        )
        
        # Precompute noise schedule
        self.register_buffer('betas', self._make_beta_schedule(num_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def _make_beta_schedule(
        self, 
        n_timestep: int, 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02
    ) -> torch.Tensor:
        """Create linear noise schedule"""
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timestep) ** 2
    
    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values"""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        time_embed = self.time_embed(emb)
        
        return time_embed
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        noisy_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise added to actions
        
        Args:
            x: Fused features of shape (B, N, D)
            t: Timestep of shape (B,)
            noisy_action: Noisy action of shape (B, total_action_dim)
            
        Returns:
            predicted_noise: Tensor of shape (B, total_action_dim)
        """
        # Pool features
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # Time embedding
        time_embed = self._embed_time(t)  # (B, hidden_dim)
        
        # Concatenate
        inputs = torch.cat([x_pooled, noisy_action, time_embed], dim=-1)
        
        # Predict noise
        noise = self.noise_net(inputs)
        
        return noise
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss
        
        Args:
            x: Fused features of shape (B, N, D)
            action: Clean action of shape (B, chunk_size, action_dim)
            
        Returns:
            loss: Scalar loss value
        """
        B = x.shape[0]
        device = x.device
        
        # Reshape action
        a0 = action.reshape(B, -1)  # (B, total_action_dim)
        
        # Sample random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(a0)
        
        # Add noise according to schedule
        alpha_cumprod_t = self.alphas_cumprod[t].unsqueeze(-1)  # (B, 1)
        noisy_action = (
            torch.sqrt(alpha_cumprod_t) * a0 + 
            torch.sqrt(1 - alpha_cumprod_t) * noise
        )
        
        # Predict noise
        predicted_noise = self.forward(x, t.float() / self.num_steps, noisy_action)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Sample actions using DDPM reverse process
        
        Args:
            x: Fused features of shape (B, N, D)
            num_steps: Number of denoising steps
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        B = x.shape[0]
        device = x.device
        
        # Start from pure noise
        a_t = torch.randn(B, self.total_action_dim, device=device)
        
        # Reverse diffusion
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i / num_steps, device=device)
            
            # Predict noise
            predicted_noise = self.forward(x, t, a_t)
            
            # Compute alpha values
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            alpha_cumprod_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)
            
            # Compute mean
            beta_t = 1 - alpha_t
            mean = (1 / torch.sqrt(alpha_t)) * (
                a_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )
            
            # Add noise (except for last step)
            if i > 0:
                noise = torch.randn_like(a_t)
                variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                a_t = mean + torch.sqrt(variance) * noise
            else:
                a_t = mean
        
        # Reshape
        actions = a_t.reshape(B, self.chunk_size, self.action_dim)
        
        return actions


class MLPHead(ActionHead):
    """
    Simple MLP Action Head
    
    Direct regression from features to actions
    Fast but unimodal output
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        input_dim = config.get('input_dim', 768)
        hidden_dim = config.get('hidden_dim', 512)
        dropout = config.get('dropout', 0.1)
        
        self.total_action_dim = self.chunk_size * self.action_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.total_action_dim)
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict actions directly
        
        Args:
            x: Fused features of shape (B, N, D)
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        # Pool features
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # Predict actions
        actions_flat = self.mlp(x_pooled)  # (B, total_action_dim)
        
        # Reshape
        actions = actions_flat.reshape(-1, self.chunk_size, self.action_dim)
        
        return actions
    
    def compute_loss(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss"""
        predicted = self.forward(x)
        return F.mse_loss(predicted, action)
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample = forward pass for MLP"""
        return self.forward(x)


def build_action_head(config: Dict[str, Any], input_dim: int) -> ActionHead:
    """
    Factory function to build action head from config
    
    Args:
        config: Configuration dictionary with 'type' key
        input_dim: Dimension of input features
        
    Returns:
        ActionHead instance
    """
    head_type = config.get('type', 'flow_matching').lower()
    
    config['input_dim'] = input_dim
    
    if head_type == 'flow_matching':
        return FlowMatchingHead(config)
    elif head_type == 'diffusion':
        return DiffusionHead(config)
    elif head_type == 'mlp':
        return MLPHead(config)
    else:
        raise ValueError(f"Unknown action head type: {head_type}")
