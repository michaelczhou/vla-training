# Physical Intelligence 技术原理 - Encoder/Decoder/Loss 深度解析

## 目录

1. [VLM Encoder 完整架构解析](#1-vlm-encoder-完整架构解析)
2. [Action Expert Decoder 完整架构解析](#2-action-expert-decoder-完整架构解析)
3. [Loss 函数完整数学推导](#3-loss-函数完整数学推导)
4. [可视化架构图](#4-可视化架构图)
5. [完整代码实现与逐行注释](#5-完整代码实现与逐行注释)

---

## 1. VLM Encoder 完整架构解析

### 1.1 PaliGemma-3B 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PaliGemma-3B VLM Encoder                               │
│                                                                                  │
│   ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐  │
│   │   RGB Image     │         │  Language Text  │         │  Proprio State  │  │
│   │   224×224×3     │         │  Token Sequence │         │  [q₁, q₂, ...]  │  │
│   └────────┬────────┘         └────────┬────────┘         └────────┬────────┘  │
│            │                           │                           │           │
│            ▼                           ▼                           ▼           │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                          Input Embedding Layer                           │  │
│   │  • Image Patches: 224/14 = 16×16 = 256 patches → Linear → 1152 dims    │  │
│   │  • Text Tokens: Token IDs → Embedding Table → 2048 dims                 │  │
│   │  • Proprio: Linear Projection → 512 dims                                │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                      │
│                                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                        Vision Encoder (SigLIP ViT)                       │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Patch Embedding + Position Embedding                            │   │  │
│   │  │  Input: [256, 1152]                                              │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Transformer Block 1                                             │   │  │
│   │  │  • LayerNorm → Multi-Head Attention (16 heads, 1152 dims)       │   │  │
│   │  │  • LayerNorm → MLP (1152 → 4608 → 1152)                         │   │  │
│   │  │  • Residual Connections                                          │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Transformer Block 2-27 (×26 more blocks)                        │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  Output: [256, 1152] Vision Features                                   │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                      │
│                                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                     Multi-Modal Projector                                │  │
│   │  • Linear: 1152 → 2048                                                   │  │
│   │  • GELU Activation                                                       │  │
│   │  • Linear: 2048 → 2048                                                   │  │
│   │  Output: [256, 2048] Projected Vision Features                          │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                      │
│                                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                     Language Encoder (Gemma-2B)                          │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Token Embedding + Vision Feature Concatenation                  │   │  │
│   │  │  Input: [<bos>, <image>×256, text_tokens] → [1+256+seq_len, 2048]│   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Transformer Block 1                                             │   │  │
│   │  │  • LayerNorm → Multi-Head Attention (8 heads, 2048 dims)        │   │  │
│   │  │    - Q: 2048 → 2048 (8 × 256 per head)                          │   │  │
│   │  │    - K: 2048 → 1024 (4 × 256 per head, GQA)                     │   │  │
│   │  │    - V: 2048 → 1024 (4 × 256 per head, GQA)                     │   │  │
│   │  │    - Attention: softmax(QK^T/√d)V                               │   │  │
│   │  │  • LayerNorm → MLP (2048 → 16384 → 2048)                        │   │  │
│   │  │  • Residual Connections                                          │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Transformer Block 2-18 (×17 more blocks)                        │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │  Output: [1+256+seq_len, 2048] Context Features                         │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                      │
│                                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                          Context Vector                                  │  │
│   │  • Extract [CLS] / First Token Representation                            │  │
│   │  • Output: [B, 2048]                                                     │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Vision Encoder 详细实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP Vision Encoder (ViT 架构)
    
    基于 Vision Transformer 的图像编码器
    用于将 RGB 图像转换为视觉特征序列
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 1152,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        intermediate_size: int = 4608,
    ):
        """
        初始化 SigLIP Vision Encoder
        
        Args:
            image_size: 输入图像尺寸 (默认 224×224)
            patch_size: Patch 尺寸 (默认 14×14)
            hidden_size: 隐藏层维度 (默认 1152)
            num_hidden_layers: Transformer 层数 (默认 27)
            num_attention_heads: 注意力头数 (默认 16)
            intermediate_size: MLP 中间层维度 (默认 4608)
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2  # 256
        
        # ========== 1. Patch Embedding ==========
        # 将图像分割为 patches 并线性投影到 hidden_size
        # 卷积实现：kernel_size=patch_size, stride=patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=3,           # RGB 3 通道
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )
        
        # ========== 2. Position Embedding ==========
        # 可学习的位置编码，为每个 patch 添加位置信息
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size) * 0.02
        )
        # +1 用于 [CLS] token
        
        # ========== 3. [CLS] Token ==========
        # 用于聚合全局图像信息的特殊 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # ========== 4. Transformer Blocks ==========
        # 堆叠多个 Transformer 编码器层
        self.layers = nn.ModuleList([
            SigLIPTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # ========== 5. LayerNorm ==========
        # 最终层归一化
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pixel_values: [B, 3, 224, 224] RGB 图像，值范围 [0, 1]
            
        Returns:
            last_hidden_state: [B, num_patches+1, hidden_size] 视觉特征序列
            pooled_output: [B, hidden_size] [CLS] token 表示 (全局图像特征)
        """
        batch_size = pixel_values.shape[0]
        
        # ========== Step 1: Patch Embedding ==========
        # pixel_values: [B, 3, 224, 224]
        patch_embeds = self.patch_embedding(pixel_values)
        # patch_embeds: [B, hidden_size, 16, 16]
        
        # 展平空间维度
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # patch_embeds: [B, 256, hidden_size]
        
        # ========== Step 2: 添加 [CLS] Token ==========
        # cls_tokens: [1, 1, hidden_size] → [B, 1, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 拼接 [CLS] + patches
        hidden_states = torch.cat([cls_tokens, patch_embeds], dim=1)
        # hidden_states: [B, 257, hidden_size]
        
        # ========== Step 3: 添加位置编码 ==========
        hidden_states = hidden_states + self.position_embedding
        # hidden_states: [B, 257, hidden_size]
        
        # ========== Step 4: Transformer 编码 ==========
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # hidden_states: [B, 257, hidden_size]
        
        # ========== Step 5: 层归一化 ==========
        hidden_states = self.final_layernorm(hidden_states)
        
        # ========== Step 6: 提取输出 ==========
        # [CLS] token 作为全局图像表示
        pooled_output = hidden_states[:, 0, :]
        # pooled_output: [B, hidden_size]
        
        # 所有 token 作为序列输出
        last_hidden_state = hidden_states
        # last_hidden_state: [B, 257, hidden_size]
        
        return last_hidden_state, pooled_output


class SigLIPTransformerBlock(nn.Module):
    """
    SigLIP Transformer 编码器块
    
    包含:
    1. LayerNorm → Multi-Head Attention → Residual
    2. LayerNorm → MLP → Residual
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads  # 1152/16 = 72
        
        # ========== 1. 归一化层 ==========
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # ========== 2. 多头自注意力 ==========
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,  # 输入格式：[B, seq_len, hidden_size]
        )
        
        # ========== 3. MLP (Feed-Forward Network) ==========
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),  # GELU 激活函数
            nn.Linear(intermediate_size, hidden_size),
        )
        
        # ========== 4. Dropout ==========
        self.dropout = nn.Dropout(hidden_dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [B, seq_len, hidden_size]
            
        Returns:
            hidden_states: [B, seq_len, hidden_size]
        """
        # ========== 1. Attention + Residual ==========
        # LayerNorm
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        # Multi-Head Attention
        # hidden_states: [B, seq_len, hidden_size]
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            need_weights=False,
        )
        # attn_output: [B, seq_len, hidden_size]
        
        # Dropout + Residual
        attn_output = self.dropout(attn_output)
        hidden_states = residual + attn_output
        # hidden_states: [B, seq_len, hidden_size]
        
        # ========== 2. MLP + Residual ==========
        # LayerNorm
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        # hidden_states: [B, seq_len, hidden_size]
        
        # Dropout + Residual
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

### 1.3 数学原理详解

#### 1.3.1 Patch Embedding

**卷积实现**:
$$\text{PatchEmbed}(I)_{i,j} = \sum_{c=0}^{2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} W_{c,m,n} \cdot I_{c, i\cdot p + m, j\cdot p + n} + b$$

其中:
- $I \in \mathbb{R}^{3 \times H \times W}$: 输入图像
- $p$: patch 尺寸 (14)
- $W \in \mathbb{R}^{3 \times p \times p \times d}$: 卷积核权重
- $d$: 隐藏层维度 (1152)

**等价于**: 将图像分割为 $H/p \times W/p$ 个 patches，每个 patch 展平后线性投影。

#### 1.3.2 自注意力机制

**QKV 投影**:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $X \in \mathbb{R}^{B \times N \times d}$ 是输入序列。

**缩放点积注意力**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**多头注意力**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

其中 $\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$

#### 1.3.3 位置编码

**可学习位置编码**:
$$\text{PositionEmbed}(i) = E_{pos}[i] \in \mathbb{R}^d$$

添加到 patch 嵌入:
$$X'_{i} = X_{i} + E_{pos}[i]$$

---

## 2. Action Expert Decoder 完整架构解析

### 2.1 Flow Matching Decoder 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Action Expert Decoder                                   │
│                         (Flow Matching U-Net)                                    │
│                                                                                  │
│   ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐  │
│   │  VLM Context    │         │  Noisy Actions  │         │   Timestep τ    │  │
│   │  [B, 2048]      │         │  [B, 50, 7]     │         │  [B,]           │  │
│   └────────┬────────┘         └────────┬────────┘         └────────┬────────┘  │
│            │                           │                           │           │
│            ▼                           ▼                           ▼           │
│   ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐  │
│   │  Context        │         │  Action         │         │  Time           │  │
│   │  Projection     │         │  Flatten        │         │  Embedding      │  │
│   │  2048 → 2048    │         │  [B, 350]       │         │  [B, 2048]      │  │
│   └────────┬────────┘         └────────┬────────┘         └────────┬────────┘  │
│            │                           │                           │           │
│            └───────────────────────────┼───────────────────────────┘           │
│                                        ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                        Condition Encoder                                 │  │
│   │  • Concat: [context, actions_flat, time_emb] → [B, 4446]                │  │
│   │  • Linear: 4446 → 2048                                                   │  │
│   │  • SiLU Activation                                                       │  │
│   │  • Linear: 2048 → 2048                                                   │  │
│   │  Output: [B, 2048] Condition Embedding                                   │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                           U-Net Backbone                                 │  │
│   │                                                                          │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Down Block 1                                                    │   │  │
│   │  │  • ResnetBlock(2048 → 2048)                                      │   │  │
│   │  │  • ResnetBlock(2048 → 2048)                                      │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │                            │                                            │  │
│   │                            ▼ Downsample                                 │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Down Block 2                                                    │   │  │
│   │  │  • ResnetBlock(2048 → 1024)                                      │   │  │
│   │  │  • ResnetBlock(1024 → 1024)                                      │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │                            │                                            │  │
│   │                            ▼ Downsample                                 │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Mid Block                                                       │   │  │
│   │  │  • ResnetBlock(1024 → 1024)                                      │   │  │
│   │  │  • ResnetBlock(1024 → 1024)                                      │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │                            │                                            │  │
│   │                            ▼ Upsample                                   │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Up Block 1                                                      │   │  │
│   │  │  • ResnetBlock(2048 → 2048) ← Skip connection from Down 1       │   │  │
│   │  │  • ResnetBlock(2048 → 2048)                                      │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   │                            │                                            │  │
│   │                            ▼                                            │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │  Output Layer                                                    │   │  │
│   │  │  • SiLU Activation                                               │   │  │
│   │  │  • Linear: 2048 → 350 (50×7)                                     │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                        Velocity Field Output                             │  │
│   │  • Reshape: [B, 350] → [B, 50, 7]                                       │  │
│   │  Output: [B, 50, 7] 预测的速度场                                         │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flow Matching Decoder 详细实现

```python
class FlowMatchingDecoder(nn.Module):
    """
    流匹配 Action Expert Decoder
    
    基于 U-Net 架构的速度场预测网络
    """
    
    def __init__(
        self,
        context_dim: int = 2048,
        action_dim: int = 7,
        horizon: int = 50,
        hidden_dim: int = 512,
        time_embed_dim: int = 512,
    ):
        """
        初始化 Flow Matching Decoder
        
        Args:
            context_dim: VLM 上下文维度 (默认 2048)
            action_dim: 动作维度 (默认 7)
            horizon: 动作块长度 (默认 50)
            hidden_dim: 隐藏层维度 (默认 512)
            time_embed_dim: 时间嵌入维度 (默认 512)
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # ========== 1. 时间嵌入 ==========
        # 使用正弦位置编码 + MLP 将标量时间映射到高维空间
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        # ========== 2. 上下文投影 ==========
        # 将 VLM 上下文投影到隐藏维度
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )
        
        # ========== 3. 条件编码器 ==========
        # 拼接 [context, actions, time] 并编码
        condition_input_dim = hidden_dim * 4 + action_dim * horizon + time_embed_dim
        
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )
        
        # ========== 4. U-Net 下采样路径 ==========
        self.down_blocks = nn.ModuleList([
            # Down Block 1
            nn.Sequential(
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
            ),
            # Downsample
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            # Down Block 2
            nn.Sequential(
                ResnetBlock(hidden_dim * 2, hidden_dim * 2),
                ResnetBlock(hidden_dim * 2, hidden_dim * 2),
            ),
        ])
        
        # ========== 5. U-Net 中间层 ==========
        self.mid_block = nn.Sequential(
            ResnetBlock(hidden_dim * 2, hidden_dim * 2),
            ResnetBlock(hidden_dim * 2, hidden_dim * 2),
        )
        
        # ========== 6. U-Net 上采样路径 ==========
        self.up_blocks = nn.ModuleList([
            # Upsample
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            # Up Block 1 (带 skip connection)
            nn.Sequential(
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
            ),
        ])
        
        # ========== 7. 输出层 ==========
        self.output_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, action_dim * horizon),
        )
        
    def forward(
        self,
        noisy_actions: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播 - 预测速度场
        
        Args:
            noisy_actions: [B, H, action_dim] 带噪声的动作块
            context: [B, context_dim] VLM 上下文向量
            timestep: [B,] 流匹配时间步 (0-1)
            
        Returns:
            velocity: [B, H, action_dim] 预测的速度场
        """
        batch_size = noisy_actions.shape[0]
        
        # ========== Step 1: 时间嵌入 ==========
        # timestep: [B,] → time_emb: [B, time_embed_dim]
        time_emb = self.time_embed(timestep)
        
        # ========== Step 2: 上下文投影 ==========
        # context: [B, context_dim] → context_emb: [B, hidden_dim*4]
        context_emb = self.context_projection(context)
        
        # ========== Step 3: 动作展平 ==========
        # noisy_actions: [B, H, action_dim] → actions_flat: [B, H*action_dim]
        actions_flat = noisy_actions.view(batch_size, -1)
        
        # ========== Step 4: 条件编码 ==========
        # 拼接 [context_emb, actions_flat, time_emb]
        condition = torch.cat([context_emb, actions_flat, time_emb], dim=1)
        # condition: [B, hidden_dim*4 + H*action_dim + time_embed_dim]
        
        # 编码条件
        x = self.condition_encoder(condition)
        # x: [B, hidden_dim*4]
        
        # ========== Step 5: U-Net 下采样 ==========
        skip_connections = []
        
        for down_block in self.down_blocks:
            if isinstance(down_block, nn.Linear):
                # 下采样层
                x = down_block(x)
                # x: [B, hidden_dim*2]
            else:
                # ResNet 块
                x = down_block(x)
                skip_connections.append(x)
                # x: [B, hidden_dim*2]
        
        # ========== Step 6: U-Net 中间层 ==========
        x = self.mid_block(x)
        # x: [B, hidden_dim*2]
        
        # ========== Step 7: U-Net 上采样 ==========
        for up_block in self.up_blocks:
            if isinstance(up_block, nn.Linear):
                # 上采样层
                x = up_block(x)
                # x: [B, hidden_dim*4]
            else:
                # ResNet 块 (带 skip connection)
                if len(skip_connections) > 0:
                    skip = skip_connections.pop()
                    x = up_block(x + skip)
                else:
                    x = up_block(x)
        
        # ========== Step 8: 输出层 ==========
        # velocity_flat: [B, H*action_dim]
        velocity_flat = self.output_layer(x)
        
        # 重塑为动作块形状
        velocity = velocity_flat.view(batch_size, self.horizon, self.action_dim)
        # velocity: [B, H, action_dim]
        
        return velocity


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置嵌入 (用于时间编码)
    
    将标量时间映射到高维周期特征空间
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time: [B,] 时间步 (0-1)
            
        Returns:
            embeddings: [B, dim] 时间嵌入
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 计算频率基数
        # embeddings: [half_dim,]
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 外积：[B, 1] × [1, half_dim] = [B, half_dim]
        embeddings = time[:, None] * embeddings[None, :]
        
        # 拼接 sin 和 cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # embeddings: [B, dim]
        
        return embeddings


class ResnetBlock(nn.Module):
    """
    ResNet 风格的残差块
    
    包含两个全连接层和残差连接
    """
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
        
        # 残差投影 (如果维度不同)
        if dim_in != dim_out:
            self.residual_conv = nn.Linear(dim_in, dim_out)
        else:
            self.residual_conv = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, dim_in] 输入
            
        Returns:
            out: [B, dim_out] 输出
        """
        residual = self.residual_conv(x)
        out = self.block(x) + residual
        return out
```

---

## 3. Loss 函数完整数学推导

### 3.1 流匹配 Loss

#### 3.1.1 基础公式

**流匹配目标**: 学习速度场 $\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{o})$ 使得：

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{o})$$

其中 $\mathbf{x}_t$ 从噪声 $\mathbf{x}_1 \sim \mathcal{N}(0, I)$ 流向数据 $\mathbf{x}_0 \sim p_{data}$。

**线性插值路径**:
$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$

**真实速度场**:
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$

**流匹配 Loss**:
$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \|\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{o}) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2_2 \right]$$

#### 3.1.2 代码实现

```python
def flow_matching_loss(
    model: FlowMatchingDecoder,
    context: torch.Tensor,
    target_actions: torch.Tensor,
    num_steps: int = 5,
) -> torch.Tensor:
    """
    计算流匹配 Loss
    
    Args:
        model: 流匹配模型
        context: [B, context_dim] VLM 上下文
        target_actions: [B, H, action_dim] 目标动作 (x_0)
        num_steps: 数值积分步数
        
    Returns:
        loss: 标量 Loss
    """
    batch_size = target_actions.shape[0]
    device = target_actions.device
    
    # ========== Step 1: 采样时间步 ==========
    # t ~ Uniform(0, 1)
    t = torch.rand(batch_size, device=device)
    # t: [B,]
    
    # ========== Step 2: 采样噪声 ==========
    # x_1 ~ N(0, I)
    noise = torch.randn_like(target_actions)
    # noise: [B, H, action_dim]
    
    # ========== Step 3: 线性插值 ==========
    # x_t = (1-t) * x_0 + t * x_1
    # 需要调整 t 的维度以广播
    t_expanded = t[:, None, None]  # [B, 1, 1]
    
    x_t = (1 - t_expanded) * target_actions + t_expanded * noise
    # x_t: [B, H, action_dim]
    
    # ========== Step 4: 预测速度场 ==========
    velocity_pred = model(
        noisy_actions=x_t,
        context=context,
        timestep=t,
    )
    # velocity_pred: [B, H, action_dim]
    
    # ========== Step 5: 计算目标速度 ==========
    # v_target = x_1 - x_0
    velocity_target = noise - target_actions
    # velocity_target: [B, H, action_dim]
    
    # ========== Step 6: 计算 MSE Loss ==========
    loss = F.mse_loss(velocity_pred, velocity_target)
    
    return loss
```

### 3.2 FAST Tokenization Loss

#### 3.2.1 重建 Loss

**DCT 重建误差**:
$$\mathcal{L}_{recon} = \|\text{IDCT}(\text{Quantize}(\text{DCT}(\mathbf{a}))) - \mathbf{a}\|^2_2$$

**代码实现**:

```python
def fast_reconstruction_loss(
    tokenizer: FASTTokenizer,
    actions: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """
    计算 FAST Tokenization 重建 Loss
    
    Args:
        tokenizer: FAST tokenizer
        actions: [B, H, action_dim] 原始动作
        gamma: 量化缩放因子
        
    Returns:
        loss: 重建 Loss
    """
    # ========== Step 1: 编码 ==========
    tokens = tokenizer.encode(actions)
    
    # ========== Step 2: 解码 ==========
    reconstructed = tokenizer.decode(tokens)
    
    # ========== Step 3: 计算 MSE ==========
    loss = F.mse_loss(
        torch.from_numpy(reconstructed),
        actions,
    )
    
    return loss
```

### 3.3 知识隔离 Loss

#### 3.3.1 联合训练 Loss

$$\mathcal{L}_{total} = \mathcal{L}_{FAST} + \lambda_1 \mathcal{L}_{flow} + \lambda_2 \mathcal{L}_{VLM}$$

**代码实现**:

```python
def knowledge_insulated_loss(
    model: KnowledgeInsulatedVLA,
    batch: dict,
    lambda_flow: float = 1.0,
    lambda_fast: float = 0.5,
    lambda_vlm: float = 0.1,
) -> torch.Tensor:
    """
    计算知识隔离 VLA 的联合 Loss
    
    Args:
        model: 知识隔离 VLA 模型
        batch: 数据批次
        lambda_*: 各 Loss 的权重
        
    Returns:
        total_loss: 总 Loss
    """
    # ========== 1. FAST Token Loss ==========
    fast_logits = model.fast_head(model.vlm_context)
    fast_loss = F.cross_entropy(
        fast_logits.view(-1, fast_logits.size(-1)),
        batch['fast_tokens'].view(-1),
    )
    
    # ========== 2. Flow Matching Loss ==========
    # 停止梯度到 VLM
    context_detached = model.vlm_context.detach()
    
    velocity_pred = model.action_expert(
        noisy_actions=batch['noisy_actions'],
        context=context_detached,
        timestep=batch['timestep'],
    )
    
    flow_loss = F.mse_loss(
        velocity_pred,
        batch['velocity_target'],
    )
    
    # ========== 3. VLM Language Loss (可选) ==========
    vlm_logits = model.vlm_outputs.logits
    vlm_loss = F.cross_entropy(
        vlm_logits.view(-1, vlm_logits.size(-1)),
        batch['labels'].view(-1),
    )
    
    # ========== 4. 总 Loss ==========
    total_loss = (
        lambda_fast * fast_loss +
        lambda_flow * flow_loss +
        lambda_vlm * vlm_loss
    )
    
    return total_loss
```

---

## 4. 可视化架构图

### 4.1 Encoder 数据流图

```
输入图像 [B, 3, 224, 224]
         │
         ▼
┌─────────────────────────┐
│   Patch Embedding       │
│   Conv2d(3, 1152, 14)   │
│   Output: [B, 256, 1152]│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   + Position Embed      │
│   [B, 257, 1152]        │
│   (257 = 1 [CLS] + 256) │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Transformer Block 1   │
│   • MHA (16 heads)      │
│   • MLP (1152→4608→1152)│
│   • LayerNorm + Residual│
└─────────────────────────┘
         │
         ▼
    ... (×26 more blocks)
         │
         ▼
┌─────────────────────────┐
│   Transformer Block 27  │
│   Output: [B, 257, 1152]│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   LayerNorm             │
│   Output: [B, 257, 1152]│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Multi-Modal Projector │
│   Linear(1152, 2048)    │
│   GELU                  │
│   Linear(2048, 2048)    │
│   Output: [B, 256, 2048]│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Concat with Text      │
│   [B, 256+seq_len, 2048]│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Gemma Language Model  │
│   18 Transformer Blocks │
│   Output: [B, L, 2048]  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Extract [CLS]         │
│   Context: [B, 2048]    │
└─────────────────────────┘
```

### 4.2 Decoder 数据流图

```
Context [B, 2048]      Noisy Actions [B, 50, 7]      Timestep [B,]
     │                        │                          │
     ▼                        ▼                          ▼
┌─────────┐            ┌─────────────┐            ┌───────────┐
│ Context │            │   Flatten   │            │    Time   │
│ Project │            │ [B, 350]    │            │  Embed    │
│ [B,2048]│            │             │            │ [B, 512]  │
└────┬────┘            └──────┬──────┘            └─────┬─────┘
     │                        │                          │
     └────────────────────────┼──────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Concatenate   │
                    │ [B, 4446]       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Condition Enc.  │
                    │ [B, 2048]       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │   Down Block 1  │           │   Skip Conn.    │
     │ [B, 2048]       │──────────▶│  Save for Up    │
     └────────┬────────┘           └─────────────────┘
              │
              ▼ Downsample
     ┌─────────────────┐
     │   Down Block 2  │
     │ [B, 1024]       │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │    Mid Block    │
     │ [B, 1024]       │
     └────────┬────────┘
              │
              ▼ Upsample
     ┌─────────────────┐           ┌─────────────────┐
     │    Up Block 1   │◀──────────│   Skip Conn.    │
     │ [B, 2048]       │  Concat   │  Add from Down  │
     └────────┬────────┘           └─────────────────┘
              │
              ▼
     ┌─────────────────┐
     │   Output Layer  │
     │ Linear → [B,350]│
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │   Reshape       │
     │ [B, 50, 7]      │
     └─────────────────┘
```

### 4.3 Loss 计算流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flow Matching Loss                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Target Actions x₀ [B, 50, 7]          Noise x₁ [B, 50, 7]      │
│            │                                │                   │
│            │                                │                   │
│            └──────────────┬─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                           │
│                  │  Sample t ~ U(0,1)│                          │
│                  │  t: [B,]         │                           │
│                  └────────┬────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                           │
│                  │  Linear Interp. │                           │
│                  │  x_t = (1-t)x₀ + tx₁ │                       │
│                  │  x_t: [B, 50, 7] │                           │
│                  └────────┬────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                           │
│                  │  Flow Decoder   │                           │
│                  │  v_θ(x_t, t, c) │                           │
│                  │  Output: [B,50,7]│                          │
│                  └────────┬────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                           │
│                  │  Target Velocity│                           │
│                  │  v* = x₁ - x₀   │                           │
│                  │  [B, 50, 7]     │                           │
│                  └────────┬────────┘                           │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                           │
│                  │  MSE Loss       │                           │
│                  │  ||v_θ - v*||²  │                           │
│                  └─────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 完整代码实现与逐行注释

### 5.1 完整 VLA 模型

```python
"""
Physical Intelligence π₀ VLA 完整实现

包含:
1. PaliGemma VLM Encoder
2. Flow Matching Action Expert Decoder
3. 知识隔离训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any


# ============================================================================
# 1. Vision Encoder
# ============================================================================

class SigLIPVisionEncoder(nn.Module):
    """SigLIP Vision Encoder (ViT 架构)"""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 1152,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        intermediate_size: int = 4608,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding: 将图像分割为 patches 并投影
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )
        
        # [CLS] Token: 用于聚合全局信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # Position Embedding: 可学习的位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size) * 0.02
        )
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            SigLIPTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final LayerNorm
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            last_hidden_state: [B, num_patches+1, hidden_size]
            pooled_output: [B, hidden_size] ([CLS] token)
        """
        batch_size = pixel_values.shape[0]
        
        # Patch Embedding
        patch_embeds = self.patch_embedding(pixel_values)  # [B, hidden_size, 16, 16]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, 256, hidden_size]
        
        # Add [CLS] Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_size]
        hidden_states = torch.cat([cls_tokens, patch_embeds], dim=1)  # [B, 257, hidden_size]
        
        # Add Position Embedding
        hidden_states = hidden_states + self.position_embedding
        
        # Transformer Blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # LayerNorm
        hidden_states = self.final_layernorm(hidden_states)
        
        # Extract outputs
        pooled_output = hidden_states[:, 0, :]  # [B, hidden_size] ([CLS])
        last_hidden_state = hidden_states  # [B, 257, hidden_size]
        
        return last_hidden_state, pooled_output


class SigLIPTransformerBlock(nn.Module):
    """SigLIP Transformer 编码器块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention + Residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states, need_weights=False
        )
        hidden_states = residual + attn_output
        
        # MLP + Residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# 2. Flow Matching Decoder
# ============================================================================

class FlowMatchingDecoder(nn.Module):
    """流匹配 Action Expert Decoder"""
    
    def __init__(
        self,
        context_dim: int = 2048,
        action_dim: int = 7,
        horizon: int = 50,
        hidden_dim: int = 512,
        time_embed_dim: int = 512,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        # Context Projection
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )
        
        # Condition Encoder
        condition_input_dim = hidden_dim * 4 + action_dim * horizon + time_embed_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
        )
        
        # U-Net Down Blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
            ),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Sequential(
                ResnetBlock(hidden_dim * 2, hidden_dim * 2),
                ResnetBlock(hidden_dim * 2, hidden_dim * 2),
            ),
        ])
        
        # U-Net Mid Block
        self.mid_block = nn.Sequential(
            ResnetBlock(hidden_dim * 2, hidden_dim * 2),
            ResnetBlock(hidden_dim * 2, hidden_dim * 2),
        )
        
        # U-Net Up Blocks
        self.up_blocks = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Sequential(
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
                ResnetBlock(hidden_dim * 4, hidden_dim * 4),
            ),
        ])
        
        # Output Layer
        self.output_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, action_dim * horizon),
        )
        
    def forward(
        self,
        noisy_actions: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        
        # Time Embedding
        time_emb = self.time_embed(timestep)  # [B, time_embed_dim]
        
        # Context Projection
        context_emb = self.context_projection(context)  # [B, hidden_dim*4]
        
        # Flatten Actions
        actions_flat = noisy_actions.view(batch_size, -1)  # [B, H*action_dim]
        
        # Condition Encoding
        condition = torch.cat([context_emb, actions_flat, time_emb], dim=1)
        x = self.condition_encoder(condition)  # [B, hidden_dim*4]
        
        # U-Net Down
        skip_connections = []
        for down_block in self.down_blocks:
            if isinstance(down_block, nn.Linear):
                x = down_block(x)
            else:
                x = down_block(x)
                skip_connections.append(x)
        
        # U-Net Mid
        x = self.mid_block(x)
        
        # U-Net Up
        for up_block in self.up_blocks:
            if isinstance(up_block, nn.Linear):
                x = up_block(x)
            else:
                if len(skip_connections) > 0:
                    skip = skip_connections.pop()
                    x = up_block(x + skip)
                else:
                    x = up_block(x)
        
        # Output
        velocity_flat = self.output_layer(x)  # [B, H*action_dim]
        velocity = velocity_flat.view(batch_size, self.horizon, self.action_dim)
        
        return velocity


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    """ResNet 残差块"""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
        self.residual_conv = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual_conv(x)


# ============================================================================
# 3. Complete VLA Model
# ============================================================================

class Pi0VLA(nn.Module):
    """
    π₀ VLA 完整模型
    
    整合 VLM Encoder 和 Flow Matching Decoder
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Vision Encoder
        self.vision_encoder = SigLIPVisionEncoder(
            image_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 14),
            hidden_size=config.get('vision_hidden_size', 1152),
            num_hidden_layers=config.get('vision_num_layers', 27),
            num_attention_heads=config.get('vision_num_heads', 16),
        )
        
        # Multi-Modal Projector
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(1152, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
        )
        
        # Language Model (简化版本，实际使用 Gemma)
        self.language_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2048,
                nhead=8,
                dim_feedforward=16384,
                batch_first=True,
            ),
            num_layers=18,
        )
        
        # Action Expert
        self.action_expert = FlowMatchingDecoder(
            context_dim=config.get('context_dim', 2048),
            action_dim=config.get('action_dim', 7),
            horizon=config.get('horizon', 50),
            hidden_dim=config.get('hidden_dim', 512),
        )
        
    def encode(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码多模态输入为上下文向量
        
        Returns:
            context: [B, 2048] 上下文向量
        """
        # Vision Encoding
        vision_features, _ = self.vision_encoder(pixel_values)
        # vision_features: [B, 257, 1152]
        
        # Project Vision Features
        image_embeddings = self.multi_modal_projector(vision_features[:, 1:, :])
        # image_embeddings: [B, 256, 2048] (跳过 [CLS])
        
        # Text Embedding
        text_embeddings = self.language_model.embeddings(input_ids)
        # text_embeddings: [B, seq_len, 2048]
        
        # Concatenate Vision and Text
        # 假设<image> token 在位置 1
        combined = torch.cat([
            text_embeddings[:, :1, :],  # [B, 1, 2048]
            image_embeddings,  # [B, 256, 2048]
            text_embeddings[:, 1:, :],  # [B, seq_len-1, 2048]
        ], dim=1)
        
        # Language Model
        outputs = self.language_model(combined, src_key_padding_mask=~attention_mask.bool())
        # outputs: [B, seq_len+256, 2048]
        
        # Extract Context ([CLS] / First Token)
        context = outputs[:, 0, :]  # [B, 2048]
        
        return context
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播 - 预测速度场
        
        Returns:
            velocity: [B, H, action_dim] 预测的速度场
        """
        # Encode
        context = self.encode(pixel_values, input_ids, attention_mask)
        
        # Decode
        velocity = self.action_expert(noisy_actions, context, timestep)
        
        return velocity
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_denoise_steps: int = 5,
    ) -> torch.Tensor:
        """
        生成动作块 (推理模式)
        
        Returns:
            actions: [B, H, action_dim] 生成的动作
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Encode
        context = self.encode(pixel_values, input_ids, attention_mask)
        
        # Sample Noise
        actions = torch.randn(
            batch_size,
            self.action_expert.horizon,
            self.action_expert.action_dim,
            device=device,
        )
        
        # Denoise
        for tau in torch.linspace(0, 1, num_denoise_steps, device=device):
            velocity = self.action_expert(actions, context, tau)
            actions = actions + (1 / num_denoise_steps) * velocity
        
        return actions


# ============================================================================
# 4. Training Functions
# ============================================================================

def train_step(
    model: Pi0VLA,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    训练一步
    
    Returns:
        loss: 当前 Loss 值
    """
    model.train()
    
    # Forward
    velocity_pred = model(
        pixel_values=batch['pixel_values'],
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        noisy_actions=batch['noisy_actions'],
        timestep=batch['timestep'],
    )
    
    # Compute Loss
    loss = F.mse_loss(velocity_pred, batch['velocity_target'])
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: Pi0VLA,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    """
    评估模型
    
    Returns:
        avg_loss: 平均 Loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        velocity_pred = model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            noisy_actions=batch['noisy_actions'],
            timestep=batch['timestep'],
        )
        
        loss = F.mse_loss(velocity_pred, batch['velocity_target'])
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

---

*详细版报告完成时间：2026 年 3 月 3 日*
*版本：3.0 - Encoder/Decoder/Loss 深度解析*
