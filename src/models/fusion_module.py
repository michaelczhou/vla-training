"""
Fusion Module
=============
Implements multi-modal fusion strategies:
- Cross-Attention (recommended)
- Concatenation + MLP
- FiLM (Feature-wise Linear Modulation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FusionModule(nn.Module):
    """Base class for fusion modules"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse vision and language features
        
        Args:
            vision_features: Tensor of shape (B, N_v, D_v)
            language_features: Tensor of shape (B, N_l, D_l)
            
        Returns:
            fused_features: Tensor of shape (B, N, D)
        """
        raise NotImplementedError


class CrossAttentionFusion(FusionModule):
    """
    Cross-Attention Fusion
    Uses language features as queries to attend to vision features
    
    This is the recommended fusion strategy for VLA models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        vision_dim = config.get('vision_dim', 768)
        language_dim = config.get('language_dim', 768)
        hidden_dim = config.get('hidden_dim', 768)
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 1)
        dropout = config.get('dropout', 0.1)
        
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.lang_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attn_layers.append(layer)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features using cross-attention
        
        Args:
            vision_features: Tensor of shape (B, N_v, D_v)
            language_features: Tensor of shape (B, N_l, D_l)
            
        Returns:
            fused_features: Tensor of shape (B, N_l, D)
        """
        # Project to common dimension
        v = self.vision_proj(vision_features)  # (B, N_v, D)
        l = self.lang_proj(language_features)  # (B, N_l, D)
        
        # Cross-attention: language queries vision
        for cross_attn in self.cross_attn_layers:
            # Self-attention on language
            l_norm = self.norm1(l)
            l_attn = cross_attn(
                query=l_norm,
                key=l_norm,
                value=l_norm,
                need_weights=False
            )[0]
            l = l + l_attn
            
            # Cross-attention: language attends to vision
            l_norm = self.norm2(l)
            cross_attn_out = cross_attn(
                query=l_norm,
                key=v,
                value=v,
                need_weights=False
            )[0]
            l = l + cross_attn_out
            
            # Feed-forward
            l_norm = self.norm3(l)
            l = l + self.ffn(l_norm)
        
        return l  # (B, N_l, D)


class ConcatFusion(FusionModule):
    """
    Concatenation Fusion
    Simple concatenation followed by MLP
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        vision_dim = config.get('vision_dim', 768)
        language_dim = config.get('language_dim', 768)
        hidden_dim = config.get('hidden_dim', 768)
        dropout = config.get('dropout', 0.1)
        
        # Concatenation dimension
        concat_dim = vision_dim + language_dim
        
        # MLP fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features by concatenation
        
        Args:
            vision_features: Tensor of shape (B, N_v, D_v)
            language_features: Tensor of shape (B, N_l, D_l)
            
        Returns:
            fused_features: Tensor of shape (B, N_l, D)
        """
        # Pool vision features to match language sequence length
        if vision_features.shape[1] != language_features.shape[1]:
            # Global average pool vision features
            v_pooled = vision_features.mean(dim=1, keepdim=True)  # (B, 1, D_v)
            v_pooled = v_pooled.expand(-1, language_features.shape[1], -1)  # (B, N_l, D_v)
        else:
            v_pooled = vision_features
        
        # Concatenate
        fused = torch.cat([v_pooled, language_features], dim=-1)  # (B, N_l, D_v + D_l)
        
        # MLP fusion
        output = self.fusion_mlp(fused)  # (B, N_l, D)
        
        return output


class FiLMFusion(FusionModule):
    """
    FiLM (Feature-wise Linear Modulation) Fusion
    Uses language features to modulate vision features
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        vision_dim = config.get('vision_dim', 768)
        language_dim = config.get('language_dim', 768)
        hidden_dim = config.get('hidden_dim', 768)
        
        self.hidden_dim = hidden_dim
        
        # Language to modulation parameters
        self.modulation_net = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim * 2)  # gamma and beta
        )
        
        # Output projection
        self.output_proj = nn.Linear(vision_dim, hidden_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features using FiLM modulation
        
        Args:
            vision_features: Tensor of shape (B, N_v, D_v)
            language_features: Tensor of shape (B, N_l, D_l)
            
        Returns:
            fused_features: Tensor of shape (B, N_v, D)
        """
        # Pool language features
        l_pooled = language_features.mean(dim=1)  # (B, D_l)
        
        # Get modulation parameters
        mod_params = self.modulation_net(l_pooled)  # (B, D_v * 2)
        gamma, beta = mod_params.chunk(2, dim=-1)  # (B, D_v), (B, D_v)
        
        # Add sequence dimension
        gamma = gamma.unsqueeze(1)  # (B, 1, D_v)
        beta = beta.unsqueeze(1)  # (B, 1, D_v)
        
        # Apply modulation
        modulated = gamma * vision_features + beta  # (B, N_v, D_v)
        
        # Project output
        output = self.output_proj(modulated)  # (B, N_v, D)
        
        return output


def build_fusion_module(config: Dict[str, Any], vision_dim: int, language_dim: int) -> FusionModule:
    """
    Factory function to build fusion module from config
    
    Args:
        config: Configuration dictionary with 'type' key
        vision_dim: Dimension of vision features
        language_dim: Dimension of language features
        
    Returns:
        FusionModule instance
    """
    fusion_type = config.get('type', 'cross_attention').lower()
    
    config['vision_dim'] = vision_dim
    config['language_dim'] = language_dim
    
    if fusion_type == 'cross_attention':
        return CrossAttentionFusion(config)
    elif fusion_type == 'concat':
        return ConcatFusion(config)
    elif fusion_type == 'film':
        return FiLMFusion(config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
