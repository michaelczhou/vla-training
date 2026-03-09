"""
RDT2 Model Implementation
Based on "RDT 2: A Diffusion Foundation Model for Bimanual Manipulation"
https://github.com/thu-ml/RDT2

Key Features:
- Based on Qwen2.5-VL architecture
- Residual VQ action tokenizer
- Support for bimanual manipulation
- Zero-shot generalization to unseen embodiments
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from transformers import AutoModel, AutoTokenizer
import math


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization for action tokenization.
    Based on SoundStream and EnCodec's RVQ design.
    
    Args:
        num_quantizers: Number of quantizer layers
        codebook_size: Size of each codebook
        embedding_dim: Dimension of embeddings
        commitment_cost: Commitment loss weight
    """
    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create multiple quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through residual VQ.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            quantized: Quantized output
            loss: Total quantization loss
            indices: List of codebook indices for each quantizer
        """
        residual = x
        total_loss = 0
        all_indices = []
        
        for quantizer in self.quantizers:
            quantized, loss, indices = quantizer(residual)
            residual = residual - quantized
            total_loss += loss
            all_indices.append(indices)
        
        # Final output is sum of all quantized residuals
        output = x - residual
        return output, total_loss, all_indices
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode actions to discrete tokens."""
        residual = x
        all_indices = []
        
        for quantizer in self.quantizers:
            _, _, indices = quantizer(residual)
            quantized = quantizer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        
        return all_indices
    
    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Decode discrete tokens to actions."""
        output = torch.zeros(
            indices[0].shape[0], indices[0].shape[1], self.embedding_dim,
            device=indices[0].device
        )
        
        for i, quantizer in enumerate(self.quantizers):
            if i < len(indices):
                output += quantizer.decode(indices[i])
        
        return output


class VectorQuantizer(nn.Module):
    """Single vector quantizer layer."""
    
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook embeddings
        self.embeddings = nn.Embedding(codebook_size, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor.
        
        Args:
            x: Input [B, T, D]
        
        Returns:
            quantized: Quantized tensor
            loss: VQ loss
            indices: Codebook indices
        """
        # Flatten input
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        
        # Get closest codebook entries
        indices = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(indices, self.codebook_size).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view_as(x)
        
        # Loss: commitment + codebook
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, loss, indices.view(x.shape[0], x.shape[1])
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to vectors."""
        return self.embeddings(indices)


class RDT2Transformer(nn.Module):
    """
    RDT2 Transformer based on Qwen2.5-VL architecture.
    Modified for diffusion-based action generation.
    """
    
    def __init__(
        self,
        vision_encoder_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        action_dim: int = 14,  # For bimanual: 7 DOF per arm
        num_diffusion_steps: int = 100,
        num_action_tokens: int = 256,
        use_rvq: bool = True,
        rvq_config: Optional[Dict] = None
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_action_tokens = num_action_tokens
        self.use_rvq = use_rvq
        
        # Load vision-language backbone
        self.vl_backbone = AutoModel.from_pretrained(
            vision_encoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            vision_encoder_name,
            trust_remote_code=True
        )
        
        # Hidden size from backbone
        hidden_size = self.vl_backbone.config.hidden_size
        
        # Action tokenizer (RVQ or simple MLP)
        if use_rvq:
            rvq_config = rvq_config or {}
            self.action_tokenizer = ResidualVQ(**rvq_config)
            action_embed_dim = rvq_config.get('embedding_dim', 256)
        else:
            self.action_tokenizer = nn.Linear(action_dim, hidden_size)
            action_embed_dim = hidden_size
        
        # Action embedding projection
        self.action_proj = nn.Linear(action_embed_dim, hidden_size)
        
        # Timestep embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Action decoder head
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_embed_dim)
        )
        
        # Denoising transformer layers
        self.denoising_layers = nn.ModuleList([
            DenoisingTransformerLayer(hidden_size, num_heads=16)
            for _ in range(6)
        ])
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for denoising.
        
        Args:
            images: Input images [B, N, C, H, W]
            text_input_ids: Text token IDs [B, L]
            noisy_actions: Noisy action tokens [B, T, D]
            timesteps: Diffusion timesteps [B]
            attention_mask: Attention mask [B, L]
        
        Returns:
            predicted_noise: Predicted noise for denoising
        """
        batch_size = images.shape[0]
        
        # Encode vision and language
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vl_outputs = self.vl_backbone(
                pixel_values=images,
                input_ids=text_input_ids,
                attention_mask=attention_mask
            )
            vl_features = vl_outputs.last_hidden_state  # [B, L_vl, D]
        
        # Tokenize actions
        if self.use_rvq:
            action_tokens, vq_loss, _ = self.action_tokenizer(noisy_actions)
        else:
            action_tokens = self.action_tokenizer(noisy_actions)
            vq_loss = 0
        
        # Project actions
        action_embeds = self.action_proj(action_tokens)
        
        # Add timestep embedding
        time_embeds = self.time_embed(timesteps.unsqueeze(-1).float() / self.num_diffusion_steps)
        action_embeds = action_embeds + time_embeds.unsqueeze(1)
        
        # Concatenate VL features and action tokens
        combined_features = torch.cat([vl_features, action_embeds], dim=1)
        
        # Apply denoising layers
        for layer in self.denoising_layers:
            combined_features = layer(combined_features)
        
        # Extract action predictions
        action_features = combined_features[:, -self.num_action_tokens:, :]
        predicted_noise = self.action_head(action_features)
        
        return predicted_noise, vq_loss


class DenoisingTransformerLayer(nn.Module):
    """Single denoising transformer layer with cross-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Self-attention for actions
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Cross-attention to VL features
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Cross-attention (already included in concatenated features)
        # In practice, we'd separate VL and action features here
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


class RDT2Policy:
    """
    High-level policy interface for RDT2.
    Handles inference with diffusion sampling.
    """
    
    def __init__(
        self,
        model: RDT2Transformer,
        num_inference_steps: int = 10,
        device: str = "cuda"
    ):
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        text: str,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Predict action using DDPM sampling.
        
        Args:
            images: Input images [B, N, C, H, W]
            text: Text instruction
            num_samples: Number of action samples
        
        Returns:
            actions: Predicted actions [B, T, D]
        """
        batch_size = images.shape[0]
        
        # Tokenize text
        text_inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Initialize random noise for actions
        shape = (batch_size, self.model.num_action_tokens, self.model.action_dim)
        actions = torch.randn(shape, device=self.device)
        
        # DDPM sampling
        for t in reversed(range(self.num_inference_steps)):
            timesteps = torch.full((batch_size,), t, device=self.device)
            
            # Predict noise
            predicted_noise, _ = self.model(
                images=images,
                text_input_ids=text_inputs.input_ids,
                noisy_actions=actions,
                timesteps=timesteps,
                attention_mask=text_inputs.attention_mask
            )
            
            # Denoise step (simplified DDPM)
            alpha = 1 - (t + 1) / self.num_inference_steps
            actions = (actions - (1 - alpha) * predicted_noise) / alpha.sqrt()
        
        return actions
