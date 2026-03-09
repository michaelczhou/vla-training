"""
VLA Model
=========
Complete Vision-Language-Action model integrating all components
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from .vision_encoder import build_vision_encoder, VisionEncoder
from .language_model import build_language_model, LanguageModel
from .fusion_module import build_fusion_module, FusionModule
from .action_head import build_action_head, ActionHead


class VLAModel(nn.Module):
    """
    Vision-Language-Action Model
    
    Complete VLA architecture integrating:
    - Vision encoder (ViT/SigLIP/ResNet)
    - Language model (Gemma/Qwen/LLaMA)
    - Fusion module (Cross-Attention/Concat/FiLM)
    - Action head (Flow Matching/Diffusion/MLP)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLA model
        
        Args:
            config: Configuration dictionary with model specifications
        """
        super().__init__()
        self.config = config
        
        # Extract sub-configs
        vision_config = config.get('vision', {})
        language_config = config.get('language', {})
        fusion_config = config.get('fusion', {})
        action_head_config = config.get('action_head', {})
        
        # Build vision encoder
        self.vision_encoder = build_vision_encoder(vision_config)
        vision_dim = self.vision_encoder.output_dim
        
        # Build language model
        self.language_model = build_language_model(language_config)
        language_dim = self.language_model.output_dim
        
        # Build fusion module
        self.fusion_module = build_fusion_module(
            fusion_config, 
            vision_dim=vision_dim, 
            language_dim=language_dim
        )
        fusion_dim = fusion_config.get('hidden_dim', 768)
        
        # Build action head
        action_head_config['input_dim'] = fusion_dim
        self.action_head = build_action_head(action_head_config, input_dim=fusion_dim)
        
        # Store dimensions
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.fusion_dim = fusion_dim
        self.action_dim = action_head_config.get('action_dim', 7)
        self.chunk_size = action_head_config.get('chunk_size', 10)
    
    def forward(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through VLA model
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            actions: Action tensor of shape (B, chunk_size, action_dim), optional
            time: Time values for diffusion/flow matching, optional
            
        Returns:
            loss: Training loss (if actions provided)
            predictions: Action predictions
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # (B, N_v, D_v)
        
        # Encode language
        language_features = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # (B, L, D_l)
        
        # Fuse features
        fused_features = self.fusion_module(
            vision_features=vision_features,
            language_features=language_features
        )  # (B, N, D)
        
        # Generate actions
        if isinstance(self.action_head, (FlowMatchingHead, DiffusionHead)):
            if actions is not None and time is not None:
                # Training mode: compute loss
                loss, _ = self.action_head.compute_loss(
                    x=fused_features,
                    action_0=torch.randn_like(actions),
                    action_1=actions
                )
                predictions = None
            else:
                # Inference mode: sample actions
                loss = None
                predictions = self.action_head.sample(fused_features)
        else:
            # MLP head
            predictions = self.action_head(fused_features)
            if actions is not None:
                loss = self.action_head.compute_loss(fused_features, actions)
            else:
                loss = None
        
        return loss, predictions
    
    def predict_actions(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Predict actions from images and text
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            num_steps: Number of sampling steps (for diffusion/flow matching)
            
        Returns:
            actions: Tensor of shape (B, chunk_size, action_dim)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode vision
            vision_features = self.vision_encoder(images)
            
            # Encode language
            language_features = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Fuse
            fused_features = self.fusion_module(
                vision_features=vision_features,
                language_features=language_features
            )
            
            # Sample actions
            if hasattr(self.action_head, 'sample'):
                actions = self.action_head.sample(fused_features, num_steps=num_steps)
            else:
                actions = self.action_head(fused_features)
        
        return actions
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to features"""
        return self.vision_encoder(images)
    
    def encode_language(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to features"""
        return self.language_model(input_ids, attention_mask)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vla_model(config: Dict[str, Any]) -> VLAModel:
    """
    Factory function to build VLA model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VLAModel instance
    """
    return VLAModel(config)


def load_vla_model_from_checkpoint(
    checkpoint_path: str, 
    config: Optional[Dict[str, Any]] = None,
    device: str = 'cpu'
) -> VLAModel:
    """
    Load VLA model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model config (if not in checkpoint)
        device: Device to load model on
        
    Returns:
        VLAModel instance
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use provided
    if config is None:
        config = checkpoint.get('config', {})
    
    # Build model
    model = build_vla_model(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
