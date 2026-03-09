"""
Vision Encoder Module
=====================
Implements various vision encoders for VLA models:
- SigLIP (State-of-the-art)
- ViT (Vision Transformer)
- ResNet (CNN-based, fast inference)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
from transformers import SiglipModel, SiglipVisionConfig, ViTModel, ViTConfig


class VisionEncoder(nn.Module):
    """Base class for vision encoders"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.freeze = config.get('freeze', False)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            features: Tensor of shape (B, D) or (B, N, D)
        """
        raise NotImplementedError
    
    def freeze_parameters(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False


class SigLipEncoder(VisionEncoder):
    """
    SigLIP (Sigmoid Loss on Image Patch) encoder
    State-of-the-art vision encoder from Google
    
    Reference: https://arxiv.org/abs/2303.15343
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        pretrained = config.get('pretrained', 'siglip-so400m-patch14-384')
        
        # Load SigLIP model
        if pretrained:
            self.model = SiglipModel.from_pretrained(pretrained)
        else:
            vision_config = SiglipVisionConfig(**config.get('vision_config', {}))
            self.model = SiglipModel(vision_config)
        
        # Get output dimension
        self.output_dim = self.model.vision_embed_dim
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using SigLIP
        
        Args:
            images: Tensor of shape (B, C, H, W), pixel values in [0, 1]
            
        Returns:
            features: Tensor of shape (B, N, D) where N is number of patches
        """
        outputs = self.model.vision_model(
            pixel_values=images,
            output_hidden_states=False
        )
        
        # Get patch features (last hidden state)
        features = outputs.last_hidden_state  # (B, N, D)
        
        return features
    
    @property
    def num_patches(self) -> int:
        """Number of image patches"""
        return self.model.vision_model.embeddings.num_patches


class ViTEncoder(VisionEncoder):
    """
    Vision Transformer (ViT) encoder
    
    Reference: https://arxiv.org/abs/2010.11929
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        pretrained = config.get('pretrained', 'google/vit-base-patch16-224')
        
        # Load ViT model
        if pretrained:
            self.model = ViTModel.from_pretrained(pretrained)
        else:
            vision_config = ViTConfig(**config.get('vision_config', {}))
            self.model = ViTModel(vision_config)
        
        self.output_dim = self.model.config.hidden_size
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using ViT
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            features: Tensor of shape (B, N, D)
        """
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=False
        )
        
        features = outputs.last_hidden_state  # (B, N, D)
        
        return features


class ResNetEncoder(VisionEncoder):
    """
    ResNet encoder (CNN-based)
    Faster inference, suitable for edge deployment
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        variant = config.get('variant', 'resnet50')
        
        # Load ResNet
        if variant == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.output_dim = 512
        elif variant == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.output_dim = 512
        elif variant == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.output_dim = 2048
        elif variant == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using ResNet
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            features: Tensor of shape (B, D)
        """
        features = self.model(images)  # (B, D, 1, 1)
        features = features.flatten(2)  # (B, D, 1)
        features = features.squeeze(-1)  # (B, D)
        
        # Add sequence dimension for compatibility
        features = features.unsqueeze(1)  # (B, 1, D)
        
        return features


def build_vision_encoder(config: Dict[str, Any]) -> VisionEncoder:
    """
    Factory function to build vision encoder from config
    
    Args:
        config: Configuration dictionary with 'type' key
        
    Returns:
        VisionEncoder instance
    """
    encoder_type = config.get('type', 'siglip').lower()
    
    if encoder_type == 'siglip':
        return SigLipEncoder(config)
    elif encoder_type == 'vit':
        return ViTEncoder(config)
    elif encoder_type == 'resnet':
        return ResNetEncoder(config)
    else:
        raise ValueError(f"Unknown vision encoder type: {encoder_type}")
