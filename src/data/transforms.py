"""
Data Transforms
===============
Image and action transformations for data augmentation
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ImageTransform:
    """
    Image transformation pipeline
    
    Supports:
    - Resizing
    - Normalization
    - Data augmentation (color jitter, random crop, flip)
    """
    
    def __init__(self, config: Dict[str, Any], training: bool = True):
        """
        Initialize image transforms
        
        Args:
            config: Configuration dictionary
            training: Whether in training mode (enables augmentation)
        """
        self.training = training
        self.height = config.get('height', 224)
        self.width = config.get('width', 224)
        self.normalize = config.get('normalize', True)
        
        # Build transform pipeline
        transforms = []
        
        # Resize
        transforms.append(T.Resize((self.height, self.width), antialias=True))
        
        # Augmentation (training only)
        if training:
            if config.get('color_jitter', False):
                transforms.append(T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ))
            
            if config.get('random_crop', False):
                transforms.append(T.RandomCrop(
                    size=(self.height, self.width),
                    padding=4,
                    pad_if_needed=True
                ))
            
            if config.get('flip', False):
                transforms.append(T.RandomHorizontalFlip(p=0.5))
        
        # Convert to tensor
        transforms.append(T.ToTensor())
        
        # Normalize
        if self.normalize:
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            ))
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to image
        
        Args:
            image: Numpy array of shape (H, W, C) or (H, W)
            
        Returns:
            transformed: Tensor of shape (C, H, W)
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Apply transforms
        transformed = self.transform(image)
        
        return transformed


class ActionTransform:
    """
    Action transformation and normalization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize action transforms
        
        Args:
            config: Configuration dictionary
        """
        self.action_dim = config.get('dim', 7)
        self.chunk_size = config.get('chunk_size', 10)
        self.normalize = config.get('normalize', True)
        
        # Normalization statistics
        self.mean = config.get('mean', None)
        self.std = config.get('std', None)
        
        # Or use min/max
        self.min_values = config.get('min_values', None)
        self.max_values = config.get('max_values', None)
        
        if self.normalize:
            if self.mean is None and self.min_values is None:
                # Default to no normalization if not specified
                self.normalize = False
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action using statistics
        
        Args:
            action: Numpy array of shape (chunk_size, action_dim)
            
        Returns:
            normalized: Normalized action
        """
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean)
            std = np.array(self.std)
            normalized = (action - mean) / (std + 1e-8)
        elif self.min_values is not None and self.max_values is not None:
            min_vals = np.array(self.min_values)
            max_vals = np.array(self.max_values)
            # Normalize to [-1, 1]
            normalized = 2 * (action - min_vals) / (max_vals - min_vals + 1e-8) - 1
        else:
            normalized = action
        
        return normalized
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize action
        
        Args:
            action: Normalized action
            
        Returns:
            denormalized: Original scale action
        """
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean)
            std = np.array(self.std)
            denormalized = action * (std + 1e-8) + mean
        elif self.min_values is not None and self.max_values is not None:
            min_vals = np.array(self.min_values)
            max_vals = np.array(self.max_values)
            # Denormalize from [-1, 1]
            denormalized = (action + 1) / 2 * (max_vals - min_vals + 1e-8) + min_vals
        else:
            denormalized = action
        
        return denormalized
    
    def __call__(self, action: np.ndarray) -> torch.Tensor:
        """
        Transform action
        
        Args:
            action: Numpy array of shape (chunk_size, action_dim)
            
        Returns:
            transformed: Tensor of shape (chunk_size, action_dim)
        """
        # Normalize
        if self.normalize:
            action = self.normalize_action(action)
        
        # Convert to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32)
        
        return action_tensor


def build_transforms(config: Dict[str, Any], training: bool = True) -> Tuple[ImageTransform, ActionTransform]:
    """
    Build image and action transforms from config
    
    Args:
        config: Configuration dictionary
        training: Whether in training mode
        
    Returns:
        image_transform, action_transform
    """
    image_config = config.get('image', {})
    action_config = config.get('action', {})
    
    image_transform = ImageTransform(image_config, training=training)
    action_transform = ActionTransform(action_config)
    
    return image_transform, action_transform
