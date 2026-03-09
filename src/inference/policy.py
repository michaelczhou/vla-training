"""
VLA Policy
==========
Policy interface for inference
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from ..models.vla_model import build_vla_model, load_vla_model_from_checkpoint
from ..models.language_model import build_language_model


class VLAPolicy:
    """
    VLA Policy for inference
    
    Provides a simple interface for:
    - Loading trained models
    - Processing images and text
    - Generating action predictions
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize policy
        
        Args:
            model: Trained VLA model
            config: Model configuration
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Set to eval mode
        self.model.eval()
        
        # Build image transform
        image_config = config.get('data', {}).get('image', {})
        self.image_transform = T.Compose([
            T.Resize((image_config.get('height', 224), image_config.get('width', 224))),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Build language model for tokenization
        lang_config = config.get('model', {}).get('language', {})
        try:
            self.language_model = build_language_model(lang_config)
            self.language_model.to(device)
            self.tokenizer = self.language_model.tokenizer
        except Exception as e:
            print(f"Warning: Could not load language model: {e}")
            self.tokenizer = None
        
        # Action normalization
        action_config = config.get('data', {}).get('action', {})
        self.action_mean = action_config.get('mean', None)
        self.action_std = action_config.get('std', None)
        self.action_min = action_config.get('min_values', None)
        self.action_max = action_config.get('max_values', None)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda'
    ) -> 'VLAPolicy':
        """
        Load policy from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on
            
        Returns:
            VLAPolicy instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('config', {})
        model = build_vla_model(config.get('model', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, config, device=device)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model
        
        Args:
            image: Numpy array of shape (H, W, C)
            
        Returns:
            Preprocessed tensor of shape (1, C, H, W)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply transform
        tensor = self.image_transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model
        
        Args:
            text: Text string
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if self.tokenizer is None:
            # Dummy tokenization
            return {
                'input_ids': torch.zeros(1, 64, dtype=torch.long, device=self.device),
                'attention_mask': torch.ones(1, 64, dtype=torch.long, device=self.device)
            }
        
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        text: str,
        num_steps: int = 50
    ) -> np.ndarray:
        """
        Predict actions from image and text
        
        Args:
            image: Input image
            text: Text instruction
            num_steps: Number of sampling steps (for diffusion/flow matching)
            
        Returns:
            Actions as numpy array of shape (chunk_size, action_dim)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        text_dict = self.preprocess_text(text)
        
        # Run model
        actions = self.model.predict_actions(
            images=image_tensor,
            input_ids=text_dict['input_ids'],
            attention_mask=text_dict['attention_mask'],
            num_steps=num_steps
        )
        
        # Convert to numpy
        actions = actions.cpu().numpy()[0]  # Remove batch dimension
        
        # Denormalize if needed
        actions = self.denormalize_actions(actions)
        
        return actions
    
    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize actions to original scale
        
        Args:
            actions: Normalized actions
            
        Returns:
            Denormalized actions
        """
        if self.action_mean is not None and self.action_std is not None:
            mean = np.array(self.action_mean)
            std = np.array(self.action_std)
            actions = actions * (std + 1e-8) + mean
        elif self.action_min is not None and self.action_max is not None:
            min_vals = np.array(self.action_min)
            max_vals = np.array(self.action_max)
            actions = (actions + 1) / 2 * (max_vals - min_vals + 1e-8) + min_vals
        
        return actions
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
        texts: List[str],
        num_steps: int = 50
    ) -> np.ndarray:
        """
        Predict actions for batch of images and texts
        
        Args:
            images: List of images
            texts: List of text instructions
            num_steps: Number of sampling steps
            
        Returns:
            Actions as numpy array of shape (B, chunk_size, action_dim)
        """
        # Preprocess
        image_tensors = torch.stack([self.preprocess_image(img) for img in images])
        
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
        else:
            input_ids = torch.zeros(len(texts), 64, dtype=torch.long, device=self.device)
            attention_mask = torch.ones(len(texts), 64, dtype=torch.long, device=self.device)
        
        # Run model
        actions = self.model.predict_actions(
            images=image_tensors,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_steps=num_steps
        )
        
        # Convert to numpy
        actions = actions.cpu().numpy()
        
        # Denormalize
        for i in range(len(actions)):
            actions[i] = self.denormalize_actions(actions[i])
        
        return actions
