"""
Language Model Module
=====================
Implements language encoders for VLA models:
- Gemma (Google, lightweight)
- Qwen (Alibaba, Chinese optimized)
- LLaMA (Meta, popular)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import (
    GemmaModel, GemmaConfig,
    Qwen2Model, Qwen2Config,
    LlamaModel, LlamaConfig,
    AutoTokenizer
)


class LanguageModel(nn.Module):
    """Base class for language models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.freeze = config.get('freeze', False)
        self.tokenizer = None
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode text tokens to features
        
        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            
        Returns:
            features: Tensor of shape (B, L, D)
        """
        raise NotImplementedError
    
    def tokenize(self, texts: List[str], max_length: int = 64) -> Dict[str, torch.Tensor]:
        """
        Tokenize text strings
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def freeze_parameters(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False


class GemmaModel(LanguageModel):
    """
    Gemma language model from Google
    Lightweight and efficient
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        pretrained = config.get('pretrained', 'google/gemma-2b')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GemmaModel.from_pretrained(pretrained)
        self.output_dim = self.model.config.hidden_size
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode tokens using Gemma
        
        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            
        Returns:
            features: Tensor of shape (B, L, D)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        features = outputs.last_hidden_state  # (B, L, D)
        
        return features


class QwenModel(LanguageModel):
    """
    Qwen language model from Alibaba
    Optimized for Chinese and multilingual tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        pretrained = config.get('pretrained', 'Qwen/Qwen2-1.5B')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = Qwen2Model.from_pretrained(pretrained)
        self.output_dim = self.model.config.hidden_size
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode tokens using Qwen
        
        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            
        Returns:
            features: Tensor of shape (B, L, D)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        features = outputs.last_hidden_state  # (B, L, D)
        
        return features


class LLaMAModel(LanguageModel):
    """
    LLaMA language model from Meta
    Popular and well-supported
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        pretrained = config.get('pretrained', 'meta-llama/Llama-2-7b-hf')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = LlamaModel.from_pretrained(pretrained)
        self.output_dim = self.model.config.hidden_size
        
        if self.freeze:
            self.freeze_parameters()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode tokens using LLaMA
        
        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            
        Returns:
            features: Tensor of shape (B, L, D)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        features = outputs.last_hidden_state  # (B, L, D)
        
        return features


def build_language_model(config: Dict[str, Any]) -> LanguageModel:
    """
    Factory function to build language model from config
    
    Args:
        config: Configuration dictionary with 'type' key
        
    Returns:
        LanguageModel instance
    """
    model_type = config.get('type', 'gemma').lower()
    
    if model_type == 'gemma':
        return GemmaModel(config)
    elif model_type == 'qwen':
        return QwenModel(config)
    elif model_type == 'llama':
        return LLaMAModel(config)
    else:
        raise ValueError(f"Unknown language model type: {model_type}")
