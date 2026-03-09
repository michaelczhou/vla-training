"""
FAST Tokenizer
==============
Frequency-space Action Sequence Tokenizer

Uses DCT (Discrete Cosine Transform) to convert action sequences
to frequency domain, then quantizes and encodes as tokens.

Reference: FAST (Frequency-space Action Sequence Tokenizer)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy.fftpack import dct, idct


class FASTTokenizer:
    """
    Frequency-space Action Sequence Tokenizer
    
    Converts continuous action sequences to discrete tokens using:
    1. DCT transformation to frequency domain
    2. Quantization of frequency coefficients
    3. BPE-style encoding to tokens
    """
    
    def __init__(
        self, 
        action_dim: int = 7, 
        chunk_size: int = 10, 
        num_tokens: int = 32,
        num_frequency_components: int = 8
    ):
        """
        Initialize FAST tokenizer
        
        Args:
            action_dim: Dimension of action space
            chunk_size: Number of timesteps per chunk
            num_tokens: Number of discrete tokens per dimension
            num_frequency_components: Number of DCT coefficients to keep
        """
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_tokens = num_tokens
        self.num_frequency_components = num_frequency_components
        
        # Total tokens per chunk
        self.total_tokens = num_frequency_components * action_dim
        
        # Quantization bins
        self.bins = torch.linspace(-1, 1, num_tokens + 1)
    
    def _dct(self, x: np.ndarray) -> np.ndarray:
        """Apply DCT along time dimension"""
        return dct(x, type=2, norm='ortho', axis=0)
    
    def _idct(self, x: np.ndarray) -> np.ndarray:
        """Apply inverse DCT along time dimension"""
        return idct(x, type=2, norm='ortho', axis=0)
    
    def encode(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Encode action chunk to tokens
        
        Args:
            action_chunk: Tensor of shape (chunk_size, action_dim)
            
        Returns:
            tokens: Tensor of shape (total_tokens,)
        """
        # Convert to numpy
        action_np = action_chunk.cpu().numpy()  # (chunk_size, action_dim)
        
        # Apply DCT along time dimension
        freq_coeffs = self._dct(action_np)  # (chunk_size, action_dim)
        
        # Keep only low-frequency components
        freq_coeffs = freq_coeffs[:self.num_frequency_components, :]  # (num_freq, action_dim)
        
        # Flatten
        freq_coeffs = freq_coeffs.flatten()  # (num_freq * action_dim,)
        
        # Quantize
        tokens = torch.zeros_like(freq_coeffs, dtype=torch.long)
        for i, coeff in enumerate(freq_coeffs):
            # Find bin
            bin_idx = torch.searchsorted(self.bins, torch.tensor(coeff)) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_tokens - 1)
            tokens[i] = bin_idx
        
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to action chunk
        
        Args:
            tokens: Tensor of shape (total_tokens,)
            
        Returns:
            action_chunk: Tensor of shape (chunk_size, action_dim)
        """
        # Dequantize
        freq_coeffs = np.zeros(self.total_tokens)
        for i, token in enumerate(tokens):
            # Get bin center
            bin_center = (self.bins[token] + self.bins[token + 1]) / 2
            freq_coeffs[i] = bin_center.item()
        
        # Reshape
        freq_coeffs = freq_coeffs.reshape(self.num_frequency_components, self.action_dim)
        
        # Pad to full length
        freq_coeffs_full = np.zeros((self.chunk_size, self.action_dim))
        freq_coeffs_full[:self.num_frequency_components, :] = freq_coeffs
        
        # Apply inverse DCT
        action_chunk = self._idct(freq_coeffs_full)
        
        return torch.tensor(action_chunk, dtype=torch.float32)
    
    def encode_batch(self, action_chunks: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of action chunks
        
        Args:
            action_chunks: Tensor of shape (B, chunk_size, action_dim)
            
        Returns:
            tokens: Tensor of shape (B, total_tokens)
        """
        B = action_chunks.shape[0]
        all_tokens = []
        
        for i in range(B):
            tokens = self.encode(action_chunks[i])
            all_tokens.append(tokens)
        
        return torch.stack(all_tokens)
    
    def decode_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode batch of tokens
        
        Args:
            tokens: Tensor of shape (B, total_tokens)
            
        Returns:
            action_chunks: Tensor of shape (B, chunk_size, action_dim)
        """
        B = tokens.shape[0]
        all_actions = []
        
        for i in range(B):
            action = self.decode(tokens[i])
            all_actions.append(action)
        
        return torch.stack(all_actions)


class FASTEmbedding(nn.Module):
    """
    Embedding layer for FAST tokens
    """
    
    def __init__(
        self, 
        num_tokens: int = 32,
        num_frequency_components: int = 8,
        action_dim: int = 7,
        embedding_dim: int = 256
    ):
        super().__init__()
        
        self.tokenizer = FASTTokenizer(
            action_dim=action_dim,
            num_tokens=num_tokens,
            num_frequency_components=num_frequency_components
        )
        
        # Embedding table
        self.total_vocab = num_tokens
        self.total_positions = num_frequency_components * action_dim
        
        self.embedding = nn.Embedding(
            num_embeddings=self.total_vocab,
            embedding_dim=embedding_dim
        )
        
        # Position embeddings
        self.position_embedding = nn.Embedding(
            num_embeddings=self.total_positions,
            embedding_dim=embedding_dim
        )
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens
        
        Args:
            tokens: Tensor of shape (B, total_positions)
            
        Returns:
            embeddings: Tensor of shape (B, total_positions, embedding_dim)
        """
        # Token embeddings
        token_emb = self.embedding(tokens)  # (B, total_positions, D)
        
        # Position embeddings
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        pos_emb = self.position_embedding(positions)  # (total_positions, D)
        
        # Combine
        embeddings = token_emb + pos_emb
        
        return embeddings


def build_fast_tokenizer(config: dict) -> FASTTokenizer:
    """
    Build FAST tokenizer from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FASTTokenizer instance
    """
    return FASTTokenizer(
        action_dim=config.get('action_dim', 7),
        chunk_size=config.get('chunk_size', 10),
        num_tokens=config.get('num_tokens', 32),
        num_frequency_components=config.get('num_frequency_components', 8)
    )
