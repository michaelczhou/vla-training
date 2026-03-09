"""
Checkpoint Utilities
====================
Save and load model checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import json


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    best_val_loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        global_step: Current global step
        config: Model/training configuration
        best_val_loss: Best validation loss
        metadata: Additional metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    
    # Save config separately for easy access
    if config is not None:
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_checkpoint(
    path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        path: Path to checkpoint file
        device: Device to load tensors on
        
    Returns:
        Checkpoint dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    return checkpoint


def load_model_from_checkpoint(
    model: nn.Module,
    path: str,
    device: str = 'cpu',
    strict: bool = True
) -> nn.Module:
    """
    Load model weights from checkpoint
    
    Args:
        model: Model to load weights into
        path: Path to checkpoint file
        device: Device to load tensors on
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Model with loaded weights
    """
    checkpoint = load_checkpoint(path, device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    return model


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """
    Get checkpoint metadata without loading full state dict
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint info
    """
    path = Path(path)
    
    if not path.exists():
        return {}
    
    # Try to load config file first
    config_path = path.with_suffix('.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    # Load checkpoint metadata
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', None),
        'config': config or checkpoint.get('config', None),
    }
    
    # Model info
    state_dict = checkpoint.get('model_state_dict', {})
    info['num_params'] = sum(p.numel() for p in state_dict.values())
    
    return info
