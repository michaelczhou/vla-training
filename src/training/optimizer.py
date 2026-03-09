"""
Optimizer & Scheduler
=====================
Optimizer and learning rate scheduler utilities
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR, LinearLR, SequentialLR,
    ReduceLROnPlateau, StepLR
)
from typing import Dict, Any, Optional, Tuple


def build_optimizer(
    model: nn.Module, 
    config: Dict[str, Any]
) -> Optimizer:
    """
    Build optimizer from config
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_type = config.get('type', 'AdamW').lower()
    lr = config.get('lr', 3e-4)
    weight_decay = config.get('weight_decay', 0.1)
    
    # Get parameters with optional different lr for different parts
    params = get_parameter_groups(model, config)
    
    if optimizer_type == 'adam':
        optimizer = Adam(
            params,
            lr=lr,
            betas=tuple(config.get('betas', [0.9, 0.999])),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(
            params,
            lr=lr,
            betas=tuple(config.get('betas', [0.9, 0.95])),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            params,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(
    model: nn.Module, 
    config: Dict[str, Any]
) -> list:
    """
    Create parameter groups with different learning rates
    
    Args:
        model: Model to get parameters from
        config: Configuration with lr multipliers
        
    Returns:
        List of parameter dictionaries
    """
    # Check for layer-specific lr multipliers
    vision_lr_mult = config.get('vision_lr_mult', 1.0)
    language_lr_mult = config.get('language_lr_mult', 1.0)
    head_lr_mult = config.get('head_lr_mult', 1.0)
    
    base_lr = config.get('lr', 3e-4)
    
    param_groups = []
    
    # Vision encoder parameters
    if hasattr(model, 'vision_encoder'):
        for name, param in model.vision_encoder.named_parameters():
            if param.requires_grad:
                param_groups.append({
                    'params': param,
                    'lr': base_lr * vision_lr_mult,
                    'name': f'vision.{name}'
                })
    
    # Language model parameters
    if hasattr(model, 'language_model'):
        for name, param in model.language_model.named_parameters():
            if param.requires_grad:
                param_groups.append({
                    'params': param,
                    'lr': base_lr * language_lr_mult,
                    'name': f'language.{name}'
                })
    
    # Action head parameters
    if hasattr(model, 'action_head'):
        for name, param in model.action_head.named_parameters():
            if param.requires_grad:
                param_groups.append({
                    'params': param,
                    'lr': base_lr * head_lr_mult,
                    'name': f'head.{name}'
                })
    
    # Other parameters (fusion, etc.)
    handled_params = set()
    for group in param_groups:
        for p in group['params']:
            handled_params.add(id(p))
    
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in handled_params:
            param_groups.append({
                'params': param,
                'lr': base_lr,
                'name': f'other.{name}'
            })
    
    if len(param_groups) == 0:
        # Fallback: all parameters
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    
    return param_groups


def build_scheduler(
    optimizer: Optimizer, 
    config: Dict[str, Any],
    num_training_steps: int
) -> Optional[Any]:
    """
    Build learning rate scheduler from config
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Scheduler instance or None
    """
    scheduler_type = config.get('type', 'cosine')
    
    if scheduler_type is None or scheduler_type == 'none':
        return None
    
    warmup_steps = config.get('warmup_steps', 1000)
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.get('min_lr', 1e-5)
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_type == 'linear':
        # Linear decay with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        linear_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.get('min_lr', 1e-5) / config.get('lr', 3e-4),
            total_iters=num_training_steps - warmup_steps
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, linear_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_type == 'step':
        # Step decay
        scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 10000),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 1000),
            min_lr=config.get('min_lr', 1e-6)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate"""
    return optimizer.param_groups[0]['lr']
