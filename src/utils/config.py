"""
Configuration Utilities
=======================
Load and merge YAML configurations
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    
    Later configs override earlier ones.
    Nested dictionaries are merged recursively.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if config is None:
            continue
        
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def load_configs(*paths: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and merge multiple configuration files
    
    Args:
        *paths: Paths to YAML files
        
    Returns:
        Merged configuration dictionary
    """
    configs = []
    
    for path in paths:
        if path is not None:
            config = load_config(path)
            configs.append(config)
    
    return merge_configs(*configs)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'vision': {
                'type': 'siglip',
                'pretrained': 'siglip-so400m-patch14-384',
                'freeze': False,
            },
            'language': {
                'type': 'gemma',
                'pretrained': 'google/gemma-2b',
                'freeze': True,
            },
            'fusion': {
                'type': 'cross_attention',
                'hidden_dim': 768,
                'num_heads': 8,
            },
            'action_head': {
                'type': 'flow_matching',
                'action_dim': 7,
                'chunk_size': 10,
                'hidden_dim': 512,
                'num_steps': 50,
            },
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'grad_accum_steps': 4,
            'grad_clip': 1.0,
            'use_amp': True,
            'optimizer': {
                'type': 'AdamW',
                'lr': 3e-4,
                'weight_decay': 0.1,
                'betas': [0.9, 0.95],
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_steps': 1000,
                'min_lr': 1e-5,
            },
            'checkpoint': {
                'save_every': 1000,
                'keep_last': 5,
            },
            'logging': {
                'log_every': 100,
            },
        },
        'data': {
            'type': 'rlds',
            'image': {
                'height': 224,
                'width': 224,
                'normalize': True,
                'color_jitter': True,
                'random_crop': True,
                'flip': False,
            },
            'action': {
                'dim': 7,
                'chunk_size': 10,
                'normalize': True,
            },
        },
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises exception otherwise
    """
    # Check required sections
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Check model config
    model_config = config['model']
    if 'action_head' not in model_config:
        raise ValueError("Missing model.action_head configuration")
    
    action_head = model_config['action_head']
    if 'action_dim' not in action_head:
        raise ValueError("Missing action_head.action_dim")
    if 'chunk_size' not in action_head:
        raise ValueError("Missing action_head.chunk_size")
    
    return True
