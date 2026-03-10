"""
Configuration Validation Utilities
===================================
Validate YAML configuration files for VLA training

Usage:
    from utils.config_validator import validate_config, ModelConfig, TrainingConfig
    
    # Validate config
    errors = validate_config(config)
    if errors:
        for error in errors:
            print(f"Error: {error}")
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Configuration validation error."""
    field: str
    message: str
    severity: str  # 'error', 'warning'
    
    def __str__(self):
        prefix = "⚠️" if self.severity == "warning" else "❌"
        return f"{prefix} {self.field}: {self.message}"


class ConfigValidator:
    """
    Configuration validator for VLA training configs.
    
    Validates model, data, and training configurations.
    """
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def add_error(self, field: str, message: str, severity: str = 'error'):
        """Add a validation error."""
        self.errors.append(ValidationError(field, message, severity))
    
    def validate_model_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            List of validation errors
        """
        self.errors = []
        
        # Check required fields
        required_fields = ['vision', 'language', 'fusion', 'action_head']
        for field in required_fields:
            if field not in config:
                self.add_error(f'model.{field}', f'Required field missing', 'error')
        
        # Vision config
        if 'vision' in config:
            vision = config['vision']
            if 'type' not in vision:
                self.add_error('model.vision.type', 'Vision encoder type required', 'error')
            if vision.get('type') not in ['vit', 'siglip', 'resnet']:
                self.add_error(
                    'model.vision.type',
                    f"Unknown type: {vision.get('type')}. Use: vit, siglip, resnet",
                    'warning'
                )
        
        # Language config
        if 'language' in config:
            language = config['language']
            if 'type' not in language:
                self.add_error('model.language.type', 'Language model type required', 'error')
            if language.get('freeze') is None:
                self.add_error(
                    'model.language.freeze',
                    'Not specified, defaulting to False',
                    'warning'
                )
        
        # Action head config
        if 'action_head' in config:
            action_head = config['action_head']
            required = ['type', 'action_dim', 'chunk_size']
            for field in required:
                if field not in action_head:
                    self.add_error(
                        f'model.action_head.{field}',
                        f'Required field missing',
                        'error'
                    )
            
            # Check action head type
            valid_types = ['flow_matching', 'diffusion', 'mlp']
            if 'type' in action_head and action_head['type'] not in valid_types:
                self.add_error(
                    'model.action_head.type',
                    f"Unknown type: {action_head['type']}",
                    'error'
                )
        
        return self.errors
    
    def validate_training_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            List of validation errors
        """
        self.errors = []
        
        # Batch size
        if 'batch_size' in config:
            bs = config['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                self.add_error('training.batch_size', 'Must be positive integer', 'error')
            elif bs > 256:
                self.add_error(
                    'training.batch_size',
                    f'Large batch size ({bs}), consider gradient accumulation',
                    'warning'
                )
        
        # Learning rate
        if 'optimizer' in config and 'lr' in config['optimizer']:
            lr = config['optimizer']['lr']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.add_error('training.optimizer.lr', 'Must be positive number', 'error')
            elif lr > 1e-2:
                self.add_error(
                    'training.optimizer.lr',
                    f'Large learning rate ({lr}), may cause instability',
                    'warning'
                )
            elif lr < 1e-6:
                self.add_error(
                    'training.optimizer.lr',
                    f'Small learning rate ({lr}), may not converge',
                    'warning'
                )
        
        # Check mixed precision
        if 'mixed_precision' in config:
            valid_types = ['fp16', 'bf16', 'fp32']
            if config['mixed_precision'] not in valid_types:
                self.add_error(
                    'training.mixed_precision',
                    f"Unknown type: {config['mixed_precision']}",
                    'warning'
                )
        
        return self.errors
    
    def validate_data_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate data configuration.
        
        Args:
            config: Data configuration dictionary
            
        Returns:
            List of validation errors
        """
        self.errors = []
        
        # Data path
        if 'path' not in config and 'data_dir' not in config:
            self.add_error('data.path', 'Data path not specified', 'error')
        
        # Image size
        if 'image_size' in config:
            size = config['image_size']
            if isinstance(size, int):
                if size not in [224, 256, 384, 448, 512]:
                    self.add_error(
                        'data.image_size',
                        f'Unusual size: {size}',
                        'warning'
                    )
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                if size[0] != size[1]:
                    self.add_error(
                        'data.image_size',
                        'Non-square images may cause issues',
                        'warning'
                    )
        
        # Action config
        if 'action_dim' in config:
            dim = config['action_dim']
            if not isinstance(dim, int) or dim <= 0:
                self.add_error('data.action_dim', 'Must be positive integer', 'error')
            elif dim > 100:
                self.add_error(
                    'data.action_dim',
                    f'Large action dimension ({dim}), unusual',
                    'warning'
                )
        
        return self.errors
    
    def validate(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None
    ) -> List[ValidationError]:
        """
        Validate all configurations.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            
        Returns:
            List of all validation errors
        """
        all_errors = []
        
        if model_config:
            all_errors.extend(self.validate_model_config(model_config))
        
        if training_config:
            all_errors.extend(self.validate_training_config(training_config))
        
        if data_config:
            all_errors.extend(self.validate_data_config(data_config))
        
        return all_errors


def validate_config(
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False
) -> Tuple[bool, List[ValidationError]]:
    """
    Validate VLA configuration files.
    
    Args:
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        data_config: Data configuration dictionary
        raise_on_error: Whether to raise exception on validation errors
        
    Returns:
        Tuple of (is_valid, errors)
        
    Example:
        is_valid, errors = validate_config(
            model_config=load_config('configs/model/pi0.yaml'),
            training_config=load_config('configs/training/droid.yaml'),
            data_config=load_config('configs/data/droid.yaml')
        )
        
        if not is_valid:
            for error in errors:
                print(error)
    """
    validator = ConfigValidator()
    errors = validator.validate(model_config, training_config, data_config)
    
    is_valid = all(e.severity == 'warning' for e in errors)
    
    if errors and raise_on_error:
        error_messages = '\n'.join(str(e) for e in errors)
        raise ValueError(f"Configuration validation failed:\n{error_messages}")
    
    return is_valid, errors


def print_validation_report(
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None
):
    """
    Print a formatted validation report.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
    """
    is_valid, errors = validate_config(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config
    )
    
    print("=" * 60)
    print("  Configuration Validation Report")
    print("=" * 60)
    
    if not errors:
        print("\n✅ All checks passed!")
    else:
        errors_count = sum(1 for e in errors if e.severity == 'error')
        warnings_count = sum(1 for e in errors if e.severity == 'warning')
        
        print(f"\n📊 Summary:")
        print(f"   Errors: {errors_count}")
        print(f"   Warnings: {warnings_count}")
        
        print(f"\n📋 Details:")
        for error in errors:
            print(f"   {error}")
    
    print("=" * 60)
    
    return is_valid