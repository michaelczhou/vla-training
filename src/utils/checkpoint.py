"""
Enhanced Checkpoint Management
===============================
Advanced checkpoint saving, loading, and management

Features:
- Best model tracking
- Checkpoint pruning/cleanup
- Resume training support
- Partial loading (transfer learning)
- Checkpoint metadata
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import shutil
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for saving and loading model checkpoints.
    
    Features:
    - Automatic best model tracking
    - Checkpoint history and pruning
    - Resume training support
    - Checkpoint metadata
    
    Usage:
        manager = CheckpointManager(
            checkpoint_dir='checkpoints/my_experiment',
            max_checkpoints=5,
            save_best=True
        )
        
        # Save checkpoint
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'val_loss': 0.5}
        )
        
        # Load best model
        manager.load_best(model)
        
        # Resume training
        start_epoch = manager.load_latest(model, optimizer, scheduler)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True,
        metric_key: str = 'val_loss',
        mode: str = 'min',  # 'min' or 'max'
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best model based on metric
            metric_key: Key to use for metric comparison
            mode: 'min' for loss, 'max' for accuracy
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_key = metric_key
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': []}
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than best."""
        if self.mode == 'min':
            return metric < self.best_metric
        return metric > self.best_metric
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Metrics dictionary
            config: Configuration dictionary
            extra_data: Extra data to save
            name: Custom checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        if name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f'checkpoint_epoch_{epoch}_{timestamp}'
        
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add optimizer state
        if optimizer is not None and self.save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None and self.save_scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add config
        if config is not None:
            checkpoint['config'] = config
        
        # Add extra data
        if extra_data is not None:
            checkpoint.update(extra_data)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        metric_value = metrics.get(self.metric_key) if metrics else None
        checkpoint_info = {
            'name': name,
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'is_best': False,
            'timestamp': checkpoint['timestamp']
        }
        
        # Check if this is the best model
        if metric_value is not None and self.save_best:
            if self._is_better(metric_value):
                self.best_metric = metric_value
                checkpoint_info['is_best'] = True
                
                # Save best model
                best_path = self.checkpoint_dir / 'best_model.pt'
                shutil.copy2(checkpoint_path, best_path)
                
                # Also save EMA best if exists
                # (handled by caller)
                
                logger.info(f"New best model! {self.metric_key}={metric_value:.4f}")
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        shutil.copy2(checkpoint_path, latest_path)
        
        # Update metadata
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['last_save'] = checkpoint['timestamp']
        
        # Prune old checkpoints
        self._prune_checkpoints()
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def _prune_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        # Keep best + latest + max_checkpoints
        to_keep = []
        for ckpt in checkpoints:
            if ckpt.get('is_best'):
                to_keep.append(ckpt['name'])
                continue
            if 'latest' in ckpt['name']:
                to_keep.append(ckpt['name'])
                continue
            if len([n for n in to_keep if 'epoch' in n]) < self.max_checkpoints:
                to_keep.append(ckpt['name'])
        
        # Remove old checkpoints
        for ckpt in checkpoints:
            if ckpt['name'] not in to_keep:
                ckpt_path = Path(ckpt['path'])
                if ckpt_path.exists():
                    ckpt_path.unlink()
                    logger.debug(f"Removed old checkpoint: {ckpt_path}")
        
        # Update metadata
        self.metadata['checkpoints'] = [
            ckpt for ckpt in checkpoints
            if ckpt['name'] in to_keep
        ]
    
    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
            strict: Whether to strictly match state dict
            
        Returns:
            Checkpoint info dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
    
    def load_best(
        self,
        model: nn.Module,
        device: str = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load best model checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return self.load(str(best_path), model, device=device, strict=strict)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cpu'
    ) -> int:
        """Load latest checkpoint and return start epoch."""
        latest_path = self.checkpoint_dir / 'latest.pt'
        
        if not latest_path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return 0
        
        info = self.load(
            str(latest_path),
            model,
            optimizer,
            scheduler,
            device=device,
            strict=False
        )
        
        epoch = info.get('epoch', 0)
        logger.info(f"Resuming from epoch {epoch}")
        
        return epoch
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return sorted(
            self.metadata.get('checkpoints', []),
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Simple checkpoint save function (backward compatible).
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Metrics dictionary
        config: Configuration
        **kwargs: Extra data
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    
    # Save config separately
    if config is not None:
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Simple checkpoint load function (backward compatible).
    
    Args:
        path: Path to checkpoint
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model
    if model is not None:
        model.load_state_dict(checkpoint.get('model_state_dict', {}))
    
    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint