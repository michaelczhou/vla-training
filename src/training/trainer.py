"""
VLA Trainer
===========
Main training loop for VLA models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
import os
import time
import json
from pathlib import Path
from tqdm import tqdm

from .optimizer import build_optimizer, build_scheduler, get_lr
from .losses import build_loss
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class VLATrainer:
    """
    VLA Model Trainer
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Logging (TensorBoard, WandB)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: VLA model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Build optimizer
        self.optimizer = build_optimizer(model, config.get('optimizer', {}))
        
        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // self.grad_accum_steps
        total_steps = steps_per_epoch * self.num_epochs
        
        # Build scheduler
        self.scheduler = build_scheduler(
            self.optimizer, 
            config.get('scheduler', {}),
            total_steps
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get('checkpoint', {}).get('save_every', 1000)
        self.keep_last = config.get('checkpoint', {}).get('keep_last', 5)
        
        # Logging
        self.log_every = config.get('logging', {}).get('log_every', 100)
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            metrics: Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {self.epoch + 1}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                loss, _ = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    actions=actions,
                    time=None
                )
                
                # Normalize loss by gradient accumulation steps
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.grad_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Logging
            if self.global_step % self.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = get_lr(self.optimizer)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.2e}'
                })
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
        
        avg_train_loss = total_loss / max(num_batches, 1)
        
        return {
            'train_loss': avg_train_loss,
            'learning_rate': get_lr(self.optimizer),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation
        
        Returns:
            metrics: Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_dataloader, desc='Validation')
        
        for batch in pbar:
            # Move batch to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                loss, _ = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    actions=actions,
                    time=None
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / max(num_batches, 1)
        
        return {
            'val_loss': avg_val_loss,
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop
        
        Returns:
            training_history: Dictionary with all metrics
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader, 'sampler') and \
               hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            if 'val_loss' in val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
                
                # Save best checkpoint
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best')
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
            
            # Save training history
            self.save_history(history)
            
            # Print summary
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} - "
                  f"Train Loss: {train_metrics['train_loss']:.4f} - "
                  f"Val Loss: {val_metrics.get('val_loss', 'N/A')} - "
                  f"Time: {elapsed:.1f}s")
        
        # Save final checkpoint
        self.save_checkpoint('final')
        
        return history
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'{name}.pt'
        
        save_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.config,
            best_val_loss=self.best_val_loss
        )
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'))
        
        if len(checkpoints) > self.keep_last:
            for checkpoint in checkpoints[:-self.keep_last]:
                if checkpoint.name not in ['best.pt', 'final.pt']:
                    checkpoint.unlink()
    
    def save_history(self, history: Dict[str, Any]):
        """Save training history to JSON"""
        history_path = self.log_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resumed from checkpoint: {checkpoint_path} "
              f"(epoch {self.epoch + 1}, step {self.global_step})")
