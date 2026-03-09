"""
Training script for VLA models.
Supports RDT2, Pi0, and custom architectures.

Usage:
    python scripts/train.py --config configs/model/pi0_base.yaml --data configs/data/droid.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm
import wandb
from datetime import datetime

from models.rdt2_model import RDT2Transformer
from models.pi0_model import PiZeroModel, FlowMatchingActionHead
from data.dataset import VLADataset
from training.losses import FlowMatchingLoss, DiffusionLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA model")
    parser.add_argument("--config", type=str, required=True, help="Model config file")
    parser.add_argument("--data", type=str, required=True, help="Data config file")
    parser.add_argument("--exp-name", type=str, default="vla_experiment", help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_config):
    """Create model based on config."""
    model_type = model_config.get('type', 'pi0')
    
    if model_type == 'rdt2':
        model = RDT2Transformer(
            vision_encoder_name=model_config.get('vision_encoder', 'Qwen/Qwen2.5-VL-7B-Instruct'),
            action_dim=model_config.get('action_dim', 14),
            num_diffusion_steps=model_config.get('num_diffusion_steps', 100),
            use_rvq=model_config.get('use_rvq', True),
            rvq_config=model_config.get('rvq_config', {})
        )
    elif model_type == 'pi0':
        model = PiZeroModel(
            vision_encoder_name=model_config.get('vision_encoder', 'google/siglip-base-patch16-224'),
            language_model_name=model_config.get('language_model', 'microsoft/phi-2'),
            action_dim=model_config.get('action_dim', 7),
            num_action_chunks=model_config.get('num_action_chunks', 10),
            use_memory=model_config.get('use_memory', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_dataloader(data_config, batch_size, num_workers):
    """Create training dataloader."""
    dataset = VLADataset(
        data_path=data_config['path'],
        seq_length=data_config.get('seq_length', 10),
        image_size=data_config.get('image_size', 224),
        augment=data_config.get('augment', True)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, logger, use_wandb=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        actions = batch['actions'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Generate random timesteps for flow matching
        batch_size = actions.shape[0]
        timesteps = torch.rand(batch_size, device=device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast(dtype=torch.bfloat16):
            outputs = model(
                images=images,
                text_input_ids=text_input_ids,
                actions=actions,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
            
            # Compute loss
            if isinstance(model, PiZeroModel):
                # Flow matching loss
                target_velocity = compute_target_velocity(actions, timesteps)
                pred_velocity = outputs['velocity']
                loss = F.mse_loss(pred_velocity, target_velocity)
            elif isinstance(model, RDT2Transformer):
                # Diffusion/flow matching loss
                predicted_noise = outputs[0] if isinstance(outputs, tuple) else outputs.get('predicted_noise')
                noise = torch.randn_like(actions)
                loss = F.mse_loss(predicted_noise, noise)
            else:
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[1]
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        if use_wandb and batch_idx % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/batch': batch_idx + epoch * num_batches
            })
    
    return total_loss / num_batches


def compute_target_velocity(actions, timesteps):
    """
    Compute target velocity for flow matching.
    For simple linear interpolation: x_t = (1-t) * x_0 + t * x_1
    where x_0 ~ N(0, I) and x_1 = actions
    
    Velocity: v_t = x_1 - x_0
    """
    # This is a simplified version
    # In practice, you'd need the actual noise sample used to create x_t
    batch_size = actions.shape[0]
    noise = torch.randn_like(actions)
    
    # x_t = (1-t) * noise + t * actions
    # v_t = actions - noise
    velocity = actions - noise
    
    return velocity


def validate(model, dataloader, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation {epoch}"):
            images = batch['images'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            actions = batch['actions'].to(device)
            
            timesteps = torch.rand(actions.shape[0], device=device)
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    images=images,
                    text_input_ids=text_input_ids,
                    actions=actions,
                    timesteps=timesteps
                )
                
                # Compute validation loss
                if isinstance(outputs, dict):
                    loss = outputs.get('loss', torch.tensor(0.0))
                else:
                    loss = outputs[1] if len(outputs) > 1 else torch.tensor(0.0)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    args = parse_args()
    
    # Setup
    logger = setup_logger(args.exp_name)
    logger.info(f"Starting training: {args.exp_name}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    model_config = load_config(args.config)
    data_config = load_config(args.data)
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="vla-training",
            name=args.exp_name,
            config={**model_config, **data_config, **vars(args)}
        )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(model_config)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloader
    logger.info("Loading dataset...")
    train_loader = create_dataloader(data_config, args.batch_size, args.num_workers)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler,
            args.device, epoch, logger, args.use_wandb
        )
        
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_path = output_dir / "best_model.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss
            )
            logger.info(f"New best model saved!")
    
    logger.info("Training complete!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
