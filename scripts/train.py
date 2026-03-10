"""
Training Script for VLA Models
================================
Supports RDT2, Pi0, and custom VLA architectures.

Features:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Exponential Moving Average (EMA)
- Learning rate warmup
- Checkpoint saving/loading
- TensorBoard & WandB logging
- Resume training from checkpoint

Usage:
    # Basic training
    python scripts/train.py --config configs/model/pi0_base.yaml --data configs/data/droid.yaml

    # With custom settings
    python scripts/train.py --config configs/model/pi0_base.yaml \
        --data configs/data/droid.yaml --batch-size 16 --lr 3e-4 \
        --epochs 100 --use-wandb --exp-name my_experiment

    # Resume from checkpoint
    python scripts/train.py --config configs/model/pi0_base.yaml \
        --data configs/data/droid.yaml --resume checkpoints/my_experiment/latest.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import wandb
from datetime import datetime
from typing import Dict, Any, Optional

from models.rdt2_model import RDT2Transformer
from models.pi0_model import PiZeroModel, FlowMatchingActionHead
from data.dataset import VLADataset
from training.losses import FlowMatchingLoss, DiffusionLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train VLA (Vision-Language-Action) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model & Data configs
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to model config YAML file"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data config YAML file"
    )
    
    # Experiment settings
    parser.add_argument(
        "--exp-name", type=str, default="vla_experiment",
        help="Experiment name for logging/checkpoints"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints",
        help="Directory to save checkpoints"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000,
        help="Number of warmup steps for learning rate"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    # EMA settings
    parser.add_argument(
        "--use-ema", action="store_true",
        help="Enable Exponential Moving Average"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999,
        help="EMA decay rate"
    )
    
    # Device settings
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--pin-memory", action="store_true", default=True,
        help="Pin memory for faster data transfer"
    )
    
    # Logging settings
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--use-tensorboard", action="store_true", default=True,
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Log every N batches"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Save checkpoint every N epochs"
    )
    
    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create VLA model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Initialized VLA model
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_config.get('type', 'pi0')
    
    if model_type == 'rdt2':
        model = RDT2Transformer(
            vision_encoder_name=model_config.get(
                'vision_encoder', 'Qwen/Qwen2.5-VL-7B-Instruct'
            ),
            action_dim=model_config.get('action_dim', 14),
            num_diffusion_steps=model_config.get('num_diffusion_steps', 100),
            use_rvq=model_config.get('use_rvq', True),
            rvq_config=model_config.get('rvq_config', {})
        )
    elif model_type == 'pi0':
        model = PiZeroModel(
            vision_encoder_name=model_config.get(
                'vision_encoder', 'google/siglip-base-patch16-224'
            ),
            language_model_name=model_config.get(
                'language_model', 'microsoft/phi-2'
            ),
            action_dim=model_config.get('action_dim', 7),
            num_action_chunks=model_config.get('num_action_chunks', 10),
            use_memory=model_config.get('use_memory', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_dataloader(
    data_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    shuffle: bool = True
) -> DataLoader:
    """
    Create training DataLoader.
    
    Args:
        data_config: Data configuration dictionary
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = VLADataset(
        data_path=data_config.get('path', data_config.get('data_dir')),
        seq_length=data_config.get('seq_length', 10),
        image_size=data_config.get('image_size', 224),
        augment=data_config.get('augment', True)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    return dataloader


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains a shadow copy of model parameters that is updated
    as an exponential moving average of the training parameters.
    
    Args:
        model: Model to track
        decay: Decay rate for EMA (typically 0.999)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with shadow (EMA) parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_target_velocity(actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Compute target velocity for flow matching training.
    
    For linear interpolation: x_t = (1-t) * x_0 + t * x_1
    where x_0 ~ N(0, I) and x_1 = actions
    
    The velocity is: v_t = x_1 - x_0
    
    Args:
        actions: Target action tensor of shape (B, T, A)
        timesteps: Time steps of shape (B,)
        
    Returns:
        Target velocity tensor
    """
    batch_size = actions.shape[0]
    noise = torch.randn_like(actions)
    
    # Ensure timesteps has correct shape for broadcasting
    timesteps = timesteps.view(batch_size, 1, 1)
    
    # Compute velocity: v_t = action - noise
    # This is used for flow matching loss
    velocity = actions - noise
    
    return velocity


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    ema: Optional[EMA],
    logger,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    log_interval: int = 10,
    tensorboard_writer: Optional[SummaryWriter] = None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: VLA model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: Mixed precision scaler
        device: Device to train on
        epoch: Current epoch number
        ema: Optional EMA instance
        logger: Logger instance
        use_wandb: Whether to log to WandB
        use_tensorboard: Whether to log to TensorBoard
        grad_clip: Maximum gradient norm for clipping
        grad_accum_steps: Gradient accumulation steps
        log_interval: Logging interval
        tensorboard_writer: TensorBoard writer
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
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
        with autocast(dtype=torch.bfloat16):
            outputs = model(
                images=images,
                text_input_ids=text_input_ids,
                actions=actions,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
            
            # Compute loss based on model type
            if isinstance(model, PiZeroModel):
                target_velocity = compute_target_velocity(actions, timesteps)
                pred_velocity = outputs['velocity']
                loss = F.mse_loss(pred_velocity, target_velocity)
            elif isinstance(model, RDT2Transformer):
                predicted_noise = outputs[0] if isinstance(outputs, tuple) else \
                    outputs.get('predicted_noise')
                noise = torch.randn_like(actions)
                loss = F.mse_loss(predicted_noise, noise)
            else:
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[1]
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        # Logging
        total_loss += loss.item() * grad_accum_steps
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Log to WandB
        if use_wandb and batch_idx % log_interval == 0:
            wandb.log({
                'train/loss': loss.item() * grad_accum_steps,
                'train/batch': batch_idx + epoch * num_batches,
                'train/lr': optimizer.param_groups[0]['lr']
            })
        
        # Log to TensorBoard
        if use_tensorboard and tensorboard_writer is not None and batch_idx % log_interval == 0:
            global_step = batch_idx + epoch * num_batches
            tensorboard_writer.add_scalar('train/loss', loss.item() * grad_accum_steps, global_step)
            tensorboard_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    ema: Optional[EMA] = None
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: VLA model (or EMA shadow model)
        dataloader: Validation data loader
        device: Device to validate on
        epoch: Current epoch number
        ema: Optional EMA instance (if validating on EMA model)
        
    Returns:
        Average validation loss
    """
    # Use EMA parameters if available
    if ema is not None:
        ema.apply_shadow()
    
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
    
    # Restore original parameters
    if ema is not None:
        ema.restore()
    
    return total_loss / num_batches


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(args.exp_name)
    logger.info("=" * 60)
    logger.info(f"VLA Training - Experiment: {args.exp_name}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs to output directory
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # Load configs
    try:
        model_config = load_config(args.config)
        data_config = load_config(args.data)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    logger.info(f"Model config: {model_config.get('name', 'unknown')}")
    logger.info(f"Data config: {data_config.get('name', 'unknown')}")
    
    # Initialize TensorBoard
    tensorboard_writer = None
    if args.use_tensorboard:
        log_dir = output_dir / "logs" / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging to: {log_dir}")
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(
            project="vla-training",
            name=args.exp_name,
            config={
                **model_config,
                **data_config,
                **vars(args)
            }
        )
        logger.info("WandB logging enabled")
    
    # Create model
    logger.info("Creating model...")
    try:
        model = create_model(model_config)
        model = model.to(args.device)
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Create dataloader
    logger.info("Loading dataset...")
    try:
        train_loader = create_dataloader(
            data_config,
            args.batch_size,
            args.num_workers,
            args.pin_memory
        )
        logger.info(f"Training samples: {len(train_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        sys.exit(1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return max(
            0.01,
            float(args.epochs - step) / float(max(1, args.epochs - args.warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        logger.info(f"EMA enabled with decay={args.ema_decay}")
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
    
    # Training loop
    best_train_loss = float('inf')
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=args.device,
            epoch=epoch,
            ema=ema,
            logger=logger,
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
            tensorboard_writer=tensorboard_writer
        )
        
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        # Step scheduler
        scheduler.step()
        
        # Log to TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar('epoch/train_loss', train_loss, epoch)
            tensorboard_writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss
            )
            
            # Also save as latest
            latest_path = output_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss
            )
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_path = output_dir / "best_model.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss
            )
            logger.info(f"★ New best model saved! Loss: {best_train_loss:.6f}")
            
            # Save EMA best model
            if ema is not None:
                ema.apply_shadow()
                ema_path = output_dir / "best_model_ema.pt"
                save_checkpoint(
                    ema_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    train_loss
                )
                ema.restore()
                logger.info(f"★ EMA best model saved: {ema_path}")
    
    # Training complete
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best training loss: {best_train_loss:.6f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 60)
    
    # Cleanup
    if args.use_wandb:
        wandb.finish()
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()