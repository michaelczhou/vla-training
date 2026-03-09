#!/usr/bin/env python3
"""
DROID Training Example
======================
Train VLA model on DROID dataset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.vla_model import build_vla_model
from src.training.trainer import VLATrainer
from src.data.dataloader import build_dataloader
from src.utils.config import load_configs, merge_configs, get_default_config


def main():
    print("=== DROID VLA Training Example ===\n")
    
    # Load configurations
    config = merge_configs(
        get_default_config(),
        load_configs('configs/model/pi0_base.yaml'),
        load_configs('configs/data/droid.yaml'),
        load_configs('configs/training/droid_finetune.yaml'),
    )
    
    # Build model
    print("Building model...")
    model = build_vla_model(config['model'])
    
    num_params = model.get_num_params()
    trainable_params = model.get_trainable_params()
    print(f"Model: {num_params:,} total params, {trainable_params:,} trainable\n")
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config['data'], training=True)
    val_loader = build_dataloader(config['data'], training=False)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")
    
    # Build trainer
    print("Building trainer...")
    trainer = VLATrainer(
        model=model,
        config=config['training'],
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    print("Starting training...\n")
    history = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
