"""
DataLoader
==========
Data loading utilities for VLA training
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Dict, Any, Optional, Tuple
from .dataset import build_dataset


class VLACollator:
    """
    Custom collate function for VLA data batches
    
    Handles:
    - Image stacking
    - Action stacking
    - Text tokenization
    """
    
    def __init__(self, tokenizer, max_length: int = 64):
        """
        Initialize collator
        
        Args:
            tokenizer: Language model tokenizer
            max_length: Maximum text sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            batch_dict: Dictionary of batched tensors
        """
        # Extract fields
        images = [item['image'] for item in batch]
        actions = [item['action'] for item in batch]
        languages = [item['language'] for item in batch]
        
        # Stack images and actions
        images = torch.stack(images)  # (B, C, H, W)
        actions = torch.stack(actions)  # (B, chunk_size, action_dim)
        
        # Tokenize text
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                languages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        else:
            # Dummy tokenization
            input_ids = torch.zeros(len(batch), self.max_length, dtype=torch.long)
            attention_mask = torch.ones(len(batch), self.max_length, dtype=torch.long)
        
        return {
            'images': images,
            'actions': actions,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'languages': languages,
        }


def build_dataloader(
    config: Dict[str, Any],
    tokenizer=None,
    training: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    Build DataLoader from config
    
    Args:
        config: Dataset configuration
        tokenizer: Language model tokenizer
        training: Whether in training mode
        distributed: Whether using distributed training
        world_size: Number of processes
        rank: Process rank
        
    Returns:
        DataLoader instance
    """
    # Build dataset
    dataset = build_dataset(config, training=training)
    
    # Build sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training
        )
        shuffle = False
    else:
        sampler = None
        shuffle = training
    
    # Build collator
    collator = VLACollator(tokenizer, max_length=config.get('max_text_length', 64))
    
    # Build DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=training,
        persistent_workers=config.get('num_workers', 4) > 0
    )
    
    return dataloader


def build_train_val_dataloaders(
    config: Dict[str, Any],
    tokenizer=None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders
    
    Args:
        config: Dataset configuration
        tokenizer: Language model tokenizer
        distributed: Whether using distributed training
        world_size: Number of processes
        rank: Process rank
        
    Returns:
        train_loader, val_loader
    """
    # Training loader
    train_config = config.copy()
    train_config['batch_size'] = config.get('train_batch_size', config.get('batch_size', 32))
    train_loader = build_dataloader(
        train_config, 
        tokenizer=tokenizer, 
        training=True,
        distributed=distributed,
        world_size=world_size,
        rank=rank
    )
    
    # Validation loader
    val_config = config.copy()
    val_config['batch_size'] = config.get('val_batch_size', config.get('batch_size', 32))
    val_loader = build_dataloader(
        val_config, 
        tokenizer=tokenizer, 
        training=False,
        distributed=distributed,
        world_size=world_size,
        rank=rank
    )
    
    return train_loader, val_loader
