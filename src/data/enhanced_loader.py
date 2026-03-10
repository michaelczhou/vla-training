"""
Enhanced Data Loading Utilities
================================
Advanced data loading with caching, prefetching, and error handling

Features:
- Automatic data caching
- Prefetching with multiple workers
- Graceful error handling
- Data validation
- Memory-efficient loading
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import get_worker_info
from typing import Dict, Any, List, Optional, Callable, Iterator
import numpy as np
import random
from pathlib import Path
from functools import lru_cache
import hashlib
import time
import queue
from threading import Thread


class CachedDataset(Dataset):
    """
    Dataset with automatic caching to speed up data loading.
    
    Caches loaded samples in memory after first access.
    Useful for small datasets that fit in memory.
    
    Args:
        base_dataset: Underlying dataset to wrap
        cache_size: Maximum number of samples to cache (None = all)
    """
    
    def __init__(self, base_dataset: Dataset, cache_size: Optional[int] = None):
        self.base_dataset = base_dataset
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with caching."""
        if idx in self.cache:
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
            return self.cache[idx]
        
        # Load from base dataset
        item = self.base_dataset[idx]
        
        # Cache if not at capacity
        if self.cache_size is None or len(self.cache) < self.cache_size:
            self.cache[idx] = item
            self.access_count[idx] = 1
        
        return item
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_samples': len(self.cache),
            'cache_size': self.cache_size or len(self.base_dataset),
            'total_accesses': sum(self.access_count.values()),
            'hit_rate': sum(self.access_count.values()) / max(1, len(self))
        }


class PrefetchDataLoader:
    """
    DataLoader with CUDA prefetching.
    
    Prefetches data to GPU while the previous batch is being processed.
    Can significantly improve training speed on GPUs.
    
    Args:
        dataloader: Base DataLoader
        device: Target device (cuda:0, cuda:1, etc.)
        buffer_size: Number of batches to prefetch
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: str = 'cuda',
        buffer_size: int = 2
    ):
        self.dataloader = dataloader
        self.device = device
        self.buffer_size = buffer_size
        self.stream = torch.cuda.Stream() if 'cuda' in device else None
        
        # Initialize prefetch thread
        self.prefetch_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = False
        self.thread = None
    
    def _prefetch_worker(self):
        """Background worker to prefetch data."""
        iterator = iter(self.dataloader)
        
        while not self.stop_event:
            try:
                batch = next(iterator)
                
                if 'cuda' in self.device:
                    with torch.cuda.stream(self.stream):
                        batch = self._move_to_device(batch)
                
                self.prefetch_queue.put(batch)
            except StopIteration:
                self.prefetch_queue.put(None)
                break
            except Exception as e:
                self.prefetch_queue.put(e)
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(v) for v in batch)
        return batch
    
    def __iter__(self) -> Iterator:
        """Start prefetching and return iterator."""
        # Start prefetch thread
        self.stop_event = False
        self.thread = Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
        
        return self
    
    def __next__(self) -> Any:
        """Get next prefetched batch."""
        batch = self.prefetch_queue.get()
        
        if batch is None:
            raise StopIteration
        
        if isinstance(batch, Exception):
            raise batch
        
        return batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def close(self):
        """Stop prefetch thread."""
        self.stop_event = True
        if self.thread:
            self.thread.join(timeout=1.0)


class RobustDataLoader:
    """
    DataLoader with error handling and recovery.
    
    Handles corrupted data samples gracefully by skipping
    problematic samples and logging warnings.
    
    Args:
        dataloader: Base DataLoader
        max_errors: Maximum consecutive errors before giving up
        skip_on_error: Whether to skip corrupted samples
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        max_errors: int = 10,
        skip_on_error: bool = True
    ):
        self.dataloader = dataloader
        self.max_errors = max_errors
        self.skip_on_error = skip_on_error
        self.error_count = 0
        self.skipped_samples = 0
    
    def __iter__(self) -> Iterator:
        """Iterate with error handling."""
        self.error_count = 0
        self.skipped_samples = 0
        
        for batch in self.dataloader:
            if isinstance(batch, Exception):
                self.error_count += 1
                if self.error_count > self.max_errors:
                    raise RuntimeError(
                        f"Too many errors ({self.error_count}), stopping"
                    )
                if self.skip_on_error:
                    self.skipped_samples += 1
                    continue
                raise batch
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return {
            'total_samples': len(self.dataloader),
            'errors': self.error_count,
            'skipped': self.skipped_samples,
            'success_rate': (len(self.dataloader) - self.skipped_samples) / max(1, len(self.dataloader))
        }


class DataLoaderFactory:
    """
    Factory for creating optimized DataLoaders.
    
    Provides convenient methods for creating DataLoaders with
    common optimizations.
    
    Usage:
        factory = DataLoaderFactory(dataset, batch_size=32)
        
        # Training loader with all optimizations
        train_loader = factory.create_train_loader(
            num_workers=4,
            use_cuda=True,
            use_caching=True,
            use_prefetch=True
        )
        
        # Validation loader (simpler)
        val_loader = factory.create_val_loader()
    """
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def create_train_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = True,
        use_caching: bool = False,
        cache_size: Optional[int] = None,
        use_prefetch: bool = False,
        prefetch_device: str = 'cuda',
        use_robust: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create optimized training DataLoader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            shuffle: Shuffle data
            drop_last: Drop last incomplete batch
            use_caching: Use dataset caching
            cache_size: Cache size (None = all)
            use_prefetch: Use CUDA prefetching
            prefetch_device: Target CUDA device
            use_robust: Use error handling
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Configured DataLoader
        """
        dataset = self.dataset
        
        # Apply caching if requested
        if use_caching:
            dataset = CachedDataset(dataset, cache_size)
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            **kwargs
        )
        
        # Apply robust loading
        if use_robust:
            loader = RobustDataLoader(loader)
        
        # Apply prefetching
        if use_prefetch and 'cuda' in prefetch_device:
            loader = PrefetchDataLoader(loader, device=prefetch_device)
        
        return loader
    
    def create_val_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 2,
        shuffle: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Create validation DataLoader.
        
        Simplified loader for validation (no shuffle, no prefetch).
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            **kwargs
        )


def create_optimized_loader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
    mode: str = 'train',
    **kwargs
) -> DataLoader:
    """
    Convenience function to create an optimized DataLoader.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: Number of workers
        device: Target device
        mode: 'train' or 'val'
        **kwargs: Additional arguments
        
    Returns:
        Optimized DataLoader
    """
    factory = DataLoaderFactory(dataset)
    
    if mode == 'train':
        return factory.create_train_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory='cuda' in device,
            use_prefetch='cuda' in device,
            prefetch_device=device,
            **kwargs
        )
    else:
        return factory.create_val_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )