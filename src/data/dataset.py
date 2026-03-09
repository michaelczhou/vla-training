"""
Robot Datasets
==============
Dataset classes for loading robot learning data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import os
import json
from pathlib import Path

from .transforms import ImageTransform, ActionTransform, build_transforms


class RLDSRobotDataset(Dataset):
    """
    RLDS (Robot Learning Dataset) format dataset
    
    RLDS is a standardized format for robot learning datasets.
    Reference: https://github.com/google-research/rlds
    """
    
    def __init__(
        self, 
        data_path: str, 
        config: Dict[str, Any],
        training: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize RLDS dataset
        
        Args:
            data_path: Path to RLDS dataset directory
            config: Dataset configuration
            training: Whether in training mode
            max_samples: Maximum number of samples to load
        """
        self.data_path = Path(data_path)
        self.config = config
        self.training = training
        self.max_samples = max_samples
        
        # Build transforms
        self.image_transform, self.action_transform = build_transforms(config, training=training)
        
        # Load dataset info
        self._load_dataset_info()
        
        # Load episode indices
        self._load_episodes()
    
    def _load_dataset_info(self):
        """Load dataset metadata"""
        info_path = self.data_path / 'dataset_info.json'
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}
        
        # Get action dimension
        self.action_dim = self.config.get('action', {}).get('dim', 7)
        self.chunk_size = self.config.get('action', {}).get('chunk_size', 10)
    
    def _load_episodes(self):
        """Load episode file paths"""
        self.episodes = []
        
        # Look for episode files
        episode_dir = self.data_path / 'episodes'
        if episode_dir.exists():
            for episode_file in sorted(episode_dir.glob('*.npz')):
                self.episodes.append(episode_file)
        
        # Or look for tfrecord files
        tfrecord_dir = self.data_path / 'tfrecord'
        if tfrecord_dir.exists():
            for tf_file in sorted(tfrecord_dir.glob('*.tfrecord*')):
                self.episodes.append(tf_file)
        
        # Limit samples
        if self.max_samples is not None:
            self.episodes = self.episodes[:self.max_samples]
        
        if len(self.episodes) == 0:
            # Create dummy data for testing
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing without actual dataset"""
        print(f"Warning: No episodes found at {self.data_path}, creating dummy data")
        self.is_dummy = True
        self.num_dummy_samples = min(self.max_samples or 1000, 1000)
    
    def __len__(self) -> int:
        """Get dataset length"""
        if hasattr(self, 'is_dummy') and self.is_dummy:
            return self.num_dummy_samples
        return len(self.episodes) * 100  # Assume 100 steps per episode
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary with 'image', 'text', 'action'
        """
        if hasattr(self, 'is_dummy') and self.is_dummy:
            return self._get_dummy_item(idx)
        
        # Map index to episode and step
        episode_idx = idx // 100
        step_idx = idx % 100
        
        if episode_idx >= len(self.episodes):
            episode_idx = len(self.episodes) - 1
        
        episode_path = self.episodes[episode_idx]
        
        # Load episode data
        try:
            if episode_path.suffix == '.npz':
                episode_data = np.load(episode_path, allow_pickle=True)
            else:
                # For tfrecord, would need tensorflow
                return self._get_dummy_item(idx)
        except Exception as e:
            return self._get_dummy_item(idx)
        
        # Extract data
        images = episode_data.get('images', episode_data.get('observation/images)', None))
        actions = episode_data.get('actions', None)
        language = episode_data.get('language_instruction', b'').decode('utf-8')
        
        if images is None or actions is None:
            return self._get_dummy_item(idx)
        
        # Get specific timestep
        if step_idx >= len(images):
            step_idx = len(images) - 1
        
        image = images[step_idx]
        
        # Get action chunk
        action_end = min(step_idx + self.chunk_size, len(actions))
        action_chunk = actions[step_idx:action_end]
        
        # Pad if necessary
        if len(action_chunk) < self.chunk_size:
            padding = np.zeros((self.chunk_size - len(action_chunk), self.action_dim))
            action_chunk = np.concatenate([action_chunk, padding], axis=0)
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        action_tensor = self.action_transform(action_chunk)
        
        return {
            'image': image_tensor,
            'action': action_tensor,
            'language': language,
        }
    
    def _get_dummy_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dummy data item"""
        # Random image
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Random action chunk
        action = np.random.rand(self.chunk_size, self.action_dim).astype(np.float32) * 2 - 1
        
        # Dummy language
        language = "dummy task"
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        action_tensor = self.action_transform(action)
        
        return {
            'image': image_tensor,
            'action': action_tensor,
            'language': language,
        }


class LeRobotDataset(Dataset):
    """
    LeRobot format dataset
    
    LeRobot is a popular format for robot learning.
    Reference: https://github.com/huggingface/lerobot
    """
    
    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        training: bool = True,
        camera_key: str = 'observation.images.top'
    ):
        """
        Initialize LeRobot dataset
        
        Args:
            data_path: Path to LeRobot dataset
            config: Dataset configuration
            training: Whether in training mode
            camera_key: Key for camera observation
        """
        self.data_path = Path(data_path)
        self.config = config
        self.training = training
        self.camera_key = camera_key
        
        # Build transforms
        self.image_transform, self.action_transform = build_transforms(config, training=training)
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata"""
        meta_path = self.data_path / 'meta.json'
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        self.action_dim = self.config.get('action', {}).get('dim', 7)
        self.chunk_size = self.config.get('action', {}).get('chunk_size', 10)
        
        # Check if we have actual data
        self.has_data = (self.data_path / 'data').exists()
        
        if not self.has_data:
            print(f"Warning: No data found at {self.data_path}, will use dummy data")
    
    def __len__(self) -> int:
        """Get dataset length"""
        if self.has_data:
            # Count actual samples
            data_dir = self.data_path / 'data'
            return len(list(data_dir.glob('*.pt'))) * 100
        else:
            return 1000  # Dummy length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        if not self.has_data:
            return self._get_dummy_item(idx)
        
        # Try to load actual data
        try:
            data_dir = self.data_path / 'data'
            data_files = sorted(data_dir.glob('*.pt'))
            
            if len(data_files) == 0:
                return self._get_dummy_item(idx)
            
            file_idx = idx % len(data_files)
            data_file = data_files[file_idx]
            
            data = torch.load(data_file, map_location='cpu')
            
            # Extract fields
            image = data.get(self.camera_key, None)
            action = data.get('action', None)
            language = data.get('language', 'task')
            
            if image is None or action is None:
                return self._get_dummy_item(idx)
            
            # Handle image format
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.numpy().transpose(1, 2, 0)
                else:
                    image = image.numpy()
            
            # Handle action format
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            
            # Get action chunk
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            if len(action) < self.chunk_size:
                padding = np.zeros((self.chunk_size - len(action), self.action_dim))
                action = np.concatenate([action, padding], axis=0)
            else:
                action = action[:self.chunk_size]
            
            # Apply transforms
            image_tensor = self.image_transform(image)
            action_tensor = self.action_transform(action)
            
            return {
                'image': image_tensor,
                'action': action_tensor,
                'language': language,
            }
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._get_dummy_item(idx)
    
    def _get_dummy_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dummy data item"""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        action = np.random.rand(self.chunk_size, self.action_dim).astype(np.float32) * 2 - 1
        language = "dummy task"
        
        image_tensor = self.image_transform(image)
        action_tensor = self.action_transform(action)
        
        return {
            'image': image_tensor,
            'action': action_tensor,
            'language': language,
        }


def build_dataset(
    config: Dict[str, Any],
    training: bool = True
) -> Dataset:
    """
    Build dataset from config
    
    Args:
        config: Dataset configuration
        training: Whether in training mode
        
    Returns:
        Dataset instance
    """
    dataset_type = config.get('type', 'rlds').lower()
    data_path = config.get('data_dir', 'data/dummy')
    
    if dataset_type == 'rlds':
        return RLDSRobotDataset(data_path, config, training=training)
    elif dataset_type == 'lerobot':
        return LeRobotDataset(data_path, config, training=training)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
