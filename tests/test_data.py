"""
Data Tests
==========
Test data loading and transforms
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.transforms import ImageTransform, ActionTransform
from src.data.dataset import RLDSRobotDataset
from src.data.dataloader import build_dataloader


class TestImageTransform:
    def test_image_transform_basic(self):
        config = {
            'height': 224,
            'width': 224,
            'normalize': True,
            'color_jitter': False,
            'random_crop': False,
            'flip': False,
        }
        transform = ImageTransform(config, training=False)
        
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = transform(image)
        
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32
    
    def test_image_transform_grayscale(self):
        config = {
            'height': 224,
            'width': 224,
            'normalize': True,
        }
        transform = ImageTransform(config, training=False)
        
        image = np.random.rand(256, 256).astype(np.float32)
        result = transform(image)
        
        assert result.shape == (3, 224, 224)


class TestActionTransform:
    def test_action_transform_normalize(self):
        config = {
            'dim': 7,
            'chunk_size': 10,
            'normalize': True,
            'mean': [0.0] * 7,
            'std': [1.0] * 7,
        }
        transform = ActionTransform(config)
        
        action = np.random.rand(10, 7).astype(np.float32)
        result = transform(action)
        
        assert result.shape == (10, 7)
        assert result.dtype == torch.float32
    
    def test_action_transform_minmax(self):
        config = {
            'dim': 7,
            'chunk_size': 10,
            'normalize': True,
            'min_values': [-1.0] * 7,
            'max_values': [1.0] * 7,
        }
        transform = ActionTransform(config)
        
        action = np.random.rand(10, 7).astype(np.float32)
        result = transform(action)
        
        assert result.shape == (10, 7)


class TestDataset:
    def test_rlds_dataset_dummy(self):
        config = {
            'type': 'rlds',
            'data_dir': '/tmp/dummy',
            'image': {
                'height': 224,
                'width': 224,
            },
            'action': {
                'dim': 7,
                'chunk_size': 10,
            },
        }
        
        dataset = RLDSRobotDataset('/tmp/nonexistent', config, max_samples=10)
        
        assert len(dataset) > 0
        
        item = dataset[0]
        
        assert 'image' in item
        assert 'action' in item
        assert 'language' in item
        
        assert item['image'].shape == (3, 224, 224)
        assert item['action'].shape == (10, 7)


class TestDataloader:
    def test_build_dataloader(self):
        config = {
            'type': 'rlds',
            'data_dir': '/tmp/dummy',
            'batch_size': 4,
            'image': {
                'height': 224,
                'width': 224,
            },
            'action': {
                'dim': 7,
                'chunk_size': 10,
            },
        }
        
        dataloader = build_dataloader(config, training=True)
        
        assert dataloader is not None
        
        # Get a batch
        batch = next(iter(dataloader))
        
        assert 'images' in batch
        assert 'actions' in batch
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        
        assert batch['images'].shape[0] == 4
        assert batch['actions'].shape[0] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
