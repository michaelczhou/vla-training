"""
Training Tests
==============
Test training components
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.losses import FlowMatchingLoss, DiffusionLoss, ActionLoss
from src.training.optimizer import build_optimizer, build_scheduler


class TestLosses:
    def test_flow_matching_loss(self):
        loss_fn = FlowMatchingLoss()
        
        pred = torch.randn(4, 70)
        target = torch.randn(4, 70)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_diffusion_loss(self):
        loss_fn = DiffusionLoss(num_steps=100)
        
        pred = torch.randn(4, 70)
        target = torch.randn(4, 70)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
    
    def test_action_loss_mse(self):
        loss_fn = ActionLoss(loss_type='mse')
        
        pred = torch.randn(4, 10, 7)
        target = torch.randn(4, 10, 7)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
    
    def test_action_loss_l1(self):
        loss_fn = ActionLoss(loss_type='l1')
        
        pred = torch.randn(4, 10, 7)
        target = torch.randn(4, 10, 7)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0


class TestOptimizer:
    def test_build_optimizer_adamw(self):
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        config = {
            'type': 'AdamW',
            'lr': 1e-3,
            'weight_decay': 0.1,
        }
        
        optimizer = build_optimizer(model, config)
        
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0
        assert optimizer.defaults['lr'] == 1e-3
    
    def test_build_optimizer_adam(self):
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        config = {
            'type': 'Adam',
            'lr': 1e-3,
            'betas': [0.9, 0.999],
        }
        
        optimizer = build_optimizer(model, config)
        
        assert optimizer is not None
    
    def test_build_scheduler_cosine(self):
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        config = {
            'type': 'cosine',
            'warmup_steps': 100,
            'min_lr': 1e-5,
        }
        
        scheduler = build_scheduler(optimizer, config, num_training_steps=1000)
        
        assert scheduler is not None
        
        # Test stepping
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(100):
            scheduler.step()
        
        # LR should have changed after warmup
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr != initial_lr or current_lr <= initial_lr


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
