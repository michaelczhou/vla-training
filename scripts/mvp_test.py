#!/usr/bin/env python3
"""
Minimal Viable Product (MVP) Test
=================================
Quick test to verify the VLA framework works end-to-end

Usage:
    python scripts/mvp_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.models.vla_model import build_vla_model
from src.models.action_head import FlowMatchingHead
from src.data.transforms import ImageTransform, ActionTransform


def test_model_creation():
    """Test model creation"""
    print("1. Testing model creation...")
    
    config = {
        'vision': {
            'type': 'resnet',
            'variant': 'resnet18',
            'freeze': True,
        },
        'language': {
            'type': 'gemma',
            'pretrained': None,
            'freeze': True,
            'language_config': {
                'hidden_size': 256,
                'num_hidden_layers': 2,
                'num_attention_heads': 4,
                'vocab_size': 1000,
            }
        },
        'fusion': {
            'type': 'cross_attention',
            'hidden_dim': 256,
            'num_heads': 4,
        },
        'action_head': {
            'type': 'flow_matching',
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 256,
        },
    }
    
    model = build_vla_model(config)
    
    num_params = model.get_num_params()
    trainable_params = model.get_trainable_params()
    
    print(f"   ✓ Model created: {num_params:,} params, {trainable_params:,} trainable")
    
    return model


def test_forward_pass(model):
    """Test forward pass"""
    print("2. Testing forward pass...")
    
    # Create dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)
    actions = torch.randn(batch_size, 10, 7)
    
    # Training mode
    model.train()
    loss, _ = model(images, input_ids, attention_mask, actions=actions)
    
    assert loss is not None
    assert loss.dim() == 0
    print(f"   ✓ Training forward: loss = {loss.item():.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        _, predictions = model(images, input_ids, attention_mask)
    
    assert predictions is not None
    assert predictions.shape == (batch_size, 10, 7)
    print(f"   ✓ Inference forward: predictions shape = {predictions.shape}")
    
    return True


def test_action_head():
    """Test action head sampling"""
    print("3. Testing action head sampling...")
    
    config = {
        'input_dim': 256,
        'action_dim': 7,
        'chunk_size': 10,
        'hidden_dim': 128,
        'num_steps': 20,
    }
    
    head = FlowMatchingHead(config)
    head.eval()
    
    x = torch.randn(2, 32, 256)
    
    with torch.no_grad():
        actions = head.sample(x, num_steps=20)
    
    assert actions.shape == (2, 10, 7)
    print(f"   ✓ Action sampling: shape = {actions.shape}")
    
    return True


def test_transforms():
    """Test data transforms"""
    print("4. Testing transforms...")
    
    image_config = {
        'height': 224,
        'width': 224,
        'normalize': True,
    }
    image_transform = ImageTransform(image_config, training=False)
    
    action_config = {
        'dim': 7,
        'chunk_size': 10,
        'normalize': True,
    }
    action_transform = ActionTransform(action_config)
    
    # Test image transform
    image = np.random.rand(256, 256, 3).astype(np.float32)
    image_tensor = image_transform(image)
    assert image_tensor.shape == (3, 224, 224)
    
    # Test action transform
    action = np.random.rand(10, 7).astype(np.float32)
    action_tensor = action_transform(action)
    assert action_tensor.shape == (10, 7)
    
    print(f"   ✓ Image transform: {image.shape} → {image_tensor.shape}")
    print(f"   ✓ Action transform: {action.shape} → {action_tensor.shape}")
    
    return True


def main():
    print("=" * 50)
    print("VLA Framework MVP Test")
    print("=" * 50)
    print()
    
    try:
        # Run tests
        model = test_model_creation()
        test_forward_pass(model)
        test_action_head()
        test_transforms()
        
        print()
        print("=" * 50)
        print("✓ All MVP tests passed!")
        print("=" * 50)
        print()
        print("The VLA framework is working correctly.")
        print("You can now:")
        print("  1. Train a model: python scripts/train.py --config configs/model/pi0_base.yaml")
        print("  2. Run inference: python scripts/inference.py --checkpoint <path>")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 50)
        print(f"✗ Test failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
