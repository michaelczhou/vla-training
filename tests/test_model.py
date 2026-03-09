"""
Model Tests
===========
Test VLA model components
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.vision_encoder import SigLipEncoder, ViTEncoder, ResNetEncoder
from src.models.language_model import GemmaModel
from src.models.fusion_module import CrossAttentionFusion
from src.models.action_head import FlowMatchingHead, DiffusionHead, MLPHead
from src.models.vla_model import VLAModel


class TestVisionEncoder:
    def test_siglip_encoder(self):
        config = {
            'type': 'siglip',
            'pretrained': None,
            'vision_config': {
                'hidden_size': 768,
                'num_hidden_layers': 4,
                'num_attention_heads': 8,
                'image_size': 224,
                'patch_size': 16,
            }
        }
        encoder = SigLipEncoder(config)
        
        images = torch.randn(2, 3, 224, 224)
        features = encoder(images)
        
        assert features.dim() == 3
        assert features.shape[0] == 2
    
    def test_resnet_encoder(self):
        config = {
            'type': 'resnet',
            'variant': 'resnet18',
            'freeze': False,
        }
        encoder = ResNetEncoder(config)
        
        images = torch.randn(2, 3, 224, 224)
        features = encoder(images)
        
        assert features.shape[0] == 2
        assert features.dim() == 3


class TestFusionModule:
    def test_cross_attention_fusion(self):
        config = {
            'type': 'cross_attention',
            'vision_dim': 768,
            'language_dim': 768,
            'hidden_dim': 512,
            'num_heads': 8,
        }
        fusion = CrossAttentionFusion(config)
        
        vision_features = torch.randn(2, 197, 768)
        language_features = torch.randn(2, 32, 768)
        
        fused = fusion(vision_features, language_features)
        
        assert fused.shape == (2, 32, 512)


class TestActionHead:
    def test_flow_matching_head(self):
        config = {
            'type': 'flow_matching',
            'input_dim': 512,
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 256,
        }
        head = FlowMatchingHead(config)
        
        x = torch.randn(2, 32, 512)
        t = torch.rand(2)
        action = torch.randn(2, 10, 7)
        
        # Test forward
        velocity = head(x, t, action)
        assert velocity.shape == (2, 70)  # 10 * 7
        
        # Test sample
        actions = head.sample(x, num_steps=10)
        assert actions.shape == (2, 10, 7)
    
    def test_mlp_head(self):
        config = {
            'type': 'mlp',
            'input_dim': 512,
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 256,
        }
        head = MLPHead(config)
        
        x = torch.randn(2, 32, 512)
        actions = head(x)
        
        assert actions.shape == (2, 10, 7)


class TestVLAModel:
    def test_vla_model_forward(self):
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
                'type': 'mlp',
                'action_dim': 7,
                'chunk_size': 10,
                'hidden_dim': 256,
            },
        }
        
        model = VLAModel(config)
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        actions = torch.randn(2, 10, 7)
        
        # Test forward with actions (training)
        loss, predictions = model(images, input_ids, attention_mask, actions=actions)
        
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        
        # Test forward without actions (inference)
        loss, predictions = model(images, input_ids, attention_mask)
        
        assert predictions is not None
        assert predictions.shape == (2, 10, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
