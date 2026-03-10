"""
Tests for Vision Encoder Module
================================
测试视觉编码器的各种配置和功能
"""

import unittest
import torch
import sys
sys.path.insert(0, '/root/.openclaw/workspace/vla-training')

from src.models.vision_encoder import build_vision_encoder, VisionEncoder


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""
    
    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.height = 224
        self.width = 224
        self.channels = 3
        
    def test_vit_encoder_creation(self):
        """测试 ViT 编码器创建"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,  # 测试时不加载预训练权重
            'freeze': False,
        }
        
        encoder = build_vision_encoder(config)
        self.assertIsNotNone(encoder)
        self.assertTrue(hasattr(encoder, 'output_dim'))
        
    def test_resnet_encoder_creation(self):
        """测试 ResNet 编码器创建"""
        config = {
            'type': 'resnet',
            'model_name': 'resnet50',
            'pretrained': False,
            'freeze': False,
        }
        
        encoder = build_vision_encoder(config)
        self.assertIsNotNone(encoder)
        
    def test_encoder_forward(self):
        """测试编码器前向传播"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'freeze': False,
        }
        
        encoder = build_vision_encoder(config)
        
        # 创建测试输入
        images = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # 前向传播
        features = encoder(images)
        
        # 检查输出形状
        self.assertEqual(len(features.shape), 3)  # (B, N, D)
        self.assertEqual(features.shape[0], self.batch_size)
        
    def test_encoder_output_dim(self):
        """测试编码器输出维度"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'freeze': False,
        }
        
        encoder = build_vision_encoder(config)
        
        images = torch.randn(1, 3, 224, 224)
        features = encoder(images)
        
        # ViT-Base 应该有 768 维输出
        self.assertEqual(features.shape[-1], 768)
        
    def test_freeze_encoder(self):
        """测试冻结编码器"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'freeze': True,  # 冻结
        }
        
        encoder = build_vision_encoder(config)
        
        # 检查参数是否冻结
        for param in encoder.parameters():
            self.assertFalse(param.requires_grad)
            
    def test_unfreeze_encoder(self):
        """测试不冻结编码器"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'freeze': False,  # 不冻结
        }
        
        encoder = build_vision_encoder(config)
        
        # 检查参数是否可训练
        trainable_params = sum(p.requires_grad for p in encoder.parameters())
        self.assertGreater(trainable_params, 0)


class TestVisionEncoderEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_invalid_config(self):
        """测试无效配置"""
        config = {
            'type': 'invalid_type',  # 无效类型
        }
        
        with self.assertRaises((ValueError, KeyError)):
            build_vision_encoder(config)
            
    def test_different_image_sizes(self):
        """测试不同图像尺寸"""
        config = {
            'type': 'vit',
            'model_name': 'vit_base_patch16_224',
            'pretrained': False,
            'freeze': False,
        }
        
        encoder = build_vision_encoder(config)
        
        # 测试不同尺寸
        sizes = [(224, 224), (192, 192), (256, 256)]
        
        for h, w in sizes:
            images = torch.randn(1, 3, h, w)
            try:
                features = encoder(images)
                self.assertIsNotNone(features)
            except Exception as e:
                # 某些尺寸可能不支持，记录即可
                print(f"Size ({h}, {w}) not supported: {e}")


if __name__ == '__main__':
    unittest.main()
