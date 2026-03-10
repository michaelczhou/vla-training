"""
Tests for Action Head Module
============================
测试动作头的各种类型和功能
"""

import unittest
import torch
import sys
sys.path.insert(0, '/root/.openclaw/workspace/vla-training')

from src.models.action_head import (
    build_action_head, ActionHead, FlowMatchingBlock
)


class TestFlowMatchingBlock(unittest.TestCase):
    """测试流匹配块"""
    
    def setUp(self):
        self.hidden_dim = 512
        self.batch_size = 4
        
    def test_block_creation(self):
        """测试块创建"""
        block = FlowMatchingBlock(self.hidden_dim, num_heads=8)
        self.assertIsNotNone(block)
        
    def test_block_forward(self):
        """测试块前向传播"""
        block = FlowMatchingBlock(self.hidden_dim, num_heads=8)
        
        # 输入
        x = torch.randn(self.batch_size, self.hidden_dim)
        
        # 前向传播
        output = block(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
    def test_block_gradient(self):
        """测试梯度传播"""
        block = FlowMatchingBlock(self.hidden_dim, num_heads=8)
        
        x = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度
        self.assertIsNotNone(x.grad)
        self.assertTrue((x.grad != 0).any())


class TestActionHeadBase(unittest.TestCase):
    """测试动作头基类"""
    
    def test_base_class_not_implemented(self):
        """测试基类方法未实现"""
        config = {
            'action_dim': 7,
            'chunk_size': 10,
        }
        
        head = ActionHead(config)
        
        x = torch.randn(2, 10, 512)
        
        # 基类方法应该抛出 NotImplementedError
        with self.assertRaises(NotImplementedError):
            head(x)
            
        with self.assertRaises(NotImplementedError):
            head.sample(x)


class TestFlowMatchingHead(unittest.TestCase):
    """测试流匹配动作头"""
    
    def setUp(self):
        self.config = {
            'type': 'flow_matching',
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 512,
            'num_steps': 50,
            'input_dim': 768,
        }
        self.batch_size = 2
        
    def test_head_creation(self):
        """测试动作头创建"""
        head = build_action_head(self.config, input_dim=768)
        self.assertIsNotNone(head)
        
    def test_head_forward(self):
        """测试前向传播"""
        head = build_action_head(self.config, input_dim=768)
        
        # 输入特征
        x = torch.randn(self.batch_size, 10, 768)
        
        # 前向传播
        output = head(x)
        
        # 检查输出形状: (B, chunk_size, action_dim)
        self.assertEqual(output.shape, (self.batch_size, 10, 7))
        
    def test_head_sampling(self):
        """测试采样"""
        head = build_action_head(self.config, input_dim=768)
        
        x = torch.randn(self.batch_size, 10, 768)
        
        # 采样
        actions = head.sample(x, num_steps=10)  # 减少步数以加快测试
        
        # 检查输出形状
        self.assertEqual(actions.shape, (self.batch_size, 10, 7))
        
    def test_head_loss(self):
        """测试损失计算"""
        head = build_action_head(self.config, input_dim=768)
        
        x = torch.randn(self.batch_size, 10, 768)
        action_0 = torch.randn(self.batch_size, 10, 7)
        action_1 = torch.randn(self.batch_size, 10, 7)
        
        # 计算损失
        loss, _ = head.compute_loss(x, action_0, action_1)
        
        # 检查损失
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() >= 0)  # 损失应该非负


class TestDiffusionHead(unittest.TestCase):
    """测试扩散动作头"""
    
    def setUp(self):
        self.config = {
            'type': 'diffusion',
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 512,
            'num_steps': 100,
            'input_dim': 768,
        }
        self.batch_size = 2
        
    def test_head_creation(self):
        """测试扩散头创建"""
        head = build_action_head(self.config, input_dim=768)
        self.assertIsNotNone(head)
        
    def test_diffusion_forward(self):
        """测试扩散前向"""
        head = build_action_head(self.config, input_dim=768)
        
        x = torch.randn(self.batch_size, 10, 768)
        output = head(x)
        
        self.assertEqual(output.shape, (self.batch_size, 10, 7))


class TestMLPHead(unittest.TestCase):
    """测试 MLP 动作头"""
    
    def setUp(self):
        self.config = {
            'type': 'mlp',
            'action_dim': 7,
            'chunk_size': 10,
            'hidden_dim': 512,
            'num_layers': 3,
            'input_dim': 768,
        }
        self.batch_size = 2
        
    def test_mlp_creation(self):
        """测试 MLP 头创建"""
        head = build_action_head(self.config, input_dim=768)
        self.assertIsNotNone(head)
        
    def test_mlp_forward(self):
        """测试 MLP 前向"""
        head = build_action_head(self.config, input_dim=768)
        
        x = torch.randn(self.batch_size, 10, 768)
        output = head(x)
        
        self.assertEqual(output.shape, (self.batch_size, 10, 7))


class TestActionHeadEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_invalid_type(self):
        """测试无效类型"""
        config = {
            'type': 'invalid',
            'action_dim': 7,
            'chunk_size': 10,
        }
        
        with self.assertRaises((ValueError, KeyError)):
            build_action_head(config, input_dim=768)
            
    def test_different_action_dims(self):
        """测试不同动作维度"""
        for action_dim in [6, 7, 8, 14]:
            config = {
                'type': 'flow_matching',
                'action_dim': action_dim,
                'chunk_size': 10,
                'hidden_dim': 256,
                'num_steps': 10,
                'input_dim': 512,
            }
            
            head = build_action_head(config, input_dim=512)
            x = torch.randn(1, 10, 512)
            output = head(x)
            
            self.assertEqual(output.shape[-1], action_dim)
            
    def test_different_chunk_sizes(self):
        """测试不同动作块大小"""
        for chunk_size in [5, 10, 20, 50]:
            config = {
                'type': 'flow_matching',
                'action_dim': 7,
                'chunk_size': chunk_size,
                'hidden_dim': 256,
                'num_steps': 10,
                'input_dim': 512,
            }
            
            head = build_action_head(config, input_dim=512)
            x = torch.randn(1, chunk_size, 512)
            output = head(x)
            
            self.assertEqual(output.shape[1], chunk_size)


if __name__ == '__main__':
    unittest.main()
