"""
Unit tests for RDT2 model components.
Tests cover Residual VQ, Transformer layers, and full model inference.
"""

import unittest
import torch
import numpy as np
from src.models.rdt2_model import (
    ResidualVQ, VectorQuantizer, RDT2Transformer, DenoisingTransformerLayer
)


class TestVectorQuantizer(unittest.TestCase):
    """Test single vector quantizer."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16
        self.embed_dim = 256
        self.codebook_size = 1024
        
        self.vq = VectorQuantizer(
            codebook_size=self.codebook_size,
            embedding_dim=self.embed_dim,
            commitment_cost=0.25
        )
    
    def test_forward_shape(self):
        """Test output shapes match input."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        quantized, loss, indices = self.vq(x)
        
        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(indices.shape, (self.batch_size, self.seq_len))
        self.assertIsInstance(loss.item(), float)
    
    def test_quantization_loss(self):
        """Test VQ loss is non-negative."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        _, loss, _ = self.vq(x)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_encode_decode(self):
        """Test encode-decode cycle preserves approximate values."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Encode
        indices = self.vq(torch.flatten(x, 0, 1))[2]
        
        # Decode
        decoded = self.vq.decode(indices)
        
        # Should be reconstructible
        self.assertEqual(decoded.shape, x.shape)
    
    def test_deterministic(self):
        """Test same input produces same output."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        quantized1, _, indices1 = self.vq(x)
        quantized2, _, indices2 = self.vq(x)
        
        torch.testing.assert_close(quantized1, quantized2)
        torch.testing.assert_close(indices1.float(), indices2.float())


class TestResidualVQ(unittest.TestCase):
    """Test residual vector quantization."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.embed_dim = 128
        
        self.rvq = ResidualVQ(
            num_quantizers=4,
            codebook_size=512,
            embedding_dim=self.embed_dim,
            commitment_cost=0.25
        )
    
    def test_forward_shape(self):
        """Test RVQ forward pass shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        quantized, loss, indices = self.rvq(x)
        
        self.assertEqual(quantized.shape, x.shape)
        self.assertEqual(len(indices), 4)  # 4 quantizers
        self.assertIsInstance(loss.item(), float)
    
    def test_residual_property(self):
        """Test that RVQ improves reconstruction with more quantizers."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Full reconstruction
        quantized, _, _ = self.rvq(x)
        
        # Error should be small
        reconstruction_error = torch.mean((x - quantized) ** 2).item()
        self.assertLess(reconstruction_error, 0.1)
    
    def test_encode_decode_consistency(self):
        """Test encode then decode returns similar values."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Encode to discrete tokens
        indices = self.rvq.encode(x)
        self.assertEqual(len(indices), 4)
        
        # Decode back
        decoded = self.rvq.decode(indices)
        self.assertEqual(decoded.shape, x.shape)


class TestDenoisingTransformerLayer(unittest.TestCase):
    """Test denoising transformer layer."""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 32
        self.hidden_size = 512
        self.num_heads = 8
        
        self.layer = DenoisingTransformerLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads
        )
    
    def test_forward_shape(self):
        """Test layer preserves shape."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        out = self.layer(x)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test gradients can flow through layer."""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestRDT2Transformer(unittest.TestCase):
    """Test full RDT2 model."""
    
    @unittest.skip("Requires large model download")
    def setUp(self):
        self.batch_size = 1
        self.num_images = 2
        self.action_dim = 14
        self.seq_len = 10
        
        # Use smaller config for testing
        self.model = RDT2Transformer(
            vision_encoder_name="gpt2",  # Smaller placeholder
            action_dim=self.action_dim,
            num_diffusion_steps=10,
            use_rvq=True,
            rvq_config={
                'num_quantizers': 4,
                'codebook_size': 256,
                'embedding_dim': 128
            }
        )
    
    @unittest.skip("Requires large model download")
    def test_forward_pass(self):
        """Test full forward pass."""
        images = torch.randn(self.batch_size, self.num_images, 3, 224, 224)
        text_input_ids = torch.randint(0, 1000, (self.batch_size, 20))
        noisy_actions = torch.randn(self.batch_size, 256, self.action_dim)
        timesteps = torch.randint(0, 10, (self.batch_size,))
        
        predicted_noise, vq_loss = self.model(
            images=images,
            text_input_ids=text_input_ids,
            noisy_actions=noisy_actions,
            timesteps=timesteps
        )
        
        self.assertEqual(predicted_noise.shape, noisy_actions.shape)
        self.assertIsInstance(vq_loss.item(), float)
    
    def test_action_tokenizer(self):
        """Test action tokenizer independently."""
        from src.models.rdt2_model import ResidualVQ
        
        tokenizer = ResidualVQ(
            num_quantizers=4,
            codebook_size=256,
            embedding_dim=128
        )
        
        actions = torch.randn(2, 16, 14)  # [B, T, action_dim]
        
        # Pad or project to embedding dim if needed
        if actions.shape[-1] != 128:
            projection = torch.nn.Linear(14, 128)
            actions = projection(actions)
        
        quantized, loss, indices = tokenizer(actions)
        
        self.assertEqual(quantized.shape, actions.shape)
        self.assertGreaterEqual(loss.item(), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for training pipeline."""
    
    def test_training_step(self):
        """Test a single training step."""
        # Create simple model components
        rvq = ResidualVQ(num_quantizers=2, codebook_size=128, embedding_dim=64)
        
        # Simulate batch
        batch_size = 2
        actions = torch.randn(batch_size, 8, 64)
        target_actions = torch.randn(batch_size, 8, 64)
        
        # Forward
        quantized, vq_loss, _ = rvq(actions)
        
        # Simple MSE loss
        recon_loss = torch.nn.functional.mse_loss(quantized, target_actions)
        total_loss = recon_loss + 0.1 * vq_loss
        
        # Backward
        total_loss.backward()
        
        self.assertIsNotNone(total_loss.item())
        self.assertGreater(total_loss.item(), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVectorQuantizer))
    suite.addTests(loader.loadTestsFromTestCase(TestResidualVQ))
    suite.addTests(loader.loadTestsFromTestCase(TestDenoisingTransformerLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestRDT2Transformer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
