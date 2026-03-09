#!/usr/bin/env python3
"""
Inference Demo Example
======================
Run VLA inference with a trained model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from src.inference.policy import VLAPolicy


def main():
    print("=== VLA Inference Demo ===\n")
    
    # Load policy from checkpoint
    checkpoint_path = 'checkpoints/my_first_vla/best.pt'
    
    try:
        policy = VLAPolicy.from_checkpoint(checkpoint_path, device='cuda')
        print(f"Loaded model from {checkpoint_path}\n")
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or update the checkpoint path\n")
        return
    
    # Create dummy image
    print("Creating test image...")
    image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Test prompts
    prompts = [
        "pick up the red block",
        "place the object in the box",
        "open the drawer",
    ]
    
    print("\nRunning inference...\n")
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        actions = policy.predict(image, prompt, num_steps=50)
        
        print(f"Actions shape: {actions.shape}")
        print(f"First action: [{', '.join(f'{a:.3f}' for a in actions[0])}]")
        print()
    
    print("Demo completed!")


if __name__ == '__main__':
    main()
