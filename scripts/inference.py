#!/usr/bin/env python3
"""
Inference Script
================
Run inference with trained VLA model

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/my_vla/best.pt \
        --image path/to/image.jpg \
        --prompt "pick up the red block"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

from src.inference.policy import VLAPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='Run VLA inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--prompt', type=str, default='pick up the object',
                        help='Text instruction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    return parser.parse_args()


def run_inference(policy, image_path, prompt, num_steps):
    """Run single inference"""
    # Load image
    if image_path:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    else:
        # Use dummy image
        image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Run inference
    actions = policy.predict(image, prompt, num_steps=num_steps)
    
    return actions


def interactive_mode(policy, num_steps):
    """Run interactive inference loop"""
    print("\n=== Interactive VLA Inference ===")
    print("Enter image path and prompt to get action predictions")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get image path
            image_path = input("Image path (or press Enter for dummy): ").strip()
            if image_path.lower() == 'quit':
                break
            
            if not image_path:
                image_path = None
            
            # Get prompt
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                prompt = "pick up the object"
            
            # Run inference
            print("\nRunning inference...")
            actions = run_inference(policy, image_path, prompt, num_steps)
            
            # Display results
            print(f"\nPredicted actions (shape: {actions.shape}):")
            for i, action in enumerate(actions):
                print(f"  Step {i}: [{', '.join(f'{a:.3f}' for a in action)}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load policy
    print(f"Loading model from {args.checkpoint}...")
    policy = VLAPolicy.from_checkpoint(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    
    if args.interactive:
        # Interactive mode
        interactive_mode(policy, args.num_steps)
    else:
        # Single inference
        actions = run_inference(policy, args.image, args.prompt, args.num_steps)
        
        print(f"\nPredicted actions (shape: {actions.shape}):")
        for i, action in enumerate(actions):
            print(f"  Step {i}: [{', '.join(f'{a:.3f}' for a in action)}]")


if __name__ == '__main__':
    main()
