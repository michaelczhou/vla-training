#!/usr/bin/env python3
"""
Inference Script for VLA Models
================================
Run inference with trained VLA (Vision-Language-Action) model.

Features:
- Single image inference
- Batch inference
- Interactive mode
- ONNX export for deployment
- Benchmark mode for performance evaluation

Usage:
    # Single image inference
    python scripts/inference.py \\
        --checkpoint checkpoints/my_vla/best.pt \\
        --image path/to/image.jpg \\
        --prompt "pick up the red block"

    # Batch inference
    python scripts/inference.py \\
        --checkpoint checkpoints/my_vla/best.pt \\
        --batch-dir /path/to/images \\
        --prompt "pick up the object"

    # Export to ONNX
    python scripts/inference.py \\
        --checkpoint checkpoints/my_vla/best.pt \\
        --export-onnx output/model.onnx

    # Interactive mode
    python scripts/inference.py \\
        --checkpoint checkpoints/my_vla/best.pt \\
        --interactive

    # Benchmark
    python scripts/inference.py \\
        --checkpoint checkpoints/my_vla/best.pt \\
        --benchmark --num-runs 100
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.inference.policy import VLAPolicy


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run VLA model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    
    # Inference mode
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to single input image'
    )
    parser.add_argument(
        '--prompt', type=str, default='pick up the object',
        help='Text instruction for the task'
    )
    parser.add_argument(
        '--batch-dir', type=str, default=None,
        help='Directory containing batch of images'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Run in interactive mode'
    )
    
    # Generation settings
    parser.add_argument(
        '--num-steps', type=int, default=50,
        help='Number of sampling steps for diffusion/flow matching'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--num-samples', type=int, default=1,
        help='Number of action samples to generate'
    )
    
    # Export settings
    parser.add_argument(
        '--export-onnx', type=str, default=None,
        help='Export model to ONNX format'
    )
    parser.add_argument(
        '--onnx-opset', type=int, default=14,
        help='ONNX opset version'
    )
    
    # Benchmark settings
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run benchmark mode'
    )
    parser.add_argument(
        '--num-runs', type=int, default=100,
        help='Number of benchmark runs'
    )
    parser.add_argument(
        '--warmup-runs', type=int, default=10,
        help='Number of warmup runs'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save inference results'
    )
    parser.add_argument(
        '--save-visualization', action='store_true',
        help='Save action visualization'
    )
    
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def load_images_from_dir(directory: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images from a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of (filename, image) tuples
    """
    directory = Path(directory)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    images = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in supported_formats:
            try:
                image = load_image(str(file_path))
                images.append((file_path.name, image))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
    
    return images


def run_single_inference(
    policy: VLAPolicy,
    image: np.ndarray,
    prompt: str,
    num_steps: int,
    temperature: float = 1.0,
    num_samples: int = 1
) -> np.ndarray:
    """
    Run inference on a single image.
    
    Args:
        policy: VLA policy instance
        image: Input image as numpy array
        prompt: Text instruction
        num_steps: Number of sampling steps
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        Predicted actions as numpy array
    """
    # Run prediction
    if num_samples == 1:
        actions = policy.predict(image, prompt, num_steps=num_steps)
    else:
        # Generate multiple samples
        actions = []
        for _ in range(num_samples):
            action = policy.predict(image, prompt, num_steps=num_steps)
            actions.append(action)
        actions = np.stack(actions, axis=0)
    
    return actions


def run_batch_inference(
    policy: VLAPolicy,
    images: List[Tuple[str, np.ndarray]],
    prompt: str,
    num_steps: int,
    output_dir: Optional[str] = None
) -> List[Tuple[str, np.ndarray]]:
    """
    Run inference on a batch of images.
    
    Args:
        policy: VLA policy instance
        images: List of (filename, image) tuples
        prompt: Text instruction
        num_steps: Number of sampling steps
        output_dir: Optional directory to save results
        
    Returns:
        List of (filename, actions) tuples
    """
    results = []
    
    print(f"\nRunning batch inference on {len(images)} images...")
    for filename, image in tqdm(images, desc="Processing"):
        actions = policy.predict(image, prompt, num_steps=num_steps)
        results.append((filename, actions))
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{Path(filename).stem}_actions.npy"
            np.save(output_path, actions)
    
    return results


def interactive_mode(policy: VLAPolicy, num_steps: int):
    """
    Run interactive inference loop.
    
    Args:
        policy: VLA policy instance
        num_steps: Number of sampling steps
    """
    print("\n" + "=" * 60)
    print("  Interactive VLA Inference Mode")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Enter image path (or press Enter for random image)")
    print("  - Enter text prompt (or use default)")
    print("  - Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get image path
            image_path = input("Image path [Enter for random]: ").strip()
            if image_path.lower() in ['quit', 'q', 'exit']:
                print("Exiting...")
                break
            
            # Load image or use random
            if image_path:
                try:
                    image = load_image(image_path)
                    print(f"  Loaded: {image_path}")
                except Exception as e:
                    print(f"  Error loading image: {e}")
                    continue
            else:
                image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                print("  Using random image")
            
            # Get prompt
            prompt = input("Prompt [pick up the object]: ").strip()
            if prompt.lower() in ['quit', 'q', 'exit']:
                print("Exiting...")
                break
            if not prompt:
                prompt = "pick up the object"
            
            # Run inference
            print("\nRunning inference...")
            start_time = time.time()
            actions = policy.predict(image, prompt, num_steps=num_steps)
            elapsed = time.time() - start_time
            
            # Display results
            print(f"\n  Inference time: {elapsed*1000:.2f} ms")
            print(f"  Action shape: {actions.shape}")
            print(f"\n  Predicted actions:")
            
            for i, action in enumerate(actions[:5]):  # Show first 5 steps
                action_str = ', '.join([f'{a:.3f}' for a in action])
                print(f"    Step {i}: [{action_str}]")
            
            if len(actions) > 5:
                print(f"    ... ({len(actions) - 5} more steps)")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


def benchmark_inference(
    policy: VLAPolicy,
    num_runs: int = 100,
    warmup_runs: int = 10,
    image_size: Tuple[int, int] = (224, 224)
) -> dict:
    """
    Run benchmark to measure inference performance.
    
    Args:
        policy: VLA policy instance
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        image_size: Size of dummy image
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("  Benchmark Mode")
    print("=" * 60)
    
    # Create dummy input
    dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    dummy_prompt = "pick up the object"
    
    # Warmup
    print(f"\nWarmup: {warmup_runs} runs...")
    for _ in range(warmup_runs):
        _ = policy.predict(dummy_image, dummy_prompt)
    
    # Benchmark
    print(f"\nRunning benchmark: {num_runs} runs...")
    times = []
    
    for _ in tqdm(range(num_runs)):
        start_time = time.time()
        _ = policy.predict(dummy_image, dummy_prompt)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    # Compute statistics
    times = np.array(times) * 1000  # Convert to ms
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times))
    }
    
    # Print results
    print("\n" + "-" * 40)
    print("  Benchmark Results")
    print("-" * 40)
    print(f"  Mean latency:   {results['mean_ms']:.2f} ms")
    print(f"  Std deviation:  {results['std_ms']:.2f} ms")
    print(f"  Min latency:    {results['min_ms']:.2f} ms")
    print(f"  Max latency:    {results['max_ms']:.2f} ms")
    print(f"  P50 (median):   {results['p50_ms']:.2f} ms")
    print(f"  P95:            {results['p95_ms']:.2f} ms")
    print(f"  P99:            {results['p99_ms']:.2f} ms")
    print(f"  Throughput:     {results['fps']:.2f} FPS")
    print("-" * 40)
    
    return results


def export_to_onnx(
    policy: VLAPolicy,
    output_path: str,
    opset_version: int = 14,
    image_size: Tuple[int, int] = (224, 224)
):
    """
    Export VLA model to ONNX format.
    
    Args:
        policy: VLA policy instance
        output_path: Output path for ONNX file
        opset_version: ONNX opset version
        image_size: Input image size
    """
    import onnx
    
    print(f"\nExporting to ONNX: {output_path}")
    
    # Create dummy input
    dummy_image = torch.randn(1, 3, *image_size)
    dummy_prompt = "pick up the object"
    
    # Get model
    model = policy.model
    model.eval()
    
    # Export
    torch.onnx.export(
        model,
        (dummy_image, dummy_prompt),
        output_path,
        input_names=['image', 'prompt'],
        output_names=['actions'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'actions': {0: 'batch_size', 1: 'chunk_size'}
        },
        opset_version=opset_version
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"  Export successful!")
    print(f"  Model saved to: {output_path}")
    
    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {file_size:.2f} MB")


def save_visualization(
    actions: np.ndarray,
    output_path: str,
    title: str = "Action Sequence"
):
    """
    Save action visualization as image.
    
    Args:
        actions: Action sequence
        output_path: Output path for image
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Heatmap
    im = axes[0].imshow(actions.T, cmap='RdBu_r', aspect='auto')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Action Dimension')
    axes[0].set_title(f'{title} - Heatmap')
    plt.colorbar(im, ax=axes[0])
    
    # Line plot
    for i in range(min(7, actions.shape[1])):
        axes[1].plot(actions[:, i], label=f'Dim {i}', marker='o', markersize=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Action Value')
    axes[1].set_title(f'{title} - Time Series')
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved: {output_path}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load policy
    print(f"Loading model from {args.checkpoint}...")
    try:
        policy = VLAPolicy.from_checkpoint(args.checkpoint, device=args.device)
        print(f"  Model loaded successfully!")
        print(f"  Device: {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Export to ONNX
    if args.export_onnx:
        export_to_onnx(policy, args.export_onnx, args.onnx_opset)
        return
    
    # Benchmark mode
    if args.benchmark:
        results = benchmark_inference(
            policy,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        # Save results
        if args.output_dir:
            import json
            output_path = Path(args.output_dir) / "benchmark_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(policy, args.num_steps)
        return
    
    # Batch inference mode
    if args.batch_dir:
        images = load_images_from_dir(args.batch_dir)
        if not images:
            print(f"Error: No images found in {args.batch_dir}")
            sys.exit(1)
        
        results = run_batch_inference(
            policy, images, args.prompt, args.num_steps, args.output_dir
        )
        
        print(f"\nProcessed {len(results)} images")
        
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
        return
    
    # Single image inference
    if args.image:
        image = load_image(args.image)
        print(f"\nInput: {args.image}")
        print(f"Prompt: {args.prompt}")
    else:
        # Use dummy image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"\nUsing random image")
        print(f"Prompt: {args.prompt}")
    
    # Run inference
    print(f"Running inference ({args.num_steps} steps)...")
    start_time = time.time()
    actions = policy.predict(image, args.prompt, num_steps=args.num_steps)
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\nInference time: {elapsed*1000:.2f} ms")
    print(f"Action shape: {actions.shape}")
    print(f"\nPredicted actions:")
    
    for i, action in enumerate(actions):
        action_str = ', '.join([f'{a:.3f}' for a in action])
        print(f"  Step {i}: [{action_str}]")
    
    # Save visualization
    if args.save_visualization and args.output_dir:
        output_path = Path(args.output_dir) / "action_visualization.png"
        save_visualization(actions, str(output_path))


if __name__ == '__main__':
    main()