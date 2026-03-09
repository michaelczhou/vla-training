# Installation Guide

## Quick Install

```bash
cd vla-training
pip install -e .
```

## Manual Install

```bash
# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow pyyaml tqdm scipy numpy

# Install in development mode
pip install -e .
```

## Verify Installation

```bash
# Run MVP test
python scripts/mvp_test.py

# Run unit tests
pytest tests/ -v
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

## GPU Support

For NVIDIA GPU support:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Optional Dependencies

### Weights & Biases (logging)
```bash
pip install wandb
```

### TensorRT (deployment)
```bash
pip install tensorrt onnx
```

### LeRobot (data loading)
```bash
pip install lerobot
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Enable gradient checkpointing
- Use gradient accumulation
- Freeze language model

### Import Errors
```bash
# Reinstall package
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

### Slow Training
- Use mixed precision (AMP) - enabled by default
- Increase batch size if GPU memory allows
- Use multiple GPUs with `torchrun`
