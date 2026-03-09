"""
Deployment Utilities
====================
Export models for deployment (ONNX, TensorRT)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def export_onnx(
    model: nn.Module,
    output_path: str,
    config: Dict[str, Any],
    opset_version: int = 14,
    dynamic_axes: bool = True
):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX file
        config: Model configuration
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    image_height = config.get('data', {}).get('image', {}).get('height', 224)
    image_width = config.get('data', {}).get('image', {}).get('width', 224)
    text_length = 64
    
    dummy_images = torch.randn(batch_size, 3, image_height, image_width)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, text_length))
    dummy_attention_mask = torch.ones(batch_size, text_length)
    
    # Define dynamic axes
    if dynamic_axes:
        dynamic_axes_dict = {
            'images': {0: 'batch_size'},
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
        }
    else:
        dynamic_axes_dict = None
    
    # Export
    torch.onnx.export(
        model,
        (dummy_images, dummy_input_ids, dummy_attention_mask),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images', 'input_ids', 'attention_mask'],
        output_names=['actions'],
        dynamic_axes=dynamic_axes_dict,
    )
    
    print(f"Exported model to ONNX: {output_path}")


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = 'fp16',
    max_batch_size: int = 1,
    workspace_size: int = 1 << 30
):
    """
    Export ONNX model to TensorRT engine
    
    Note: Requires tensorrt Python package
    
    Args:
        onnx_path: Path to ONNX file
        output_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size
        workspace_size: Maximum workspace size in bytes
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed. Install with: pip install tensorrt")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"TensorRT Error: {parser.get_error(error)}")
            return
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Would need calibration for INT8
    
    # Build engine
    engine = builder.build_serialized_network(network, config)
    
    # Save engine
    with open(output_path, 'wb') as f:
        f.write(engine)
    
    print(f"Exported TensorRT engine: {output_path}")


class TensorRTRunner:
    """
    TensorRT inference runner
    """
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT runner
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        try:
            import tensorrt as trt
            import cupy as cp
        except ImportError:
            raise ImportError("TensorRT or CuPy not installed")
        
        self.trt = trt
        self.cp = cp
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create runtime
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Allocate GPU buffers for I/O"""
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_buffer = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            
            self.bindings.append(int(device_buffer))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_buffer, 'device': device_buffer})
            else:
                self.outputs.append({'host': host_buffer, 'device': device_buffer})
    
    def infer(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Output tensor
        """
        import pycuda.driver as cuda
        
        # Copy inputs to GPU
        for i, input_tensor in enumerate(inputs.values()):
            cuda.memcpy_htod(
                self.inputs[i]['device'],
                input_tensor.cpu().numpy()
            )
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy outputs from GPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh(output['host'], output['device'])
            outputs.append(torch.tensor(output['host']))
        
        return outputs[0] if len(outputs) == 1 else outputs
