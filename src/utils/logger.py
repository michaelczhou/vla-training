"""
Enhanced Logging Utilities
===========================
Setup logging for training with TensorBoard support
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class TensorBoardLogger:
    """
    TensorBoard logger for VLA training.
    
    Provides a simple interface for logging scalars, images, histograms,
    and other data to TensorBoard.
    
    Usage:
        tb_logger = TensorBoardLogger(log_dir='logs/tensorboard')
        tb_logger.log_scalar('train/loss', 0.5, step=100)
        tb_logger.log_image('input/image', image_tensor, step=100)
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Import TensorBoard only when needed
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images, step: int):
        """Log multiple images."""
        if self.enabled:
            self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def log_graph(self, model, input_to_model):
        """Log model graph."""
        if self.enabled:
            self.writer.add_graph(model, input_to_model)
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        if self.enabled:
            # Convert config to string for logging
            config_str = '\n'.join([f'{k}: {v}' for k, v in config.items()])
            self.writer.add_text('hyperparameters', config_str)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class Logger:
    """
    Custom logger for VLA training.
    
    Features:
    - Console and file output
    - Multiple log levels
    - Structured logging
    - TensorBoard integration
    """
    
    def __init__(
        self,
        name: str = 'vla-training',
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        use_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            use_tensorboard: Whether to enable TensorBoard
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.name = name
        self.level = level
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        self.log_file = None
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'{name}_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        
        # TensorBoard logger
        self.tb_logger = None
        if use_tensorboard and tensorboard_dir:
            try:
                self.tb_logger = TensorBoardLogger(tensorboard_dir)
                self.info(f"TensorBoard logging to: {tensorboard_dir}")
            except Exception as e:
                self.warning(f"Failed to initialize TensorBoard: {e}")
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics in structured format."""
        metrics_str = ' | '.join(
            f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' 
            for k, v in metrics.items()
        )
        self.info(f'[Step {step}] {metrics_str}')
        
        # Also log to TensorBoard
        if self.tb_logger and self.tb_logger.enabled:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_logger.log_scalar(key, value, step)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar to TensorBoard."""
        if self.tb_logger and self.tb_logger.enabled:
            self.tb_logger.log_scalar(tag, value, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image to TensorBoard."""
        if self.tb_logger and self.tb_logger.enabled:
            self.tb_logger.log_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard."""
        if self.tb_logger and self.tb_logger.enabled:
            self.tb_logger.log_histogram(tag, values, step)
    
    def close(self):
        """Close all handlers."""
        if self.tb_logger:
            self.tb_logger.close()


def setup_logger(
    name: str = 'vla-training',
    log_dir: Optional[str] = None,
    level: str = 'INFO',
    use_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None
) -> Logger:
    """
    Setup logger with optional TensorBoard support.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        use_tensorboard: Whether to enable TensorBoard
        tensorboard_dir: Directory for TensorBoard logs
        
    Returns:
        Logger instance
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    logger = Logger(
        name=name,
        log_dir=log_dir,
        level=level_map.get(level.upper(), logging.INFO),
        use_tensorboard=use_tensorboard,
        tensorboard_dir=tensorboard_dir
    )
    
    return logger