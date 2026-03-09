"""
Logging Utilities
=================
Setup logging for training
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class Logger:
    """
    Custom logger for VLA training
    
    Features:
    - Console and file output
    - Multiple log levels
    - Structured logging
    """
    
    def __init__(
        self,
        name: str = 'vla-training',
        log_dir: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
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
        """Log metrics in structured format"""
        metrics_str = ' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' 
                                  for k, v in metrics.items())
        self.info(f'[Step {step}] {metrics_str}')


def setup_logger(
    name: str = 'vla-training',
    log_dir: Optional[str] = None,
    level: str = 'INFO'
) -> Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level string
        
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
        level=level_map.get(level.upper(), logging.INFO)
    )
    
    return logger
