#!/usr/bin/env python3
"""
VLA Training Framework - Setup
"""

from setuptools import setup, find_packages

setup(
    name='vla-training',
    version='0.1.0',
    description='Vision-Language-Action Model Training Framework',
    author='VLA Team',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'transformers>=4.35.0',
        'pillow>=10.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.66.0',
        'scipy>=1.11.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'logging': [
            'wandb>=0.16.0',
            'tensorboard>=2.14.0',
        ],
        'deploy': [
            'tensorrt>=8.6.0',
            'onnx>=1.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'vla-train=scripts.train:main',
            'vla-infer=scripts.inference:main',
        ],
    },
)
