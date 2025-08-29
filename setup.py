#!/usr/bin/env python3
"""
KnowRL: Knowledge-enhanced Reinforcement Learning setup
Handles complex PyTorch dependencies with proper installation order
"""

import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom install command that handles PyTorch dependencies first"""
    
    def run(self):
        # Install PyTorch with CUDA first
        print("🔧 Installing PyTorch ecosystem with CUDA support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch==2.5.1+cu121",
            "torchvision==0.20.1+cu121", 
            "torchaudio==2.5.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        
        # Install xformers separately to avoid conflicts
        print("🔧 Installing xformers...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "xformers==0.0.28.post3",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        
        # Now run normal installation
        print("📚 Installing remaining dependencies...")
        install.run(self)


# Core dependencies (excluding PyTorch ecosystem)
install_requires = [
    # Core ML/AI frameworks  
    "unsloth==2025.3.18",
    "accelerate==1.3.0",
    "bitsandbytes==0.45.1",
    "transformers==4.50.0",
    "tokenizers==0.21.0",
    "datasets==3.1.0",
    "safetensors==0.5.2",
    "huggingface-hub==0.26.0",
    
    # Training frameworks
    "peft==0.14.0", 
    "trl==0.14.0",
    "deepspeed==0.15.4",
    "flash_attn==2.7.4.post1",
    
    # NLP utilities
    "sentence-transformers==2.3.0",
    "sentencepiece==0.2.0",
    "tiktoken==0.7.0",
    "einops==0.8.0",
    
    # Experiment tracking
    "swanlab==0.4.6",
    "wandb==0.19.5",
    "tensorboard==2.18.0",
    
    # Inference engines
    "vllm==0.7.0",
    "outlines==0.1.11",
    
    # API integrations
    "openai==1.61.0",
    
    # Core utilities
    "numpy==1.26.4",
    "pandas==2.2.3", 
    "requests==2.32.3",
    "pyyaml==6.0.2",
    "tqdm==4.67.1",
]

extras_require = {
    "dev": [
        "jupyter==1.1.1",
        "jupyterlab==4.3.5",
        "notebook==7.3.2",
        "ipython==8.12.3",
        "ipykernel==6.29.5",
    ],
    "full": [
        "ray==2.41.0",
        "fastapi==0.115.8",
        "uvicorn==0.34.0",
    ]
}

setup(
    name="knowrl",
    version="1.0.0",
    description="Knowledge-enhanced Reinforcement Learning for reducing hallucinations in LLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="KnowRL Team",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    cmdclass={
        "install": CustomInstallCommand,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)