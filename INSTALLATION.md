# KnowRL Installation Guide

Multiple elegant installation options to avoid dependency conflicts.

## 🚀 Quick Start (Recommended)

### Option 1: Automated Script
```bash
bash setup.sh conda    # Uses environment.yml (most robust)
bash setup.sh pip      # Uses staged pip installation
```

### Option 2: Conda Environment (Single Command)
```bash
conda env create -f environment.yml
conda activate knowrl
```

### Option 3: Python Package Installation
```bash
pip install -e .    # Installs with proper dependency ordering
```

## 📋 Manual Installation (Advanced)

### Staged Requirements Approach
```bash
# 1. Create environment
conda create -n knowrl python=3.12
conda activate knowrl

# 2. Install PyTorch first (with CUDA)
pip install -r requirements-torch.txt

# 3. Install remaining dependencies
pip install -r requirements-main.txt
```

## ✅ Verification

After installation, verify everything works:
```bash
conda activate knowrl
python -c "
import torch, unsloth, transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('✅ Installation successful!')
"
```

## 🔧 Troubleshooting

**Dependency Conflicts**: Use the automated script or environment.yml - they handle installation order properly.

**CUDA Issues**: Ensure you have CUDA 12.1 compatible drivers, or modify PyTorch versions in the requirements files.

**Memory Issues**: Some packages (flash_attn, xformers) require significant RAM during compilation.