#!/bin/bash

# KnowRL Environment Setup Script
# Handles dependencies with proper installation order

set -e  # Exit on any error

echo "🚀 Setting up KnowRL environment..."

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found. Please install conda/miniconda first."
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    echo "📦 Creating conda environment..."
    
    # Option 1: Use environment.yml (recommended)
    if [ "$1" = "conda" ]; then
        echo "Using conda environment.yml..."
        conda env create -f environment.yml
        echo "✅ Environment created. Activate with: conda activate knowrl"
        
    # Option 2: Use pip with proper ordering
    elif [ "$1" = "pip" ]; then
        echo "Using pip with staged installation..."
        
        # Check if environment exists
        if conda info --envs | grep -q knowrl; then
            echo "⚠️  Environment 'knowrl' already exists. Remove it? (y/N)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                conda env remove -n knowrl
            else
                echo "Aborting setup."
                exit 1
            fi
        fi
        
        # Create base environment
        conda create -n knowrl python=3.12 -y
        
        # Activate environment
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate knowrl
        
        echo "🔧 Installing PyTorch ecosystem..."
        pip install -r requirements-torch.txt
        
        echo "📚 Installing main dependencies..."
        pip install -r requirements-main.txt
        
        echo "✅ Installation complete!"
        
    else
        echo "❌ Invalid option. Use: bash setup.sh [conda|pip]"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate knowrl
    
    # Test key imports
    python -c "
import torch
import unsloth
import transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Unsloth: {unsloth.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print('🎉 All key packages imported successfully!')
"
}

# Main execution
main() {
    check_conda
    
    if [ $# -eq 0 ]; then
        echo "Choose installation method:"
        echo "1. conda (recommended) - uses environment.yml"
        echo "2. pip - staged pip installation"
        read -p "Enter choice (1/2): " choice
        
        case $choice in
            1) setup_environment "conda" ;;
            2) setup_environment "pip" ;;
            *) echo "❌ Invalid choice"; exit 1 ;;
        esac
    else
        setup_environment "$1"
    fi
    
    verify_installation
    
    echo ""
    echo "🎊 Setup complete! Next steps:"
    echo "1. conda activate knowrl"
    echo "2. Follow the COMPLETE_REPRODUCTION_GUIDE.md for training"
}

main "$@"