#!/bin/bash

# KnowRL Environment Setup Script
# Handles dependencies with proper installation order

set -e  # Exit on any error

echo "🚀 Setting up KnowRL environment..."

# Function to show progress dots
show_progress() {
    local pid=$1
    local delay=2
    local spinstr='|/-\'
    echo -n "Working... "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
    echo "Done!"
}

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found. Please install conda/miniconda first."
        exit 1
    fi
    echo "✅ Conda found: $(conda --version)"
}

# Function to setup environment
setup_environment() {
    echo "📦 Creating conda environment..."
    
    # Option 1: Use environment.yml (recommended)
    if [ "$1" = "conda" ]; then
        echo "🔄 Using conda environment.yml..."
        echo "📋 This will install 150+ packages and may take 5-15 minutes..."
        echo "💡 Tip: You can press Ctrl+C and run 'bash setup.sh pip' for faster installation"
        echo ""
        
        # Ask if user wants to continue
        read -p "Continue with conda installation? (Y/n): " -t 10 response || response="Y"
        if [[ "$response" =~ ^([nN][oO]|[nN])$ ]]; then
            echo "Switching to pip installation..."
            setup_environment "pip"
            return
        fi
        
        echo "⏱️  Starting conda environment creation at $(date)..."
        echo "🔍 Running: conda env create -f environment.yml --verbose"
        echo ""
        
        # Run with verbose output and show progress
        conda env create -f environment.yml --verbose 2>&1 | while IFS= read -r line; do
            echo "[$(date +'%H:%M:%S')] $line"
        done
        
        echo ""
        echo "✅ Environment created at $(date). Activate with: conda activate knowrl"
        
    # Option 2: Use pip with proper ordering
    elif [ "$1" = "pip" ]; then
        echo "🔄 Using pip with staged installation (faster, ~3-5 minutes)..."
        
        # Check if environment exists
        if conda info --envs | grep -q knowrl; then
            echo "⚠️  Environment 'knowrl' already exists. Remove it? (y/N)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                echo "🗑️  Removing existing environment..."
                conda env remove -n knowrl --verbose
            else
                echo "Aborting setup."
                exit 1
            fi
        fi
        
        # Create base environment
        echo "🐍 Creating base Python 3.12 environment..."
        conda create -n knowrl python=3.12 -y --verbose
        
        # Activate environment
        echo "🔌 Activating environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate knowrl
        
        echo "🔧 Installing PyTorch ecosystem with CUDA support..."
        echo "📦 Installing: $(wc -l < requirements-torch.txt) PyTorch packages"
        pip install -r requirements-torch.txt --verbose --progress-bar on
        
        echo ""
        echo "📚 Installing main dependencies..."
        echo "📦 Installing: $(grep -v '^#' requirements-main.txt | grep -v '^$' | wc -l) packages"
        pip install -r requirements-main.txt --verbose --progress-bar on
        
        echo ""
        echo "✅ Installation complete at $(date)!"
        
    else
        echo "❌ Invalid option. Use: bash setup.sh [conda|pip]"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    echo ""
    echo "🔍 Verifying installation..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate knowrl
    
    echo "🧪 Testing package imports..."
    
    # Test key imports with progress
    python -c "
import sys
print('🐍 Python version:', sys.version.split()[0])

print('🔄 Testing PyTorch...')
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA devices: {torch.cuda.device_count()}')
    print(f'✅ Current device: {torch.cuda.get_device_name(0)}')

print('🔄 Testing Unsloth...')
import unsloth
print(f'✅ Unsloth: {unsloth.__version__}')

print('🔄 Testing Transformers...')
import transformers
print(f'✅ Transformers: {transformers.__version__}')

print('🔄 Testing training dependencies...')
import datasets, peft, trl
print(f'✅ Datasets: {datasets.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print(f'✅ TRL: {trl.__version__}')

print()
print('🎉 All key packages imported successfully!')
print('🚀 Environment ready for KnowRL training!')
"
}

# Function to show help
show_help() {
    echo "KnowRL Environment Setup Script"
    echo ""
    echo "Usage:"
    echo "  bash setup.sh [METHOD]"
    echo ""
    echo "Methods:"
    echo "  conda  - Use conda environment.yml (most robust, slower)"
    echo "  pip    - Use staged pip installation (faster, ~3-5 min)"
    echo "  help   - Show this help message"
    echo ""
    echo "If no method specified, interactive mode will prompt you."
    echo ""
    echo "Examples:"
    echo "  bash setup.sh pip      # Fast installation"
    echo "  bash setup.sh conda    # Robust installation"
}

# Main execution
main() {
    # Record start time
    start_time=$(date +%s)
    
    check_conda
    
    if [ $# -eq 0 ]; then
        echo ""
        echo "⚡ Choose installation method:"
        echo "   1. pip (recommended) - Fast staged installation (~3-5 minutes)"
        echo "   2. conda - Robust but slower (~10-15 minutes)"  
        echo "   3. help - Show detailed help"
        echo ""
        read -p "Enter choice (1/2/3): " choice
        
        case $choice in
            1) setup_environment "pip" ;;
            2) setup_environment "conda" ;;
            3) show_help; exit 0 ;;
            *) echo "❌ Invalid choice. Use: bash setup.sh help"; exit 1 ;;
        esac
    else
        case "$1" in
            "help"|"-h"|"--help") show_help; exit 0 ;;
            "conda"|"pip") setup_environment "$1" ;;
            *) echo "❌ Unknown option: $1"; show_help; exit 1 ;;
        esac
    fi
    
    verify_installation
    
    # Calculate total time
    end_time=$(date +%s)
    total_time=$((end_time - start_time))
    
    echo ""
    echo "🎊 Setup complete in ${total_time} seconds!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. conda activate knowrl"
    echo "  2. Follow COMPLETE_REPRODUCTION_GUIDE.md for training"
    echo "  3. Test with: python -c 'import torch; print(torch.cuda.is_available())'"
}

main "$@"