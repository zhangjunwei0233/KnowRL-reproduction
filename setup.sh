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

# Function to check and setup CUDA environment
check_cuda_env() {
    echo "🔍 Checking CUDA environment..."
    
    # Check NVIDIA driver
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  nvidia-smi not found. CUDA training may not work."
        echo "💡 Continuing setup - CUDA might still be available through runtime."
        return 0  # Continue setup, don't exit
    fi
    
    # Get GPU info
    local gpu_info=$(nvidia-smi --query-gpu=name,count --format=csv,noheader,nounits | head -1)
    echo "✅ GPU detected: $gpu_info"
    
    # Check if CUDA_HOME is set and valid
    if [[ -n "$CUDA_HOME" ]] && [[ -f "$CUDA_HOME/bin/nvcc" ]]; then
        echo "✅ CUDA_HOME already set: $CUDA_HOME"
        return 0
    fi
    
    # Try to find system CUDA installation
    local cuda_paths=("/usr/local/cuda" "/usr/local/cuda-12" "/usr/local/cuda-11" "/opt/cuda")
    for path in "${cuda_paths[@]}"; do
        if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
            export CUDA_HOME="$path"
            export PATH="$CUDA_HOME/bin:$PATH" 
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            echo "✅ Found system CUDA at: $CUDA_HOME"
            return 0
        fi
    done
    
    echo "⚠️  System CUDA toolkit not found (common in containerized environments)."
    echo "💡 Will install cudatoolkit-dev via conda during environment setup."
    return 0  # This is expected in containerized environments, not an error
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
            echo "✅ Environment 'knowrl' already exists. Continue installation? (Y/n)"
            read -r response
            if [[ "$response" =~ ^([nN][oO]|[nN])$ ]]; then
                echo "Setup canceled."
                exit 0
            fi
            echo "🔄 Continuing installation in existing environment..."
            echo "💡 Tip: You can run this multiple times to resume installation"
            echo "💡 Tip: You can switch networks/proxies between runs for faster downloads"
            SKIP_ENV_CREATION=true
        else
            SKIP_ENV_CREATION=false
        fi
        
        # Create base environment with CUDA toolkit for containerized environments
        if [[ "$SKIP_ENV_CREATION" == "false" ]]; then
            echo "🐍 Creating base Python 3.12 environment..."
            if [[ -z "$CUDA_HOME" ]]; then
                echo "📦 Installing cudatoolkit-dev for DeepSpeed compilation in containerized environment..."
                conda create -n knowrl python=3.12 cudatoolkit-dev -c conda-forge -y --verbose
            else
                echo "📦 Using existing system CUDA..."
                conda create -n knowrl python=3.12 -y --verbose
            fi
        else
            echo "🔄 Using existing environment (skipping creation)..."
        fi
        
        # Activate environment and setup CUDA
        echo "🔌 Activating environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate knowrl
        
        # Setup CUDA environment for conda-installed toolkit
        if [[ -z "$CUDA_HOME" ]] && [[ -f "$CONDA_PREFIX/bin/nvcc" ]]; then
            echo "🔧 Setting up conda CUDA environment..."
            export CUDA_HOME=$CONDA_PREFIX
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
            echo "✅ CUDA_HOME set to: $CUDA_HOME"
            
            # Create activation script so CUDA_HOME is set automatically on conda activate
            echo "🔧 Creating conda activation script..."
            mkdir -p $CONDA_PREFIX/etc/conda/activate.d
            mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
            
            # Activation script
            cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'EOF'
#!/bin/bash
# Auto-setup CUDA environment when activating knowrl conda environment
if [[ -f "$CONDA_PREFIX/bin/nvcc" ]]; then
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi
EOF
            
            # Deactivation script  
            cat > $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh << 'EOF'
#!/bin/bash
# Clean up CUDA environment when deactivating
unset CUDA_HOME
# Note: PATH and LD_LIBRARY_PATH will be restored by conda automatically
EOF
            
            chmod +x $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
            chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh
            echo "✅ Conda activation script created - CUDA_HOME will be set automatically"
        fi
        
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

# Function to verify environment only (no installation)
verify_only() {
    echo "🔍 KnowRL Environment Verification"
    echo ""
    
    # Check conda
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found. Please install conda/miniconda first."
        return 1
    fi
    echo "✅ Conda found: $(conda --version)"
    
    # Check if knowrl environment exists
    if ! conda info --envs | grep -q "knowrl"; then
        echo "❌ Environment 'knowrl' not found."
        echo "💡 Run: bash setup.sh pip  (to create it)"
        return 1
    fi
    echo "✅ Environment 'knowrl' exists"
    
    # Activate and test
    echo ""
    echo "🧪 Testing package imports..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate knowrl
    
    # Test with error handling
    python -c "
import sys
import traceback

def test_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        module = __import__(module_name)
        if hasattr(module, '__version__'):
            print(f'✅ {display_name}: {module.__version__}')
        else:
            print(f'✅ {display_name}: imported successfully')
        return True
    except ImportError as e:
        print(f'❌ {display_name}: {e}')
        return False

print('🐍 Python version:', sys.version.split()[0])
print()

success = True
success &= test_import('torch', 'PyTorch')
success &= test_import('unsloth', 'Unsloth')  
success &= test_import('transformers', 'Transformers')
success &= test_import('datasets', 'Datasets')
success &= test_import('peft', 'PEFT')
success &= test_import('trl', 'TRL')

if success:
    # Additional CUDA check
    try:
        import torch
        print()
        print(f'🔥 CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'🔥 CUDA devices: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                print(f'🔥 Device {i}: {torch.cuda.get_device_name(i)}')
    except:
        pass
    
    print()
    print('🎉 All core packages working!')
    print('🚀 Environment ready for KnowRL training!')
else:
    print()
    print('❌ Some packages missing. Try reinstalling:')
    print('   bash setup.sh pip')
    sys.exit(1)
" || return 1
}

# Function to show help
show_help() {
    echo "KnowRL Environment Setup Script"
    echo ""
    echo "Usage:"
    echo "  bash setup.sh [METHOD]"
    echo ""
    echo "Methods:"
    echo "  conda   - Use conda environment.yml (most robust, slower)"
    echo "  pip     - Use staged pip installation (faster, ~3-5 min)"
    echo "  verify  - Only verify existing installation (no setup)"
    echo "  help    - Show this help message"
    echo ""
    echo "If no method specified, interactive mode will prompt you."
    echo ""
    echo "Examples:"
    echo "  bash setup.sh pip      # Fast installation"
    echo "  bash setup.sh conda    # Robust installation"
    echo "  bash setup.sh verify   # Just check if environment works"
}

# Main execution
main() {
    # Record start time
    start_time=$(date +%s)
    
    check_conda
    check_cuda_env  # Check CUDA environment for DeepSpeed compilation
    
    if [ $# -eq 0 ]; then
        echo ""
        echo "⚡ Choose option:"
        echo "   1. pip (recommended) - Fast staged installation (~3-5 minutes)"
        echo "   2. conda - Robust but slower (~10-15 minutes)"  
        echo "   3. verify - Check existing environment (no installation)"
        echo "   4. help - Show detailed help"
        echo ""
        read -p "Enter choice (1/2/3/4): " choice
        
        case $choice in
            1) setup_environment "pip" ;;
            2) setup_environment "conda" ;;
            3) verify_only; exit $? ;;
            4) show_help; exit 0 ;;
            *) echo "❌ Invalid choice. Use: bash setup.sh help"; exit 1 ;;
        esac
    else
        case "$1" in
            "help"|"-h"|"--help") show_help; exit 0 ;;
            "verify") verify_only; exit $? ;;
            "conda"|"pip") setup_environment "$1" ;;
            *) echo "❌ Unknown option: $1"; show_help; exit 1 ;;
        esac
    fi
    
    # Only run full verification after installation (not for verify-only mode)
    if [ "$1" != "verify" ]; then
        verify_installation
        
        # Calculate total time and show completion message
        end_time=$(date +%s)
        total_time=$((end_time - start_time))
        
        echo ""
        echo "🎊 Setup complete in ${total_time} seconds!"
        echo ""
        echo "📋 Next steps:"
        echo "  1. conda activate knowrl"
        echo "  2. Follow COMPLETE_REPRODUCTION_GUIDE.md for training"
        echo "  3. Test with: python -c 'import torch; print(torch.cuda.is_available())'"
    fi
}

main "$@"
