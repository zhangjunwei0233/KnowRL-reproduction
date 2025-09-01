#!/bin/bash

# KnowRL A100 Server Directory Setup Script
# Creates the required directory structure for the new server environment

set -e  # Exit on any error

echo "🏗️  Setting up KnowRL directory structure on A100 server..."

# Base directories
WORK_DIR="$HOME/a100x4"
DATA_DIR="$WORK_DIR/KnowRL-reproduction-data"

echo "📁 Creating directory structure..."

# Create main data directory structure
mkdir -p "$DATA_DIR"/{knowledge_base,training,huggingface}

# Create training subdirectories  
mkdir -p "$DATA_DIR/training"/{models,outputs,logs,checkpoints}
mkdir -p "$DATA_DIR/training/models"/{sft_output,rl_output}
mkdir -p "$DATA_DIR/training/outputs"
mkdir -p "$DATA_DIR/training/models/sft_output/deepseek-r1-distill-qwen-7b-sft"
mkdir -p "$DATA_DIR/training/outputs/deepseek-r1-distill-qwen-7b-rl"

# Create huggingface cache directories
mkdir -p "$DATA_DIR/huggingface"/{datasets,models,cache}

echo "✅ Directory structure created!"
echo ""
echo "📋 Directory layout:"
echo "├── $WORK_DIR/"
echo "│   ├── DeepSeek-R1-Distill-Qwen-7B/          (model files - already exists)"
echo "│   └── KnowRL-reproduction-data/"
echo "│       ├── knowledge_base/                    (FActScore database)"
echo "│       ├── training/"
echo "│       │   ├── models/"
echo "│       │   │   ├── sft_output/"
echo "│       │   │   │   └── deepseek-r1-distill-qwen-7b-sft/"
echo "│       │   │   └── rl_output/"
echo "│       │   ├── outputs/"
echo "│       │   │   └── deepseek-r1-distill-qwen-7b-rl/"
echo "│       │   ├── logs/"
echo "│       │   └── checkpoints/"
echo "│       └── huggingface/"
echo "│           ├── datasets/"
echo "│           ├── models/"
echo "│           └── cache/"
echo ""

# Check if model directory exists
if [ -d "$WORK_DIR/DeepSeek-R1-Distill-Qwen-7B" ]; then
    echo "✅ Model directory found: $WORK_DIR/DeepSeek-R1-Distill-Qwen-7B"
    
    # List model files
    echo "📁 Model files:"
    ls -la "$WORK_DIR/DeepSeek-R1-Distill-Qwen-7B/" | head -10
else
    echo "⚠️  Model directory not found: $WORK_DIR/DeepSeek-R1-Distill-Qwen-7B"
    echo "    Please ensure the model files are available at this location."
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Set up knowledge base (if needed):"
echo "   cd ~/KnowRL-reproduction/train/reward_function/FActScore/build_knowledge/"
echo "   # Option 1: Download pre-built (recommended)"
echo "   gdown https://drive.google.com/uc?id=1EVFkzuFvqE8AOEcdfSSm03vvvbVDa7bI"
echo "   mv knowledge_base.db ~/a100x4/KnowRL-reproduction-data/knowledge_base/"
echo ""
echo "   # Option 2: Build from scratch (if you have knowledge data)"
echo "   # Edit DATA_PATH in build_db.sh to point to your knowledge data"
echo "   # bash build_db.sh"
echo ""
echo "2. Configure API keys in train/train.sh (already done)"
echo ""
echo "3. Start training:"
echo "   # Stage 1: Cold-start SFT"
echo "   cd ~/KnowRL-reproduction/train/"
echo "   conda activate knowrl"
echo "   CUDA_VISIBLE_DEVICES=0,1,2,3 llama-factory-cli train script/llama_factory_sft.yaml"
echo ""
echo "   # Stage 2: Knowledge RL"
echo "   bash train.sh"
echo ""
echo "🎉 Setup complete! Ready for KnowRL training on A100 cluster."