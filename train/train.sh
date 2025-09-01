# ============================================================================
# API Configuration - Replace with your actual credentials
# ============================================================================
export OPENAI_API_KEY_FACTSCORE="sk-47cc346af1af4ede93091bec6ea83038"
export OPENAI_BASE_URL_FACTSCORE="https://api.deepseek.com"

export OPENAI_API_KEY_JUDGE="sk-47cc346af1af4ede93091bec6ea83038"
export OPENAI_API_BASE_JUDGE="https://api.deepseek.com"

export SWANLAB_API_KEY="dgIXV7EsCZL3OMGq5lKvZ"
# ============================================================================
# Configuration
# ============================================================================
export FACTSCORE_DB_PATH="/home/jovyan/a100x4/KnowRL-reproduction-data/knowledge_base/knowledge_base.db"
export USE_API_MANAGER_FOR_LLM_EVAL=True
export USE_API_MANAGER_FOR_FACTSCORE=True

# Set GPU device and CPU threading
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=12

# HuggingFace cache configuration - centralized cache for all projects
export HF_HOME=/home/jovyan/a100x4/huggingface
export HF_DATASETS_CACHE=/home/jovyan/a100x4/huggingface/datasets

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration file
CONFIG_FILE="./script/grpo.yaml"

# ============================================================================
# Run Training
# ============================================================================
echo "Starting GRPO training..."
echo "Config: $CONFIG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set accelerate config and launch distributed training
export ACCELERATE_CONFIG_FILE="../accelerate_config.yaml"
accelerate launch --config_file ../accelerate_config.yaml main.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi
