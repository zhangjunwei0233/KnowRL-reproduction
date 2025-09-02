# KnowRL Complete Reproduction Guide

This guide provides step-by-step instructions for reproducing KnowRL research on A100 clusters, based on tested configurations.

## Overview: Paper to Code Mapping

KnowRL addresses LLM hallucinations through knowledge-integrated reinforcement learning with a two-stage approach:

1. **Cold-Start SFT** → Aligns models with factual reasoning patterns  
2. **Knowledgeable RL** → Uses external knowledge in reward functions during RL training

### Core Architecture Components

#### Two-Stage Training Pipeline

**Stage 1: Cold-Start SFT**
- **Implementation**: LLaMA-Factory framework
- **Dataset**: `data/coldstart/knowrl_coldstart.json`
- **Configuration**: `train/script/llama_factory_sft.yaml`
- **Purpose**: Establishes foundational factual reasoning patterns

**Stage 2: Knowledge RL**  
- **Entry point**: `train/main.py`
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Configuration**: `train/script/grpo.yaml`
- **Execution**: `train/train.sh`

#### Multi-Component Reward System

1. **Format Reward** (`train/reward_function/format_reward.py`): Enforces `<think>` and `<answer>` tags
2. **Factuality Reward** (`train/reward_function/fact_reward.py`): External knowledge verification using FActScore
3. **Correctness Reward** (`train/reward_function/correct_reward.py`): LLM-based correctness evaluation
4. **Combined Reward** (`train/reward_function/combined_reward.py`): Intelligent combination of signals

## Hardware Requirements

### Tested Configuration (A100 Cluster)
- **GPUs**: 4× NVIDIA A100-PCIE-40GB (Kubernetes cluster)
- **CUDA Runtime**: 12.2+ (containerized environment)
- **System RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space

### Memory Usage per GPU
- **Training Model**: ~14GB (base) + ~8GB (training overhead) = ~22GB
- **vLLM Inference**: ~14GB (base) + ~4GB (KV cache) = ~18GB
- **Total**: Up to ~40GB (full A100-40GB utilization)

## Step 1: Environment Setup

### 1.1 Create Conda Environment
```bash
conda create -n knowrl python=3.12
conda activate knowrl
```

### 1.2 Clone Repository
```bash
git clone https://github.com/zhangjunwei0233/KnowRL-reproduction.git
cd KnowRL-reproduction
```

### 1.3 Install Dependencies (Containerized Environment)
```bash
# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install main requirements
pip install -r requirements.txt

# For containerized A100 clusters - install CUDA toolkit and compilers
conda install -c conda-forge gcc_linux-64 gxx_linux-64 cudatoolkit-dev

# Set up automatic conda activation with CUDA environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_setup.sh << 'EOF'
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/cuda_setup.sh

# Install LLaMA-Factory for Stage 1
pip install llamafactory
```

### 1.4 Verify Installation
```bash
# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Test compiler setup (required for Triton/vLLM)
echo $CC
$CC --version

# Test LLaMA-Factory
llamafactory-cli --help
```

## Step 2: Directory Structure Setup

### 2.1 Create Directory Structure
```bash
# Create organized storage structure
mkdir -p ~/a100x4/{huggingface/{datasets,models,cache},KnowRL-reproduction-data/{knowledge_base,training/{models/sft_output,outputs}}}

# Verify structure
ls -la ~/a100x4/
```

**Final structure:**
```
~/a100x4/
├── DeepSeek-R1-Distill-Qwen-7B/           # Model files (download separately)
├── huggingface/                            # HF cache (shared)
│   ├── datasets/, models/, cache/
└── KnowRL-reproduction-data/
    ├── knowledge_base/knowledge_base.db    # FActScore database
    └── training/{models/sft_output/,outputs/}
```

### 2.2 Configure Environment Variables
```bash
# Add to your shell profile (~/.bashrc)
export HF_HOME=/home/jovyan/a100x4/huggingface
export HF_DATASETS_CACHE=/home/jovyan/a100x4/huggingface/datasets
export HF_ENDPOINT=https://hf-mirror.com  # For faster downloads in Asia
```

## Step 3: Download Model and Knowledge Base

### 3.1 Download Base Model
```bash
# Option 1: Download to local directory
cd ~/a100x4/
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Option 2: Let HuggingFace auto-download (will use HF_HOME cache)
# Model will be downloaded automatically during training
```

### 3.2 Download Knowledge Base
```bash
cd ~/a100x4/KnowRL-reproduction-data/knowledge_base/

# Download from HuggingFace dataset
hf download zjunlp/KnowRL-Knowledge-Base --repo-type dataset --local-dir ./

# Verify
ls -lh knowledge_base.db
# Expected: ~100MB-1GB file
```

## Step 4: Configure API Keys and Paths

### 4.1 API Keys Setup
You need **three** API keys:
1. **OpenAI API Key (FActScore)**: For factuality evaluation
2. **OpenAI API Key (Judge)**: For correctness evaluation
3. **SwanLab API Key**: For experiment tracking

Get OpenAI keys from: https://platform.openai.com/api-keys
Get SwanLab key from: https://swanlab.cn/

### 4.2 Update Configuration Files

**Edit `train/train.sh`:**
```bash
# Replace with your actual API keys
export OPENAI_API_KEY_FACTSCORE="sk-your-factscore-key-here"
export OPENAI_API_KEY_JUDGE="sk-your-judge-key-here"
export SWANLAB_API_KEY="your-swanlab-key-here"

# Update paths to match your directory structure
export FACTSCORE_DB_PATH="/home/jovyan/a100x4/KnowRL-reproduction-data/knowledge_base/knowledge_base.db"
```

**Edit `train/script/grpo.yaml`:**
```yaml
# Update model paths
model_name_or_path: "/home/jovyan/a100x4/DeepSeek-R1-Distill-Qwen-7B"
adapter_path: "/home/jovyan/a100x4/KnowRL-reproduction-data/training/models/sft_output/deepseek-r1-distill-qwen-7b-sft"
output_dir: "/home/jovyan/a100x4/KnowRL-reproduction-data/training/outputs/deepseek-r1-distill-qwen-7b-rl"

# Update dataset path
dataset_id_or_path: /home/jovyan/KnowRL-reproduction/data/rl/knowrl_RLdata.json

# Configure SwanLab
swanlab_project: "knowrl_reproduction_a100"
swanlab_experiment_name: "test"
```

**Edit `train/script/llama_factory_sft.yaml`:**
```yaml
# Update model and output paths
model_name_or_path: /home/jovyan/a100x4/DeepSeek-R1-Distill-Qwen-7B
dataset_dir: /home/jovyan/KnowRL-reproduction/data/coldstart
output_dir: /home/jovyan/a100x4/KnowRL-reproduction-data/training/models/sft_output/deepseek-r1-distill-qwen-7b-sft

# Add experiment tracking
report_to: swanlab
swanlab_project: "knowrl_reproduction_a100"
run_name: "deepseek-r1-7b-coldstart-sft"
```

## Step 5: Training Execution

### 5.1 Stage 1: Cold-Start SFT
```bash
cd ~/KnowRL-reproduction/train/
conda activate knowrl

# Run SFT training (4-6 hours)
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train script/llama_factory_sft.yaml

# Verify SFT completion
ls -la ~/a100x4/KnowRL-reproduction-data/training/models/sft_output/deepseek-r1-distill-qwen-7b-sft/
# Should contain: adapter_config.json, adapter_model.safetensors, etc.
```

### 5.2 Stage 2: Knowledge RL Training
```bash
# Run GRPO training (6-12 hours for full training, 2-3 minutes for minimal test)
bash train.sh

# Monitor training
tail -f ~/a100x4/KnowRL-reproduction-data/training/outputs/*/runs/*/events.out.tfevents.*
```

## Step 6: Testing with Minimal Configuration

For initial testing and validation, use the minimal test configuration:

### 6.1 Switch to Minimal Test Branch
```bash
git checkout minimal-test
```

### 6.2 Minimal Test Parameters
The minimal-test branch contains ultra-conservative settings:
```yaml
max_steps: 5                          # Just 5 steps for testing
per_device_train_batch_size: 1        # Minimal memory usage
gradient_accumulation_steps: 1        # No accumulation
max_completion_length: 128            # Short completions
num_generations: 1                    # Single generation per step
vllm_gpu_memory_utilization: 0.2     # Conservative vLLM memory
```

### 6.3 Run Minimal Test
```bash
cd ~/KnowRL-reproduction/train/
bash train.sh
```

**Expected results:**
- Runtime: ~2-3 minutes
- Memory usage: ~20-25GB per GPU
- Should complete without OOM errors
- Tests complete GRPO pipeline

## Step 7: Scaling Up and Optimization

### 7.1 Memory Optimization Strategies
If you encounter memory issues, adjust these parameters:

```yaml
# Reduce batch sizes
per_device_train_batch_size: 6        # From 12
gradient_accumulation_steps: 4        # Increase to maintain effective batch size

# Reduce vLLM memory allocation  
vllm_gpu_memory_utilization: 0.25     # From 0.5

# Reduce generation parameters
num_generations: 2                     # From 4
max_completion_length: 256             # From 512
```

### 7.2 GPU Utilization
For optimal GPU utilization:
- Use all 4 A100 GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Monitor with: `watch -n 1 nvidia-smi`
- Check memory usage and temperature

## Step 8: Monitoring and Results

### 8.1 SwanLab Dashboard
Visit your SwanLab dashboard to monitor:
- Training loss curves
- Reward function outputs (format, factuality, correctness)
- GPU utilization metrics
- Training progress

### 8.2 Local Log Analysis
```bash
# Check training logs
cd ~/a100x4/KnowRL-reproduction-data/training/outputs/

# Analyze reward outputs
ls -la reward_outputs/  # JSON files with detailed reward breakdowns
```

## Troubleshooting

### Common Issues and Solutions

**CUDA Compiler Not Found:**
```bash
# Install and configure compilers (already covered in setup)
conda install -c conda-forge gcc_linux-64 gxx_linux-64 cudatoolkit-dev
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
```

**Out of Memory (OOM):**
- Reduce `per_device_train_batch_size` to 1
- Set `vllm_gpu_memory_utilization` to 0.2
- Use minimal test configuration first

**vLLM Compatibility Issues:**
- Set `use_vllm: false` in `grpo.yaml` for testing
- Ensure `vllm_enforce_eager: true` if using vLLM

**Knowledge Base Issues:**
```bash
# Verify database exists and is readable
ls -lh ~/a100x4/KnowRL-reproduction-data/knowledge_base/knowledge_base.db
sqlite3 knowledge_base.db "SELECT COUNT(*) FROM documents;"
```

**API Rate Limits:**
- Use different API keys for FActScore and Judge
- Implement delays between API calls if needed

## Expected Results

Based on successful reproduction:
- **Minimal Test**: Completes in 2-3 minutes, uses ~25GB per GPU
- **Full Training**: Takes 6-12 hours, achieves improved factuality scores
- **Memory Usage**: Efficiently utilizes A100-40GB GPUs
- **Factuality**: Improved performance on knowledge-intensive tasks

## Next Steps

After successful reproduction:
1. **Scale up** from minimal to full configuration
2. **Experiment** with different reward weightings
3. **Evaluate** on downstream tasks (TruthfulQA, SimpleQA, etc.)
4. **Customize** knowledge base for domain-specific applications

## Support

- **Paper**: https://arxiv.org/abs/2506.19807
- **Models**: https://huggingface.co/collections/zjunlp/knowrl-68485613feca77696d252a1d
- **Current Branch**: `minimal-test` for initial testing
- **Git Workflow**: Develop on laptop, sync to server for execution