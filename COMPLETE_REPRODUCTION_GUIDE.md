# KnowRL Complete Reproduction Guide

This comprehensive guide provides both theoretical understanding and practical step-by-step instructions for reproducing the KnowRL research, connecting the paper's methodology to its codebase implementation.

## Overview: Paper to Code Mapping

KnowRL addresses LLM hallucinations through knowledge-integrated reinforcement learning with a two-stage approach:

1. **Cold-Start SFT** → Aligns models with factual reasoning patterns  
2. **Knowledgeable RL** → Uses external knowledge in reward functions during RL training

The codebase is a **direct, faithful implementation** of the paper's methodology, with every theoretical component having a corresponding code module.

## Core Architecture Components

### 1. Two-Stage Training Pipeline

#### Stage 1: Cold-Start SFT
- **Paper Concept**: "Pre-aligns the model using high-quality factual Q&A pairs"
- **Implementation**: 
  - Framework: LLaMA-Factory (external dependency)
  - Dataset: `data/coldstart/knowrl_coldstart.json`
  - Configuration: Custom YAML with LoRA settings (rank=256, alpha=512)
- **Purpose**: Establishes foundational factual reasoning patterns before RL

#### Stage 2: Knowledge RL  
- **Paper Concept**: "Integrates external knowledge into RL training loop"
- **Implementation**:
  - Entry point: `train/main.py`
  - Algorithm: GRPO (Group Relative Policy Optimization)
  - Configuration: `train/script/grpo.yaml`
  - Execution: `train/train.sh`
- **Key Code**: `train/main.py:184-194` sets up GRPO trainer with reward functions

### 2. Knowledge Integration System

The paper's core innovation - real-time external knowledge integration - is implemented through the FActScore framework:

**Knowledge Base Construction**:
```
train/reward_function/FActScore/build_knowledge/
├── build_db.sh                 # Knowledge base building script
├── build_knowledge_base.py     # Core database creation logic
└── knowledge_base.db          # SQLite database (downloaded/built)
```

**Knowledge Retrieval Implementation** (`train/reward_function/fact_reward.py:34-48`):
```python
fs = FactScorer(
    openai_key=openai_api_key,
    base_url=base_url,
    af_model_version='gpt-4o-mini',
    use_nli=True,
    nli_model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)
fs.register_knowledge_source(db_path=db_path)
```

## Multi-Component Reward System

The paper's composite reward function is implemented through four interconnected components:

### A. Format Reward (`train/reward_function/format_reward.py`)
- **Paper Purpose**: Enforces structured reasoning with `<think>` and `<answer>` tags
- **Implementation**: Regex-based format validation (`format_reward.py:18-29`)
- **Scoring**: +1.0 for correct format, -1.0 for incorrect
- **Paper Connection**: Implements "slow thinking" pattern with explicit reasoning steps

### B. Factuality Reward (`train/reward_function/fact_reward.py`)
- **Paper Purpose**: Core knowledge integration - measures factual accuracy against external knowledge
- **Implementation Strategy**:
  1. Atomic fact decomposition using FActScore
  2. Real-time knowledge base querying
  3. NLI-based fact verification
- **Scoring Formula**: `supported_facts_count / 15.0` (capped at 1.0)
- **Key Code Section**: `fact_reward.py:122-129` performs batch fact-checking

### C. Correctness Reward (`train/reward_function/correct_reward.py`)
- **Paper Purpose**: LLM-based correctness evaluation against reference answers
- **Multi-Stage Evaluation Process**:
  1. **Refusal Detection** (`correct_reward.py:51-59`): Identifies non-responsive answers
  2. **Numeric Handling** (`correct_reward.py:217-241`): Special processing for mathematical answers
  3. **Keyword Matching** (`correct_reward.py:244-259`): Preliminary relevance check
  4. **LLM Judge** (`correct_reward.py:262-284`): Final correctness assessment
- **Scoring**: +2.0 (correct), -1.0 (incorrect/not attempted), 0.0 (error)

### D. Combined Reward (`train/reward_function/combined_reward.py`)
- **Paper Purpose**: Intelligent combination of individual reward signals
- **Combination Strategy** (`combined_reward.py:24-50`):
```python
if llm_reward == -1:
    final_rewards[i] = -1 + fact_score  # Factuality can partially rescue format issues
```
- **Innovation**: Allows factuality score to moderate format penalties

## Training Process Implementation

### Data Flow Pipeline

1. **Data Loading** (`main.py:139-157`):
   - Source: `data/rl/knowrl_RLdata.json`
   - Format validation and field mapping
   - Dataset preprocessing for GRPO training

2. **Prompt Generation** (`main.py:71-97`):
   - Implements paper's structured prompting strategy
   - Enforces `<think>` and `<answer>` tag usage
   - Template: Explicit format instructions with examples

3. **Model Configuration** (`main.py:110-131`):
   - **Unsloth Integration**: Fast training framework (paper's efficiency claims)
   - **LoRA Setup**: rank=256, alpha=512 (parameter-efficient training)
   - **Memory Optimization**: Gradient checkpointing and mixed precision

4. **GRPO Training Loop** (`main.py:184-194`):
   - Multiple reward functions integrated into single trainer
   - Batch generation with factuality checking
   - Real-time knowledge retrieval during training

## Critical Configuration Files

### Primary Configuration (`train/script/grpo.yaml`)
Maps directly to paper's experimental setup:

| Paper Specification | Code Parameter | Location |
|---------------------|----------------|----------|
| 7B Models (DeepSeek-R1, Skywork-OR1) | `model_name_or_path` | `grpo.yaml:2` |
| LoRA rank=256, alpha=512 | `lora_r`, `lora_alpha` | `grpo.yaml:5-6` |
| Batch size optimization | `per_device_train_batch_size: 24` | `grpo.yaml:22` |
| GRPO generations | `num_generations: 12` | `grpo.yaml:37` |
| SwanLab tracking | `report_to: swanlab` | `grpo.yaml:16` |

### Environment Setup (`train/train.sh`)
Essential environment variables for reproduction:
- `OPENAI_API_KEY_FACTSCORE`: For FActScore factuality evaluation
- `OPENAI_API_KEY_JUDGE`: For LLM-based correctness evaluation  
- `SWANLAB_API_KEY`: For experiment tracking and visualization
- `FACTSCORE_DB_PATH`: Path to knowledge base database
- `CUDA_VISIBLE_DEVICES`: GPU configuration (single A800 setup)

## Key Innovation Implementations

### Test-time Scaling Law
- **Paper Theory**: External knowledge reduces hallucinations during generation
- **Code Implementation**: Real-time knowledge retrieval in `fact_reward.py:122-129` during RL training
- **Mechanism**: Each generation triggers factuality checking against knowledge base

### Knowledge Boundary Learning  
- **Paper Theory**: Model learns when to rely on internal vs external knowledge
- **Code Implementation**: Refusal detection + partial matching in `correct_reward.py:200-259`
- **Strategy**: Multi-tier evaluation prevents over-reliance on either knowledge source

### Fact-based Slow Thinking
- **Paper Theory**: Structured reasoning with factual grounding
- **Code Implementation**: Format enforcement + factuality scoring in reward combination
- **Result**: Models develop explicit reasoning chains validated against external knowledge

## File Structure for Reproduction

### Essential Files by Function

| Component | Primary Files | Purpose |
|-----------|---------------|---------|
| **Cold-Start SFT** | `data/coldstart/`, LLaMA-Factory config | Stage 1 training |
| **Knowledge RL** | `main.py`, `grpo.yaml`, `train.sh` | Stage 2 orchestration |
| **Knowledge Base** | `FActScore/build_knowledge/` | External knowledge source |
| **Reward System** | `reward_function/*.py` | Multi-component scoring |
| **GRPO Framework** | `trl/` directory | Customized RL training |
| **Experiment Tracking** | SwanLab integration | Results monitoring |

### Dependencies and Frameworks
- **Unsloth**: Fast training with memory optimization
- **TRL**: Transformers Reinforcement Learning (customized)
- **FActScore**: Factuality scoring with knowledge retrieval
- **SwanLab**: Experiment tracking and visualization
- **LLaMA-Factory**: Cold-start SFT framework

## Pre-Reproduction Checklist

Before attempting reproduction, ensure understanding of:

1. **Two-stage training dependency**: Stage 2 requires Stage 1 completion
2. **Knowledge base requirement**: External database must be built/downloaded
3. **API key configuration**: Multiple OpenAI endpoints needed
4. **Hardware specifications**: Optimized for single A800 GPU
5. **Environment variables**: Critical for component communication
6. **Dataset format**: Specific JSON structure required
7. **Reward function interplay**: Understanding how components combine

## Part 2: Detailed Reproduction Steps

This section provides step-by-step instructions for reproducing the KnowRL research results.

### Step 1: Hardware and Environment Preparation

### Hardware Requirements
- **Minimum**: 1x A800 (80GB VRAM) or A100 80GB
- **Alternative**: Multiple smaller GPUs (e.g., 4x RTX 4090) with DeepSpeed
- **System RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for models, datasets, and knowledge base

### Environment Setup

#### 1.1 Create Conda Environment
```bash
conda create -n knowrl python=3.12
conda activate knowrl
```

#### 1.2 Clone Repository
```bash
git clone https://github.com/zjunlp/KnowRL.git
cd KnowRL
```

#### 1.3 Install Dependencies

**🚀 Multiple Elegant Installation Options:**

**Option A: Automated Script (Recommended)**
```bash
bash setup.sh conda    # Most robust dependency resolution
bash setup.sh pip      # Staged pip installation
```

**Option B: Conda Environment File**  
```bash
conda env create -f environment.yml
```

**Option C: Python Package**
```bash
pip install -e .    # Modern setup with dependency ordering
```

**Option D: Manual (Original Method)**
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install unsloth separately (requires PyTorch to be installed first)
pip install unsloth==2025.3.18

# Create modified requirements file (excluding PyTorch and unsloth)
sed '1,4d' requirements.txt > requirements_no_torch.txt

# Install remaining dependencies
pip install -r requirements_no_torch.txt
```

**💡 Why Multiple Options?** Each method handles the PyTorch dependency conflict differently:
- **Script**: Full automation with verification  
- **Conda env**: Best dependency resolution
- **Package install**: Modern Python standards
- **Manual**: Full control and understanding

See `INSTALLATION.md` for detailed troubleshooting.

#### 1.4 Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
python -c "import unsloth; print('Unsloth installed successfully')"
```

#### 1.5 HuggingFace Cache Configuration

**For automatic model downloads**, configure HuggingFace cache location (especially important for server users):

**Standard setup (optional):**
```bash
# Models will download to ~/.cache/huggingface/ by default
# No configuration needed for basic setup
```

**Server storage setup (recommended):**
```bash
# Set HuggingFace cache to your storage location
export HF_HOME="/data22/zhangjunwei/huggingface_cache"
export TRANSFORMERS_CACHE="/data22/zhangjunwei/huggingface_cache"

# Create cache directory
mkdir -p /data22/zhangjunwei/huggingface_cache

# Make permanent by adding to ~/.bashrc
echo 'export HF_HOME="/data22/zhangjunwei/huggingface_cache"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="/data22/zhangjunwei/huggingface_cache"' >> ~/.bashrc
source ~/.bashrc
```

**Model Download Behavior:**
- **Automatic**: Models download automatically when using HuggingFace IDs (e.g., `"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"`)
- **First time only**: ~13GB download for 7B models (10-30 minutes)
- **Subsequent runs**: Use cached version (no re-download)

**Test pre-download (optional):**
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Testing model download...')
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B') 
print('Model cached successfully!')
"
```

#### 1.6 Session Persistence Setup (Critical for Long Training)

Since KnowRL training runs for 6-12 hours, use **tmux** to prevent losing progress from SSH disconnections:

**Install tmux (if needed):**
```bash
sudo apt install tmux  # Ubuntu/Debian
# or
sudo yum install tmux  # CentOS/RHEL
```

**Essential tmux Commands:**

| Action | Command | Use Case |
|--------|---------|----------|
| **Create session** | `tmux new -s knowrl` | Start training session |
| **Detach safely** | `Ctrl+b, d` | Disconnect without stopping |
| **Reattach** | `tmux attach -t knowrl` | Reconnect to training |
| **List sessions** | `tmux ls` | See all active sessions |
| **Kill session** | `tmux kill-session -t knowrl` | Clean up after training |
| **Split horizontal** | `Ctrl+b, "` | Monitor multiple outputs |
| **Split vertical** | `Ctrl+b, %` | Side-by-side monitoring |
| **Switch panes** | `Ctrl+b, arrows` | Navigate between views |
| **New window** | `Ctrl+b, c` | Add monitoring window |

**Recommended workflow:**
```bash
# 1. Start training session
tmux new-session -s knowrl-training

# 2. Run your training commands inside tmux
cd ~/KnowRL/train/
bash train_server.sh

# 3. Detach safely when needed
Ctrl+b, d

# 4. Reconnect later
ssh username@your-server
tmux attach -t knowrl-training
```

#### 1.7 Server Storage Configuration (For Users with Separate Storage)

If you're using a server environment where you need to keep the codebase in your home folder but store large files (datasets, models, knowledge base) in a separate storage location, follow this configuration.

##### 1.5.1 Recommended Directory Structure

```bash
# Codebase (in home folder)
~/KnowRL/
├── train/
├── data/                    # Will be symlinked to storage
└── requirements.txt

# Data and Models (in storage folder - example: /data22/username/)
/data22/zhangjunwei/
├── models/
│   ├── base_models/
│   │   ├── deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/
│   │   └── skywork--Skywork-OR1-7B-Preview/
│   ├── sft_output/
│   └── final_knowrl_models/
├── datasets/
│   ├── coldstart/
│   │   └── knowrl_coldstart.json
│   └── rl/
│       └── knowrl_RLdata.json
├── knowledge_base/
│   └── knowledge_base.db
└── experiment_outputs/
    └── knowrl_experiments/
```

##### 1.5.2 Storage Setup Process

**Step A: Create Storage Directory Structure**
```bash
# Replace '/data22/zhangjunwei' with your storage path
STORAGE_ROOT="/data22/zhangjunwei"

mkdir -p $STORAGE_ROOT/{models/{base_models,sft_output,final_knowrl_models},datasets/{coldstart,rl},knowledge_base,experiment_outputs}

# Verify structure
ls -la $STORAGE_ROOT/
```

**Step B: Check Available Storage Space**
```bash
# Ensure you have sufficient space
df -h /data22/zhangjunwei/

# Minimum requirements:
# - Base model (7B): ~13GB
# - SFT output: ~2GB (LoRA adapters)  
# - Knowledge base: ~1.2GB
# - Training outputs: ~5GB
# - Total needed: ~25GB minimum
```

##### 1.5.3 Configuration Approach Options

**Option 1: Symlinks Approach (Recommended - Minimal Code Changes)**

```bash
cd ~/KnowRL/

# Move original data to storage and create symlink
mv data /data22/zhangjunwei/datasets_original
ln -s /data22/zhangjunwei/datasets_original data

# Verify symlink works
ls -la data/  # Should show contents from storage location
```

**Benefits**: Most configuration files won't need path changes.

**Option 2: Direct Path Configuration (More Control)**

You'll need to update multiple configuration files with absolute paths:

```bash
# Create server-specific config files
cd ~/KnowRL/train/script/
cp grpo.yaml grpo_server.yaml

cd ~/KnowRL/train/
cp train.sh train_server.sh
```

##### 1.5.4 Key Files to Update (Option 2 Only)

**A. Main Training Configuration** (`train/script/grpo_server.yaml`):
```yaml
# Update these paths
model_name_or_path: "/data22/zhangjunwei/models/base_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
output_dir: "/data22/zhangjunwei/experiment_outputs/knowrl_experiments/grpo_run1"  
dataset_id_or_path: "/data22/zhangjunwei/datasets/rl/knowrl_RLdata.json"
```

**B. Environment Variables** (`train/train_server.sh`):
```bash
# Update knowledge base path
export FACTSCORE_DB_PATH="/data22/zhangjunwei/knowledge_base/knowledge_base.db"

# Update config file reference
CONFIG_FILE="./script/grpo_server.yaml"
```

**C. Knowledge Base Build Script** (`train/reward_function/FActScore/build_knowledge/build_db.sh`):
```bash
# If building knowledge base from scratch
DATA_PATH="/data22/zhangjunwei/datasets/raw/wikipedia.jsonl"
DB_PATH="/data22/zhangjunwei/knowledge_base/knowledge_base.db"
```

##### 1.5.5 Data Migration Commands

```bash
# Set your storage root
STORAGE_ROOT="/data22/zhangjunwei"

# Copy datasets (if they exist in the repo)
if [ -d "~/KnowRL/data/coldstart" ]; then
    cp ~/KnowRL/data/coldstart/knowrl_coldstart.json $STORAGE_ROOT/datasets/coldstart/
fi

if [ -d "~/KnowRL/data/rl" ]; then  
    cp ~/KnowRL/data/rl/knowrl_RLdata.json $STORAGE_ROOT/datasets/rl/
fi

# Create placeholders for models (will be downloaded later)
echo "Base models will be downloaded here" > $STORAGE_ROOT/models/base_models/README.txt

# Verify setup
ls -la $STORAGE_ROOT/datasets/*/
ls -la $STORAGE_ROOT/models/*/
```

##### 1.5.6 Environment Variables Setup (Alternative Approach)

Add to your `~/.bashrc` for persistent configuration:
```bash
# Add these lines to ~/.bashrc
export KNOWRL_DATA_ROOT="/data22/zhangjunwei"
export KNOWRL_MODELS_ROOT="/data22/zhangjunwei/models"  
export KNOWRL_DATASETS_ROOT="/data22/zhangjunwei/datasets"
export KNOWRL_KB_ROOT="/data22/zhangjunwei/knowledge_base"

# Apply changes
source ~/.bashrc
```

Then reference in configuration files:
```yaml
# In grpo.yaml
model_name_or_path: "${KNOWRL_MODELS_ROOT}/base_models/DeepSeek-R1-Distill-Qwen-7B"
output_dir: "${KNOWRL_DATA_ROOT}/experiment_outputs/grpo_run1"
dataset_id_or_path: "${KNOWRL_DATASETS_ROOT}/rl/knowrl_RLdata.json"
```

##### 1.5.7 Verification Commands

```bash
# Test data access
ls -la /data22/zhangjunwei/datasets/rl/knowrl_RLdata.json

# Test symlink (if using Option 1)
ls -la ~/KnowRL/data/rl/knowrl_RLdata.json

# Test environment variables (if using)
echo "Data root: $KNOWRL_DATA_ROOT"
echo "Models root: $KNOWRL_MODELS_ROOT"

# Test write permissions
touch /data22/zhangjunwei/test_write_permission.txt && rm /data22/zhangjunwei/test_write_permission.txt && echo "Write permission OK" || echo "Write permission FAILED"
```

**Important Notes for Later Steps**:
- When downloading the knowledge base (Step 3), use: `cd /data22/zhangjunwei/knowledge_base/`
- When configuring model paths (Step 4), use your storage paths
- When running training, ensure all paths point to your storage location
- Monitor storage space usage during training

### Step 2: API Keys and Service Setup

### 2.1 Obtain Required API Keys
You need **THREE** API keys:
1. **OpenAI API Key (FActScore)**: For factuality evaluation
2. **OpenAI API Key (Judge)**: For LLM-based correctness evaluation  
3. **SwanLab API Key**: For experiment tracking

**Get OpenAI API Keys**:
- Visit: https://platform.openai.com/api-keys
- Create two separate keys (recommended for rate limiting)
- Ensure sufficient credits for evaluation calls

**Get SwanLab API Key**:
- Visit: https://swanlab.cn/ and create account
- Generate API key from dashboard
- Alternative: Use `wandb` by modifying configuration

### 2.2 Configure Environment Variables
Edit `train/train.sh` with your API keys:

```bash
cd train/
cp train.sh train.sh.backup  # Create backup
nano train.sh  # Edit with your preferred editor
```

**Replace placeholders with actual keys**:
```bash
export OPENAI_API_KEY_FACTSCORE="sk-your-factscore-key-here"
export OPENAI_API_KEY_JUDGE="sk-your-judge-key-here"  
export SWANLAB_API_KEY="your-swanlab-key-here"
```

### Step 3: Knowledge Base Setup

**Important**: If you configured server storage in Step 1.5, download the knowledge base to your storage location instead of the default path.

### 3.1 Option 1: Download Pre-built Knowledge Base (Recommended)

**For standard setup:**
```bash
cd train/reward_function/FActScore/build_knowledge/

# Download pre-built knowledge base (1.2GB)
gdown https://drive.google.com/uc?id=1EVFkzuFvqE8AOEcdfSSm03vvvbVDa7bI

# Verify download
ls -lh knowledge_base.db
# Expected: ~1.2GB file size
```

**For server storage setup (if you completed Step 1.5):**
```bash
# Download to your storage location
cd /data22/zhangjunwei/knowledge_base/  # Use your storage path

# Download pre-built knowledge base (1.2GB)
gdown https://drive.google.com/uc?id=1EVFkzuFvqE8AOEcdfSSm03vvvbVDa7bI

# Verify download
ls -lh knowledge_base.db
# Expected: ~1.2GB file size

# If using symlinks, create symlink in code location
ln -s /data22/zhangjunwei/knowledge_base/knowledge_base.db ~/KnowRL/train/reward_function/FActScore/build_knowledge/knowledge_base.db
```

### 3.2 Option 2: Build Knowledge Base from Scratch

**If you have your own Wikipedia dump or want to customize**:

```bash
cd train/reward_function/FActScore/build_knowledge/

# 1. Prepare your data file (e.g., wikipedia.jsonl)
# Expected format: {"title": "Article Title", "text": "Article content..."}

# 2. Edit build_db.sh configuration
nano build_db.sh
# Modify: DATA_PATH="./data/your_data_file.json"

# 3. Run build script
bash build_db.sh
```

### 3.3 Verify Knowledge Base
```bash
# Check database exists and is accessible
python -c "
import sqlite3
conn = sqlite3.connect('knowledge_base.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM documents')
print(f'Documents in database: {cursor.fetchone()[0]}')
conn.close()
"
```

### Step 4: Configuration Customization

**Important**: If you configured server storage in Step 1.5, use the server-specific config files and paths you created, or use the symlink approach for seamless configuration.

### 4.1 Configure Model Paths in grpo.yaml
```bash
cd train/script/
cp grpo.yaml grpo.yaml.backup  # Create backup
nano grpo.yaml
```

**Essential configurations to modify**:

**For standard setup:**
```yaml
# Line 2: Replace with your base model path
model_name_or_path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # HuggingFace model ID

# Line 10: Set your output directory  
output_dir: "output/knowrl_model_run1"

# Lines 17-18: Configure SwanLab project details
swanlab_project: "knowrl_reproduction"
swanlab_experiment_name: "run1"
```

**For server storage setup (edit grpo_server.yaml instead):**
```yaml
# Line 2: Use your storage paths
model_name_or_path: "/data22/zhangjunwei/models/base_models/DeepSeek-R1-Distill-Qwen-7B"

# Line 10: Output to storage location
output_dir: "/data22/zhangjunwei/experiment_outputs/knowrl_experiments/grpo_run1"

# Line 13: Update dataset path  
dataset_id_or_path: "/data22/zhangjunwei/datasets/rl/knowrl_RLdata.json"

# Lines 17-18: Configure SwanLab project details
swanlab_project: "knowrl_reproduction" 
swanlab_experiment_name: "server_run1"
```

**For different hardware configurations**:
```yaml
# For smaller GPUs, reduce batch size
per_device_train_batch_size: 12  # Reduce from 24
gradient_accumulation_steps: 8   # Increase from 4

# For multiple GPUs, adjust VLLM memory
vllm_gpu_memory_utilization: 0.3  # Reduce from 0.5
```

### 4.2 Verify Dataset Paths
```bash
# Check training datasets exist
ls -la ../data/coldstart/knowrl_coldstart.json
ls -la ../data/rl/knowrl_RLdata.json

# Verify dataset format
head -n 3 ../data/rl/knowrl_RLdata.json
```

### Step 5: Stage 1 - Cold-Start SFT Training

### 5.1 Install LLaMA-Factory
```bash
# In a separate directory or use pip
pip install llamafactory
```

### 5.2 Create SFT Configuration
Create `llama_factory_sft.yaml`:
```yaml
### model
model_name_or_path: /path/to/your/base_model  # Same as grpo.yaml
adapter_name_or_path: /path/to/your/sft_adapter_output

### method  
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 256
lora_alpha: 512

### dataset
dataset: knowrl_coldstart  # Register this dataset name in LLaMA-Factory
template: qwen  # Or appropriate template for your model
cutoff_len: 3072
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /path/to/sft_output
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 4.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000
```

### 5.3 Run Cold-Start SFT
```bash
# Multi-GPU training (adjust CUDA_VISIBLE_DEVICES as needed)
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llama_factory_sft.yaml

# Monitor training progress
tail -f /path/to/sft_output/trainer_log.jsonl
```

### 5.4 Verify SFT Completion
```bash
# Check output directory
ls -la /path/to/sft_output/
# Should contain: adapter_config.json, adapter_model.bin, training_args.bin

# Test loading the adapter
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('/path/to/base_model')
model = PeftModel.from_pretrained(base_model, '/path/to/sft_output')
print('SFT model loaded successfully')
"
```

### Step 6: Stage 2 - Knowledge RL Training

### 6.1 Update GRPO Configuration
Update `grpo.yaml` with SFT model path:
```yaml
# Use SFT-trained model as base for RL
model_name_or_path: "/path/to/sft_output"  # Path to your SFT adapter
```

### 6.2 Pre-training Verification
```bash
cd train/

# Verify all environment variables are set
bash -c "
source train.sh
echo 'FACTSCORE_DB_PATH: $FACTSCORE_DB_PATH'
echo 'OPENAI_API_KEY_FACTSCORE: '${OPENAI_API_KEY_FACTSCORE:0:10}'...'
echo 'OPENAI_API_KEY_JUDGE: '${OPENAI_API_KEY_JUDGE:0:10}'...'
echo 'SWANLAB_API_KEY: '${SWANLAB_API_KEY:0:10}'...'
"

# Test knowledge base connectivity
python -c "
import os
from reward_function.fact_reward import FactualityScorer
os.environ['FACTSCORE_DB_PATH'] = 'reward_function/FActScore/build_knowledge/knowledge_base.db'
scorer = FactualityScorer()
print('Knowledge base connection successful' if scorer.get_fact_scorer() else 'Connection failed')
"
```

### 6.3 Run Knowledge RL Training

**For standard setup:**
```bash
# Start training
bash train.sh

# Monitor training in separate terminal  
tail -f output/your_output_directory_name/trainer_log.jsonl
```

**For server storage setup:**
```bash
# Start training with server-specific configuration
bash train_server.sh  # Uses grpo_server.yaml and storage paths

# Monitor training in separate terminal
tail -f /data22/zhangjunwei/experiment_outputs/knowrl_experiments/grpo_run1/trainer_log.jsonl

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 6.4 Training Monitoring and Logs
**SwanLab Dashboard**: Visit https://swanlab.cn/workspace to monitor:
- Loss curves
- Reward function outputs
- Training metrics
- GPU utilization

**Local Log Files**:
```bash
# Training logs
ls -la output/your_output_directory_name/

# Reward function outputs (JSON format)
ls -la reward_function_outputs/
# Contains: format_reward_*.json, factuality_reward_*.json, etc.
```

### Step 7: Results Analysis and Validation

### 7.1 Training Completion Verification
```bash
# Check final model output
ls -la output/your_output_directory_name/
# Should contain: pytorch_model.bin, config.json, tokenizer files

# Verify training completed successfully
grep "Training completed" output/your_output_directory_name/trainer_log.jsonl
```

### 7.2 Model Testing
```bash
# Quick inference test
python -c "
import os
os.chdir('train')
from main import *
# Load trained model and test basic inference
"

# Or create a simple inference script
python test_inference.py  # Create this based on main.py structure
```

### 7.3 Reward Function Analysis
```bash
# Analyze reward function outputs
cd reward_function_outputs/

# Check format reward distribution
python -c "
import json
import glob
import numpy as np

for file in glob.glob('format_reward_*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
    rewards = [entry['reward'] for entry in data['format_checks']]
    print(f'{file}: Mean reward = {np.mean(rewards):.3f}')
"

# Analyze factuality scores
python -c "
import json
import glob

for file in glob.glob('factuality_reward_*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
    if 'summary' in data:
        print(f'{file}: Overall score = {data[\"summary\"][\"overall_score\"]:.3f}')
"
```

### Step 8: Evaluation and Benchmarking

### 8.1 Prepare Evaluation Environment
```bash
# Install OpenCompass for evaluation (optional)
pip install opencompass

# Or prepare your own evaluation scripts based on paper benchmarks:
# - TruthfulQA
# - SimpleQA  
# - ChineseSimpleQA
# - GPQA
# - AIME 2025
```

### 8.2 Generate Sample Outputs
```bash
# Create evaluation script based on main.py inference logic
python evaluate_model.py --model_path output/your_output_directory_name \
                        --eval_dataset path/to/evaluation/data.json \
                        --output_file evaluation_results.json
```

### Troubleshooting Common Issues

#### Memory Issues
```bash
# Reduce batch size in grpo.yaml
per_device_train_batch_size: 8  # From 24
gradient_accumulation_steps: 12  # Increase accordingly

# Enable gradient checkpointing
gradient_checkpointing: true

# Use DeepSpeed for multi-GPU
# Copy appropriate config from train/trl/accelerate_configs/
```

#### API Rate Limiting
```bash
# Use different API keys for different functions
# Add delays in API calls by modifying api_client_manager.py
```

#### Knowledge Base Issues
```bash
# Rebuild database if corrupted
cd train/reward_function/FActScore/build_knowledge/
rm knowledge_base.db
bash build_db.sh
```

#### Training Crashes
```bash
# Enable checkpointing
resume_from_checkpoint: true  # In grpo.yaml

# Check logs for specific errors
grep -i error output/your_output_directory_name/trainer_log.jsonl
```

#### SwanLab Connection Issues
```bash
# Test SwanLab connection
python -c "
import swanlab
swanlab.login()  # Enter your API key
print('SwanLab connection successful')
"

# Alternative: Use wandb instead
# Modify grpo.yaml: report_to: wandb
```

### Expected Results and Benchmarks

Based on the paper, you should expect:
- **Format Reward**: ~90%+ compliance after training
- **Factuality Score**: Improved factual accuracy compared to baseline
- **Overall Performance**: Better results on TruthfulQA, SimpleQA benchmarks
- **Training Time**: ~6-12 hours for 150 steps on A800 (depends on dataset size)

### Next Steps and Extensions

After successful reproduction:
1. **Model Analysis**: Compare outputs with paper results
2. **Hyperparameter Tuning**: Experiment with different reward weightings
3. **Knowledge Base Customization**: Use domain-specific knowledge bases
4. **Evaluation Extensions**: Test on additional benchmarks
5. **Model Scaling**: Try with different model sizes

### Support and Resources

- **Paper**: https://arxiv.org/abs/2506.19807
- **HuggingFace**: https://huggingface.co/collections/zjunlp/knowrl-68485613feca77696d252a1d  
- **Issues**: Report problems to the original repository
- **Documentation**: This guide covers reproduction; refer to paper for theoretical details

This comprehensive guide provides all necessary steps for reproducing the KnowRL research. Follow each step carefully, and don't hesitate to adapt configurations based on your specific hardware and requirements.
