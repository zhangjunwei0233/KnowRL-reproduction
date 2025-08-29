#!/usr/bin/env python3
"""
Comprehensive Pre-Training Checklist for KnowRL
Validates all components before starting training

Usage: python pre_training_checklist.py
"""

import os
import sys
import json
import yaml
import sqlite3
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_check(item, status, details=""):
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {item}")
    if details:
        print(f"   {details}")
    return status

def check_environment():
    """Check Python environment and dependencies"""
    print_header("ENVIRONMENT & DEPENDENCIES")
    
    issues = []
    
    # Check Python version
    python_ver = sys.version_info
    python_ok = python_ver >= (3, 8)
    print_check(f"Python {python_ver.major}.{python_ver.minor}.{python_ver.micro}", 
                python_ok, "Requires Python 3.8+")
    if not python_ok:
        issues.append("Python version too old")
    
    # Check critical packages
    critical_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('unsloth', 'Unsloth'),
        ('datasets', 'Datasets'),
        ('peft', 'PEFT'),
        ('trl', 'TRL'),
        ('openai', 'OpenAI client'),
        ('rank_bm25', 'BM25 for FActScore'),
        ('zhipuai', 'ZhipuAI (optional fallback)'),
        ('sqlite3', 'SQLite3'),
        ('nltk', 'NLTK'),
        ('yaml', 'PyYAML')
    ]
    
    for package, name in critical_packages:
        try:
            __import__(package)
            print_check(f"{name} installed", True)
        except ImportError:
            print_check(f"{name} missing", False)
            if package not in ['zhipuai']:  # Optional packages
                issues.append(f"Missing {name}")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_check("CUDA available", cuda_available)
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_check(f"CUDA devices: {device_count}", True, 
                       f"Device 0: {torch.cuda.get_device_name(0)}")
        else:
            issues.append("CUDA not available")
    except:
        issues.append("Cannot check CUDA")
    
    return issues

def check_storage_structure():
    """Check storage directory structure"""
    print_header("STORAGE STRUCTURE")
    
    issues = []
    base_path = Path("/data22/zhangjunwei")
    
    required_dirs = [
        "knowrl_training/outputs",
        "knowrl_training/models", 
        "knowrl_training/logs",
        "knowrl_knowledge_base"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        print_check(f"Directory: {full_path}", exists)
        if not exists:
            issues.append(f"Missing directory: {full_path}")
    
    # Check write permissions
    try:
        test_file = base_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print_check("Write permissions", True)
    except Exception as e:
        print_check("Write permissions", False, str(e))
        issues.append("No write permission to storage")
    
    return issues

def check_knowledge_base():
    """Check knowledge base integrity"""
    print_header("KNOWLEDGE BASE")
    
    issues = []
    
    db_path = os.environ.get("FACTSCORE_DB_PATH")
    if not db_path:
        print_check("FACTSCORE_DB_PATH set", False)
        issues.append("FACTSCORE_DB_PATH not configured")
        return issues
    
    print_check(f"Database path: {db_path}", True)
    
    # Check file exists
    db_exists = os.path.exists(db_path)
    print_check("Knowledge base file exists", db_exists)
    if not db_exists:
        issues.append(f"Knowledge base missing: {db_path}")
        return issues
    
    # Check file size
    size_gb = os.path.getsize(db_path) / (1024**3)
    size_ok = size_gb > 1.0  # Should be ~1.2GB
    print_check(f"Database size: {size_gb:.2f}GB", size_ok)
    if not size_ok:
        issues.append("Knowledge base file too small")
    
    # Check database integrity
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        conn.close()
        
        tables_ok = table_count > 0
        print_check(f"Database tables: {table_count}", tables_ok)
        if not tables_ok:
            issues.append("Knowledge base has no tables")
    except Exception as e:
        print_check("Database integrity", False, str(e))
        issues.append("Knowledge base corrupted")
    
    return issues

def check_api_configuration():
    """Check API keys and endpoints"""
    print_header("API CONFIGURATION")
    
    issues = []
    
    required_vars = [
        ("OPENAI_API_KEY_FACTSCORE", "DeepSeek API key for FActScore"),
        ("OPENAI_BASE_URL_FACTSCORE", "DeepSeek base URL for FActScore"),
        ("OPENAI_API_KEY_JUDGE", "DeepSeek API key for Judge"),
        ("OPENAI_API_BASE_JUDGE", "DeepSeek base URL for Judge"),
        ("SWANLAB_API_KEY", "SwanLab API key"),
        ("HF_HOME", "HuggingFace cache directory")
    ]
    
    for var_name, description in required_vars:
        value = os.environ.get(var_name)
        has_value = value is not None and value != "" and "your_" not in value.lower()
        print_check(f"{description}", has_value)
        if not has_value:
            issues.append(f"Missing {var_name}")
    
    # Check HF_HOME accessibility
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_accessible = os.path.exists(hf_home) and os.access(hf_home, os.R_OK)
        print_check(f"HF_HOME accessible: {hf_home}", hf_accessible)
        if not hf_accessible:
            issues.append("HF_HOME not accessible")
    
    return issues

def check_training_data():
    """Check training datasets"""
    print_header("TRAINING DATA")
    
    issues = []
    
    # Check coldstart data
    coldstart_path = Path("../data/coldstart/knowrl_coldstart.json")
    coldstart_exists = coldstart_path.exists()
    print_check("Cold-start SFT data", coldstart_exists, str(coldstart_path))
    
    if coldstart_exists:
        try:
            with open(coldstart_path) as f:
                coldstart_data = json.load(f)
            sample_count = len(coldstart_data)
            print_check(f"Cold-start samples: {sample_count}", sample_count > 0)
            
            # Check data format
            if sample_count > 0:
                sample = coldstart_data[0]
                has_required_fields = all(key in sample for key in ['conversations'])
                print_check("Cold-start data format", has_required_fields)
                if not has_required_fields:
                    issues.append("Cold-start data missing required fields")
        except Exception as e:
            print_check("Cold-start data parsing", False, str(e))
            issues.append("Cold-start data corrupted")
    else:
        issues.append("Cold-start data missing")
    
    # Check RL data
    rl_path = Path("../data/rl/knowrl_RLdata.json")
    rl_exists = rl_path.exists()
    print_check("RL training data", rl_exists, str(rl_path))
    
    if rl_exists:
        try:
            with open(rl_path) as f:
                rl_data = json.load(f)
            sample_count = len(rl_data)
            print_check(f"RL samples: {sample_count}", sample_count > 0)
        except Exception as e:
            print_check("RL data parsing", False, str(e))
            issues.append("RL data corrupted")
    else:
        issues.append("RL data missing")
    
    return issues

def check_configuration_consistency():
    """Check configuration file consistency"""
    print_header("CONFIGURATION CONSISTENCY")
    
    issues = []
    
    # Load configurations
    try:
        with open("script/grpo.yaml") as f:
            grpo_config = yaml.safe_load(f)
        print_check("GRPO config loaded", True)
    except Exception as e:
        print_check("GRPO config loading", False, str(e))
        issues.append("Cannot load grpo.yaml")
        return issues
    
    try:
        with open("script/llama_factory_sft.yaml") as f:
            sft_config = yaml.safe_load(f)
        print_check("SFT config loaded", True)
    except Exception as e:
        print_check("SFT config loading", False, str(e))
        issues.append("Cannot load llama_factory_sft.yaml")
        return issues
    
    # Check model consistency
    grpo_model = grpo_config.get('model_name_or_path')
    sft_model = sft_config.get('model_name_or_path')
    models_match = grpo_model == sft_model
    print_check(f"Model consistency: {sft_model} -> {grpo_model}", models_match)
    if not models_match:
        issues.append("Model mismatch between SFT and GRPO configs")
    
    # Check LoRA settings consistency  
    grpo_lora_r = grpo_config.get('lora_r')
    sft_lora_r = sft_config.get('lora_rank')
    lora_r_match = grpo_lora_r == sft_lora_r
    print_check(f"LoRA rank consistency: {sft_lora_r} -> {grpo_lora_r}", lora_r_match)
    if not lora_r_match:
        issues.append("LoRA rank mismatch")
    
    grpo_lora_alpha = grpo_config.get('lora_alpha')
    sft_lora_alpha = sft_config.get('lora_alpha')
    lora_alpha_match = grpo_lora_alpha == sft_lora_alpha
    print_check(f"LoRA alpha consistency: {sft_lora_alpha} -> {grpo_lora_alpha}", lora_alpha_match)
    if not lora_alpha_match:
        issues.append("LoRA alpha mismatch")
    
    # Check output directories exist
    grpo_output = grpo_config.get('output_dir')
    sft_output = sft_config.get('output_dir')
    
    if grpo_output:
        grpo_output_parent = Path(grpo_output).parent
        grpo_output_ok = grpo_output_parent.exists()
        print_check(f"GRPO output parent exists: {grpo_output_parent}", grpo_output_ok)
        if not grpo_output_ok:
            issues.append(f"GRPO output parent missing: {grpo_output_parent}")
    
    if sft_output:
        sft_output_parent = Path(sft_output).parent  
        sft_output_ok = sft_output_parent.exists()
        print_check(f"SFT output parent exists: {sft_output_parent}", sft_output_ok)
        if not sft_output_ok:
            issues.append(f"SFT output parent missing: {sft_output_parent}")
    
    # Check save strategy
    save_strategy = sft_config.get('save_strategy')
    save_ok = save_strategy in ['steps', 'epoch']
    print_check(f"SFT save strategy: {save_strategy}", save_ok)
    if not save_ok:
        issues.append("SFT save_strategy must be 'steps' or 'epoch', not 'no'")
    
    return issues

def check_hardware_requirements():
    """Check hardware requirements"""
    print_header("HARDWARE REQUIREMENTS")
    
    issues = []
    
    try:
        import torch
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_ok = gpu_memory >= 20.0  # A800 has ~80GB, minimum ~20GB for 7B model
            print_check(f"GPU memory: {gpu_memory:.1f}GB", memory_ok)
            if not memory_ok:
                issues.append("GPU memory may be insufficient for 7B model training")
        
        # Check system memory
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_kb = int(line.split()[1])
                        mem_gb = mem_kb / (1024**2)
                        mem_ok = mem_gb >= 32.0
                        print_check(f"System RAM: {mem_gb:.1f}GB", mem_ok)
                        if not mem_ok:
                            issues.append("System RAM may be insufficient")
                        break
        except:
            print_check("System RAM check", False, "Cannot check system memory")
        
        # Check disk space
        storage_path = Path("/data22/zhangjunwei")
        if storage_path.exists():
            import shutil
            free_space = shutil.disk_usage(storage_path).free / (1024**3)
            space_ok = free_space >= 50.0  # Need ~50GB for training
            print_check(f"Free storage: {free_space:.1f}GB", space_ok)
            if not space_ok:
                issues.append("Insufficient storage space")
        
    except Exception as e:
        print_check("Hardware checks", False, str(e))
        issues.append("Cannot check hardware requirements")
    
    return issues

def main():
    """Run comprehensive pre-training checklist"""
    print("🚀 KnowRL Comprehensive Pre-Training Checklist")
    print("=" * 60)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_environment())
    all_issues.extend(check_storage_structure())
    all_issues.extend(check_knowledge_base())
    all_issues.extend(check_api_configuration())
    all_issues.extend(check_training_data())
    all_issues.extend(check_configuration_consistency())
    all_issues.extend(check_hardware_requirements())
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    if not all_issues:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your KnowRL training setup is ready!")
        print("\n📋 Next Steps:")
        print("   1. Stage 1 SFT: llamafactory-cli train script/llama_factory_sft.yaml")
        print("   2. Stage 2 RL:  bash train.sh")
        print("   3. Monitor via SwanLab dashboard")
        return 0
    else:
        print("⚠️  ISSUES FOUND - Fix before training:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 Action Required:")
        print("   Fix all issues above before starting training")
        print("   Re-run this checklist until all checks pass")
        return 1

if __name__ == "__main__":
    sys.exit(main())