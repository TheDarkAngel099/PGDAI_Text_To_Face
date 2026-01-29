#!/usr/bin/env python3
"""
Download LLaVA 1.5 model from HuggingFace
"""

import os
import sys
from pathlib import Path

# Set up environment
MODEL_CACHE_DIR = "/home/dai01/Text_To_Face/vlm_llava/model_cache"
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache directory
os.environ['HF_HOME'] = MODEL_CACHE_DIR

print(f"[INFO] Setting HuggingFace cache to: {MODEL_CACHE_DIR}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    
    # LLaVA 1.5 13B model ID
    # MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    MODEL_ID = "llava-hf/llava-1.5-13b-hf"
    
    print(f"[INFO] Downloading {MODEL_ID}...")
    print(f"[INFO] This may take a while (model is ~26GB)...")
    
    # Download the model
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        resume_download=True
    )
    
    print(f"[SUCCESS] Model downloaded to: {model_path}")
    
except ImportError as e:
    print(f"[ERROR] Missing required packages: {e}")
    print("[INFO] Installing required packages...")
    os.system("pip install --quiet transformers huggingface_hub torch")
    print("[INFO] Please run the script again.")
    sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Failed to download model: {e}")
    sys.exit(1)
