#!/usr/bin/env python3

"""
Script to download the official Stable Diffusion 1.5 LORA training script
from the Hugging Face Diffusers repository.

This fetches the latest training script from:
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

Run with:
    python download_sd15_training_script.py
"""

import os
import urllib.request
from pathlib import Path

# --------------------------------------------------------
# Script configuration
# --------------------------------------------------------

SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py"
OUTPUT_FILE = "./train_text_to_image_lora_sd15.py"


# --------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# --------------------------------------------------------

def main():
    print(f"[INFO] Downloading SD 1.5 LORA training script...")
    print(f"[INFO] Source: {SCRIPT_URL}")
    
    output_path = Path(OUTPUT_FILE).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"[INFO] Downloading...")
        urllib.request.urlretrieve(SCRIPT_URL, output_path)
        
        # Make it executable
        os.chmod(output_path, 0o755)
        
        print(f"[SUCCESS] Script saved at: {output_path}")
        print(f"\nUsage:")
        print(f"  accelerate launch {output_path} \\")
        print(f"    --pretrained_model_name_or_path ./sd1_5_model \\")
        print(f"    --dataset_name <dataset> \\")
        print(f"    --output_dir ./sd15_lora_output \\")
        print(f"    --rank 4")
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

if __name__ == "__main__":
    main()
