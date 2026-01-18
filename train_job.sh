
#!/bin/bash
#SBATCH -J realvizxl_lora_train
#SBATCH -A <YOUR_ACCOUNT>            # TODO: Param project/account
#SBATCH -p <YOUR_GPU_PARTITION>      # TODO: e.g., a100q / gpuq
#SBATCH -N 1
#SBATCH --gpus-per-node=4            # Change to 1/2/4/8 as needed
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

################################################################################
#                          >>> EDIT THESE VARIABLES <<<
################################################################################

# Base workspace (prefer fast scratch if available on Param)
export BASE_DIR=${SCRATCH:-/scratch/$USER}/realviz_xl_project

# Path where you saved the model once (downloaded earlier)
export MODEL_DIR="$BASE_DIR/cache/models/RealVisXL_V4.0"   # e.g., /scratch/$USER/models/RealVisXL_V4.0

# Path to your dataset root with images/, captions/, metadata.jsonl
export DATASET_DIR="$BASE_DIR/data/lora_dataset"

# Python environment (conda path)
export ENV_DIR="$BASE_DIR/env/py310"

# Where to put outputs and checkpoints
export OUT_DIR="$BASE_DIR/outputs"
export CKPT_DIR="$BASE_DIR/checkpoints"

# Training hyperparameters
export RESOLUTION=512                  # SDXL LoRA common setting; increase later if desired
export BATCH_PER_GPU=2                 # Adjust per A100 VRAM; try 2–4 at 512
export GRAD_ACCUM=2                    # Effective batch = gpus * BATCH_PER_GPU * GRAD_ACCUM
export LEARNING_RATE=5e-5
export MAX_TRAIN_STEPS=5000
export CHECKPOINT_STEPS=500
export SEED=42

# Validation prompt (quick sanity check)
export VAL_PROMPT="portrait photo, centered mugshot, neutral expression"

################################################################################
#                        >>> END OF EDITABLE VARIABLES <<<
################################################################################

set -euo pipefail

echo "[INFO] Job starting on: $(hostname)"
date

# ------------------------------ Modules & Env ---------------------------------
module purge
module load cuda/12.1        # Adjust if Param provides a different CUDA module
module load anaconda         # Or the module name Param uses for conda

# Activate your Python env (created previously)
# If you need to create it the first time:
# conda create -y -p "$ENV_DIR" python=3.10
# conda activate "$ENV_DIR"
# pip install --upgrade pip
# pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# pip install diffusers[torch] transformers accelerate datasets xformers safetensors pillow tqdm peft huggingface_hub
# pip install lpips scikit-image tensorboard
source activate "$ENV_DIR" 2>/dev/null || conda activate "$ENV_DIR"

# HF cache (optional but recommended for speed)
export HF_CACHE="$BASE_DIR/cache/huggingface"
mkdir -p "$HF_CACHE" "$OUT_DIR" "$CKPT_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

# -------------------------- Training Script Staging ---------------------------
# If you don't already have the SDXL LoRA trainer, try to fetch it now.
# (Replace with a local copy if your compute nodes have no internet.)
export CODE_DIR="$BASE_DIR/code"
mkdir -p "$CODE_DIR"
if [ ! -f "$CODE_DIR/train_text_to_image_lora_sdxl.py" ]; then
  echo "[INFO] SDXL LoRA training script not found; attempting to download..."
  if [ -f "$CODE_DIR/sdxl_script_download.py" ]; then
    python "$CODE_DIR/sdxl_script_download.py" || {
      echo "[ERROR] Could not fetch training script. Pre-stage it into $CODE_DIR."
      exit 1
    }
  else
    echo "[ERROR] sdxl_script_download.py not present at $CODE_DIR. Please copy it or the trainer script."
    exit 1
  fi
fi

# ------------------------------ Sanity Checks ---------------------------------
if [ ! -d "$MODEL_DIR" ]; then
  echo "[ERROR] MODEL_DIR does not exist: $MODEL_DIR"
  echo "        Download the model once (e.g., using download_realvizxl_static.py) and update MODEL_DIR."
  exit 1
fi

if [ ! -f "$DATASET_DIR/metadata.jsonl" ]; then
  echo "[ERROR] metadata.jsonl not found under: $DATASET_DIR"
  echo "        Make sure your dataset has images/, captions/ and run create_metadata.py beforehand."
  exit 1
fi

# ------------------------------ Accelerate Config -----------------------------
# Create a default accelerate config if not present.
ACCEL_CONFIG="$BASE_DIR/accelerate/default_config.yaml"
if [ ! -f "$ACCEL_CONFIG" ]; then
  echo "[INFO] Creating default accelerate config..."
  accelerate config default
fi

# ------------------------------ Launch Training -------------------------------
echo "[INFO] Launching training with Accelerate..."
cd "$CODE_DIR"

# Notes:
# - The official Diffusers SDXL LoRA script accepts JSONL datasets with columns:
#   image path (relative from root) and text caption, which your metadata generator provides. [1](blob:https://m365.cloud.microsoft/fb3a56ef-1cd9-419f-8b35-2cc1eee10e6f)
# - SDXL LoRA at 512px is a common and resource-friendly setting; the capstone used Accelerate + fp16 + LoRA rank=4
#   and checkpointed every ~500 steps for early validation. [3](blob:https://m365.cloud.microsoft/ddedcaa1-41b6-4d14-a854-0c5f22fb8634)

accelerate launch --mixed_precision=fp16 --num_processes=auto --num_machines=1 \
  "$CODE_DIR/train_text_to_image_lora_sdxl.py" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --train_data_dir "$DATASET_DIR" \
  --caption_column "text" \
  --image_column "file_name" \
  --resolution "$RESOLUTION" \
  --train_batch_size "$BATCH_PER_GPU" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler "constant" \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --checkpointing_steps "$CHECKPOINT_STEPS" \
  --resume_from_checkpoint "latest" \
  --seed "$SEED" \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision fp16 \
  --report_to "none" \
  --validation_prompt "$VAL_PROMPT" \
  --num_validation_images 4 \
  --validation_epochs 1 \
  --checkpoint_output_dir "$CKPT_DIR" \
  --output_dir "$OUT_DIR"

echo "[INFO] Training completed."
date
``
