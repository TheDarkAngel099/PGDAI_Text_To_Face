#!/bin/bash
#SBATCH -J realviz_lora_train
#SBATCH -A cdac
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

################################################################################
#                          >>> EDIT THESE VARIABLES <<<
################################################################################

# Base workspace
export BASE_DIR=/home/dai01/Text_To_Face

# Path where you saved the model (downloaded earlier)
export MODEL_DIR="$BASE_DIR/sd_training/RealVizXL_Model"

# Path to your dataset root with images/, captions/, metadata.jsonl
export DATASET_DIR="$BASE_DIR/vlm_llava/project_results/lora_dataset"

# Python environment (conda path)
export ENV_DIR=sd_training

# Where to put outputs and checkpoints
export OUT_DIR="$BASE_DIR/sd_training/outputs_512_2epochs"
export CKPT_DIR="$BASE_DIR/sd_training/checkpoints_512_2epochs"

# Training hyperparameters for RealViz
export RESOLUTION=512                  # 512x512 (dataset will be upscaled)
export BATCH_PER_GPU=4                 # Keep batch_size=4 for 512x512
export GRAD_ACCUM=2                    # Match report's grad_accum=2 (effective batch = 4 * 2 = 8)
export LEARNING_RATE=1e-4              # Match report's LR
export LR_SCHEDULER="constant"        # Match report: use constant scheduler
export LR_WARMUP_STEPS=100             # Brief warmup for stability
export MAX_TRAIN_STEPS=1810            # Two complete epochs: 7238 images / (batch_size=4 × grad_accum=2) × 2 = 1810 steps
export VALIDATION_STEPS=225            # Validate every ~1/8 epoch
export CHECKPOINT_STEPS=450            # Save 4 checkpoints (every half-epoch: 1810/4 = 452)
export CHECKPOINTS_TOTAL_LIMIT=4       # Keep 4 checkpoints for comparison
export SEED=42

# Validation prompt (quick sanity check)
export VAL_PROMPT="a person, frontal mugshot view, centered face, neutral expression, plain white background, high quality, detailed"

################################################################################
#                        >>> END OF EDITABLE VARIABLES <<<
################################################################################

set -euo pipefail

echo "[INFO] Job starting on: $(hostname)"
date

# ------------------------------ Modules & Env ---------------------------------
# Initialize conda from local installation
source /home/dai01/Text_To_Face/miniconda_new/bin/activate

# Activate your Python env (created previously)
conda activate "$ENV_DIR"

echo "[INFO] Using existing dependencies from conda environment."

# HF cache (optional but recommended for speed)
export HF_CACHE="$BASE_DIR/cache/huggingface"
mkdir -p "$HF_CACHE" "$OUT_DIR" "$CKPT_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

# -------------------------- Training Script Staging ---------------------------
# Training script is in sd_training folder
export CODE_DIR="$BASE_DIR/sd_training"
if [ ! -f "$CODE_DIR/train_text_to_image_lora_sdxl.py" ]; then
  echo "[ERROR] Training script not found at: $CODE_DIR/train_text_to_image_lora_sdxl.py"
  echo "        Please ensure the script is present in the sd_training directory."
  exit 1
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

accelerate launch --mixed_precision=no --num_processes=1 --num_machines=1 \
  "$CODE_DIR/train_text_to_image_lora_sdxl.py" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --train_data_dir "$DATASET_DIR" \
  --caption_column "text" \
  --image_column "image" \
  --resolution "$RESOLUTION" \
  --train_batch_size "$BATCH_PER_GPU" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler "$LR_SCHEDULER" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --checkpointing_steps "$CHECKPOINT_STEPS" \
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT" \
  --resume_from_checkpoint "latest" \
  --seed "$SEED" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --validation_prompt "$VAL_PROMPT" \
  --num_validation_images 4 \
  --center_crop \
  --random_flip \
  --report_to tensorboard \
  --output_dir "$OUT_DIR"