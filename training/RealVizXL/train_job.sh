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

# Path saved the model (downloaded earlier)
export MODEL_DIR="$BASE_DIR/sd_training/RealVizXL_Model"

# Path to your dataset root with images/, captions/, metadata.jsonl
export DATASET_DIR="$BASE_DIR/vlm_llava/project_results/lora_dataset"

# Python environment (conda path)
export ENV_DIR=sd_training

# Where to put outputs and checkpoints
export OUT_DIR="$BASE_DIR/sd_training/outputs_l320_3epochs_new"
export CKPT_DIR="$BASE_DIR/sd_training/checkpoints_l320_3epochs_new"

# Training hyperparameters for RealViz
export RESOLUTION=320                  # 320x320 (native dataset resolution, no upscaling)
export BATCH_PER_GPU=4                 # Keep batch_size=4 for 320 x 320
export GRAD_ACCUM=2                    # Match report's grad_accum=2 (effective batch = 4 * 2 = 8)
export LEARNING_RATE=1e-4              # Match report's LR
export LR_SCHEDULER="linear"           # Linear decay scheduler for refined convergence
export LR_WARMUP_STEPS=100             # Brief warmup for stability
export MAX_TRAIN_STEPS=2715            # Three complete epochs: 7238 images / (batch_size=4 × grad_accum=2) × 3 = 2715 steps
export VALIDATION_STEPS=225            # Validate every ~1/4 epoch
export CHECKPOINT_STEPS=450            # Save checkpoint every half-epoch (~452 steps)
export CHECKPOINTS_TOTAL_LIMIT=10      # Save all checkpoints (6 total expected)
export RANK=4                          # LoRA rank (dimension of update matrices)
export SEED=42

# Validation prompt (quick sanity check) - detailed forensic format matching training data
export VAL_PROMPT="25-year-old Indian male, short black hair, full beard, smooth forehead, brown eyes, medium size nose, thin lips, average size ears, no visible tattoos or scars"

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


# The official Diffusers SDXL LoRA script accepts JSONL datasets with columns:



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
  --rank "$RANK" \
  --validation_prompt "$VAL_PROMPT" \
  --num_validation_images 4 \
  --center_crop \
  --report_to tensorboard \
  --output_dir "$OUT_DIR"