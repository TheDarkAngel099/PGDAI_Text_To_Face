#!/bin/bash
#SBATCH -J sd15_lora_train_320
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

# Path to SD 1.5 cached model
export MODEL_PATH="/home/dai01/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# Path to your dataset root with images/, captions/, metadata.jsonl
export DATASET_DIR="$BASE_DIR/vlm_llava/project_results/lora_dataset"

# Path to validation images for evaluation during training
# export VALIDATION_IMAGES_DIR="$BASE_DIR/vlm_llava/dataset_split/val/images"
# export VALIDATION_CAPTIONS_DIR="$BASE_DIR/vlm_llava/dataset_split/val/captions"

# Python environment (conda path)
export ENV_DIR=sd_training

# Where to put outputs and checkpoints
export OUT_DIR="$BASE_DIR/sd_training/sd1.5/outputs_l320_5000steps"
export CKPT_DIR="$BASE_DIR/sd_training/sd1.5/checkpoints_l320_5000steps"

# Training hyperparameters for SD 1.5
export RESOLUTION=320                  # 320x320 resolution
export BATCH_PER_GPU=16                # Batch size per GPU
export GRAD_ACCUM=1                    # Gradient accumulation steps
export LEARNING_RATE=5e-5              # Learning rate
export LR_SCHEDULER="constant"         # Constant learning rate (no decay)
export LR_WARMUP_STEPS=0               # No warmup
export MAX_TRAIN_STEPS=5000            # Maximum training steps
# export VALIDATION_STEPS=250            # Evaluate on validation set every 250 steps
export CHECKPOINT_STEPS=500            # Save checkpoint every 500 steps
export RANK=4                          # LoRA rank
export SEED=42                         # Random seed

# Validation prompt
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

# Activate your Python env
conda activate "$ENV_DIR"

echo "[INFO] Python environment activated"

# HF cache
export HF_CACHE="$BASE_DIR/cache/huggingface"
mkdir -p "$HF_CACHE" "$OUT_DIR" "$CKPT_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

# ------------------------------ Sanity Checks ---------------------------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "[ERROR] SD 1.5 model not found at: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$DATASET_DIR/metadata.jsonl" ]; then
  echo "[ERROR] metadata.jsonl not found under: $DATASET_DIR"
  echo "        Make sure your dataset has images/, captions/ and run create_metadata.py beforehand."
  exit 1
fi

# Check validation dataset
# if [ ! -d "$VALIDATION_IMAGES_DIR" ]; then
#   echo "[WARNING] VALIDATION_IMAGES_DIR does not exist: $VALIDATION_IMAGES_DIR"
#   echo "          Validation on actual images will be skipped. Create the validation split using split_dataset.py"
#   export VALIDATION_IMAGES_DIR=""
# fi
#
# if [ ! -d "$VALIDATION_CAPTIONS_DIR" ]; then
#   echo "[WARNING] VALIDATION_CAPTIONS_DIR does not exist: $VALIDATION_CAPTIONS_DIR"
#   echo "          Validation captions will not be used."
#   export VALIDATION_CAPTIONS_DIR=""
# fi

# ------------------------------ Training Script Check -------------------------
TRAIN_SCRIPT="$BASE_DIR/sd_training/sd1.5/train_text_to_image_lora_sd15.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "[ERROR] Training script not found at: $TRAIN_SCRIPT"
  echo "        Download it before submitting the job (compute nodes have no internet)."
  exit 1
fi

# ------------------------------ Launch Training -------------------------------
echo "[INFO] Launching SD 1.5 LORA training with Accelerate..."
cd "$BASE_DIR/sd_training/sd1.5"

accelerate launch --mixed_precision=no --num_processes=1 --num_machines=1 \
  "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path "$MODEL_PATH" \
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
  --seed "$SEED" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --rank "$RANK" \
  --validation_prompt "$VAL_PROMPT" \
  --num_validation_images 4 \
  --center_crop \
  --report_to tensorboard \
  --output_dir "$OUT_DIR"

echo "[INFO] Training completed."
date
