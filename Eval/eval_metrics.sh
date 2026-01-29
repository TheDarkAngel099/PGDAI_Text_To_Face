#!/bin/bash

#SBATCH --job-name=metrix_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --account=cdac
#SBATCH --output=metrix_eval-%j.out
#SBATCH --error=metrix_eval-%j.err

set -e

# Setup
cd /home/dai01/Text_To_Face
source miniconda_new/bin/activate sd_training

# Configuration
GEN_DIR="/home/dai01/Text_To_Face/eval_generated_detailed"
GT_DIR="/home/dai01/Text_To_Face/vlm_llava/project_results/val/lora_dataset/images"
OUTPUT_DIR="/home/dai01/Text_To_Face/metrics_results"

# Metrics parameters
CLIP_MODEL="ViT-B-16"
CLIP_PRETRAINED="openai"
RESIZE=256
DEVICE="cuda"

echo "=========================================="
echo "Running Metrics Evaluation (metrix.py)"
echo "=========================================="
echo "Generated Images: $GEN_DIR"
echo "Ground Truth: $GT_DIR"
echo "Output: $OUTPUT_DIR"
echo "CLIP Model: $CLIP_MODEL ($CLIP_PRETRAINED)"
echo "Resize: $RESIZE"
echo "Device: $DEVICE"
echo ""

python vlm_llava/metrix.py \
    --gen_dir "$GEN_DIR" \
    --gt_dir "$GT_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --clip_model "$CLIP_MODEL" \
    --clip_pretrained "$CLIP_PRETRAINED" \
    --resize $RESIZE \
    --device $DEVICE \
    --image_exts ".png,.jpg,.jpeg,.webp"

echo ""
echo "=========================================="
echo "Metrics evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
