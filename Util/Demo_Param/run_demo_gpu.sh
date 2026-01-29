#!/bin/bash

#SBATCH --job-name=demo_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=cdac
#SBATCH --output=demo_gen-%j.out
#SBATCH --error=demo_gen-%j.err

set -e

# Setup
cd /home/dai01/Text_To_Face
source miniconda_new/bin/activate sd_training

# Configuration
BASE_MODEL="/home/dai01/Text_To_Face/sd_training/RealVizXL_Model"
CHECKPOINT="/home/dai01/Text_To_Face/sd_training/outputs_indian_finetuned_ckpt2700/checkpoint-2700"
OUTPUT_DIR="/home/dai01/Text_To_Face/demo/demo_result"

# Your prompt here (EDIT THIS)
PROMPT="a 45-year-old Indian male with black hair, almond shaped brown eyes, full beard, manbun hairstyle, stubbled face, olive skin tone"

# Generation parameters
HEIGHT=320
WIDTH=320
NUM_STEPS=50
GUIDANCE_SCALE=7.5
SEED=42

echo "=========================================="
echo "RealViz Demo - Single Image Generation"
echo "=========================================="
echo "Prompt: $PROMPT"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Steps: $NUM_STEPS"
echo ""

python demo/generate_demo.py \
    --base_model "$BASE_MODEL" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --prompt "$PROMPT" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --seed $SEED \
    --device cuda

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Output saved to: $OUTPUT_DIR/demo_output.png"
echo "=========================================="
