#!/bin/bash
#SBATCH -J detect_faces_crop
#SBATCH -A cdac
#SBATCH -p standard
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

################################################################################
#                          >>> EDIT THESE VARIABLES <<<
################################################################################

# Base workspace
export BASE_DIR=/home/dai01/Text_To_Face

# Input directory containing images to process
export INPUT_DIR="$BASE_DIR/raw_images"

# Output directory for cropped faces
export OUTPUT_DIR="$BASE_DIR/cropped_faces"

# Python environment (conda path)
export ENV_DIR=sd_training

# Processing options
export SAVE_ALL_FACES=false         # false: only largest face per image; true: save all faces
export TARGET_SIZE=320               # Target size for cropped faces (square)
export MARGIN=2                      # Margin around face bbox
export CONF_THRESHOLD=0.5            # Confidence threshold for face detection

################################################################################
#                        >>> END OF EDITABLE VARIABLES <<<
################################################################################

set -euo pipefail

echo "=========================================="
echo "Face Detection and Cropping Job"
echo "=========================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $(date)"
echo "=========================================="

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_DIR

echo "[*] Python: $(which python)"
echo "[*] Python version: $(python --version)"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command with arguments
CMD="python detect_faces_and_crop.py \
    --input_dir '$INPUT_DIR' \
    --output_dir '$OUTPUT_DIR' \
    --target_size $TARGET_SIZE \
    --margin $MARGIN \
    --conf_threshold $CONF_THRESHOLD"

if [ "$SAVE_ALL_FACES" = true ]; then
    CMD="$CMD --save_all_faces"
fi

echo "[*] Running: $CMD"
echo ""

# Run the script
$CMD

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
