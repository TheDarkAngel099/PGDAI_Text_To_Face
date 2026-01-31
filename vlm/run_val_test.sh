#!/bin/bash
#SBATCH --job-name=vlm_process_val_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=vlm_output_%j.log

# 1. Load Environment
source /home/dai01/Text_To_Face/miniconda_new/bin/activate

# 2. Go to Directory
cd $SLURM_SUBMIT_DIR

echo "Starting Processing.."
echo "Using local model cache: ./model_cache"

# ==========================================
# PHASE 1: PROCESS TRAINING SET
# ==========================================
echo "----------------------------------------"
echo "Processing TRAINING Set..."
echo "----------------------------------------"

python vlm_pipeline.py \
  --input_dir ./dataset_split/train/images \
  --output_dir ./project_results/train \
  --csv_file ./dataset_split/train/train.csv \
  --cache_dir "./model_cache"

# ==========================================
# PHASE 2: PROCESS VALIDATION SET
# ==========================================
echo "----------------------------------------"
echo "Processing VALIDATION Set..."
echo "----------------------------------------"

python vlm_pipeline.py \
  --input_dir ./dataset_split/val/images \
  --output_dir ./project_results/val \
  --csv_file ./dataset_split/val/val.csv \
  --cache_dir "./model_cache"

# ==========================================
# PHASE 3: PROCESS TEST SET
# ==========================================
echo "----------------------------------------"
echo "Processing TEST Set..."
echo "----------------------------------------"

python vlm_pipeline.py \
  --input_dir ./dataset_split/test/images \
  --output_dir ./project_results/test \
  --csv_file ./dataset_split/test/test.csv \
  --cache_dir "./model_cache"

echo "All Jobs Complete!"
