#!/usr/bin/env python3
"""
Generate images using the trained RealViz LoRA model and captions from test_captions.jsonl
"""

import json
import os
import sys
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from compel import Compel
from compel.embeddings_provider import ReturnedEmbeddingsType

# ======================== Configuration ========================
BASE_MODEL = "/home/dai01/Text_To_Face/sd_training/RealVizXL_Model"
CHECKPOINT_DIR = "/home/dai01/Text_To_Face/sd_training/outputs"
TEST_CAPTIONS = "/home/dai01/Text_To_Face/PGDAI_Text_To_Face/tests/orignal images/test_captions.jsonl"
OUTPUT_DIR = "/home/dai01/Text_To_Face/PGDAI_Text_To_Face/tests/generated_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Negative prompt to avoid common artifacts
NEGATIVE_PROMPT = (
    "multiple people, face duplication, duplicated features, "
    "fused facial features, skewed face, smudged face, distorted features, "
    "asymmetrical face, deformed face, disfigured, long face, bad anatomy, "
    "unrealistic, blurry, low detail, low resolution, out of focus, artifact, "
    "poor lighting, unnatural blur"
)

# Parse command-line arguments
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate images using trained LoRA model")
parser.add_argument("checkpoint", type=str, 
                    help="Checkpoint to use (e.g., checkpoint-450, checkpoint-1800)")
parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                    help=f"Path to checkpoints directory (default: {CHECKPOINT_DIR})")
parser.add_argument("--new-epoch", action="store_true",
                    help="Use checkpoints from outputs_320_1epoch instead of outputs")
parser.add_argument("--height", type=int, default=512,
                    help="Height of generated image (default: 512)")
parser.add_argument("--width", type=int, default=512,
                    help="Width of generated image (default: 512)")
parser.add_argument("--steps", type=int, default=50,
                    help="Number of inference steps (default: 50)")
parser.add_argument("--guidance", type=float, default=7.5,
                    help="Guidance scale (default: 7.5)")
args = parser.parse_args()

# Validate checkpoint was provided
if not args.checkpoint:
    print("[ERROR] Please specify a checkpoint: python3 generate_images_from_captions.py checkpoint-1800")
    sys.exit(1)

# Update checkpoint directory if requested
if args.new_epoch:
    CHECKPOINT_DIR = "/home/dai01/Text_To_Face/sd_training/outputs_320_1epoch"
elif args.checkpoint_dir != CHECKPOINT_DIR:
    CHECKPOINT_DIR = args.checkpoint_dir

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading base model from: {BASE_MODEL}")

# ======================== Load Model ========================
# Load base SDXL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    safety_checker=None,
)

# Find and validate checkpoint
checkpoints = sorted([d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")])
if not checkpoints:
    raise ValueError(f"No checkpoints found in {CHECKPOINT_DIR}")

if args.checkpoint not in checkpoints:
    print(f"[ERROR] Checkpoint '{args.checkpoint}' not found!")
    print(f"[ERROR] Available checkpoints: {', '.join(checkpoints)}")
    sys.exit(1)

checkpoint_path = os.path.join(CHECKPOINT_DIR, args.checkpoint)
print(f"[INFO] Loading LoRA weights from: {checkpoint_path}")

# Create output directory with checkpoint name under workspace root
OUTPUT_DIR = os.path.join("/home/dai01/Text_To_Face/generated_images", args.checkpoint)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load LoRA weights using updated method
pipe.load_lora_weights(checkpoint_path)

pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

print("[INFO] Model loaded successfully!")

# Initialize Compel for handling long prompts
print("[INFO] Initializing Compel for prompt handling...")
# Explicitly set embeddings type compatible with current Compel version
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    requires_pooled=[False, True],
)

# ======================== Load Captions ========================
captions = []
with open(TEST_CAPTIONS, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            captions.append(data)

print(f"[INFO] Loaded {len(captions)} captions from {TEST_CAPTIONS}")

# ======================== Generate Images ========================
print(f"[INFO] Starting image generation. Output will be saved to: {OUTPUT_DIR}")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

for idx, caption_data in enumerate(captions, 1):
    text = caption_data.get('text', '')
    image_name = caption_data.get('image', f'image_{idx}')
    
    # Extract filename without path
    image_base = os.path.basename(image_name)
    output_name = f"generated_{image_base}"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    print(f"\n[{idx}/{len(captions)}] Generating: {output_name}")
    print(f"    Caption: {text[:100]}...")
    
    # Parse prompt with Compel (handles long prompts and weighting)
    prompt_embeds, pooled_prompt_embeds = compel(text)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(NEGATIVE_PROMPT)
    
    # Generate image
    image = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        height=args.height,
        width=args.width,
    ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"    ✓ Saved to: {output_path}")

print(f"\n[SUCCESS] All images generated! Total: {len(captions)} images")
print(f"[INFO] Images saved in: {OUTPUT_DIR}")
