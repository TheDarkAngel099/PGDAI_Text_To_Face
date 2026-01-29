#!/usr/bin/env python3
"""
Generate images using the trained Stable Diffusion 1.5 LoRA model and captions from test_captions.jsonl
"""

import json
import os
import sys
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from compel import Compel

# ======================== Configuration ========================
BASE_MODEL = "/home/dai01/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CHECKPOINT_DIR = "/home/dai01/Text_To_Face/sd_training/sd1.5/outputs_l320_5000steps"
TEST_CAPTIONS = "/home/dai01/Text_To_Face/infer_test/test_captions.jsonl"
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
parser = argparse.ArgumentParser(description="Generate images using trained Stable Diffusion 1.5 LoRA model")
parser.add_argument("checkpoint", type=str, 
                    help="Checkpoint to use (e.g., checkpoint-500, checkpoint-1000, checkpoint-5000)")
parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                    help=f"Path to checkpoints directory (default: {CHECKPOINT_DIR})")
parser.add_argument("--captions-file", type=str, default=TEST_CAPTIONS,
                    help=f"Path to JSONL captions file (default: {TEST_CAPTIONS})")
parser.add_argument("--height", type=int, default=320,
                    help="Height of generated image (default: 320)")
parser.add_argument("--width", type=int, default=320,
                    help="Width of generated image (default: 320)")
parser.add_argument("--steps", type=int, default=50,
                    help="Number of inference steps (default: 50)")
parser.add_argument("--guidance", type=float, default=7.5,
                    help="Guidance scale (default: 7.5)")
parser.add_argument("--negative-prompt", type=str, default=NEGATIVE_PROMPT,
                    help="Negative prompt to avoid in generation")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for generated images (default: generated_images/<checkpoint>)")
parser.add_argument("--max-images", type=int, default=None,
                    help="Maximum number of images to generate (default: all)")
args = parser.parse_args()

# Validate checkpoint was provided
if not args.checkpoint:
    print("[ERROR] Please specify a checkpoint: python3 generate_images_from_captions_sd15.py checkpoint-5000")
    sys.exit(1)

# Validate captions file
if not os.path.isfile(args.captions_file):
    print(f"[ERROR] Captions file not found: {args.captions_file}")
    sys.exit(1)

# Update checkpoint directory if needed
if args.checkpoint_dir != CHECKPOINT_DIR:
    CHECKPOINT_DIR = args.checkpoint_dir

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading base model from: {BASE_MODEL}")

# ======================== Load Model ========================
# Load base SD 1.5 model
pipe = StableDiffusionPipeline.from_pretrained(
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

# Create output directory - use custom path if provided, otherwise use checkpoint-based path
if args.output_dir:
    OUTPUT_DIR = args.output_dir
else:
    OUTPUT_DIR = os.path.join("/home/dai01/Text_To_Face/infer_test", f"generated_sd15_{args.checkpoint}")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load LoRA weights
# For SD 1.5, LoRA weights are loaded into the UNet
pipe.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

print("[INFO] Model loaded successfully!")

# Initialize Compel for handling long prompts
print("[INFO] Initializing Compel for prompt handling...")
compel = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
)

# ======================== Load Captions from JSONL ========================
captions = []
with open(args.captions_file, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            captions.append(data)

if not captions:
    print(f"[ERROR] No captions found in {args.captions_file}")
    sys.exit(1)

print(f"[INFO] Loaded {len(captions)} captions from {args.captions_file}")

# ======================== Generate Images ========================
print(f"[INFO] Starting image generation. Output will be saved to: {OUTPUT_DIR}")

# Limit captions if max_images is specified
captions_to_process = captions
if args.max_images and args.max_images > 0:
    captions_to_process = captions[:args.max_images]
    print(f"[INFO] Limiting to {args.max_images} images")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

for idx, caption_data in enumerate(captions_to_process, 1):
    text = caption_data.get('text', '')
    image_name = caption_data.get('image', f'image_{idx}')
    
    if not text:
        print(f"\n[{idx}/{len(captions_to_process)}] Skipping (empty caption)")
        continue
    
    # Extract filename without path
    image_base = os.path.basename(image_name)
    # Ensure .jpg extension
    if not image_base.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_base = f"{image_base}.jpg"
    output_name = f"generated_{image_base}"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    print(f"\n[{idx}/{len(captions_to_process)}] Generating: {output_name}")
    print(f"    Caption: {text[:100]}...")
    
    # Parse prompt with Compel (handles long prompts and weighting)
    prompt_embeds = compel(text)
    negative_prompt_embeds = compel(args.negative_prompt)
    
    # Generate image
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        height=args.height,
        width=args.width,
    ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"    ✓ Saved to: {output_path}")

print(f"\n[SUCCESS] All images generated! Total: {len(captions_to_process)} images")
print(f"[INFO] Images saved in: {OUTPUT_DIR}")
