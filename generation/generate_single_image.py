#!/usr/bin/env python3
"""
Generate a single image from a given prompt using the trained RealViz LoRA model
"""

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
CHECKPOINT_DIR = "/home/dai01/Text_To_Face/sd_training/outputs_l320_3epochs"
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
parser = argparse.ArgumentParser(description="Generate a single image from a prompt using trained LoRA model")
parser.add_argument("prompt", type=str, 
                    help="The text prompt to generate an image from")
parser.add_argument("--checkpoint", type=str, default="checkpoint-2700",
                    help="Checkpoint to use (default: checkpoint-2700)")
parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                    help=f"Path to checkpoints directory (default: {CHECKPOINT_DIR})")
parser.add_argument("--height", type=int, default=320,
                    help="Height of generated image (default: 320)")
parser.add_argument("--width", type=int, default=320,
                    help="Width of generated image (default: 320)")
parser.add_argument("--steps", type=int, default=50,
                    help="Number of inference steps (default: 50)")
parser.add_argument("--guidance", type=float, default=7.5,
                    help="Guidance scale (default: 7.5)")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for generated image (default: generated_images/single_image)")
parser.add_argument("--output-name", type=str, default="generated_image.png",
                    help="Output filename (default: generated_image.png)")
args = parser.parse_args()

# Validate prompt
if not args.prompt or not args.prompt.strip():
    print("[ERROR] Please provide a prompt")
    sys.exit(1)

# Update checkpoint directory if needed
if args.checkpoint_dir != CHECKPOINT_DIR:
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

# Create output directory
if args.output_dir:
    OUTPUT_DIR = args.output_dir
else:
    OUTPUT_DIR = "/home/dai01/Text_To_Face/generated_images/single_image"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load LoRA weights
pipe.load_lora_weights(checkpoint_path)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

print("[INFO] Model loaded successfully!")

# Initialize Compel for handling long prompts
print("[INFO] Initializing Compel for prompt handling...")
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    requires_pooled=[False, True],
)

# ======================== Generate Image ========================
print(f"[INFO] Generating image...")
print(f"[INFO] Prompt: {args.prompt}")
print(f"[INFO] Output will be saved to: {OUTPUT_DIR}")

output_path = os.path.join(OUTPUT_DIR, args.output_name)

# Parse prompt with Compel (handles long prompts and weighting)
prompt_embeds, pooled_prompt_embeds = compel(args.prompt)
negative_prompt_embeds, negative_pooled_prompt_embeds = compel(NEGATIVE_PROMPT)

# Generate image
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
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
print(f"[SUCCESS] Image generated and saved to: {output_path}")
