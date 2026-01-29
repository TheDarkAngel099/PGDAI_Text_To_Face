#!/usr/bin/env python3
"""
Generate images using base RealVizXL model with detailed captions
"""

import os
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from compel import Compel

# Configuration
CAPTIONS_DIR = Path("/home/dai01/Text_To_Face/ground_truth/detailed_captions")
OUTPUT_DIR = Path("/home/dai01/Text_To_Face/ground_truth/generated_images_base_model_with_mugshot_context")
MODEL_ID = "/home/dai01/Text_To_Face/sd_training/RealVizXL_Model"
RESOLUTION = 320
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
SEED = 42

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set seed for reproducibility
torch.manual_seed(SEED)

# Load pipeline
print(f"Loading base model: {MODEL_ID}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe = pipe.to(device)

# Initialize Compel for long prompt handling
print("Initializing Compel for long prompt handling...")
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    requires_pooled=[False, True]
)

# Get all caption files
caption_files = sorted(CAPTIONS_DIR.glob("*_detailed.txt"))
print(f"Found {len(caption_files)} caption files")

# Generate images
generated_count = 0
failed_count = 0
metadata = []

for idx, caption_file in enumerate(caption_files, 1):
    try:
        # Read caption
        with open(caption_file, "r") as f:
            prompt = f.read().strip()
        
        # Add mugshot context to guide the generation
        prompt_with_context = f"A professional forensic mugshot photograph of a person with: {prompt}"
        
        image_stem = caption_file.stem.replace("_detailed", "")
        output_path = OUTPUT_DIR / f"{image_stem}.png"
        
        print(f"[{idx}/{len(caption_files)}] Generating {image_stem}...")
        
        # Generate with Compel
        prompt_embeds, pooled_prompt_embeds = compel(prompt_with_context)
        
        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device=device).manual_seed(SEED)
        ).images[0]
        
        image.save(output_path)
        generated_count += 1
        print(f"  ✓ Saved")
        
        metadata.append({
            "image": image_stem,
            "status": "success"
        })
        
    except Exception as e:
        failed_count += 1
        print(f"  ✗ ERROR: {str(e)}")
        metadata.append({
            "image": image_stem,
            "status": "failed",
            "error": str(e)
        })

# Save metadata
metadata_file = OUTPUT_DIR / "generation_metadata.json"
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"Generation Complete!")
print(f"Generated: {generated_count}/{len(caption_files)}")
print(f"Failed: {failed_count}/{len(caption_files)}")
print(f"Output: {OUTPUT_DIR}")
print(f"Metadata: {metadata_file}")
print(f"{'='*60}")
