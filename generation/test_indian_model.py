#!/usr/bin/env python3
"""
Test newly trained Indian RealViz model by generating images from 5 captions
"""

import json
import os
import sys
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
from compel import Compel
from compel.embeddings_provider import ReturnedEmbeddingsType

BASE_MODEL = "/home/dai01/Text_To_Face/sd_training/RealVizXL_Model"
# Use the fine-tuned Indian checkpoint (3 epochs resume from 2700)
CHECKPOINT = "/home/dai01/Text_To_Face/sd_training/outputs_indian_finetuned_ckpt2700/checkpoint-96"
OUTPUT_DIR = "/home/dai01/Text_To_Face/infer_test_realviz_indian_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NEGATIVE_PROMPT = (
    "multiple people, face duplication, duplicated features, "
    "fused facial features, skewed face, smudged face, distorted features, "
    "asymmetrical face, deformed face, disfigured, long face, bad anatomy, "
    "unrealistic, blurry, low detail, low resolution, out of focus, artifact, "
    "poor lighting, unnatural blur"
)

# 5 Indian captions from metadata
captions = [
    "26-year-old Indian male, short black goatee, smooth forehead, receding hairline, brown eyes, average size nose, thin lips, average size ears, no visible tattoos or scars.",
    "23-year-old Indian male, black short straight hair, full beard, wrinkled forehead, receding hairline, brown eyes, medium size slightly curved nose, thin lips, average size ears, no visible tattoos or scars.",
    "23-year-old Indian male, short black goatee, smooth forehead, receding hairline, brown eyes, average size nose, thin lips, average size ears, no visible tattoos or scars.",
    "23-year-old Indian female, black shoulder-length straight hair, clean shaven, wrinkled forehead, receding hairline, brown eyes, small round nose, thin lips, average size ears, no visible tattoos or scars.",
    "22-year-old Indian male, black short straight hair, full beard, smooth forehead, receding hairline, brown eyes, large round nose, medium thick lips, average size ears, no visible tattoos or scars."
]

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading base model from: {BASE_MODEL}")

# Load base SDXL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    safety_checker=None,
)

print(f"[INFO] Loading LoRA weights from: {CHECKPOINT}")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load LoRA weights
pipe.load_lora_weights(CHECKPOINT)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=False)

print("[INFO] Model loaded successfully!")

# Initialize Compel
print("[INFO] Initializing Compel for prompt handling...")
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    requires_pooled=[False, True],
)

print(f"[INFO] Generating {len(captions)} test images...\n")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

for idx, caption in enumerate(captions, 1):
    output_name = f"test_indian_{idx}.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    print(f"[{idx}/{len(captions)}] Generating: {output_name}")
    print(f"    Caption: {caption[:80]}...")
    
    # Parse prompt with Compel
    prompt_embeds, pooled_prompt_embeds = compel(caption)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel(NEGATIVE_PROMPT)
    
    # Generate image
    image = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
        height=320,
        width=320,
    ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"    ✓ Saved to: {output_path}\n")

print(f"[SUCCESS] All test images generated!")
print(f"[INFO] Output directory: {OUTPUT_DIR}")
