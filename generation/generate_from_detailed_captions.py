#!/usr/bin/env python3

"""
Script to generate images from detailed captions using fine-tuned model.

Reads detailed captions from .txt files, uses Compel for prompt handling,
and generates images with matching filenames for metrics evaluation.

Usage:
    python generate_from_detailed_captions.py \
        --captions_dir /path/to/detailed_captions \
        --images_dir /path/to/original_images \
        --checkpoint /path/to/checkpoint \
        --output_dir /path/to/output \
        --num_inference_steps 50 \
        --guidance_scale 7.5
"""

import argparse
import os
from pathlib import Path
import json
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from compel import Compel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from detailed captions using fine-tuned model"
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        required=True,
        help="Directory containing detailed caption .txt files"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing original images (for reference)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint (e.g., checkpoint-96)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/dai01/Text_To_Face/sd_training/RealVizXL_Model",
        help="Path to base model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (default: 7.5)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Image height (default: 320)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Image width (default: 320)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images to generate (0 = all)"
    )
    
    return parser.parse_args()


def load_caption(caption_path: Path) -> str:
    """Load caption from text file"""
    with open(caption_path, 'r') as f:
        caption = f.read().strip()
    return caption


def get_image_base_name(caption_file: Path) -> str:
    """Extract base name from caption file (remove _detailed.txt)"""
    name = caption_file.stem  # Remove .txt
    if name.endswith("_detailed"):
        name = name[:-len("_detailed")]
    return name


def setup_pipeline(base_model: str, checkpoint: str, device: str):
    """Load and setup the pipeline with LoRA weights and Compel"""
    print(f"[1] Loading pretrained base model from: {base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    print(f"    ✓ Base model loaded (pretrained SDXL weights)")
    
    print(f"\n[2] Loading LoRA fine-tuned weights from: {checkpoint}")
    lora_weights_path = checkpoint + "/pytorch_lora_weights.safetensors"
    if not os.path.exists(lora_weights_path):
        print(f"    ERROR: LoRA weights not found at {lora_weights_path}")
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")
    
    pipe.load_lora_weights(lora_weights_path)
    print(f"    ✓ LoRA weights loaded and applied to base model")
    
    # Note: NOT fusing LoRA weights to keep them separate from base model
    # pipe.fuse_lora()  # Uncomment only if you want to fuse for faster inference
    
    print(f"\n[3] Setting up Compel for advanced long prompt handling")
    try:
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            requires_pooled=[False, True],
        )
        print(f"    ✓ Compel initialized successfully")
    except Exception as e:
        print(f"    WARNING: Compel initialization failed: {e}")
        print(f"    Falling back to standard SDXL prompt handling")
        compel = None
    
    return pipe, compel


def generate_image(pipe, compel, prompt: str, args) -> Image.Image:
    """Generate a single image from prompt using Compel for advanced handling"""
    try:
        if compel is not None:
            # Use Compel for advanced prompt handling (long prompts, weights, etc.)
            prompt_embeds, pooled_prompt_embeds = compel(prompt)
        else:
            # Fallback to standard SDXL pipeline
            prompt_embeds = None
            pooled_prompt_embeds = None
    except Exception as e:
        print(f"[WARNING] Compel processing failed, falling back to standard: {e}")
        prompt_embeds = None
        pooled_prompt_embeds = None
    
    # Generate image
    try:
        with torch.no_grad():
            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                # Use Compel-generated embeddings
                image = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=torch.Generator(device=args.device).manual_seed(args.seed),
                ).images[0]
            else:
                # Use standard prompt string
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    generator=torch.Generator(device=args.device).manual_seed(args.seed),
                ).images[0]
    except Exception as e:
        print(f"[ERROR] Pipeline generation failed: {e}")
        raise
    
    return image


def main():
    args = parse_args()
    
    captions_dir = Path(args.captions_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint)
    
    # Validate inputs
    if not captions_dir.exists():
        print(f"ERROR: Captions directory not found: {captions_dir}")
        return
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get list of caption files
    caption_files = sorted(captions_dir.glob("*_detailed.txt"))
    
    if args.limit > 0:
        caption_files = caption_files[:args.limit]
    
    print(f"Found {len(caption_files)} caption files to process")
    
    # Setup pipeline
    pipe, compel = setup_pipeline(args.base_model, str(checkpoint_dir), args.device)
    
    # Metadata for evaluation
    metadata = []
    
    # Generate images
    print(f"\nGenerating images with Compel (advanced long prompt handling)...")
    for caption_file in tqdm(caption_files, desc="Generating"):
        try:
            # Get base name (matching original image)
            base_name = get_image_base_name(caption_file)
            
            # Load caption
            caption = load_caption(caption_file)
            
            if not caption:
                print(f"[WARNING] Empty caption in {caption_file.name}")
                continue
            
            print(f"\n{base_name}: {caption[:80]}...")
            
            # Generate image with Compel
            image = generate_image(pipe, compel, caption, args)
            
            # Save with original image extension (assuming .jpg)
            output_path = output_dir / f"{base_name}.png"
            image.save(output_path)
            
            # Record metadata
            metadata.append({
                "base_name": base_name,
                "generated_image": str(output_path.name),
                "caption": caption,
                "resolution": f"{args.height}x{args.width}",
                "inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "checkpoint": str(checkpoint_dir.name)
            })
            
            print(f"✓ Saved to {output_path.name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {caption_file.name}: {e}")
            continue
    
    # Save metadata
    metadata_path = output_dir / "generation_metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n[SUCCESS] Generation complete!")
    print(f"Generated {len(metadata)} images")
    print(f"Saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Summary
    summary = {
        "total_generated": len(metadata),
        "output_directory": str(output_dir),
        "checkpoint": str(checkpoint_dir),
        "resolution": f"{args.height}x{args.width}",
        "inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed
    }
    
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
