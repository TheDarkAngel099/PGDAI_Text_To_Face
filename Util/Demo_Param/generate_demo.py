#!/usr/bin/env python3

"""
Demo script to generate a single image from text prompt.

Uses RealViz checkpoint with fine-tuned LoRA weights and Compel for advanced prompt handling.

Features:
- Interactive prompt input from user
- Image generation with Compel
- Display and save functionality
- Single image generation (quick feedback)

Usage:
    python demo/generate_demo.py
"""

import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from compel import Compel


def setup_pipeline(base_model: str, checkpoint: str, device: str):
    """Load and setup the pipeline with LoRA weights and Compel"""
    print(f"[1] Loading pretrained base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    print(f"    ✓ Base model loaded")
    
    print(f"\n[2] Loading LoRA fine-tuned weights...")
    lora_weights_path = checkpoint + "/pytorch_lora_weights.safetensors"
    if not os.path.exists(lora_weights_path):
        print(f"    ERROR: LoRA weights not found at {lora_weights_path}")
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")
    
    pipe.load_lora_weights(lora_weights_path)
    print(f"    ✓ LoRA weights loaded")
    
    print(f"\n[3] Setting up Compel for advanced prompt handling...")
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


def generate_image(pipe, compel, prompt: str, device: str, height: int, width: int, 
                   num_steps: int, guidance_scale: float, seed: int) -> Image.Image:
    """Generate a single image from prompt using Compel"""
    print(f"\n[4] Generating image from prompt...")
    print(f"    Prompt: {prompt[:100]}...")
    
    try:
        if compel is not None:
            # Use Compel for advanced prompt handling
            prompt_embeds, pooled_prompt_embeds = compel(prompt)
            print(f"    ✓ Prompt processed with Compel")
        else:
            prompt_embeds = None
            pooled_prompt_embeds = None
    except Exception as e:
        print(f"    WARNING: Compel processing failed, falling back: {e}")
        prompt_embeds = None
        pooled_prompt_embeds = None
    
    try:
        with torch.no_grad():
            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                # Use Compel-generated embeddings
                image = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
            else:
                # Use standard prompt string
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
        
        print(f"    ✓ Image generated successfully")
        return image
    except Exception as e:
        print(f"    ERROR: Pipeline generation failed: {e}")
        raise


def display_image(image: Image.Image):
    """Display image (basic implementation)"""
    try:
        image.show()
        print(f"\n[5] Image displayed")
    except Exception as e:
        print(f"    [INFO] Could not display image: {e}")
        print(f"    Image will be saved instead")


def save_image(image: Image.Image, output_dir: str, filename: str = "demo_output.png") -> str:
    """Save image to output directory"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    image.save(str(file_path))
    
    print(f"[6] Image saved to: {file_path}")
    return str(file_path)


def get_prompt_from_user() -> str:
    """Get prompt from user input"""
    print("\n" + "="*60)
    print("Enter your prompt for image generation:")
    print("(This can be a detailed description of the face you want)")
    print("="*60)
    
    prompt = input("\nPrompt: ").strip()
    
    if not prompt:
        print("ERROR: Prompt cannot be empty!")
        return get_prompt_from_user()
    
    return prompt


def main():
    """Main function: prompt -> generate -> display -> save"""
    
    parser = argparse.ArgumentParser(description="Generate a single demo image from text prompt")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/dai01/Text_To_Face/sd_training/RealVizXL_Model",
        help="Path to base model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/dai01/Text_To_Face/sd_training/outputs_indian_finetuned_ckpt2700/checkpoint-2700",
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dai01/Text_To_Face/demo/demo_result",
        help="Output directory for generated images"
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
        "--num_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
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
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional: provide prompt as argument (skips interactive input)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RealViz Demo - Single Image Generation")
    print("="*60)
    
    # Setup pipeline
    print(f"\n[SETUP] Loading models...")
    pipe, compel = setup_pipeline(args.base_model, args.checkpoint, args.device)
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
        print(f"\nUsing provided prompt: {prompt[:100]}...")
    else:
        prompt = get_prompt_from_user()
    
    # Generate image
    image = generate_image(
        pipe, compel, prompt, args.device,
        args.height, args.width, args.num_steps,
        args.guidance_scale, args.seed
    )
    
    # Display image
    #display_image(image)
    
    # Save image
    save_image(image, args.output_dir)
    
    print("\n" + "="*60)
    print("✓ Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
