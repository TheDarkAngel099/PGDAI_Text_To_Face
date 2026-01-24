#!/usr/bin/env python3
import os
import shutil
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPTokenizer
import logging
from datetime import datetime

# ==========================================
# 1. SETUP & ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Forensic Annotation Pipeline for Param Rudra")
    
    # Required Arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path where the 'lora_dataset' will be created")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file with metadata")
    
    # Optional Arguments
    # CHANGED DEFAULT TO 13B MODEL BELOW
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-13b-hf", help="Hugging Face Model ID")
    parser.add_argument("--cache_dir", type=str, default="./model_cache", help="Local path to cache downloaded models")
    parser.add_argument("--max_retries", type=int, default=2, help="Retry failed captions")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already processed images")
    
    return parser.parse_args()

# ==========================================
# 2. TOKEN COUNTING & VALIDATION
# ==========================================
def count_tokens(text, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
    """Count actual SDXL tokens using CLIP tokenizer"""
    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback: rough estimate
        return len(text.split()) + 2

def validate_caption(caption, max_tokens=75, min_tokens=10):
    """Ensure caption meets SDXL requirements"""
    token_count = count_tokens(caption)
    
    if token_count > max_tokens:
        return False, f"Too long ({token_count} tokens)"
    if token_count < min_tokens:
        return False, f"Too short ({token_count} tokens)"
    
    # Check for unwanted content
    forbidden_words = ['wearing', 'clothing', 'shirt', 'background', 'standing', 'sitting', 'smiling']
    caption_lower = caption.lower()
    for word in forbidden_words:
        if word in caption_lower:
            return False, f"Contains forbidden word: {word}"
    
    return True, token_count

def clean_caption(caption):
    """Remove common VLM artifacts and clean up caption"""
    # Remove model artifacts
    caption = caption.replace("ASSISTANT:", "").replace("USER:", "")
    
    # Remove redundant phrases
    removals = [
        "in the image", "in this photo", "the person", "the subject",
        "appears to be", "seems to", "looks like", "is visible"
    ]
    for phrase in removals:
        caption = caption.replace(phrase, "")
    
    # Normalize spacing
    caption = ' '.join(caption.split())
    
    # Remove trailing punctuation (SDXL doesn't need it)
    caption = caption.rstrip('.,;:')
    
    return caption.strip()

def setup_logging(output_dir):
    """Create detailed log file"""
    log_file = os.path.join(output_dir, f"caption_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# ==========================================
# 3. LOAD MODEL
# ==========================================
def load_model(model_id, cache_dir):
    print(f"⏳ Initializing LLaVA Pipeline (Float16 Mode)...")
    print(f"   Model: {model_id}")
    print(f"   Cache: {cache_dir}")
    
    os.makedirs(cache_dir, exist_ok=True)

    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        print("✅ Model loaded successfully on GPU (Float16)!")
        return model, processor
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Tip: Ensure you have internet access on the login node or pre-download weights.")
        exit(1)

# ==========================================
# 4. INFERENCE FUNCTIONS
# ==========================================

# Function 1: VLM Inference (Image + Text) -> Enable sampling for diversity
def run_vlm_inference(model, processor, image, text_prompt, max_tokens=500):
    full_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,  
        temperature=0.0,  
        top_p=0.9,
        repetition_penalty=1.05 
    )
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text

# Function 2: Text Inference (Text Only) -> UPDATED with sampling for diversity
def run_text_inference(model, processor, text_prompt, max_tokens=200):
    # CRITICAL FIX: Create a black dummy image (336x336)
    # This prevents the "Missing pixel_values" crash
    dummy_image = Image.new('RGB', (336, 336), (0, 0, 0))
    
    # We add <image> so the model consumes the dummy image
    full_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    
    inputs = processor(text=full_prompt, images=dummy_image, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0, 
        top_p=0.9,
        repetition_penalty=1.05 
    )
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text

# ==========================================
# 5. OPTIMIZED SINGLE-PHASE CAPTION GENERATION
# ==========================================

def generate_sdxl_caption(model, processor, image, metadata_str):
    """
    Single-phase optimized caption generation for SDXL/RealViz.
    Generates captions that fit within 75 token SDXL limit.
    """
    prompt = f"""Create a photo-realistic portrait description for AI image generation.

SUBJECT DATA: {metadata_str}

OUTPUT FORMAT: "[Age]-year-old [Race] [Gender], [Hair details], [Facial hair if male], [2-3 distinctive features]"

RULES:
1. MAX 60 WORDS (must fit in 75 tokens)
2. ONLY facial features visible in the photo
3. NO clothing, background, emotions, or actions
4. For MALES: Always include facial hair status (clean-shaven, stubble, beard, goatee)
5. Prioritize: Hair (color, length, texture) > Face shape > Eyes > Distinctive marks

HAIR TEXTURE:
- Straight: No waves/curls
- Wavy: Gentle S-curves
- Curly: Tight coils/loops
- Kinky/Coily: Very tight curls
- Bald: Smooth scalp visible

FACIAL HAIR (for males):
- Clean-shaven: No visible facial hair
- Stubble: Light shadow
- Goatee: Chin and mustache only
- Full beard: Chin, mustache, and cheeks

AVOID: "average", "normal", "smooth forehead", "small ears", unknown features, articles (a/an/the)

EXAMPLE: "32-year-old Black male, short kinky black hair, full beard, brown eyes, broad nose, strong jawline"

Generate caption:"""

    return run_vlm_inference(
        model=model,
        processor=processor,
        image=image,
        text_prompt=prompt,
        max_tokens=150
    )

def generate_validated_caption(model, processor, image, metadata_str, max_retries=2):
    """
    Generate caption with automatic validation and retry logic.
    Returns (caption, token_count) or (None, error_message)
    """
    for attempt in range(max_retries + 1):
        try:
            # Generate caption
            raw_caption = generate_sdxl_caption(model, processor, image, metadata_str)
            
            # Clean up caption
            caption = clean_caption(raw_caption)
            
            # Validate
            is_valid, result = validate_caption(caption)
            
            if is_valid:
                logging.info(f"  ✓ Valid caption ({result} tokens)")
                return caption, result
            else:
                logging.warning(f"  ⚠️ Attempt {attempt+1} failed: {result}")
                if attempt < max_retries:
                    continue
        
        except Exception as e:
            logging.error(f"  ❌ Generation error on attempt {attempt+1}: {e}")
            if attempt < max_retries:
                continue
    
    return None, "Failed validation after all retries"

# Legacy function for detailed annotations (kept for backward compatibility)
def get_detailed_annotation(model, processor, image, metadata_str):
    """
    Generates detailed forensic annotation for archival purposes.
    """
    prompt = f"""
Analyze this facial photograph and extract detailed attributes.

SUBJECT DATA: {metadata_str}

RULES:
- Use the subject data for demographic information
- Only describe clearly visible facial features
- Be specific about hair texture, facial hair, and distinctive features
- Do NOT describe clothing, background, or expressions

Provide details on:
1. Demographics (use subject data)
2. Hair: Color, length, style, texture
3. Hairline: Receding, straight, widow's peak
4. Facial hair: Style and length (mandatory for males)
5. Eyes: Color and shape
6. Eyebrows: Color and style
7. Nose: Size and shape
8. Lips: Thickness
9. Chin/Jaw: Shape
10. Distinctive features: Wrinkles, scars, tattoos, etc.
"""

    return run_vlm_inference(
        model=model,
        processor=processor,
        image=image,
        text_prompt=prompt,
        max_tokens=500
    )


# ==========================================
# 6. QUALITY METRICS
# ==========================================
def save_quality_report(output_dir, stats):
    """Generate report on caption quality"""
    report_path = os.path.join(output_dir, "caption_quality_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== CAPTION QUALITY REPORT ===\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Processed: {stats['total']}\n")
        f.write(f"Successful: {stats['success']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Skipped: {stats['skipped']}\n\n")
        
        if stats['token_counts']:
            f.write(f"Avg Token Count: {sum(stats['token_counts'])/len(stats['token_counts']):.1f}\n")
            f.write(f"Token Range: {min(stats['token_counts'])}-{max(stats['token_counts'])}\n")
            f.write(f"Over 75 tokens: {sum(1 for t in stats['token_counts'] if t > 75)}\n\n")
        
        if stats['failed_files']:
            f.write("\nFailed Files:\n")
            for filename, reason in stats['failed_files']:
                f.write(f"  - {filename}: {reason}\n")
    
    logging.info(f"Quality report saved to: {report_path}")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    args = parse_args()

    # Check Input
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        exit(1)
    
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV file not found: {args.csv_file}")
        exit(1)

    # LOAD CSV DATA
    print("📊 Loading CSV Data...")
    df = pd.read_csv(args.csv_file)
    df['inmate_id'] = df['inmate_id'].astype(str).str.strip()
    meta_lookup = df.set_index('inmate_id')[['sex', 'race', 'age']].to_dict('index')
    print(f"✅ Loaded metadata for {len(meta_lookup)} inmates.")

    # Setup Output Folders
    lora_root = os.path.join(args.output_dir, "lora_dataset")
    captions_dir = os.path.join(lora_root, "captions")
    detailed_dir = os.path.join(lora_root, "detailed_captions")
    images_dir = os.path.join(lora_root, "images")
    
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"📂 Output will be saved to: {lora_root}")
    
    # Setup logging
    log_file = setup_logging(lora_root)
    logging.info(f"Starting caption generation pipeline")
    logging.info(f"Output directory: {lora_root}")

    # Load Model
    model, processor = load_model(args.model_id, args.cache_dir)

    # Get Images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    all_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"🚀 Starting processing for {len(all_files)} images...")

    # Process Loop with Quality Tracking
    success_count = 0
    skipped_count = 0
    failed_count = 0
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'token_counts': [],
        'failed_files': []
    }
    
    for filename in tqdm(all_files, desc="Processing"):
        
        # --- TEST LIMIT: STOP AFTER 5 IMAGES ---
        if success_count >= 5:
            print("\n🛑 TEST MODE: Reached 5 images. Stopping.")
            break
        # ---------------------------------------

        img_path = os.path.join(args.input_dir, filename)
        file_id = os.path.splitext(filename)[0]

        # LOOKUP METADATA
        if file_id in meta_lookup:
            meta = meta_lookup[file_id]
            metadata_str = f"Sex: {meta['sex']}, Race: {meta['race']}, Age: {meta['age']}"
        else:
            print(f"⚠️ Warning: No metadata found for {file_id}. Using model estimation.")
            metadata_str = "Unknown (Estimate visually)"

        # SKIP LOGIC
        txt_path = os.path.join(captions_dir, f"{file_id}.txt")
        detailed_path = os.path.join(detailed_dir, f"{file_id}_detailed.txt")
        img_dest_path = os.path.join(images_dir, filename)

        if args.skip_existing and os.path.exists(txt_path) and os.path.exists(img_dest_path):
            skipped_count += 1
            stats['skipped'] += 1
            continue

        stats['total'] += 1
        logging.info(f"Processing: {filename}")

        try:
            image = Image.open(img_path).convert("RGB")

            # Generate validated SDXL caption (single-phase)
            final_caption, result = generate_validated_caption(
                model, processor, image, metadata_str, max_retries=args.max_retries
            )

            if final_caption is None:
                # Failed validation
                logging.error(f"Failed to generate valid caption: {result}")
                stats['failed'] += 1
                stats['failed_files'].append((filename, result))
                failed_count += 1
                continue

            # Save Final Caption
            with open(txt_path, "w", encoding='utf-8') as f:
                f.write(final_caption)
            
            # Track token count
            stats['token_counts'].append(result)

            # Generate detailed annotation for archival (optional)
            try:
                annotation = get_detailed_annotation(model, processor, image, metadata_str)
                with open(detailed_path, "w", encoding='utf-8') as f:
                    f.write(annotation)
            except Exception as e:
                logging.warning(f"Could not generate detailed annotation: {e}")

            # Copy Image
            shutil.copy(img_path, img_dest_path)
            
            success_count += 1
            stats['success'] += 1
            logging.info(f"✓ Success: {final_caption[:60]}... ({result} tokens)")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            stats['failed'] += 1
            stats['failed_files'].append((filename, str(e)))
            failed_count += 1

    print(f"\n✅ Processing Complete!")
    print(f"   Skipped (Already Done): {skipped_count}")
    print(f"   Successfully processed: {success_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Results located in: {lora_root}")
    
    # Generate quality report
    if stats['total'] > 0:
        save_quality_report(lora_root, stats)
    
    logging.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()