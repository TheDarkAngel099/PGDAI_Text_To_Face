#!/usr/bin/env python3
import os
import shutil
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ==========================================
# 1. SETUP & ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Forensic Annotation Pipeline (Single GPU)")
    
    # Required Arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for output dataset")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to metadata CSV")
    
    # Optional Arguments
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-13b-hf", help="Model ID")
    parser.add_argument("--cache_dir", type=str, default="./model_cache", help="Cache directory")
    
    return parser.parse_args()

# ==========================================
# 2. LOAD MODEL
# ==========================================
def load_model(model_id, cache_dir):
    print(f"⏳ Initializing LLaVA Pipeline (Float16)...")
    
    os.makedirs(cache_dir, exist_ok=True)

    try:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        # device_map="auto" will automatically find the available GPU
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

# ==========================================
# 3. INFERENCE FUNCTIONS
# ==========================================
def run_vlm_inference(model, processor, image, text_prompt, max_tokens=600):
    full_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,  
        temperature=0.0,  
        top_p=1,
        repetition_penalty=1.05 
    )
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text

def run_text_inference(model, processor, text_prompt, max_tokens=200):
    dummy_image = Image.new('RGB', (336, 336), (0, 0, 0))
    full_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    
    inputs = processor(text=full_prompt, images=dummy_image, return_tensors="pt").to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.0, 
        top_p=1,
        repetition_penalty=1.05 
    )
    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text

# ==========================================
# 4. TASK WRAPPERS
# ==========================================
def get_detailed_annotation(model, processor, image, metadata_str):
    prompt = f"""Act as a forensic feature extractor.

KNOWN TRUTH: {metadata_str}

INSTRUCTIONS:
1. **VISUAL EVIDENCE ONLY:** Base answers STRICTLY on the image.
2. **GENDER LOGIC:**
   - **FEMALES:** Q3 and Q4 (Facial Hair) MUST be "None".
   - **MALES:** Q3 and Q4 are MANDATORY.

- **ANALYZE FACIAL HAIR (CRITICAL STEP - LOOK AT SIDEBURNS):**
  * **Trace the line from the EAR down to the CHIN.**
  * **Full Beard:** Hair connects from the **SIDEBURNS -> JAW -> CHIN**. Continuous line.
  * **Goatee:** Hair on Chin/Mouth. **NO connection to sideburns.** GAP on jaw/cheek.
  * **Stubble:** Shadow or short rough texture. No distinct long hairs.
  * **Clean Shaven:** 100% Smooth skin.

- **ANALYZE HAIR STYLE:**
  * **Bald/Balding:** Scalp visible.
  * **Buzz Cut:** Very short, uniform length (military).
  * **Crew Cut/Fade:** Short on sides, longer on top.
  * **Slicked Back:** Combed straight back.
  * **Long/Straight:** Below ears, no curl.
  * **Wavy/Curly:** Visible texture/loops.
  * **Afro/Textured:** Tight coils, high volume.

EXTRACT DATA:
1. Race/Gender:
2. Age (from database):
3. Facial Hair Style (Full Beard, Goatee, Stubble, Clean Shaven, or None):
4. Facial Hair Length (None, Scruff, Short, Medium, Long, Bushy):
5. Hair Description (Color + Length + Style):
6. Forehead (Smooth or Wrinkled):
7. Hairline (Straight, Receding, or Balding):
8. Eyes (Color):
9. Eyebrows (Color):
10. Nose (Size + Shape):
11. Lips (Thickness):
12. Chin/Jaw (Shape):
13. Ears (Size):
14. Tattoos (Visible?):
15. Scars (Visible?):"""
    
    return run_vlm_inference(model, processor, image, prompt, max_tokens=600)

def compress_for_sd(model, processor, detailed_text):
    prompt = f"""Task: Create a dense SDXL prompt (MAX 40 WORDS).

INPUT DATA:
{detailed_text}

CRITICAL RULES (TOKEN LIMIT < 74):
1. **MAX LENGTH**: Output must be under 40 words.
2. **FORMAT**: `[Age]-year-old [Race] [Gender], [Hair], [Facial Hair], [Features]`
3. **SINGLE MENTION RULE (CRITICAL)**:
   - **NEVER** output multiple facial hair terms.
   - If style is "Goatee", DO NOT say "Beard".
   - RIGHT: "short goatee, smooth forehead"

4. **PRUNING (Save Space)**:
   - **DELETE** "Smooth forehead", "Average/Normal" features.
   - **DELETE** "None" for Tattoos/Scars.
   - **DELETE** Articles: a, an, the.

5. **MALE FACIAL HAIR (Combine Q3 + Q4)**:
   - Example: "Full Beard" + "Bushy" -> **"bushy full beard"**
   - **MUST be included.**

6. **FEMALE FACIAL HAIR**:
   - DELETE completely.

EXAMPLE OUTPUT:
41-year-old white male, short brown buzz cut, receding hairline, bushy full beard, brown eyes, square chin

REAL TASK OUTPUT:"""

    return run_text_inference(
        model=model,
        processor=processor,
        text_prompt=prompt,
        max_tokens=100
    )

# ==========================================
# 5. MAIN EXECUTION
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

    # Setup Output Folders
    lora_root = os.path.join(args.output_dir, "lora_dataset")
    captions_dir = os.path.join(lora_root, "captions")
    detailed_dir = os.path.join(lora_root, "detailed_captions")
    images_dir = os.path.join(lora_root, "images")
    
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(detailed_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Load Model
    model, processor = load_model(args.model_id, args.cache_dir)

    # Get Images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)])
    
    print(f"🚀 Processing {len(all_files)} images...")

    # Process Loop
    success_count = 0
    skipped_count = 0
    
    for filename in tqdm(all_files, desc="Annotating"):
        
        img_path = os.path.join(args.input_dir, filename)
        file_id = os.path.splitext(filename)[0]

        # LOOKUP METADATA
        if file_id in meta_lookup:
            meta = meta_lookup[file_id]
            metadata_str = f"Sex: {meta['sex']}, Race: {meta['race']}, Age: {meta['age']}"
        else:
            metadata_str = "Unknown (Estimate visually)"

        # SKIP LOGIC
        txt_path = os.path.join(captions_dir, f"{file_id}.txt")
        detailed_path = os.path.join(detailed_dir, f"{file_id}_detailed.txt")
        img_dest_path = os.path.join(images_dir, filename)

        if os.path.exists(txt_path) and os.path.exists(detailed_path) and os.path.exists(img_dest_path):
            skipped_count += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            
            # Phase 1: Annotation
            annotation = get_detailed_annotation(model, processor, image, metadata_str)
            with open(detailed_path, "w") as f:
                f.write(annotation)

            # Phase 2: Compression
            final_prompt = compress_for_sd(model, processor, annotation)
            with open(txt_path, "w") as f:
                f.write(final_prompt)

            shutil.copy(img_path, img_dest_path)
            success_count += 1

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    print(f"\n✅ Processing Complete!")
    print(f"   Skipped (Already Done): {skipped_count}")
    print(f"   Successfully processed: {success_count}")

if __name__ == "__main__":
    main()