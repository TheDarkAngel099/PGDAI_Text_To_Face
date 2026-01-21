
#!/usr/bin/env python3
import os
import shutil
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ============================================================
# 1. ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA-1.5 Forensic Annotation Pipeline (Deterministic, SDXL-safe)"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder with input face images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where lora_dataset/ will be created")
    parser.add_argument("--model_id", type=str,
                        default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace LLaVA model id")
    parser.add_argument("--cache_dir", type=str,
                        default="./model_cache",
                        help="Model cache directory")
    return parser.parse_args()

# ============================================================
# 2. LOAD MODEL (FP16, NO QUANTIZATION)
# ============================================================

def load_model(model_id, cache_dir):
    print(f"🔄 Loading LLaVA model: {model_id}")

    os.makedirs(cache_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir
    )

    model.eval()
    print("✅ Model loaded (FP16, eval mode)")
    return model, processor

# ============================================================
# 3. DETERMINISTIC LLaVA INFERENCE
# ============================================================

@torch.no_grad()
def run_llava(model, processor, image, prompt, max_new_tokens=400):
    """
    Single-pass visual inference.
    NO sampling. NO chat wrapper. Deterministic.
    """
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0
    )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

# ============================================================
# 4. PHASE 1 — FORENSIC ATTRIBUTE EXTRACTION (VISION)
# ============================================================

FORENSIC_PROMPT = """
You are a forensic attribute extraction engine.

RULES (STRICT):
- Use ONLY visible information.
- If uncertain, choose the most visually likely option.
- DO NOT explain.
- DO NOT add attributes not visible.
- DO NOT use full sentences.
- DO NOT break format.

OUTPUT FORMAT (EXACT):
1. Skin color and gender: <phrase>.
2. Overall facial appearance: <phrase>.
3. Hair: <phrase>.
4. Forehead: <phrase>.
5. Hairline: <phrase>.
6. Beard: <phrase>.
7. Eyes: <phrase>.
8. Eyebrows: <phrase>.
9. Nose: <phrase>.
10. Lips: <phrase>.
11. Chin and jawline: <phrase>.
12. Ears: <phrase>.
13. Scars: <phrase>.
"""

def extract_attributes(model, processor, image):
    return run_llava(
        model=model,
        processor=processor,
        image=image,
        prompt=FORENSIC_PROMPT,
        max_new_tokens=350
    )

# ============================================================
# 5. PHASE 2 — TEXT-ONLY SDXL PROMPT COMPRESSION
# ============================================================

@torch.no_grad()
def compress_to_sdxl(model, processor, structured_text):
    """
    Text-only compression.
    No image passed → no hallucination.
    """
    prompt = f"""
Compress the following forensic attributes into a single dense SDXL prompt.

RULES:
- Include race, gender, age, hair, beard (if any), eyes, face shape.
- No conversational phrases.
- No uncertainty language.
- Under 74 words.
- One paragraph.

INPUT:
{structured_text}

OUTPUT:
"""

    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0
    )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Output structure
    lora_root = os.path.join(args.output_dir, "lora_dataset")
    captions_dir = os.path.join(lora_root, "captions")
    images_dir = os.path.join(lora_root, "images")

    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"📂 Output directory: {lora_root}")

    model, processor = load_model(args.model_id, args.cache_dir)

    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"🚀 Processing {len(image_files)} images")

    success, skipped = 0, 0

    for fname in tqdm(image_files, desc="Annotating"):
        src_img = os.path.join(args.input_dir, fname)
        base = os.path.splitext(fname)[0]
        cap_path = os.path.join(captions_dir, f"{base}.txt")
        dst_img = os.path.join(images_dir, fname)

        if os.path.exists(cap_path) and os.path.exists(dst_img):
            skipped += 1
            continue

        try:
            image = Image.open(src_img).convert("RGB")

            # Phase 1: Vision → Structured attributes
            attributes = extract_attributes(model, processor, image)

            # Phase 2: Text → SDXL training caption
            final_caption = compress_to_sdxl(
                model, processor, attributes
            )

            with open(cap_path, "w") as f:
                f.write(final_caption)

            shutil.copy(src_img, dst_img)

            success += 1

        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    print("\n✅ PIPELINE COMPLETE")
    print(f"   Processed : {success}")
    print(f"   Skipped   : {skipped}")
    print(f"📁 Dataset at: {lora_root}")

# ============================================================
if __name__ == "__main__":
    main()
