
#!/usr/bin/env python3
import os
import shutil
import argparse
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ============================================================
# 1. ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Forensic LLaVA → SDXL Dataset Builder (RealVis XL)")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--cache_dir", default="./model_cache")
    return p.parse_args()

# ============================================================
# 2. LOAD MODEL (FP16, DETERMINISTIC)
# ============================================================

def load_model(model_id, cache_dir):
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    return model, processor

# ============================================================
# 3. FORENSIC PROMPT (VISION, SINGLE PASS)
# ============================================================

FORENSIC_PROMPT = """
You are a forensic attribute extraction engine.

STRICT RULES:
- Use ONLY visible information from the image.
- If uncertain, choose the most visually likely option.
- DO NOT explain.
- DO NOT add attributes not visible.
- DO NOT use full sentences.
- DO NOT change the format.

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

@torch.no_grad()
def run_llava(model, processor, image, prompt, limit=350):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=limit,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.0
    )
    return processor.decode(out[0], skip_special_tokens=True).strip()

# ============================================================
# 4. ATTRIBUTE QA (REJECT BAD ANNOTATIONS)
# ============================================================

ATTR_KEYS = [
    "Skin color and gender",
    "Overall facial appearance",
    "Hair",
    "Forehead",
    "Hairline",
    "Beard",
    "Eyes",
    "Eyebrows",
    "Nose",
    "Lips",
    "Chin and jawline",
    "Ears",
    "Scars"
]

def parse_attributes(text):
    attrs = {}
    for line in text.splitlines():
        m = re.match(r"\d+\.\s*(.+?):\s*(.+?)\.$", line.strip())
        if m:
            attrs[m.group(1)] = m.group(2)
    return attrs

def validate_attributes(attrs):
    if len(attrs) != 13:
        return False
    for k in ATTR_KEYS:
        if k not in attrs or len(attrs[k]) < 2:
            return False
    return True

# ============================================================
# 5. RULE‑BASED SDXL PROMPT COMPILER (NO LLM)
# ============================================================

def compile_sdxl_prompt(attrs):
    parts = [
        f"{attrs['Skin color and gender']}",
        f"{attrs['Overall facial appearance']}",
        f"{attrs['Hair']}",
        f"{attrs['Hairline']}",
        f"{attrs['Beard']}" if attrs["Beard"].lower() != "none" else "",
        f"{attrs['Eyes']}",
        f"{attrs['Eyebrows']}",
        f"{attrs['Nose']}",
        f"{attrs['Lips']}",
        f"{attrs['Chin and jawline']}",
        f"{attrs['Forehead']}",
        f"{attrs['Ears']}",
        f"{attrs['Scars']}" if attrs["Scars"].lower() != "none" else ""
    ]

    core = ", ".join([p for p in parts if p])
    prompt = (
        "realistic forensic mugshot, centered, neutral lighting, "
        + core.lower()
    )

    return truncate_clip(prompt)

# ============================================================
# 6. CLIP TOKEN BUDGET (≤77 SAFE)
# ============================================================

def truncate_clip(text, limit=77):
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])

# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    args = parse_args()

    out_root = os.path.join(args.output_dir, "lora_dataset")
    img_out = os.path.join(out_root, "images")
    cap_out = os.path.join(out_root, "captions")
    stats_path = os.path.join(out_root, "stats.json")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(cap_out, exist_ok=True)

    model, processor = load_model(args.model_id, args.cache_dir)

    stats = {k: Counter() for k in ATTR_KEYS}

    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]

    success = skipped = rejected = 0

    for f in tqdm(files, desc="Processing"):
        src = os.path.join(args.input_dir, f)
        base = os.path.splitext(f)[0]
        cap_path = os.path.join(cap_out, base + ".txt")
        img_dest = os.path.join(img_out, f)

        if os.path.exists(cap_path):
            skipped += 1
            continue

        try:
            img = Image.open(src).convert("RGB")
            txt = run_llava(model, processor, img, FORENSIC_PROMPT)
            attrs = parse_attributes(txt)

            if not validate_attributes(attrs):
                rejected += 1
                continue

            for k, v in attrs.items():
                stats[k][v] += 1

            final_prompt = compile_sdxl_prompt(attrs)

            with open(cap_path, "w") as f:
                f.write(final_prompt)
            shutil.copy(src, img_dest)

            success += 1

        except Exception as e:
            print(f"⚠️ Error {f}: {e}")
            rejected += 1

    with open(stats_path, "w") as f:
        json.dump({k: dict(v) for k, v in stats.items()}, f, indent=2)

    print("\n✅ COMPLETE")
    print(f"Processed : {success}")
    print(f"Skipped   : {skipped}")
    print(f"Rejected  : {rejected}")
    print(f"Dataset   : {out_root}")

# ============================================================
if __name__ == "__main__":
    main()
