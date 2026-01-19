
#!/usr/bin/env python3
# --- Create metadata.jsonl ---
# Requirement for the HuggingFace 'train_text_to_image_lora.py' script

import os
import json

dataset_root = "../lora_dataset"
images_dir = os.path.join(dataset_root, "images")
captions_dir = os.path.join(dataset_root, "captions")

# -------------------------------------------------------------------
# Do NOT create captions directory — just check if it exists
# -------------------------------------------------------------------
if not os.path.isdir(captions_dir):
    print(f"[ERROR] Captions folder not found: {captions_dir}")
    print("Please create it and add caption .txt files before running this script.")
    exit(1)

# Optional: also check if images directory exists
if not os.path.isdir(images_dir):
    print(f"[ERROR] Images folder not found: {images_dir}")
    exit(1)

metadata = []

print("[INFO] Scanning image files and matching captions...")

for img_file in os.listdir(images_dir):
    base = os.path.splitext(img_file)[0]
    txt_file = f"{base}.txt"
    txt_path = os.path.join(captions_dir, txt_file)

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            saved_caption = f.read().strip()

        # file_name must be relative to dataset root
        metadata.append({
            "file_name": f"images/{img_file}",
            "text": saved_caption
        })
    else:
        print(f"[WARNING] Missing caption for image: {img_file}")

# Write metadata.jsonl
metadata_path = os.path.join(dataset_root, "metadata.jsonl")

with open(metadata_path, "w") as f:
    for entry in metadata:
        json.dump(entry, f)
        f.write("\n")

print(f"[SUCCESS] metadata.jsonl created at: {metadata_path}")
print(f"[INFO] Total entries written: {len(metadata)}")
