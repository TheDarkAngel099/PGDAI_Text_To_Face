# --- Create metadata.jsonl ---
# Requirement for the HuggingFace 'train_text_to_image_lora.py' script

import os
import json


dataset_root = "lora_dataset"
images_dir = os.path.join(dataset_root, "images")
captions_dir = os.path.join(dataset_root, "captions")
os.makedirs(captions_dir, exist_ok=True)


metadata = []
for img_file in os.listdir(images_dir):
    base = os.path.splitext(img_file)[0]
    txt_file = f"{base}.txt"

    txt_path = os.path.join(captions_dir, txt_file)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            saved_caption = f.read().strip()

        # Note: file_name should be relative to where metadata.jsonl is (the root)
        metadata.append({"file_name": f"images/{img_file}", "text": saved_caption})

with open(os.path.join(dataset_root, "metadata.jsonl"), "w") as f:
    for entry in metadata:
        json.dump(entry, f)
        f.write("\n")

print(f"--- Dataset structured and metadata.jsonl generated in {dataset_root} ---")