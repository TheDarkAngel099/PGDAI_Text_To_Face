
#!/usr/bin/env python3

"""
Script to download a RealVizXL (or any HuggingFace) model
and save the full snapshot to a target directory.

Just edit the variables below, then run:
    python download_model_static.py
"""

import os
import shutil
from pathlib import Path

# --------------------------------------------------------
# 🔧 EDIT THESE TWO VARIABLES
# --------------------------------------------------------
#!/usr/bin/env python3

"""
Script to download a RealVizXL (or any HuggingFace) model
and save the full snapshot to a target directory.

Just edit the variables below, then run:
    python download_model_static.py
"""

import os
import shutil
from pathlib import Path

# --------------------------------------------------------
# 🔧 EDIT THESE TWO VARIABLES
# --------------------------------------------------------

REPO_ID = "SG161222/RealVisXL_V4.0"        # HuggingFace repo ID
TARGET_DIR = "./RealVizXL_Model"           # Local folder where model will be saved


# REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # HuggingFace repo ID
# TARGET_DIR = "./SDXL_Model"                           # Local folder where model will be saved

# Optional: HF token for private repos (or set env var HUGGINGFACE_TOKEN)
HF_TOKEN = None  # Example: "hf_abc123..." or leave None to auto-use env var


# --------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# --------------------------------------------------------

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: Install dependency:\n   pip install huggingface_hub")
        return

    token = HF_TOKEN or os.getenv("HUGGINGFACE_TOKEN")

    target = Path(TARGET_DIR).resolve()
    if target.exists():
        print(f"[INFO] Removing existing directory: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading model from {REPO_ID} ...")

    try:
        snapshot_path = snapshot_download(
            repo_id=REPO_ID,
            token=token,
            local_files_only=False
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    snapshot_path = Path(snapshot_path)
    print(f"[INFO] Cached snapshot path: {snapshot_path}")

    print(f"[INFO] Saving full model to: {target}")
    shutil.copytree(snapshot_path, target)

    print(f"[SUCCESS] Model saved at: {target}")
    print(f"Use this path for training:\n   --pretrained_model_name_or_path {target}")

if __name__ == "__main__":
    main()


REPO_ID = "SG161222/RealVisXL_V4.0"        # HuggingFace repo ID
TARGET_DIR = "./RealVizXL_Model"           # Local folder where model will be saved

# Optional: HF token for private repos (or set env var HUGGINGFACE_TOKEN)
HF_TOKEN = None  # Example: "hf_abc123..." or leave None to auto-use env var


# --------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# --------------------------------------------------------

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: Install dependency:\n   pip install huggingface_hub")
        return

    token = HF_TOKEN or os.getenv("HUGGINGFACE_TOKEN")

    target = Path(TARGET_DIR).resolve()
    if target.exists():
        print(f"[INFO] Removing existing directory: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading model from {REPO_ID} ...")

    try:
        snapshot_path = snapshot_download(
            repo_id=REPO_ID,
            token=token,
            local_files_only=False
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    snapshot_path = Path(snapshot_path)
    print(f"[INFO] Cached snapshot path: {snapshot_path}")

    print(f"[INFO] Saving full model to: {target}")
    shutil.copytree(snapshot_path, target)

    print(f"[SUCCESS] Model saved at: {target}")
    print(f"Use this path for training:\n   --pretrained_model_name_or_path {target}")

if __name__ == "__main__":
    main()
