
#!/usr/bin/env python3
"""
Download any Hugging Face model and save it into a target directory.

Usage:
------
python download_model.py --repo SG161222/RealVisXL_V4.0 --out /path/to/save/RealVizXL
"""

import argparse
import os
import shutil
from pathlib import Path

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: Install dependency first:\n  pip install huggingface_hub")
        return

    parser = argparse.ArgumentParser(description="Download and save a HuggingFace model repo.")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace model repo ID (e.g., SG161222/RealVisXL_V4.0)")
    parser.add_argument("--out", type=str, required=True,
                        help="Target directory to save the full model")
    parser.add_argument("--revision", type=str, default=None,
                        help="Optional: branch, tag or commit hash")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (or set env var HUGGINGFACE_TOKEN)")
    parser.add_argument("--offline", action="store_true",
                        help="Use only local cache, no internet")
    args = parser.parse_args()

    repo = args.repo
    out_dir = Path(args.out).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("HUGGINGFACE_TOKEN")

    print(f"[INFO] Downloading model '{repo}'...")
    try:
        cached_path = snapshot_download(
            repo_id=repo,
            revision=args.revision,
            local_files_only=args.offline,
            token=token
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return

    cached_path = Path(cached_path)

    # Delete old folder if exists
    if out_dir.exists():
        print(f"[INFO] Removing existing directory: {out_dir}")
        shutil.rmtree(out_dir)

    print(f"[INFO] Saving model to: {out_dir}")
    shutil.copytree(cached_path, out_dir)

    print(f"[SUCCESS] Model saved at: {out_dir}")

if __name__ == "__main__":
    main()
