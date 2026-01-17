
#!/usr/bin/env python3
"""
download_model.py — snapshot a Hugging Face model repo into a local folder.

Features
--------
- Downloads (or reuses cached) files for a model repo using huggingface_hub.snapshot_download
- Saves a *fully materialized* copy in the target directory (not just cache refs)
- Works with private models via HF token (env: HUGGINGFACE_TOKEN or --token)
- Supports offline mode if artifacts are already cached (--offline)
- Optional filters: --allow-patterns / --ignore-patterns
- Safe to re-run; it will reuse the same destination unless --force is provided

Examples
--------
# 1) RealViz-XL into ./RealViz-XL
python download_model.py realviz/RealViz-XL --local-dir RealViz-XL

# 2) Specific revision / branch
python download_model.py SG161222/RealVisXL_V4.0 --revision main --local-dir RealVisXL_V4.0

# 3) Use a token (private repos)
export HUGGINGFACE_TOKEN=hf_********************************
python download_model.py my/private-repo --local-dir my_model

# 4) Offline reuse of previously cached files
python download_model.py realviz/RealViz-XL --local-dir RealViz-XL --offline
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("[ERROR] huggingface_hub is required. Install with: pip install huggingface_hub", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo into the current directory.")
    parser.add_argument("repo_or_model", type=str, help="Model repo id (e.g., 'stabilityai/stable-diffusion-xl-base-1.0') or local path")
    parser.add_argument("--local-dir", type=str, default=None, help="Destination folder name (default: derive from repo name)")
    parser.add_argument("--revision", type=str, default=None, help="Branch, tag, or commit hash")
    parser.add_argument("--token", type=str, default=None, help="HF token; if not provided, uses HUGGINGFACE_TOKEN env var")
    parser.add_argument("--offline", action="store_true", help="Enable offline mode (requires files to exist in local cache)")
    parser.add_argument("--allow-patterns", nargs="*",
                        default=None, help="Only download files matching these glob patterns")
    parser.add_argument("--ignore-patterns", nargs="*",
                        default=None, help="Ignore files matching these glob patterns")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional explicit HF cache dir")
    parser.add_argument("--force", action="store_true", help="If destination exists and is non-empty, overwrite it")
    args = parser.parse_args()

    # Resolve destination directory
    repo = args.repo_or_model.rstrip("/")
    default_name = repo.split("/")[-1]
    dest = Path(args.local_dir or default_name).resolve()
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    # Token handling
    token = args.token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    # Offline / cache configuration
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads when available
    if args.cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_HOME"] = args.cache_dir

    # If destination exists
    if dest.exists():
        if args.force:
            print(f"[INFO] Removing existing destination: {dest}")
            shutil.rmtree(dest)
        else:
            if any(dest.iterdir()):
                print(f"[INFO] Destination already exists and is non-empty: {dest}\n"
                      f"       Reusing it. Use --force to overwrite.")
                sys.exit(0)

    # Perform snapshot download to cache
    try:
        print(f"[INFO] Downloading snapshot: repo={repo} revision={args.revision or 'default'} offline={args.offline}")
        snapshot_path = snapshot_download(
            repo_id=repo,
            revision=args.revision,
            local_files_only=args.offline,
            token=token,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        print(f"[ERROR] snapshot_download failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Materialize a copy of the snapshot (which lives inside cache) into dest
    snapshot_path = Path(snapshot_path)
    print(f"[INFO] Snapshot cache path: {snapshot_path}")

    # Copytree with dirs_exist_ok for Python >=3.8
    try:
        shutil.copytree(snapshot_path, dest)
    except TypeError:
        # Fallback for very old Python
        shutil.copytree(str(snapshot_path), str(dest))

    print(f"[SUCCESS] Model saved to: {dest}")
    # Print a quick contents hint
    try:
        files = sorted([p.relative_to(dest) for p in dest.rglob('*') if p.is_file()])[:10]
        if files:
            print("[INFO] Example files:")
            for f in files:
                print(f"  - {f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
