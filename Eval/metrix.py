#!/usr/bin/env python3
"""
Text-to-Face Evaluation Script
--------------------------------
Computes CLIP cosine similarity (image-image), SSIM, LPIPS, and the composite score used
in the referenced capstone report:

    composite = ((1 - lpips) + ssim + clip_cosine) / 3

It works in two modes:
1) Flat folder mode: --gen_dir contains all generated images; --gt_dir contains ground-truth images
   with matching filenames.
2) Epoch mode: --gen_dir contains subfolders per epoch (e.g., epoch_1, epoch_2, ...). The script
   evaluates each epoch separately and produces per-epoch curves and summaries.

Outputs:
- metrics.csv: per-image metrics (and epoch column if applicable)
- summary.csv: mean/std metrics overall or per epoch
- plots: mean_metrics.png, distributions.png, and (if multiple epochs) curves_by_epoch.png

Dependencies (install before running):
    pip install torch torchvision open-clip-torch lpips scikit-image pillow pandas matplotlib tqdm


"""
import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Image similarity metrics
from skimage.metrics import structural_similarity as ssim
import lpips  # Perceptual similarity

# CLIP image encoder (OpenCLIP)
import open_clip
import torchvision.transforms as T


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate text-to-face generations against ground truth.")
    p.add_argument("--gen_dir", required=True, type=str, help="Directory of generated images. Can contain epoch_* subfolders.")
    p.add_argument("--gt_dir", required=True, type=str, help="Directory of ground-truth images.")
    p.add_argument("--out_dir", required=True, type=str, help="Directory to store CSVs and plots.")
    p.add_argument("--image_exts", type=str, default=".png,.jpg,.jpeg,.webp", help="Comma-separated allowed image extensions.")
    p.add_argument("--resize", type=int, default=256, help="Resize WxH used for SSIM and LPIPS (both images are resized).")
    p.add_argument("--clip_model", type=str, default="ViT-B-16", help="OpenCLIP model name (e.g., ViT-B-16, ViT-B-32).")
    p.add_argument("--clip_pretrained", type=str, default="openai", help="OpenCLIP pretrained tag (e.g., openai, laion2b_s34b_b79k).")
    p.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    p.add_argument("--limit", type=int, default=0, help="If >0, only evaluate this many pairs per (epoch) for quick tests.")
    p.add_argument("--save_per_image_plots", action="store_true", help="Save small per-image plot strips (can be heavy).")
    return p.parse_args()


def is_image(path: Path, allowed_exts: List[str]) -> bool:
    return path.suffix.lower() in allowed_exts


def list_epoch_dirs(gen_dir: Path) -> List[Path]:
    # Detect subdirectories that look like epochs, else return empty
    subs = [d for d in gen_dir.iterdir() if d.is_dir()]
    epoch_like = [d for d in subs if d.name.lower().startswith("epoch")] 
    return sorted(epoch_like, key=lambda p: p.name)


def load_image_any(path: Path) -> Image.Image:
    img = Image.open(path)
    # Convert to RGB for consistency
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_common_filenames(gen_dir: Path, gt_dir: Path, allowed_exts: List[str]) -> List[str]:
    # Build set of basenames in gen and gt
    gen_files = {}
    for p in gen_dir.rglob("*"):
        if p.is_file() and is_image(p, allowed_exts):
            gen_files[p.name] = p

    gt_files = {}
    for p in gt_dir.rglob("*"):
        if p.is_file() and is_image(p, allowed_exts):
            gt_files[p.name] = p

    common = sorted(set(gen_files.keys()) & set(gt_files.keys()))
    return common


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # Convert PIL to float tensor [0,1]
    return T.ToTensor()(img)


def compute_ssim(img1: Image.Image, img2: Image.Image, resize: int) -> float:
    # Resize both
    img1_r = img1.resize((resize, resize), Image.BICUBIC)
    img2_r = img2.resize((resize, resize), Image.BICUBIC)
    arr1 = np.array(img1_r)
    arr2 = np.array(img2_r)
    # skimage >= 0.19 uses channel_axis instead of multichannel
    return float(ssim(arr1, arr2, channel_axis=2, data_range=255))


def compute_lpips(lpips_net, img1: Image.Image, img2: Image.Image, resize: int, device: torch.device) -> float:
    img1_r = img1.resize((resize, resize), Image.BICUBIC)
    img2_r = img2.resize((resize, resize), Image.BICUBIC)
    t1 = pil_to_tensor(img1_r).unsqueeze(0).to(device)
    t2 = pil_to_tensor(img2_r).unsqueeze(0).to(device)
    with torch.no_grad():
        d = lpips_net(t1*2-1, t2*2-1)  # LPIPS expects images in [-1,1]
    return float(d.item())


def build_clip(pretrained_model: str, pretrained_tag: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(pretrained_model, pretrained=pretrained_tag, device=device)
    model.eval()
    return model, preprocess


def compute_clip_cosine(model, preprocess, img1: Image.Image, img2: Image.Image, device: torch.device) -> float:
    # Image-image cosine similarity using CLIP image encoder
    with torch.no_grad():
        i1 = preprocess(img1).unsqueeze(0).to(device)
        i2 = preprocess(img2).unsqueeze(0).to(device)
        e1 = model.encode_image(i1)
        e2 = model.encode_image(i2)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = e2 / e2.norm(dim=-1, keepdim=True)
        cos = (e1 * e2).sum(dim=-1)
    return float(cos.item())


def evaluate_pair(paths: Tuple[Path, Path], resize: int, lpips_net, clip_model, clip_preprocess, device: torch.device) -> Dict[str, float]:
    gen_path, gt_path = paths
    gen_img = load_image_any(gen_path)
    gt_img = load_image_any(gt_path)

    # Metrics
    clip_cos = compute_clip_cosine(clip_model, clip_preprocess, gen_img, gt_img, device)
    ssim_val = compute_ssim(gen_img, gt_img, resize)
    lpips_val = compute_lpips(lpips_net, gen_img, gt_img, resize, device)

    composite = ((1.0 - lpips_val) + ssim_val + clip_cos) / 3.0

    return {
        "clip_cosine": clip_cos,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "composite": composite,
    }


def save_plots(df: pd.DataFrame, out_dir: Path, by_epoch: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean metrics bar plot (overall or per epoch)
    plt.figure(figsize=(8, 5))
    if by_epoch:
        means = df.groupby("epoch")["clip_cosine", "ssim", "lpips", "composite"].mean()
        means.rename(columns={"clip_cosine":"CLIP", "ssim":"SSIM", "lpips":"LPIPS", "composite":"Composite"}).plot(kind='bar')
        plt.title("Mean Metrics by Epoch")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(out_dir / "mean_metrics_by_epoch.png", dpi=200)
    else:
        means = df[["clip_cosine", "ssim", "lpips", "composite"]].mean()
        plt.bar(means.index, means.values)
        plt.title("Mean Metrics (Overall)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(out_dir / "mean_metrics.png", dpi=200)
    plt.close()

    # Distribution plots
    plt.figure(figsize=(10, 6))
    for col in ["clip_cosine", "ssim", "lpips", "composite"]:
        df[col].plot(kind='kde', label=col.upper())
    plt.legend()
    plt.title("Metric Distributions")
    plt.tight_layout()
    plt.savefig(out_dir / "distributions.png", dpi=200)
    plt.close()

    # If multiple epochs, line curves across epochs
    if by_epoch:
        plt.figure(figsize=(8, 5))
        means = df.groupby("epoch")["clip_cosine", "ssim", "lpips", "composite"].mean()
        for col in ["clip_cosine", "ssim", "lpips", "composite"]:
            plt.plot(means.index, means[col], marker='o', label=col.upper())
        plt.title("Mean Metrics Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score (mean)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "curves_by_epoch.png", dpi=200)
        plt.close()


def main():
    args = parse_args()
    gen_dir = Path(args.gen_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = [e.strip().lower() for e in args.image_exts.split(",") if e.strip()]

    # Device
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Build models
    print("Loading CLIP (OpenCLIP)...")
    clip_model, clip_preprocess = build_clip(args.clip_model, args.clip_pretrained, device)

    print("Loading LPIPS...")
    lpips_net = lpips.LPIPS(net='alex').to(device)
    lpips_net.eval()

    # Determine if epoch mode
    epoch_dirs = list_epoch_dirs(gen_dir)
    per_rows = []

    if epoch_dirs:
        print(f"Found {len(epoch_dirs)} epoch directories under {gen_dir}.")
        for ed in epoch_dirs:
            print(f"\nEvaluating {ed.name}")
            common_fns = get_common_filenames(ed, gt_dir, allowed_exts)
            if args.limit > 0:
                common_fns = common_fns[:args.limit]
            for fname in tqdm(common_fns, desc=f"{ed.name}"):
                gen_path = ed / fname
                gt_path = (gt_dir / fname) if (gt_dir / fname).exists() else None
                if gt_path is None:
                    continue
                metrics = evaluate_pair((gen_path, gt_path), args.resize, lpips_net, clip_model, clip_preprocess, device)
                row = {"epoch": _extract_epoch_number(ed.name), "filename": fname}
                row.update(metrics)
                per_rows.append(row)
    else:
        # Flat folder mode
        common_fns = get_common_filenames(gen_dir, gt_dir, allowed_exts)
        if args.limit > 0:
            common_fns = common_fns[:args.limit]
        for fname in tqdm(common_fns, desc="Evaluating"):
            gen_path = gen_dir / fname
            gt_path = (gt_dir / fname) if (gt_dir / fname).exists() else None
            if gt_path is None:
                continue
            metrics = evaluate_pair((gen_path, gt_path), args.resize, lpips_net, clip_model, clip_preprocess, device)
            row = {"filename": fname}
            row.update(metrics)
            per_rows.append(row)

    if not per_rows:
        print("No pairs evaluated. Check that filenames in gen_dir match those in gt_dir.")
        sys.exit(1)

    df = pd.DataFrame(per_rows)
    df.sort_values(by=[col for col in ("epoch", "filename") if col in df.columns], inplace=True)
    df.to_csv(out_dir / "metrics.csv", index=False)

    # Summaries
    if "epoch" in df.columns:
        summary = df.groupby("epoch")["clip_cosine", "ssim", "lpips", "composite"].agg(['mean','std'])
    else:
        summary = df[["clip_cosine", "ssim", "lpips", "composite"]].agg(['mean','std'])
    summary.to_csv(out_dir / "summary.csv")

    # Plots
    save_plots(df, out_dir, by_epoch=("epoch" in df.columns))

    print(f"\nSaved per-image metrics to: {out_dir / 'metrics.csv'}")
    print(f"Saved summary metrics to: {out_dir / 'summary.csv'}")
    if (out_dir / 'mean_metrics_by_epoch.png').exists():
        print(f"Saved plot: {out_dir / 'mean_metrics_by_epoch.png'}")
        print(f"Saved plot: {out_dir / 'curves_by_epoch.png'}")
    else:
        print(f"Saved plot: {out_dir / 'mean_metrics.png'}")
    print(f"Saved plot: {out_dir / 'distributions.png'}")


def _extract_epoch_number(name: str) -> int:
    # Try to parse trailing number from names like 'epoch_1', 'epoch4', etc.
    import re
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1))
    # Fallback: 0
    return 0


if __name__ == "__main__":
    main()
