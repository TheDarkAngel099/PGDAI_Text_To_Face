#!/usr/bin/env python3
"""
Compare Base Model vs Fine-tuned Model Metrics
Plot 4 metrics with two lines each (base vs fine-tuned)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
BASE_MODEL_CSV = "/home/dai01/Text_To_Face/base_model_results/metrics.csv"
FINETUNED_MODEL_CSV = "/home/dai01/Text_To_Face/metrics_results/metrics.csv"
OUTPUT_DIR = "/home/dai01/Text_To_Face/comparison_plots"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load the CSV files
print("📊 Loading CSV files...")
df_base = pd.read_csv(BASE_MODEL_CSV)
df_finetuned = pd.read_csv(FINETUNED_MODEL_CSV)

print(f"✅ Base model: {len(df_base)} samples")
print(f"✅ Fine-tuned model: {len(df_finetuned)} samples")

# Merge on filename to get matching pairs
print("\n🔗 Merging datasets on common filenames...")
df_merged = pd.merge(df_base, df_finetuned, on='filename', suffixes=('_base', '_finetuned'))
print(f"✅ Found {len(df_merged)} matching samples")

# Metrics to compare
metrics = ['clip_cosine', 'ssim', 'lpips', 'composite']
metric_labels = {
    'clip_cosine': 'CLIP Cosine Similarity',
    'ssim': 'SSIM (Structural Similarity)',
    'lpips': 'LPIPS (Perceptual Distance)',
    'composite': 'Composite Score'
}

# Create 4 subplots (2x2 grid) - KDE Distribution plots like metrix.py
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    base_col = f"{metric}_base"
    finetuned_col = f"{metric}_finetuned"
    
    # Get values
    base_values = df_merged[base_col]
    finetuned_values = df_merged[finetuned_col]
    
    # Plot KDE distributions (like metrix.py)
    base_values.plot(kind='kde', ax=ax, label='Base Model', color='#E74C3C', linewidth=3, alpha=0.8)
    finetuned_values.plot(kind='kde', ax=ax, label='Fine-tuned Model', color='#3498DB', linewidth=3, alpha=0.8)
    
    # Calculate statistics
    base_mean = base_values.mean()
    finetuned_mean = finetuned_values.mean()
    improvement = ((finetuned_mean - base_mean) / base_mean) * 100 if metric != 'lpips' else ((base_mean - finetuned_mean) / base_mean) * 100
    
    # Add vertical lines for means
    ax.axvline(base_mean, color='#E74C3C', linestyle='--', alpha=0.6, linewidth=2, label=f'Base Mean: {base_mean:.3f}')
    ax.axvline(finetuned_mean, color='#3498DB', linestyle='--', alpha=0.6, linewidth=2, label=f'Fine-tuned Mean: {finetuned_mean:.3f}')
    
    # Styling
    ax.set_title(f"{metric_labels[metric]}\n(Improvement: {improvement:+.1f}%)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Special note for LPIPS (lower is better)
    if metric == 'lpips':
        ax.text(0.5, 0.95, '(Lower is Better)', transform=ax.transAxes, 
                ha='center', fontsize=9, style='italic', color='gray', va='top')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_comparison_4plots.png", dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {OUTPUT_DIR}/metrics_comparison_4plots.png")
plt.close()

# Create individual KDE distribution plots for each metric (like metrix.py)
print("\n📊 Creating individual metric distribution plots...")
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    base_col = f"{metric}_base"
    finetuned_col = f"{metric}_finetuned"
    
    base_values = df_merged[base_col]
    finetuned_values = df_merged[finetuned_col]
    
    # Plot KDE distributions
    base_values.plot(kind='kde', ax=ax, label='Base Model', color='#E74C3C', linewidth=3, alpha=0.8)
    finetuned_values.plot(kind='kde', ax=ax, label='Fine-tuned Model', color='#3498DB', linewidth=3, alpha=0.8)
    
    # Calculate statistics
    base_mean = base_values.mean()
    finetuned_mean = finetuned_values.mean()
    base_std = base_values.std()
    finetuned_std = finetuned_values.std()
    improvement = ((finetuned_mean - base_mean) / base_mean) * 100 if metric != 'lpips' else ((base_mean - finetuned_mean) / base_mean) * 100
    
    # Add vertical lines for means
    ax.axvline(base_mean, color='#E74C3C', linestyle='--', alpha=0.6, linewidth=2.5, label=f'Base Mean: {base_mean:.4f} (±{base_std:.4f})')
    ax.axvline(finetuned_mean, color='#3498DB', linestyle='--', alpha=0.6, linewidth=2.5, label=f'Fine-tuned Mean: {finetuned_mean:.4f} (±{finetuned_std:.4f})')
    
    # Title with improvement
    title = f"{metric_labels[metric]} Distribution Comparison\n"
    if metric == 'lpips':
        title += f"Fine-tuned model shows {improvement:+.1f}% improvement (Lower LPIPS = Better)"
    else:
        title += f"Fine-tuned model shows {improvement:+.1f}% improvement"
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{metric_labels[metric]} Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{metric}_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/{metric}_comparison.png")
    plt.close()

# Create summary statistics table
print("\n📊 Summary Statistics:")
print("=" * 80)
summary_data = []
for metric in metrics:
    base_col = f"{metric}_base"
    finetuned_col = f"{metric}_finetuned"
    
    base_mean = df_merged[base_col].mean()
    base_std = df_merged[base_col].std()
    finetuned_mean = df_merged[finetuned_col].mean()
    finetuned_std = df_merged[finetuned_col].std()
    
    if metric == 'lpips':
        improvement = ((base_mean - finetuned_mean) / base_mean) * 100  # Lower is better
    else:
        improvement = ((finetuned_mean - base_mean) / base_mean) * 100
    
    summary_data.append({
        'Metric': metric_labels[metric],
        'Base Mean': f"{base_mean:.4f}",
        'Base Std': f"{base_std:.4f}",
        'Fine-tuned Mean': f"{finetuned_mean:.4f}",
        'Fine-tuned Std': f"{finetuned_std:.4f}",
        'Improvement (%)': f"{improvement:+.2f}%"
    })
    
    print(f"{metric_labels[metric]:30s} | Base: {base_mean:.4f}±{base_std:.4f} | Fine-tuned: {finetuned_mean:.4f}±{finetuned_std:.4f} | Improvement: {improvement:+.2f}%")

print("=" * 80)

# Save summary to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{OUTPUT_DIR}/summary_comparison.csv", index=False)
print(f"\n✅ Summary saved to: {OUTPUT_DIR}/summary_comparison.csv")

# Create mean metrics bar chart (like metrix.py style)
fig, ax = plt.subplots(figsize=(10, 6))

# Create DataFrame for easier plotting
mean_data = pd.DataFrame({
    'Base Model': [df_merged[f"{m}_base"].mean() for m in metrics],
    'Fine-tuned Model': [df_merged[f"{m}_finetuned"].mean() for m in metrics]
}, index=[metric_labels[m] for m in metrics])

mean_data.plot(kind='bar', ax=ax, color=['#E74C3C', '#3498DB'], alpha=0.8, width=0.7)

ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Mean Metrics: Base Model vs Fine-tuned Model', fontsize=16, fontweight='bold')
ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mean_metrics_bar_chart.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/mean_metrics_bar_chart.png")
plt.close()

print("\n" + "=" * 80)
print("🎉 All comparison plots generated successfully!")
print(f"📂 Output directory: {OUTPUT_DIR}")
print("=" * 80)
