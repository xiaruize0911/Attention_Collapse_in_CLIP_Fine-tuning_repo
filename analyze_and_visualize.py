#!/usr/bin/env python3
"""
Comprehensive Analysis & Visualization Script
==============================================
Generates all 9 planned figures + summary tables for the CLIP attention
structural collapse experiment report.

Figures:
  Fig 1: Project overview (skipped - conceptual diagram)
  Fig 2: Per-layer entropy heatmap (baseline)
  Fig 3: Accuracy + Entropy + Zero-shot triple curve (Full FT)
  Fig 4: Layer-wise entropy heatmap over training steps
  Fig 5: Full FT vs LoRA collapse comparison
  Fig 6: Learning rate vs collapse speed (A1 sweep)
  Fig 7: Collapse degree vs zero-shot accuracy scatter
  Fig 8: Regularization effectiveness (APR / Entropy Floor / WD)
  Fig 9: Head diversity evolution
  Extra: ERF@0.95 evolution, Gini coefficient trends, frozen layers
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

METRICS_DIR = 'outputs/metrics'
FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def load_history(name):
    """Load a history JSON file."""
    path = os.path.join(METRICS_DIR, f'{name}_history.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_metric_series(history, metric_key):
    """Extract a time series of a metric from attention_metrics list."""
    if history is None:
        return []
    return [am[metric_key] for am in history['attention_metrics'] if metric_key in am]

def get_layer_metric_series(history, metric_key):
    """Extract per-layer metric series -> shape (T, 12)."""
    if history is None:
        return np.array([])
    vals = []
    for am in history['attention_metrics']:
        if metric_key in am:
            vals.append(am[metric_key])
    return np.array(vals)

def normalize_steps(n):
    """Create normalized step axis [0, 1] for n data points."""
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0, 1, n)

# ============================================================
# Load all data
# ============================================================
print("Loading all experiment histories...")

# Core experiments
E2 = load_history('E2_full_ft_eurosat')
E3 = load_history('E3_full_ft_pets')
E4 = load_history('E4_lora_r4_eurosat')
E5 = load_history('E5_lora_r8_eurosat')
E6 = load_history('E6_lora_r16_eurosat')
E7 = load_history('E7_lora_r8_pets')

# LR sweep
A1_1e6 = load_history('A1_lr_1e-06')
A1_5e6 = load_history('A1_lr_5e-06')
A1_5e5 = load_history('A1_lr_5e-05')
A1_1e4 = load_history('A1_lr_0.0001')

# LoRA QKVO
A2 = load_history('A2_lora_qkvo')

# Frozen layers
A3_3 = load_history('A3_freeze_3')
A3_6 = load_history('A3_freeze_6')
A3_9 = load_history('A3_freeze_9')

# Regularization
R1_001 = load_history('reg_apr_lambda0.01_eurosat')
R1_01 = load_history('reg_apr_lambda0.1_eurosat')
R1_10 = load_history('reg_apr_lambda1.0_eurosat')
R2_001 = load_history('reg_entropy_floor_lambda0.01_eurosat')
R2_01 = load_history('reg_entropy_floor_lambda0.1_eurosat')
R3_00 = load_history('R3_wd_0.0')
R3_01 = load_history('R3_wd_0.1')

# Zero-shot results
zs_baseline = load_json(os.path.join(METRICS_DIR, 'zero_shot_baseline_results.json'))
zs_e2 = load_json(os.path.join(METRICS_DIR, 'zero_shot_E2_full_ft_eurosat_results.json'))
zs_e3 = load_json(os.path.join(METRICS_DIR, 'zero_shot_E3_full_ft_pets_results.json'))
zs_e5 = load_json(os.path.join(METRICS_DIR, 'zero_shot_E5_lora_r8_eurosat_results.json'))

# Baseline stats
baseline_stats = load_json(os.path.join(METRICS_DIR, 'E1_baseline_stats.json'))

print("All data loaded.\n")

# ============================================================
# Fig 2: Baseline Per-Layer Attention Structure
# ============================================================
print("Generating Fig 2: Baseline per-layer attention structure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fig 2: Pre-trained CLIP ViT-B/32 Baseline Attention Structure', fontweight='bold', fontsize=14)

summary = baseline_stats['summary']
layers = list(range(1, 13))

# Entropy per layer
ax = axes[0, 0]
ax.bar(layers, summary['entropy_per_layer'], color='steelblue', alpha=0.8)
ax.axhline(summary['entropy_mean'], color='red', linestyle='--', label=f"Mean={summary['entropy_mean']:.4f}")
ax.set_xlabel('Layer')
ax.set_ylabel('Attention Entropy (bits)')
ax.set_title('Per-Layer Attention Entropy')
ax.set_xticks(layers)
ax.legend()

# ERF@0.95 per layer
ax = axes[0, 1]
ax.bar(layers, summary['erf95_per_layer'], color='darkorange', alpha=0.8)
ax.axhline(summary['erf95_mean'], color='red', linestyle='--', label=f"Mean={summary['erf95_mean']:.4f}")
ax.set_xlabel('Layer')
ax.set_ylabel('ERF@0.95')
ax.set_title('Per-Layer Effective Receptive Field')
ax.set_xticks(layers)
ax.legend()

# Gini per layer
ax = axes[1, 0]
ax.bar(layers, summary['gini_per_layer'], color='seagreen', alpha=0.8)
ax.axhline(summary['gini_mean'], color='red', linestyle='--', label=f"Mean={summary['gini_mean']:.4f}")
ax.set_xlabel('Layer')
ax.set_ylabel('Gini Coefficient')
ax.set_title('Per-Layer Attention Concentration (Gini)')
ax.set_xticks(layers)
ax.legend()

# Head diversity per layer
ax = axes[1, 1]
ax.bar(layers, summary['head_diversity_per_layer'], color='mediumpurple', alpha=0.8)
ax.axhline(summary['head_diversity_mean'], color='red', linestyle='--', label=f"Mean={summary['head_diversity_mean']:.4f}")
ax.set_xlabel('Layer')
ax.set_ylabel('Head Diversity (cosine)')
ax.set_title('Per-Layer Head Diversity')
ax.set_xticks(layers)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_baseline_structure.png'))
plt.close()
print("  Saved fig2_baseline_structure.png")

# ============================================================
# Fig 3: Triple Curve - Accuracy, Entropy, Zero-shot
# ============================================================
print("Generating Fig 3: Accuracy + Entropy triple curve...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fig 3: Accuracy ↑, Entropy ↓, Zero-shot Capacity ↓ During Fine-tuning', fontweight='bold', fontsize=14)

for idx, (hist, title, zs_val) in enumerate([
    (E2, 'Full FT on EuroSAT (20 epochs)', zs_e2['cifar100']),
    (E3, 'Full FT on Pets (30 epochs)', zs_e3['cifar100']),
]):
    ax1 = axes[idx]
    
    # Use epochs for accuracy (1 per epoch)
    n_epochs = len(hist['val_acc'])
    epoch_x = np.arange(1, n_epochs + 1)
    
    # Accuracy (left y-axis)
    color_acc = 'tab:blue'
    ax1.plot(epoch_x, hist['val_acc'], color=color_acc, linewidth=2, label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy', color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.3, 1.05)
    
    # Entropy (right y-axis) - attention_metrics has more entries, use normalized x
    ax2 = ax1.twinx()
    entropy_series = get_metric_series(hist, 'entropy_mean')
    color_ent = 'tab:red'
    # Map attention metric steps to epoch scale
    ent_x = np.linspace(0, n_epochs, len(entropy_series))
    ax2.plot(ent_x, entropy_series, color=color_ent, linewidth=2, linestyle='--', label='Entropy (mean)')
    ax2.set_ylabel('Attention Entropy', color=color_ent)
    ax2.tick_params(axis='y', labelcolor=color_ent)
    
    # Zero-shot annotation
    baseline_zs = zs_baseline['cifar100']
    ax1.annotate(f'Zero-shot: {baseline_zs:.1%} → {zs_val:.1%}',
                xy=(0.5, 0.02), xycoords='axes fraction', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                ha='center')
    
    ax1.set_title(title)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_triple_curve.png'))
plt.close()
print("  Saved fig3_triple_curve.png")

# ============================================================
# Fig 4: Layer-wise Entropy Heatmap Over Training
# ============================================================
print("Generating Fig 4: Layer-wise entropy heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Fig 4: Layer-wise Attention Entropy Evolution During Training', fontweight='bold', fontsize=14)

for idx, (hist, title) in enumerate([
    (E2, 'E2: Full FT EuroSAT'),
    (E3, 'E3: Full FT Pets'),
    (E5, 'E5: LoRA r=8 EuroSAT'),
    (E7, 'E7: LoRA r=8 Pets'),
]):
    ax = axes[idx // 2, idx % 2]
    layer_entropy = get_layer_metric_series(hist, 'entropy_per_layer')
    if layer_entropy.size == 0:
        continue
    
    im = ax.imshow(layer_entropy.T, aspect='auto', cmap='RdYlBu', origin='lower',
                   extent=[0, layer_entropy.shape[0]-1, 0.5, 12.5])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    ax.set_yticks(range(1, 13))
    plt.colorbar(im, ax=ax, label='Entropy')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_layer_entropy_heatmap.png'))
plt.close()
print("  Saved fig4_layer_entropy_heatmap.png")

# ============================================================
# Fig 5: Full FT vs LoRA Collapse Comparison
# ============================================================
print("Generating Fig 5: Full FT vs LoRA comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Fig 5: Full Fine-tuning vs LoRA — Attention Collapse Comparison (EuroSAT)', fontweight='bold', fontsize=14)

experiments = [
    (E2, 'Full FT', 'tab:red'),
    (E4, 'LoRA r=4', 'tab:blue'),
    (E5, 'LoRA r=8', 'tab:green'),
    (E6, 'LoRA r=16', 'tab:orange'),
]

metrics_to_plot = [
    ('entropy_mean', 'Attention Entropy', axes[0, 0]),
    ('erf95_mean', 'ERF@0.95', axes[0, 1]),
    ('gini_mean', 'Gini Coefficient', axes[1, 0]),
    ('head_diversity_mean', 'Head Diversity', axes[1, 1]),
]

for metric_key, metric_label, ax in metrics_to_plot:
    for hist, label, color in experiments:
        series = get_metric_series(hist, metric_key)
        if not series:
            continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_fullft_vs_lora.png'))
plt.close()
print("  Saved fig5_fullft_vs_lora.png")

# Also do Pets comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 5b: Full FT vs LoRA — Attention Collapse Comparison (Pets)', fontweight='bold', fontsize=14)

pets_experiments = [
    (E3, 'Full FT', 'tab:red'),
    (E7, 'LoRA r=8', 'tab:green'),
]

for metric_key, metric_label, ax in [
    ('entropy_mean', 'Attention Entropy', axes[0]),
    ('gini_mean', 'Gini Coefficient', axes[1]),
]:
    for hist, label, color in pets_experiments:
        series = get_metric_series(hist, metric_key)
        if not series:
            continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5b_fullft_vs_lora_pets.png'))
plt.close()
print("  Saved fig5b_fullft_vs_lora_pets.png")

# ============================================================
# Fig 6: Learning Rate vs Collapse Speed
# ============================================================
print("Generating Fig 6: Learning rate sweep analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Fig 6: Learning Rate vs Attention Collapse (EuroSAT Full FT)', fontweight='bold', fontsize=14)

lr_experiments = [
    (A1_1e6, 'lr=1e-6', 'tab:blue'),
    (A1_5e6, 'lr=5e-6', 'tab:green'),
    (E2,     'lr=1e-5 (default)', 'tab:orange'),
    (A1_5e5, 'lr=5e-5', 'tab:red'),
    (A1_1e4, 'lr=1e-4', 'tab:purple'),
]

# Entropy evolution
ax = axes[0]
for hist, label, color in lr_experiments:
    series = get_metric_series(hist, 'entropy_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Attention Entropy')
ax.set_title('Entropy Evolution')
ax.legend(fontsize=8)

# Gini evolution
ax = axes[1]
for hist, label, color in lr_experiments:
    series = get_metric_series(hist, 'gini_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Gini Coefficient')
ax.set_title('Gini Evolution')
ax.legend(fontsize=8)

# Final entropy vs LR bar chart
ax = axes[2]
lr_labels = ['1e-6', '5e-6', '1e-5', '5e-5', '1e-4']
final_entropies = []
baseline_ent = E2['baseline_metrics']['entropy_mean']
for hist in [A1_1e6, A1_5e6, E2, A1_5e5, A1_1e4]:
    final_entropies.append(hist['final_metrics']['entropy_mean'])

entropy_changes = [(fe - baseline_ent) / baseline_ent * 100 for fe in final_entropies]
colors = ['green' if c > 0 else 'red' for c in entropy_changes]
bars = ax.bar(lr_labels, entropy_changes, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Entropy Change (%)')
ax.set_title('Final Entropy Change vs Baseline')
for bar, val in zip(bars, entropy_changes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:+.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_lr_sweep.png'))
plt.close()
print("  Saved fig6_lr_sweep.png")

# ============================================================
# Fig 7: Collapse Degree vs Zero-shot Accuracy Scatter
# ============================================================
print("Generating Fig 7: Collapse vs zero-shot scatter...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fig 7: Attention Structural Change vs Zero-shot Transfer Performance', fontweight='bold', fontsize=14)

# Data points
models = ['Baseline', 'Full FT\nEuroSAT', 'Full FT\nPets', 'LoRA r=8\nEuroSAT']
zs_accs = [
    zs_baseline['cifar100'],
    zs_e2['cifar100'],
    zs_e3['cifar100'],
    zs_e5['cifar100'],
]
# Compute entropy change from baseline
entropy_changes_abs = [0]  # baseline
for hist in [E2, E3, E5]:
    bl = hist['baseline_metrics']['entropy_mean']
    fi = hist['final_metrics']['entropy_mean']
    entropy_changes_abs.append(fi - bl)

# Scatter: entropy change vs zero-shot
ax = axes[0]
colors_scatter = ['black', 'red', 'darkred', 'blue']
markers = ['o', 's', '^', 'D']
for i, (name, zs, ec) in enumerate(zip(models, zs_accs, entropy_changes_abs)):
    ax.scatter(ec, zs, c=colors_scatter[i], s=120, marker=markers[i], label=name, zorder=5)
ax.set_xlabel('Entropy Change (final - baseline)')
ax.set_ylabel('CIFAR-100 Zero-shot Accuracy')
ax.set_title('Entropy Change vs Zero-shot')
ax.legend()
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

# Bar chart comparison
ax = axes[1]
x_pos = np.arange(len(models))
bars = ax.bar(x_pos, [z*100 for z in zs_accs], color=['gray', 'red', 'darkred', 'blue'], alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('CIFAR-100 Zero-shot Accuracy (%)')
ax.set_title('Zero-shot Transfer Comparison')
for bar, val in zip(bars, zs_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_collapse_vs_zeroshot.png'))
plt.close()
print("  Saved fig7_collapse_vs_zeroshot.png")

# ============================================================
# Fig 8: Regularization Effectiveness
# ============================================================
print("Generating Fig 8: Regularization effectiveness...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fig 8: Regularization Methods — Effect on Attention Structure', fontweight='bold', fontsize=14)

# Row 1: Entropy evolution
# APR
ax = axes[0, 0]
for hist, label, color in [
    (E2,     'No reg (baseline)', 'gray'),
    (R1_001, 'APR λ=0.01', 'tab:blue'),
    (R1_01,  'APR λ=0.1', 'tab:green'),
    (R1_10,  'APR λ=1.0', 'tab:red'),
]:
    series = get_metric_series(hist, 'entropy_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Entropy')
ax.set_title('APR Regularization — Entropy')
ax.legend(fontsize=8)

# Entropy Floor
ax = axes[0, 1]
for hist, label, color in [
    (E2,     'No reg (baseline)', 'gray'),
    (R2_001, 'EntFloor λ=0.01', 'tab:blue'),
    (R2_01,  'EntFloor λ=0.1', 'tab:green'),
]:
    series = get_metric_series(hist, 'entropy_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Entropy')
ax.set_title('Entropy Floor Regularization — Entropy')
ax.legend(fontsize=8)

# Weight Decay
ax = axes[0, 2]
for hist, label, color in [
    (E2,    'WD=0.01 (default)', 'gray'),
    (R3_00, 'WD=0.0', 'tab:blue'),
    (R3_01, 'WD=0.1', 'tab:red'),
]:
    series = get_metric_series(hist, 'entropy_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Entropy')
ax.set_title('Weight Decay — Entropy')
ax.legend(fontsize=8)

# Row 2: Accuracy comparison bars
# Collect final accuracy and entropy for all regularization methods
reg_configs = [
    ('No Reg', E2),
    ('APR 0.01', R1_001),
    ('APR 0.1', R1_01),
    ('APR 1.0', R1_10),
    ('EntFloor 0.01', R2_001),
    ('EntFloor 0.1', R2_01),
    ('WD 0.0', R3_00),
    ('WD 0.1', R3_01),
]

names = [n for n, _ in reg_configs]
best_accs = [h['best_val_acc'] for _, h in reg_configs]
final_ents = [h['final_metrics']['entropy_mean'] for _, h in reg_configs]
bl_ent = E2['baseline_metrics']['entropy_mean']
ent_changes_pct = [(fe - bl_ent) / bl_ent * 100 for fe in final_ents]

# Accuracy bars
ax = axes[1, 0]
x = np.arange(len(names))
bars = ax.bar(x, [a * 100 for a in best_accs], color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Best Val Accuracy (%)')
ax.set_title('Task Accuracy')
ax.set_ylim(97, 100)

# Entropy change bars
ax = axes[1, 1]
colors = ['green' if c > 0 else 'red' for c in ent_changes_pct]
bars = ax.bar(x, ent_changes_pct, color=colors, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Entropy Change (%)')
ax.set_title('Entropy Preservation')
ax.axhline(0, color='black', linewidth=0.8)

# Accuracy vs Entropy scatter for reg methods
ax = axes[1, 2]
for i, (name, _) in enumerate(reg_configs):
    color = 'red' if i == 0 else 'steelblue'
    marker = 's' if i == 0 else 'o'
    ax.scatter(ent_changes_pct[i], best_accs[i] * 100, c=color, s=80, marker=marker, zorder=5)
    ax.annotate(name, (ent_changes_pct[i], best_accs[i] * 100), fontsize=7,
                textcoords='offset points', xytext=(5, 5))
ax.set_xlabel('Entropy Change (%)')
ax.set_ylabel('Best Val Accuracy (%)')
ax.set_title('Accuracy vs Entropy Preservation')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig8_regularization.png'))
plt.close()
print("  Saved fig8_regularization.png")

# ============================================================
# Fig 9: Head Diversity Evolution
# ============================================================
print("Generating Fig 9: Head diversity evolution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 9: Attention Head Diversity During Training', fontweight='bold', fontsize=14)

# EuroSAT comparison
ax = axes[0]
for hist, label, color in [
    (E2, 'Full FT', 'tab:red'),
    (E4, 'LoRA r=4', 'tab:blue'),
    (E5, 'LoRA r=8', 'tab:green'),
    (E6, 'LoRA r=16', 'tab:orange'),
]:
    series = get_metric_series(hist, 'head_diversity_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Head Diversity (cosine distance)')
ax.set_title('EuroSAT')
ax.legend()

# Pets comparison
ax = axes[1]
for hist, label, color in [
    (E3, 'Full FT', 'tab:red'),
    (E7, 'LoRA r=8', 'tab:green'),
]:
    series = get_metric_series(hist, 'head_diversity_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('Head Diversity (cosine distance)')
ax.set_title('Oxford-IIIT Pets')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig9_head_diversity.png'))
plt.close()
print("  Saved fig9_head_diversity.png")

# ============================================================
# Extra Fig: ERF@0.95 Evolution
# ============================================================
print("Generating Extra Fig: ERF@0.95 evolution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ERF@0.95 Evolution During Training', fontweight='bold', fontsize=14)

ax = axes[0]
for hist, label, color in [
    (E2, 'Full FT', 'tab:red'),
    (E4, 'LoRA r=4', 'tab:blue'),
    (E5, 'LoRA r=8', 'tab:green'),
    (E6, 'LoRA r=16', 'tab:orange'),
]:
    series = get_metric_series(hist, 'erf95_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('ERF@0.95')
ax.set_title('EuroSAT')
ax.legend()

ax = axes[1]
for hist, label, color in [
    (E3, 'Full FT', 'tab:red'),
    (E7, 'LoRA r=8', 'tab:green'),
]:
    series = get_metric_series(hist, 'erf95_mean')
    if not series:
        continue
    x = normalize_steps(len(series))
    ax.plot(x, series, label=label, color=color, linewidth=1.5)
ax.set_xlabel('Normalized Training Progress')
ax.set_ylabel('ERF@0.95')
ax.set_title('Oxford-IIIT Pets')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'extra_erf95_evolution.png'))
plt.close()
print("  Saved extra_erf95_evolution.png")

# ============================================================
# Extra Fig: Frozen Layers Analysis
# ============================================================
print("Generating Extra Fig: Frozen layers analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Frozen Layers Analysis (EuroSAT)', fontweight='bold', fontsize=14)

freeze_experiments = [
    (E2,   'No Freeze', 'tab:red'),
    (A3_3, 'Freeze 3 layers', 'tab:blue'),
    (A3_6, 'Freeze 6 layers', 'tab:green'),
    (A3_9, 'Freeze 9 layers', 'tab:orange'),
]

for metric_key, metric_label, ax in [
    ('entropy_mean', 'Entropy', axes[0]),
    ('gini_mean', 'Gini', axes[1]),
    ('head_diversity_mean', 'Head Diversity', axes[2]),
]:
    for hist, label, color in freeze_experiments:
        series = get_metric_series(hist, metric_key)
        if not series:
            continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'extra_frozen_layers.png'))
plt.close()
print("  Saved extra_frozen_layers.png")

# ============================================================
# Extra Fig: LoRA QKVO comparison
# ============================================================
print("Generating Extra Fig: LoRA QKVO analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LoRA Target Modules Comparison (EuroSAT)', fontweight='bold', fontsize=14)

qkvo_experiments = [
    (E5, 'LoRA QV (default r=8)', 'tab:blue'),
    (A2, 'LoRA QKVO (r=8)', 'tab:red'),
]

for metric_key, metric_label, ax in [
    ('entropy_mean', 'Entropy', axes[0]),
    ('head_diversity_mean', 'Head Diversity', axes[1]),
]:
    for hist, label, color in qkvo_experiments:
        series = get_metric_series(hist, metric_key)
        if not series:
            continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'extra_lora_qkvo.png'))
plt.close()
print("  Saved extra_lora_qkvo.png")

# ============================================================
# Grand Summary Table
# ============================================================
print("\n" + "="*100)
print("GRAND SUMMARY TABLE")
print("="*100)

all_experiments = [
    ('E2: Full FT EuroSAT', E2),
    ('E3: Full FT Pets', E3),
    ('E4: LoRA r=4 EuroSAT', E4),
    ('E5: LoRA r=8 EuroSAT', E5),
    ('E6: LoRA r=16 EuroSAT', E6),
    ('E7: LoRA r=8 Pets', E7),
    ('A1: lr=1e-6', A1_1e6),
    ('A1: lr=5e-6', A1_5e6),
    ('A1: lr=5e-5', A1_5e5),
    ('A1: lr=1e-4', A1_1e4),
    ('A2: LoRA QKVO', A2),
    ('A3: Freeze 3', A3_3),
    ('A3: Freeze 6', A3_6),
    ('A3: Freeze 9', A3_9),
    ('R1: APR λ=0.01', R1_001),
    ('R1: APR λ=0.1', R1_01),
    ('R1: APR λ=1.0', R1_10),
    ('R2: EntFloor λ=0.01', R2_001),
    ('R2: EntFloor λ=0.1', R2_01),
    ('R3: WD=0.0', R3_00),
    ('R3: WD=0.1', R3_01),
]

print(f"{'Experiment':<28s} {'Best Acc':>10s} {'Ent Base':>10s} {'Ent Final':>10s} {'ΔEnt%':>8s} {'ERF Base':>10s} {'ERF Final':>10s} {'Gini Base':>10s} {'Gini Final':>10s} {'HDiv Base':>10s} {'HDiv Final':>10s}")
print("-" * 138)

for name, hist in all_experiments:
    if hist is None:
        continue
    bm = hist['baseline_metrics']
    fm = hist['final_metrics']
    bacc = hist['best_val_acc']
    ent_change = (fm['entropy_mean'] - bm['entropy_mean']) / bm['entropy_mean'] * 100
    print(f"{name:<28s} {bacc:>10.4f} {bm['entropy_mean']:>10.4f} {fm['entropy_mean']:>10.4f} {ent_change:>+8.2f} {bm['erf95_mean']:>10.4f} {fm['erf95_mean']:>10.4f} {bm['gini_mean']:>10.4f} {fm['gini_mean']:>10.4f} {bm['head_diversity_mean']:>10.4f} {fm['head_diversity_mean']:>10.4f}")

print()
print("="*80)
print("ZERO-SHOT TRANSFER RESULTS (CIFAR-100)")
print("="*80)
zs_results = {
    'Baseline (pre-trained)': zs_baseline['cifar100'],
    'Full FT EuroSAT': zs_e2['cifar100'],
    'Full FT Pets': zs_e3['cifar100'],
    'LoRA r=8 EuroSAT': zs_e5['cifar100'],
}
for name, acc in zs_results.items():
    print(f"  {name:<30s}: {acc:.4f} ({acc:.1%})")

print()
print("="*80)
print("KEY FINDINGS")
print("="*80)

# Compute statistics
full_ft_entropy_changes = []
lora_entropy_changes = []
for name, hist in all_experiments:
    if hist is None:
        continue
    bm = hist['baseline_metrics']
    fm = hist['final_metrics']
    ec = (fm['entropy_mean'] - bm['entropy_mean']) / bm['entropy_mean'] * 100
    if 'Full FT' in name or name.startswith('E2') or name.startswith('E3'):
        full_ft_entropy_changes.append(ec)
    if 'LoRA' in name or name.startswith('E4') or name.startswith('E5') or name.startswith('E6') or name.startswith('E7'):
        lora_entropy_changes.append(ec)

print(f"\n1. FULL FT entropy change range: {min(full_ft_entropy_changes):.2f}% to {max(full_ft_entropy_changes):.2f}%")
print(f"   Mean change: {np.mean(full_ft_entropy_changes):.2f}%")
print(f"\n2. LoRA entropy change range: {min(lora_entropy_changes):.2f}% to {max(lora_entropy_changes):.2f}%")
print(f"   Mean change: {np.mean(lora_entropy_changes):.2f}%")
print(f"\n3. Zero-shot catastrophic forgetting:")
print(f"   Full FT EuroSAT: {zs_baseline['cifar100']:.1%} → {zs_e2['cifar100']:.1%} ({(zs_e2['cifar100'] - zs_baseline['cifar100']) / zs_baseline['cifar100'] * 100:+.1f}%)")
print(f"   Full FT Pets:    {zs_baseline['cifar100']:.1%} → {zs_e3['cifar100']:.1%} ({(zs_e3['cifar100'] - zs_baseline['cifar100']) / zs_baseline['cifar100'] * 100:+.1f}%)")
print(f"   LoRA r=8:        {zs_baseline['cifar100']:.1%} → {zs_e5['cifar100']:.1%} ({(zs_e5['cifar100'] - zs_baseline['cifar100']) / zs_baseline['cifar100'] * 100:+.1f}%)")

# APR effectiveness
print(f"\n4. APR Regularization - entropy preservation:")
for name, hist in [('λ=0.01', R1_001), ('λ=0.1', R1_01), ('λ=1.0', R1_10)]:
    ec = (hist['final_metrics']['entropy_mean'] - hist['baseline_metrics']['entropy_mean']) / hist['baseline_metrics']['entropy_mean'] * 100
    print(f"   APR {name}: entropy change = {ec:+.2f}%, val_acc = {hist['best_val_acc']:.4f}")

# LR effect
print(f"\n5. Learning Rate effect on entropy:")
for name, hist in [('1e-6', A1_1e6), ('5e-6', A1_5e6), ('1e-5', E2), ('5e-5', A1_5e5), ('1e-4', A1_1e4)]:
    ec = (hist['final_metrics']['entropy_mean'] - hist['baseline_metrics']['entropy_mean']) / hist['baseline_metrics']['entropy_mean'] * 100
    print(f"   lr={name}: entropy change = {ec:+.2f}%, val_acc = {hist['best_val_acc']:.4f}")

print(f"\n6. Correlation: Higher LR → more entropy decrease (stronger collapse)")
lrs_log = [np.log10(lr) for lr in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]]
ent_changes_lr = []
for hist in [A1_1e6, A1_5e6, E2, A1_5e5, A1_1e4]:
    ec = (hist['final_metrics']['entropy_mean'] - hist['baseline_metrics']['entropy_mean']) / hist['baseline_metrics']['entropy_mean'] * 100
    ent_changes_lr.append(ec)
r, p = stats.pearsonr(lrs_log, ent_changes_lr)
print(f"   Pearson r = {r:.4f}, p-value = {p:.6f}")

print("\n\nAll figures saved to outputs/figures/")
print("Analysis complete!")
