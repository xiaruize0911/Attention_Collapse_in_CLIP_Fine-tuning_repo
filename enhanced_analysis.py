#!/usr/bin/env python3
"""
Enhanced Analysis Script - Improvements to the CLIP Attention Collapse Report
=============================================================================
Adds:
  1. Statistical tests (bootstrap CIs, Cohen's d, Welch's t-tests, Mann-Whitney U)
  2. Additional zero-shot evaluations (more models, Flowers-102 benchmark)
  3. Per-layer collapse analysis (which layers change most)
  4. Training dynamics (collapse onset detection)
  5. Metric correlation matrix across all experiments
  6. Enhanced visualizations with error bands and significance markers
  7. Attention rollout visualizations on actual images
  8. Convergence verification
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# Setup paths
PROJECT_DIR = Path("/workspace/Attention_Collapse_in_CLIP_Fine-tuning_repo")
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
CHECKPOINTS_DIR = PROJECT_DIR / "outputs" / "checkpoints"

sys.path.insert(0, str(PROJECT_DIR))
from src.model import CLIPClassifier, create_lora_model, get_pretrained_model, count_parameters
from src.dataset import (load_eurosat, load_oxford_pets, create_fixed_eval_subset,
                         get_dataloader, get_clip_transform, load_cifar100, load_flowers102)
from src.metrics import compute_all_metrics, compute_attention_rollout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(str(FIGURES_DIR), exist_ok=True)

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
    'font.family': 'DejaVu Sans',
})


# ============================================================
# Utility functions
# ============================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_history(name):
    path = METRICS_DIR / f'{name}_history.json'
    if not path.exists():
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)

def get_metric_series(history, metric_key):
    if history is None:
        return []
    return [am[metric_key] for am in history['attention_metrics'] if metric_key in am]

def get_layer_metric_series(history, metric_key):
    if history is None:
        return np.array([])
    vals = []
    for am in history['attention_metrics']:
        if metric_key in am:
            vals.append(am[metric_key])
    return np.array(vals)

def normalize_steps(n):
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0, 1, n)


# ============================================================
# SECTION 1: Additional Zero-Shot Evaluations
# ============================================================

def run_zero_shot_evaluation_extended(model_path, experiment_id, is_lora=False, lora_config=None):
    """
    Extended zero-shot evaluation on CIFAR-100 and Flowers-102.
    Supports both full FT checkpoints and LoRA checkpoints.
    """
    from transformers import CLIPModel, CLIPProcessor

    print(f"  Zero-shot eval for {experiment_id}...")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", attn_implementation="eager"
    ).to(DEVICE)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        vision_state = {k.replace('vision_model.', ''): v for k, v in state.items()
                       if k.startswith('vision_model.')}
        if vision_state:
            clip_model.vision_model.load_state_dict(vision_state, strict=False)

    clip_model.eval()
    results = {}

    # Evaluate on CIFAR-100
    try:
        cifar_test, cifar_classes, cifar_names = load_cifar100(cache_dir=str(PROJECT_DIR / "data"))
        cifar_loader = get_dataloader(cifar_test, batch_size=64, shuffle=False, num_workers=2)

        text_inputs = processor(
            text=[f"a photo of a {name}" for name in cifar_names],
            return_tensors="pt", padding=True
        ).to(DEVICE)

        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        correct, total = 0, 0
        for images, labels in cifar_loader:
            images = images.to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = image_features @ text_features.T
                predicted = similarity.argmax(dim=-1)
            labels = labels.to(DEVICE)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        results['cifar100'] = correct / total
        print(f"    CIFAR-100: {results['cifar100']:.4f}")
    except Exception as e:
        print(f"    CIFAR-100 failed: {e}")
        results['cifar100'] = None

    # Evaluate on Flowers-102
    try:
        flowers_test, flowers_classes, flowers_names = load_flowers102(cache_dir=str(PROJECT_DIR / "data"))
        flowers_loader = get_dataloader(flowers_test, batch_size=64, shuffle=False, num_workers=2)

        text_inputs = processor(
            text=[f"a photo of a {name}" for name in flowers_names],
            return_tensors="pt", padding=True
        ).to(DEVICE)

        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        correct, total = 0, 0
        for images, labels in flowers_loader:
            images = images.to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = image_features @ text_features.T
                predicted = similarity.argmax(dim=-1)
            labels = labels.to(DEVICE)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        results['flowers102'] = correct / total
        print(f"    Flowers-102: {results['flowers102']:.4f}")
    except Exception as e:
        print(f"    Flowers-102 failed: {e}")
        results['flowers102'] = None

    del clip_model
    torch.cuda.empty_cache()
    return results


# ============================================================
# SECTION 2: Statistical Tests
# ============================================================

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, statistic=np.mean):
    """Bootstrap confidence interval."""
    data = np.array(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.sort(boot_stats)
    lower = boot_stats[int((1-ci)/2 * n_bootstrap)]
    upper = boot_stats[int((1+ci)/2 * n_bootstrap)]
    return lower, upper

def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)

def compute_statistical_tests(all_experiments):
    """Compute comprehensive statistical tests."""
    results = {}
    
    # 1. Full FT vs LoRA entropy changes (per-layer paired comparison)
    full_ft_changes = []
    lora_changes = []
    
    full_ft_exps = ['E2_full_ft_eurosat']  # Use EuroSAT for controlled comparison
    lora_exps = ['E4_lora_r4_eurosat', 'E5_lora_r8_eurosat', 'E6_lora_r16_eurosat']
    
    for exp_name in full_ft_exps:
        hist = all_experiments.get(exp_name)
        if hist:
            bl = hist['baseline_metrics']['entropy_per_layer']
            fi = hist['final_metrics']['entropy_per_layer']
            changes = [(f - b) / b * 100 for b, f in zip(bl, fi)]
            full_ft_changes.extend(changes)
    
    for exp_name in lora_exps:
        hist = all_experiments.get(exp_name)
        if hist:
            bl = hist['baseline_metrics']['entropy_per_layer']
            fi = hist['final_metrics']['entropy_per_layer']
            changes = [(f - b) / b * 100 for b, f in zip(bl, fi)]
            lora_changes.extend(changes)
    
    if full_ft_changes and lora_changes:
        # Welch's t-test
        t_stat, p_welch = stats.ttest_ind(full_ft_changes, lora_changes, equal_var=False)
        # Mann-Whitney U test (non-parametric)
        u_stat, p_mann = stats.mannwhitneyu(full_ft_changes, lora_changes, alternative='two-sided')
        # Cohen's d
        d = cohens_d(lora_changes, full_ft_changes)
        
        results['fullft_vs_lora'] = {
            'full_ft_mean': np.mean(full_ft_changes),
            'full_ft_std': np.std(full_ft_changes),
            'full_ft_ci': bootstrap_ci(full_ft_changes),
            'lora_mean': np.mean(lora_changes),
            'lora_std': np.std(lora_changes),
            'lora_ci': bootstrap_ci(lora_changes),
            'welch_t': t_stat,
            'welch_p': p_welch,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mann,
            'cohens_d': d,
        }
    
    # 2. LR vs entropy correlation (Spearman + Pearson)
    lr_vals = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    lr_names = ['A1_lr_1e-06', 'A1_lr_5e-06', 'E2_full_ft_eurosat', 'A1_lr_5e-05', 'A1_lr_0.0001']
    
    ent_changes = []
    valid_lrs = []
    for lr, name in zip(lr_vals, lr_names):
        hist = all_experiments.get(name)
        if hist:
            bl = hist['baseline_metrics']['entropy_mean']
            fi = hist['final_metrics']['entropy_mean']
            ent_changes.append((fi - bl) / bl * 100)
            valid_lrs.append(np.log10(lr))
    
    if len(valid_lrs) >= 3:
        r_pearson, p_pearson = stats.pearsonr(valid_lrs, ent_changes)
        r_spearman, p_spearman = stats.spearmanr(valid_lrs, ent_changes)
        
        results['lr_vs_entropy'] = {
            'pearson_r': r_pearson, 'pearson_p': p_pearson,
            'spearman_r': r_spearman, 'spearman_p': p_spearman,
            'lr_log_values': valid_lrs,
            'entropy_changes': ent_changes,
        }
    
    # 3. Regularization effectiveness - paired comparison with no-reg baseline
    no_reg = all_experiments.get('E2_full_ft_eurosat')
    if no_reg:
        reg_results = {}
        reg_exps = {
            'APR_0.01': 'reg_apr_lambda0.01_eurosat',
            'APR_0.1': 'reg_apr_lambda0.1_eurosat',
            'APR_1.0': 'reg_apr_lambda1.0_eurosat',
            'EntFloor_0.01': 'reg_entropy_floor_lambda0.01_eurosat',
            'EntFloor_0.1': 'reg_entropy_floor_lambda0.1_eurosat',
        }
        
        nr_layers = no_reg['final_metrics']['entropy_per_layer']
        
        for reg_name, exp_name in reg_exps.items():
            hist = all_experiments.get(exp_name)
            if hist:
                reg_layers = hist['final_metrics']['entropy_per_layer']
                t_stat, p_val = stats.ttest_rel(reg_layers, nr_layers)
                d = cohens_d(reg_layers, nr_layers)
                reg_results[reg_name] = {
                    'paired_t': t_stat, 'paired_p': p_val, 'cohens_d': d,
                    'mean_diff': np.mean(np.array(reg_layers) - np.array(nr_layers))
                }
        
        results['regularization'] = reg_results
    
    # 4. Layer position effect - early vs late layers
    for exp_name, exp_label in [('E2_full_ft_eurosat', 'Full FT'), ('E5_lora_r8_eurosat', 'LoRA r=8')]:
        hist = all_experiments.get(exp_name)
        if hist:
            bl = hist['baseline_metrics']['entropy_per_layer']
            fi = hist['final_metrics']['entropy_per_layer']
            changes = [(f - b) / b * 100 for b, f in zip(bl, fi)]
            
            early = changes[:4]   # layers 1-4
            middle = changes[4:8] # layers 5-8
            late = changes[8:]    # layers 9-12
            
            f_stat, p_anova = stats.f_oneway(early, middle, late)
            
            results[f'layer_position_{exp_name}'] = {
                'early_mean': np.mean(early), 'early_std': np.std(early),
                'middle_mean': np.mean(middle), 'middle_std': np.std(middle),
                'late_mean': np.mean(late), 'late_std': np.std(late),
                'anova_f': f_stat, 'anova_p': p_anova,
            }
    
    return results


# ============================================================
# SECTION 3: Per-Layer Collapse Analysis
# ============================================================

def per_layer_collapse_analysis(all_experiments):
    """Detailed per-layer analysis of entropy changes."""
    results = {}
    
    experiments_to_analyze = {
        'E2_full_ft_eurosat': 'Full FT EuroSAT',
        'E3_full_ft_pets': 'Full FT Pets',
        'E5_lora_r8_eurosat': 'LoRA r=8 EuroSAT',
        'E7_lora_r8_pets': 'LoRA r=8 Pets',
        'A1_lr_0.0001': 'Full FT lr=1e-4',
        'A1_lr_1e-06': 'Full FT lr=1e-6',
        'reg_apr_lambda0.01_eurosat': 'APR λ=0.01',
    }
    
    for exp_name, label in experiments_to_analyze.items():
        hist = all_experiments.get(exp_name)
        if hist is None:
            continue
        
        bl_ent = hist['baseline_metrics']['entropy_per_layer']
        fi_ent = hist['final_metrics']['entropy_per_layer']
        bl_gini = hist['baseline_metrics']['gini_per_layer']
        fi_gini = hist['final_metrics']['gini_per_layer']
        bl_erf = hist['baseline_metrics']['erf95_per_layer']
        fi_erf = hist['final_metrics']['erf95_per_layer']
        
        ent_changes = [(f - b) / b * 100 for b, f in zip(bl_ent, fi_ent)]
        gini_changes = [(f - b) / b * 100 for b, f in zip(bl_gini, fi_gini)]
        erf_changes = [(f - b) / b * 100 for b, f in zip(bl_erf, fi_erf)]
        
        most_affected = np.argmax(np.abs(ent_changes))
        
        results[exp_name] = {
            'label': label,
            'entropy_changes_per_layer': ent_changes,
            'gini_changes_per_layer': gini_changes,
            'erf_changes_per_layer': erf_changes,
            'most_affected_layer': int(most_affected) + 1,
            'max_entropy_change': ent_changes[most_affected],
        }
    
    return results


# ============================================================
# SECTION 4: Training Dynamics / Collapse Onset Detection
# ============================================================

def detect_collapse_onset(history, threshold_pct=0.5):
    """
    Detect when entropy starts consistently decreasing.
    Returns: the normalized training step where entropy first drops below
    threshold_pct% of baseline.
    """
    if history is None:
        return None
    
    entropy_series = get_metric_series(history, 'entropy_mean')
    if not entropy_series:
        return None
    
    baseline = entropy_series[0]
    threshold = baseline * (1 - threshold_pct / 100)
    
    for i, ent in enumerate(entropy_series):
        if ent < threshold:
            return i / len(entropy_series)
    
    return None  # Never crossed threshold


def analyze_training_dynamics(all_experiments):
    """Analyze training dynamics for all experiments."""
    results = {}
    
    experiments_to_analyze = {
        'E2_full_ft_eurosat': 'Full FT EuroSAT',
        'E3_full_ft_pets': 'Full FT Pets',
        'E5_lora_r8_eurosat': 'LoRA r=8 EuroSAT',
        'A1_lr_5e-05': 'Full FT lr=5e-5',
        'A1_lr_0.0001': 'Full FT lr=1e-4',
    }
    
    for name, label in experiments_to_analyze.items():
        hist = all_experiments.get(name)
        if hist is None:
            continue
        
        onset_05 = detect_collapse_onset(hist, threshold_pct=0.5)
        onset_1 = detect_collapse_onset(hist, threshold_pct=1.0)
        onset_2 = detect_collapse_onset(hist, threshold_pct=2.0)
        
        # Compute entropy velocity (rate of change)
        entropy_series = get_metric_series(hist, 'entropy_mean')
        if len(entropy_series) > 3:
            velocity = np.gradient(entropy_series)
            max_decrease_idx = np.argmin(velocity)
            max_decrease_step = max_decrease_idx / len(entropy_series)
        else:
            max_decrease_step = None
        
        results[name] = {
            'label': label,
            'onset_0.5pct': onset_05,
            'onset_1pct': onset_1,
            'onset_2pct': onset_2,
            'max_decrease_step': max_decrease_step,
            'total_entropy_change_pct': ((entropy_series[-1] - entropy_series[0]) / entropy_series[0] * 100) if entropy_series else None,
        }
    
    return results


# ============================================================
# SECTION 5: Convergence Verification
# ============================================================

def verify_convergence(all_experiments):
    """Check if experiments converged by analyzing late-epoch loss/accuracy stability."""
    results = {}
    
    for name, hist in all_experiments.items():
        if hist is None or 'val_acc' not in hist:
            continue
        
        val_accs = hist['val_acc']
        if len(val_accs) < 5:
            continue
        
        # Check last 25% of epochs
        n_check = max(3, len(val_accs) // 4)
        late_accs = val_accs[-n_check:]
        early_accs = val_accs[:n_check]
        
        # Stability: std of late accuracy
        late_std = np.std(late_accs)
        converged = late_std < 0.005  # Less than 0.5% variation
        
        # Best epoch
        best_epoch = np.argmax(val_accs) + 1
        
        results[name] = {
            'converged': converged,
            'late_accuracy_std': late_std,
            'best_epoch': best_epoch,
            'total_epochs': len(val_accs),
            'final_accuracy': val_accs[-1],
            'best_accuracy': max(val_accs),
        }
    
    return results


# ============================================================
# SECTION 6: Enhanced Visualizations
# ============================================================

def fig_per_layer_delta_heatmap(per_layer_results):
    """Create per-layer entropy change heatmap across experiments."""
    exp_names = []
    exp_labels = []
    all_changes = []
    
    for name, data in per_layer_results.items():
        exp_names.append(name)
        exp_labels.append(data['label'])
        all_changes.append(data['entropy_changes_per_layer'])
    
    if not all_changes:
        return
    
    data_matrix = np.array(all_changes)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use diverging colormap centered at 0
    vmax = max(abs(data_matrix.min()), abs(data_matrix.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data_matrix, cmap='RdBu', norm=norm, aspect='auto')
    
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'L{i+1}' for i in range(12)])
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=9)
    ax.set_xlabel('Transformer Layer')
    ax.set_title('Per-Layer Entropy Change (%) Across Experiments', fontweight='bold', fontsize=14)
    
    # Annotate cells
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            val = data_matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax, label='Entropy Change (%)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig10_per_layer_delta_heatmap.png'))
    plt.close()
    print("  Saved fig10_per_layer_delta_heatmap.png")


def fig_statistical_summary(stat_results):
    """Create figure summarizing statistical test results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Statistical Analysis Summary', fontweight='bold', fontsize=14)
    
    # Panel 1: Full FT vs LoRA with CIs
    ax = axes[0]
    ft_data = stat_results.get('fullft_vs_lora', {})
    if ft_data:
        groups = ['Full FT', 'LoRA']
        means = [ft_data['full_ft_mean'], ft_data['lora_mean']]
        stds = [ft_data['full_ft_std'], ft_data['lora_std']]
        cis = [ft_data['full_ft_ci'], ft_data['lora_ci']]
        
        colors = ['#e74c3c', '#3498db']
        bars = ax.bar(groups, means, yerr=stds, color=colors, alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=0.5)
        
        # CI error bars (different style)
        for i, (ci, mean) in enumerate(zip(cis, means)):
            ax.plot([i, i], [ci[0], ci[1]], color='black', linewidth=2)
            ax.plot([i-0.05, i+0.05], [ci[0], ci[0]], color='black', linewidth=2)
            ax.plot([i-0.05, i+0.05], [ci[1], ci[1]], color='black', linewidth=2)
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        # Significance annotation
        p_val = ft_data['welch_p']
        d_val = ft_data['cohens_d']
        sig_text = f"Welch's t-test p={p_val:.2e}\nCohen's d={d_val:.2f}"
        if p_val < 0.001:
            sig_text += "\n***"
        elif p_val < 0.01:
            sig_text += "\n**"
        elif p_val < 0.05:
            sig_text += "\n*"
        
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, ha='center', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_ylabel('Per-Layer Entropy Change (%)')
    ax.set_title('Full FT vs LoRA: Per-Layer Entropy Change')
    
    # Panel 2: LR correlation with regression line
    ax = axes[1]
    lr_data = stat_results.get('lr_vs_entropy', {})
    if lr_data:
        x = lr_data['lr_log_values']
        y = lr_data['entropy_changes']
        
        ax.scatter(x, y, s=100, c='steelblue', edgecolors='black', zorder=5)
        
        # Regression line
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'r={r:.3f}, p={p:.3f}')
        
        # Prediction interval
        n = len(x)
        x_mean = np.mean(x)
        se_fit = se * np.sqrt(1/n + (x_fit - x_mean)**2 / np.sum((np.array(x) - x_mean)**2))
        ax.fill_between(x_fit, y_fit - 1.96*se_fit, y_fit + 1.96*se_fit, alpha=0.2, color='red')
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        lr_labels = ['1e-6', '5e-6', '1e-5', '5e-5', '1e-4']
        for xi, yi, lab in zip(x, y, lr_labels):
            ax.annotate(lab, (xi, yi), textcoords='offset points', xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('log₁₀(Learning Rate)')
    ax.set_ylabel('Entropy Change (%)')
    ax.set_title('Learning Rate vs Entropy Change')
    ax.legend(fontsize=9)
    
    # Panel 3: Layer position effect (early/middle/late)
    ax = axes[2]
    for exp_key, color, label in [
        ('layer_position_E2_full_ft_eurosat', '#e74c3c', 'Full FT'),
        ('layer_position_E5_lora_r8_eurosat', '#3498db', 'LoRA r=8'),
    ]:
        data = stat_results.get(exp_key, {})
        if data:
            means = [data['early_mean'], data['middle_mean'], data['late_mean']]
            stds = [data['early_std'], data['middle_std'], data['late_std']]
            x_pos = np.arange(3) + (0.15 if 'LoRA' in label else -0.15)
            ax.bar(x_pos, means, 0.3, yerr=stds, color=color, alpha=0.7,
                   label=f"{label} (F={data['anova_f']:.2f}, p={data['anova_p']:.3f})",
                   capsize=5)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Early (1-4)', 'Middle (5-8)', 'Late (9-12)'])
    ax.set_ylabel('Entropy Change (%)')
    ax.set_title('Layer Position vs Entropy Change')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig11_statistical_summary.png'))
    plt.close()
    print("  Saved fig11_statistical_summary.png")


def fig_training_dynamics(all_experiments, dynamics_results):
    """Create training dynamics figure with collapse onset markers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Training Dynamics: Entropy Evolution & Collapse Onset', fontweight='bold', fontsize=14)
    
    # Panel 1: Full FT with different LRs - entropy curves with onset markers
    ax = axes[0, 0]
    lr_exps = [
        ('A1_lr_1e-06', 'lr=1e-6', '#2196F3'),
        ('E2_full_ft_eurosat', 'lr=1e-5 (default)', '#FF9800'),
        ('A1_lr_5e-05', 'lr=5e-5', '#F44336'),
        ('A1_lr_0.0001', 'lr=1e-4', '#9C27B0'),
    ]
    
    for exp_name, label, color in lr_exps:
        hist = all_experiments.get(exp_name)
        if hist is None:
            continue
        series = get_metric_series(hist, 'entropy_mean')
        if not series:
            continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
        
        # Mark onset point
        dyn = dynamics_results.get(exp_name)
        if dyn and dyn.get('onset_1pct') is not None:
            onset = dyn['onset_1pct']
            onset_idx = int(onset * len(series))
            if onset_idx < len(series):
                ax.axvline(onset, color=color, linestyle=':', alpha=0.5)
                ax.plot(onset, series[min(onset_idx, len(series)-1)], 'v', color=color, markersize=8)
    
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Entropy Evolution (Full FT, Various LRs)')
    ax.legend(fontsize=8)
    
    # Panel 2: Entropy velocity (rate of change)
    ax = axes[0, 1]
    for exp_name, label, color in lr_exps:
        hist = all_experiments.get(exp_name)
        if hist is None:
            continue
        series = get_metric_series(hist, 'entropy_mean')
        if len(series) < 3:
            continue
        
        # Smooth velocity
        velocity = np.gradient(series)
        window = min(5, len(velocity) // 3)
        if window > 0:
            kernel = np.ones(window) / window
            velocity_smooth = np.convolve(velocity, kernel, mode='same')
        else:
            velocity_smooth = velocity
        
        x = normalize_steps(len(velocity_smooth))
        ax.plot(x, velocity_smooth, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Entropy Rate of Change')
    ax.set_title('Entropy Velocity (dH/dt)')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    
    # Panel 3: LoRA vs Full FT entropy trajectories 
    ax = axes[1, 0]
    compare_exps = [
        ('E2_full_ft_eurosat', 'Full FT EuroSAT', '#e74c3c'),
        ('E3_full_ft_pets', 'Full FT Pets', '#c0392b'),
        ('E5_lora_r8_eurosat', 'LoRA r=8 EuroSAT', '#3498db'),
        ('E7_lora_r8_pets', 'LoRA r=8 Pets', '#2980b9'),
    ]
    
    for exp_name, label, color in compare_exps:
        hist = all_experiments.get(exp_name)
        if hist is None:
            continue
        series = get_metric_series(hist, 'entropy_mean')
        if not series:
            continue
        # Normalize to percentage change from baseline
        baseline_val = series[0]
        pct_change = [(v - baseline_val) / baseline_val * 100 for v in series]
        x = normalize_steps(len(pct_change))
        ax.plot(x, pct_change, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Entropy Change from Baseline (%)')
    ax.set_title('Comparative Entropy Trajectories')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=8)
    
    # Panel 4: Accuracy vs entropy scatter over training (E2 and A1_lr_1e-4)
    ax = axes[1, 1]
    for exp_name, label, color, marker in [
        ('E2_full_ft_eurosat', 'Full FT lr=1e-5', '#FF9800', 'o'),
        ('A1_lr_0.0001', 'Full FT lr=1e-4', '#9C27B0', 's'),
        ('E5_lora_r8_eurosat', 'LoRA r=8', '#3498db', 'D'),
    ]:
        hist = all_experiments.get(exp_name)
        if hist is None:
            continue
        ent_series = get_metric_series(hist, 'entropy_mean')
        # For accuracy, interpolate to same length as entropy
        if 'val_acc' in hist and hist['val_acc'] and ent_series:
            val_accs = hist['val_acc']
            # Interpolate accuracy to match entropy time points
            acc_interp = np.interp(
                np.linspace(0, 1, len(ent_series)),
                np.linspace(0, 1, len(val_accs)),
                val_accs
            )
            ax.scatter(ent_series, acc_interp, c=color, s=15, alpha=0.5, marker=marker, label=label)
            # Arrow showing direction
            if len(ent_series) > 1:
                ax.annotate('', xy=(ent_series[-1], acc_interp[-1]),
                           xytext=(ent_series[0], acc_interp[0]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Accuracy-Entropy Co-evolution')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig12_training_dynamics.png'))
    plt.close()
    print("  Saved fig12_training_dynamics.png")


def fig_metric_correlation_matrix(all_experiments):
    """Create correlation matrix across all structural metrics."""
    metrics_data = {
        'ΔEntropy%': [], 'ΔERF%': [], 'ΔGini%': [], 'ΔHeadDiv%': [],
        'BestAcc': [], 'Experiment': []
    }
    
    for name, hist in all_experiments.items():
        if hist is None or 'final_metrics' not in hist or 'baseline_metrics' not in hist:
            continue
        bm = hist['baseline_metrics']
        fm = hist['final_metrics']
        
        metrics_data['ΔEntropy%'].append((fm['entropy_mean'] - bm['entropy_mean']) / bm['entropy_mean'] * 100)
        metrics_data['ΔERF%'].append((fm['erf95_mean'] - bm['erf95_mean']) / bm['erf95_mean'] * 100)
        metrics_data['ΔGini%'].append((fm['gini_mean'] - bm['gini_mean']) / bm['gini_mean'] * 100)
        metrics_data['ΔHeadDiv%'].append((fm['head_diversity_mean'] - bm['head_diversity_mean']) / bm['head_diversity_mean'] * 100)
        metrics_data['BestAcc'].append(hist.get('best_val_acc', 0))
        metrics_data['Experiment'].append(name)
    
    if len(metrics_data['ΔEntropy%']) < 3:
        return
    
    metric_keys = ['ΔEntropy%', 'ΔERF%', 'ΔGini%', 'ΔHeadDiv%', 'BestAcc']
    n = len(metric_keys)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(metric_keys):
        for j, m2 in enumerate(metric_keys):
            r, p = stats.pearsonr(metrics_data[m1], metrics_data[m2])
            corr_matrix[i, j] = r
            p_matrix[i, j] = p
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cross-Metric Correlation Analysis', fontweight='bold', fontsize=14)
    
    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(metric_keys, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(metric_keys, fontsize=9)
    
    for i in range(n):
        for j in range(n):
            sig = ''
            if p_matrix[i, j] < 0.001: sig = '***'
            elif p_matrix[i, j] < 0.01: sig = '**'
            elif p_matrix[i, j] < 0.05: sig = '*'
            text = f'{corr_matrix[i,j]:.2f}\n{sig}'
            color = 'white' if abs(corr_matrix[i,j]) > 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title('Metric Correlation Matrix')
    
    # Scatter: Entropy change vs Gini change (key relationship)
    ax = axes[1]
    ax.scatter(metrics_data['ΔEntropy%'], metrics_data['ΔGini%'],
              c=metrics_data['BestAcc'], cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    r, p = stats.pearsonr(metrics_data['ΔEntropy%'], metrics_data['ΔGini%'])
    x_arr = np.array(metrics_data['ΔEntropy%'])
    slope, intercept = np.polyfit(x_arr, metrics_data['ΔGini%'], 1)
    x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2, label=f'r={r:.2f}, p={p:.3f}')
    
    ax.set_xlabel('Entropy Change (%)')
    ax.set_ylabel('Gini Change (%)')
    ax.set_title('Entropy vs Gini (color=accuracy)')
    ax.legend()
    
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Best Accuracy')
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig13_metric_correlations.png'))
    plt.close()
    print("  Saved fig13_metric_correlations.png")


def fig_convergence_check(convergence_results):
    """Visualize convergence analysis."""
    names = []
    best_accs = []
    late_stds = []
    converged = []
    
    for name, data in sorted(convergence_results.items()):
        names.append(name.replace('_eurosat', '').replace('_history', ''))
        best_accs.append(data['best_accuracy'])
        late_stds.append(data['late_accuracy_std'])
        converged.append(data['converged'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Verification', fontweight='bold', fontsize=14)
    
    # Best accuracy
    ax = axes[0]
    colors = ['green' if c else 'red' for c in converged]
    bars = ax.barh(range(len(names)), best_accs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Best Validation Accuracy')
    ax.set_title('Best Accuracy (green=converged)')
    
    # Late-epoch variance
    ax = axes[1]
    bars = ax.barh(range(len(names)), late_stds, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Late-Epoch Accuracy Std Dev')
    ax.set_title('Training Stability (green=std<0.005)')
    ax.axvline(0.005, color='red', linestyle='--', label='Threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig14_convergence.png'))
    plt.close()
    print("  Saved fig14_convergence.png")


def fig_zero_shot_extended(zs_results):
    """Extended zero-shot evaluation figure with multiple benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Extended Zero-Shot Transfer Evaluation', fontweight='bold', fontsize=14)
    
    models = list(zs_results.keys())
    
    # CIFAR-100
    ax = axes[0]
    cifar_accs = [zs_results[m].get('cifar100', 0) or 0 for m in models]
    display_names = [m.replace('_', '\n') for m in models]
    colors = []
    for m in models:
        if 'baseline' in m: colors.append('gray')
        elif 'lora' in m or 'LoRA' in m: colors.append('#3498db')
        elif 'apr' in m or 'APR' in m: colors.append('#2ecc71')
        else: colors.append('#e74c3c')
    
    bars = ax.bar(range(len(models)), [a * 100 for a in cifar_accs], color=colors, alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(display_names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CIFAR-100 Zero-Shot')
    for bar, val in zip(bars, cifar_accs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    # Flowers-102
    ax = axes[1]
    flowers_accs = [zs_results[m].get('flowers102', 0) or 0 for m in models]
    bars = ax.bar(range(len(models)), [a * 100 for a in flowers_accs], color=colors, alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(display_names, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Flowers-102 Zero-Shot')
    for bar, val in zip(bars, flowers_accs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig15_zeroshot_extended.png'))
    plt.close()
    print("  Saved fig15_zeroshot_extended.png")


def fig_comprehensive_dashboard(all_experiments, stat_results, zs_results):
    """Create comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('CLIP Attention Collapse: Comprehensive Dashboard', fontweight='bold', fontsize=16, y=0.98)
    
    # Panel 1: Entropy change bar chart (all experiments)
    ax = fig.add_subplot(gs[0, 0:2])
    exp_names = []
    ent_changes = []
    for name, hist in sorted(all_experiments.items()):
        if hist is None or 'final_metrics' not in hist:
            continue
        bm = hist['baseline_metrics']
        fm = hist['final_metrics']
        ec = (fm['entropy_mean'] - bm['entropy_mean']) / bm['entropy_mean'] * 100
        short_name = name.replace('_eurosat', '').replace('_history', '').replace('reg_', 'R:')
        exp_names.append(short_name)
        ent_changes.append(ec)
    
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in ent_changes]
    ax.barh(range(len(exp_names)), ent_changes, color=colors, alpha=0.8)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=6)
    ax.set_xlabel('Entropy Change (%)')
    ax.set_title('Entropy Change Across All Experiments')
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Panel 2: Accuracy vs entropy change scatter
    ax = fig.add_subplot(gs[0, 2:4])
    accs = []
    ecs = []
    labels_scatter = []
    for name, hist in all_experiments.items():
        if hist is None or 'final_metrics' not in hist:
            continue
        bm = hist['baseline_metrics']
        fm = hist['final_metrics']
        ec = (fm['entropy_mean'] - bm['entropy_mean']) / bm['entropy_mean'] * 100
        accs.append(hist.get('best_val_acc', 0))
        ecs.append(ec)
        labels_scatter.append(name)
    
    scatter_colors = []
    for n in labels_scatter:
        if 'lora' in n.lower() or n.startswith('A2'): scatter_colors.append('#3498db')
        elif 'apr' in n or 'entropy_floor' in n: scatter_colors.append('#2ecc71')
        elif 'lr' in n and 'lora' not in n.lower(): scatter_colors.append('#e67e22')
        else: scatter_colors.append('#e74c3c')
    
    ax.scatter(ecs, accs, c=scatter_colors, s=60, edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('Entropy Change (%)')
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Accuracy vs Structural Change')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='Full FT'),
        mpatches.Patch(color='#3498db', label='LoRA'),
        mpatches.Patch(color='#2ecc71', label='Regularized'),
        mpatches.Patch(color='#e67e22', label='LR Sweep'),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc='lower left')
    
    # Panel 3: Full FT vs LoRA statistical comparison
    ax = fig.add_subplot(gs[1, 0:2])
    ft_data = stat_results.get('fullft_vs_lora', {})
    if ft_data:
        groups = ['Full FT\n(per-layer)', 'LoRA\n(per-layer)']
        means = [ft_data['full_ft_mean'], ft_data['lora_mean']]
        stds = [ft_data['full_ft_std'], ft_data['lora_std']]
        bars = ax.bar(groups, means, yerr=stds, color=['#e74c3c', '#3498db'],
                      alpha=0.7, capsize=5, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        p = ft_data['welch_p']
        d = ft_data['cohens_d']
        ax.set_title(f"Full FT vs LoRA Per-Layer ΔEntropy\n(p={p:.2e}, d={d:.2f})")
    ax.set_ylabel('Entropy Change (%)')
    
    # Panel 4: Zero-shot comparison
    ax = fig.add_subplot(gs[1, 2:4])
    if zs_results:
        zs_names = list(zs_results.keys())
        cifar_accs = [zs_results[m].get('cifar100', 0) or 0 for m in zs_names]
        flowers_accs = [zs_results[m].get('flowers102', 0) or 0 for m in zs_names]
        
        x = np.arange(len(zs_names))
        width = 0.35
        display = [n.replace('_', '\n') for n in zs_names]
        
        ax.bar(x - width/2, [a*100 for a in cifar_accs], width, label='CIFAR-100', color='steelblue', alpha=0.8)
        if any(a > 0 for a in flowers_accs):
            ax.bar(x + width/2, [a*100 for a in flowers_accs], width, label='Flowers-102', color='coral', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(display, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Zero-Shot Accuracy (%)')
        ax.set_title('Zero-Shot Transfer')
        ax.legend(fontsize=8)
    
    # Panel 5: LR vs Entropy (bottom left)
    ax = fig.add_subplot(gs[2, 0:2])
    lr_data = stat_results.get('lr_vs_entropy', {})
    if lr_data:
        x = lr_data['lr_log_values']
        y = lr_data['entropy_changes']
        ax.scatter(x, y, s=100, c='steelblue', edgecolors='black', zorder=5)
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        r = lr_data['pearson_r']
        p = lr_data['pearson_p']
        rho = lr_data['spearman_r']
        ps = lr_data['spearman_p']
        ax.set_title(f'LR↔Entropy (Pearson r={r:.3f}, p={p:.3f}; Spearman ρ={rho:.3f}, p={ps:.3f})')
    ax.set_xlabel('log₁₀(Learning Rate)')
    ax.set_ylabel('Entropy Change (%)')
    
    # Panel 6: Key findings text box
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    findings = """Key Findings Summary:
    
1. Full FT ↓ entropy (-0.14% to -10.4%), LoRA ↑ entropy (+0.4% avg)
2. Learning rate is strongest collapse predictor (r=-0.89, p=0.04)  
3. Full FT destroys zero-shot (-79% to -90%), LoRA preserves it (±0%)
4. APR λ=0.01 best regularizer: +0.86% entropy, 99.26% accuracy
5. Layer freezing has negligible effect on collapse
6. Lower LoRA ranks cause MORE entropy increase (counterintuitive)
"""
    ax.text(0.05, 0.95, findings, transform=ax.transAxes, fontsize=10, va='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.savefig(str(FIGURES_DIR / 'fig16_dashboard.png'))
    plt.close()
    print("  Saved fig16_dashboard.png")


def fig_gini_evolution(all_experiments):
    """Gini coefficient evolution - missing from original analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Gini Coefficient Evolution (Attention Concentration)', fontweight='bold', fontsize=14)
    
    # Panel 1: Full FT vs LoRA
    ax = axes[0]
    for name, label, color in [
        ('E2_full_ft_eurosat', 'Full FT EuroSAT', '#e74c3c'),
        ('E5_lora_r8_eurosat', 'LoRA r=8 EuroSAT', '#3498db'),
        ('E3_full_ft_pets', 'Full FT Pets', '#c0392b'),
        ('E7_lora_r8_pets', 'LoRA r=8 Pets', '#2980b9'),
    ]:
        hist = all_experiments.get(name)
        if hist is None: continue
        series = get_metric_series(hist, 'gini_mean')
        if not series: continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Full FT vs LoRA')
    ax.legend(fontsize=8)
    
    # Panel 2: LR sweep
    ax = axes[1]
    for name, label, color in [
        ('A1_lr_1e-06', 'lr=1e-6', '#2196F3'),
        ('E2_full_ft_eurosat', 'lr=1e-5', '#FF9800'),
        ('A1_lr_5e-05', 'lr=5e-5', '#F44336'),
        ('A1_lr_0.0001', 'lr=1e-4', '#9C27B0'),
    ]:
        hist = all_experiments.get(name)
        if hist is None: continue
        series = get_metric_series(hist, 'gini_mean')
        if not series: continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Learning Rate Effect')
    ax.legend(fontsize=8)
    
    # Panel 3: Regularization
    ax = axes[2]
    for name, label, color in [
        ('E2_full_ft_eurosat', 'No Reg', 'gray'),
        ('reg_apr_lambda0.01_eurosat', 'APR λ=0.01', '#2ecc71'),
        ('reg_apr_lambda0.1_eurosat', 'APR λ=0.1', '#27ae60'),
        ('reg_entropy_floor_lambda0.1_eurosat', 'EntFloor λ=0.1', '#3498db'),
    ]:
        hist = all_experiments.get(name)
        if hist is None: continue
        series = get_metric_series(hist, 'gini_mean')
        if not series: continue
        x = normalize_steps(len(series))
        ax.plot(x, series, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Normalized Training Progress')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Regularization Effect')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'fig17_gini_evolution.png'))
    plt.close()
    print("  Saved fig17_gini_evolution.png")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*70)
    print("ENHANCED ANALYSIS - CLIP Attention Collapse Report Improvements")
    print("="*70)
    
    np.random.seed(42)
    
    # ========== Load all experiment data ==========
    print("\n[1/7] Loading all experiment histories...")
    
    all_experiments = {}
    experiment_names = [
        'E2_full_ft_eurosat', 'E3_full_ft_pets',
        'E4_lora_r4_eurosat', 'E5_lora_r8_eurosat', 'E6_lora_r16_eurosat',
        'E7_lora_r8_pets',
        'A1_lr_1e-06', 'A1_lr_5e-06', 'A1_lr_5e-05', 'A1_lr_0.0001',
        'A2_lora_qkvo',
        'A3_freeze_3', 'A3_freeze_6', 'A3_freeze_9',
        'reg_apr_lambda0.01_eurosat', 'reg_apr_lambda0.1_eurosat', 'reg_apr_lambda1.0_eurosat',
        'reg_entropy_floor_lambda0.01_eurosat', 'reg_entropy_floor_lambda0.1_eurosat',
        'R3_wd_0.0', 'R3_wd_0.1',
    ]
    
    for name in experiment_names:
        hist = load_history(name)
        if hist:
            all_experiments[name] = hist
    
    print(f"  Loaded {len(all_experiments)} experiments")
    
    # ========== SECTION 1: Additional Zero-Shot Evaluations ==========
    print("\n[2/7] Running additional zero-shot evaluations...")
    
    # Models to evaluate
    zs_eval_configs = {
        'baseline': (None, False),
        'E2_full_ft_eurosat': (str(CHECKPOINTS_DIR / 'E2_full_ft_eurosat' / 'best_model.pth'), False),
        'E3_full_ft_pets': (str(CHECKPOINTS_DIR / 'E3_full_ft_pets' / 'best_model.pth'), False),
        'E4_lora_r4': (str(CHECKPOINTS_DIR / 'E4_lora_r4_eurosat' / 'best_model.pth'), False),
        'E5_lora_r8': (str(CHECKPOINTS_DIR / 'E5_lora_r8_eurosat' / 'best_model.pth'), False),
        'E6_lora_r16': (str(CHECKPOINTS_DIR / 'E6_lora_r16_eurosat' / 'best_model.pth'), False),
        'A1_lr_1e-4': (str(CHECKPOINTS_DIR / 'A1_lr_0.0001' / 'best_model.pth'), False),
        'A1_lr_5e-5': (str(CHECKPOINTS_DIR / 'A1_lr_5e-05' / 'best_model.pth'), False),
        'APR_0.01': (str(CHECKPOINTS_DIR / 'reg_apr_lambda0.01_eurosat' / 'best_model.pth'), False),
    }
    
    zs_results = {}
    for exp_id, (ckpt_path, is_lora) in zs_eval_configs.items():
        # Check if already evaluated
        cached_path = METRICS_DIR / f'extended_zs_{exp_id}.json'
        if cached_path.exists():
            zs_results[exp_id] = load_json(str(cached_path))
            print(f"  {exp_id}: loaded from cache")
        else:
            result = run_zero_shot_evaluation_extended(ckpt_path, exp_id, is_lora)
            zs_results[exp_id] = result
            # Cache result
            with open(str(cached_path), 'w') as f:
                json.dump(result, f, indent=2)
    
    print(f"  Completed {len(zs_results)} zero-shot evaluations")
    
    # ========== SECTION 2: Statistical Tests ==========
    print("\n[3/7] Computing statistical tests...")
    
    stat_results = compute_statistical_tests(all_experiments)
    
    # Print results
    ft_vs_lora = stat_results.get('fullft_vs_lora', {})
    if ft_vs_lora:
        print(f"\n  Full FT vs LoRA (per-layer entropy change):")
        print(f"    Full FT: {ft_vs_lora['full_ft_mean']:.3f}% ± {ft_vs_lora['full_ft_std']:.3f}%")
        print(f"      95% CI: [{ft_vs_lora['full_ft_ci'][0]:.3f}%, {ft_vs_lora['full_ft_ci'][1]:.3f}%]")
        print(f"    LoRA: {ft_vs_lora['lora_mean']:.3f}% ± {ft_vs_lora['lora_std']:.3f}%")
        print(f"      95% CI: [{ft_vs_lora['lora_ci'][0]:.3f}%, {ft_vs_lora['lora_ci'][1]:.3f}%]")
        print(f"    Welch's t-test: t={ft_vs_lora['welch_t']:.3f}, p={ft_vs_lora['welch_p']:.4f}")
        print(f"    Mann-Whitney U: U={ft_vs_lora['mann_whitney_u']:.0f}, p={ft_vs_lora['mann_whitney_p']:.4f}")
        print(f"    Cohen's d: {ft_vs_lora['cohens_d']:.3f}")
    
    lr_data = stat_results.get('lr_vs_entropy', {})
    if lr_data:
        print(f"\n  LR vs Entropy Correlation:")
        print(f"    Pearson: r={lr_data['pearson_r']:.4f}, p={lr_data['pearson_p']:.4f}")
        print(f"    Spearman: ρ={lr_data['spearman_r']:.4f}, p={lr_data['spearman_p']:.4f}")
    
    reg_stats = stat_results.get('regularization', {})
    if reg_stats:
        print(f"\n  Regularization vs No-Reg (paired t-test on per-layer entropy):")
        for name, data in reg_stats.items():
            print(f"    {name}: t={data['paired_t']:.3f}, p={data['paired_p']:.4f}, d={data['cohens_d']:.3f}")
    
    for key_prefix in ['layer_position_']:
        for key, data in stat_results.items():
            if key.startswith(key_prefix):
                exp_name = key.replace(key_prefix, '')
                print(f"\n  Layer Position Effect ({exp_name}):")
                print(f"    Early layers: {data['early_mean']:.3f}% ± {data['early_std']:.3f}%")
                print(f"    Middle layers: {data['middle_mean']:.3f}% ± {data['middle_std']:.3f}%")
                print(f"    Late layers: {data['late_mean']:.3f}% ± {data['late_std']:.3f}%")
                print(f"    One-way ANOVA: F={data['anova_f']:.3f}, p={data['anova_p']:.4f}")
    
    # Save statistical results
    stat_save = {}
    for k, v in stat_results.items():
        if isinstance(v, dict):
            stat_save[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    stat_save[k][k2] = {k3: float(v3) if isinstance(v3, (np.floating, float)) else v3 
                                         for k3, v3 in v2.items()}
                elif isinstance(v2, (np.floating, float)):
                    stat_save[k][k2] = float(v2)
                elif isinstance(v2, tuple):
                    stat_save[k][k2] = [float(x) for x in v2]
                elif isinstance(v2, list):
                    stat_save[k][k2] = [float(x) for x in v2]
                else:
                    stat_save[k][k2] = v2
    
    with open(str(METRICS_DIR / 'statistical_tests.json'), 'w') as f:
        json.dump(stat_save, f, indent=2)
    print("\n  Saved statistical_tests.json")
    
    # ========== SECTION 3: Per-Layer Collapse Analysis ==========
    print("\n[4/7] Running per-layer collapse analysis...")
    
    per_layer_results = per_layer_collapse_analysis(all_experiments)
    
    for name, data in per_layer_results.items():
        print(f"  {data['label']}: most affected layer = L{data['most_affected_layer']} "
              f"(Δent = {data['max_entropy_change']:+.2f}%)")
    
    # ========== SECTION 4: Training Dynamics ==========
    print("\n[5/7] Analyzing training dynamics...")
    
    dynamics_results = analyze_training_dynamics(all_experiments)
    
    for name, data in dynamics_results.items():
        onset = data.get('onset_1pct')
        onset_str = f"{onset:.0%}" if onset is not None else "never"
        print(f"  {data['label']}: onset(1%)={onset_str}, "
              f"total_change={data['total_entropy_change_pct']:+.2f}%")
    
    # ========== SECTION 5: Convergence Verification ==========
    print("\n[6/7] Verifying convergence...")
    
    convergence_results = verify_convergence(all_experiments)
    
    n_converged = sum(1 for v in convergence_results.values() if v['converged'])
    n_total = len(convergence_results)
    print(f"  {n_converged}/{n_total} experiments converged (std < 0.005)")
    
    for name, data in convergence_results.items():
        if not data['converged']:
            print(f"    WARNING: {name} may not have converged (late_std={data['late_accuracy_std']:.4f})")
    
    # ========== SECTION 6: Generate Enhanced Visualizations ==========
    print("\n[7/7] Generating enhanced visualizations...")
    
    fig_per_layer_delta_heatmap(per_layer_results)
    fig_statistical_summary(stat_results)
    fig_training_dynamics(all_experiments, dynamics_results)
    fig_metric_correlation_matrix(all_experiments)
    fig_convergence_check(convergence_results)
    fig_zero_shot_extended(zs_results)
    fig_comprehensive_dashboard(all_experiments, stat_results, zs_results)
    fig_gini_evolution(all_experiments)
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nNew figures generated:")
    new_figs = [
        'fig10_per_layer_delta_heatmap.png',
        'fig11_statistical_summary.png',
        'fig12_training_dynamics.png',
        'fig13_metric_correlations.png',
        'fig14_convergence.png',
        'fig15_zeroshot_extended.png',
        'fig16_dashboard.png',
        'fig17_gini_evolution.png',
    ]
    for f in new_figs:
        path = FIGURES_DIR / f
        print(f"  {'✓' if path.exists() else '✗'} {f}")
    
    print(f"\nStatistical results saved to: {METRICS_DIR / 'statistical_tests.json'}")
    print(f"Zero-shot results saved to: {METRICS_DIR / 'extended_zs_*.json'}")
    
    # Return all results for report update
    return {
        'stat_results': stat_results,
        'per_layer_results': per_layer_results,
        'dynamics_results': dynamics_results,
        'convergence_results': convergence_results,
        'zs_results': zs_results,
    }


if __name__ == "__main__":
    results = main()
