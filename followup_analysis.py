#!/usr/bin/env python3
"""Follow-up analysis addressing reviewer-style technical concerns.

Adds:
1. Exact small-n correlation tests and bootstrap CIs.
2. Holm-Bonferroni correction for regularizer comparisons.
3. Adapter-aware zero-shot evaluation for LoRA checkpoints.
4. Attention rollout and patch-to-patch structural metrics.
5. Representation-drift analysis via layerwise linear CKA.
6. Eval-subset sensitivity analysis.
7. Multi-seed run-level comparison once follow-up runs are available.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.dataset import (
    create_fixed_eval_subset,
    get_dataloader,
    load_cifar100,
    load_eurosat,
    load_flowers102,
    load_oxford_pets,
)
from src.metrics import (
    attention_entropy,
    compute_attention_rollout,
    erf_at_threshold,
    gini_coefficient,
)
from src.model import CLIPClassifier, create_lora_model, get_pretrained_model

PROJECT_DIR = Path("/workspace/Attention_Collapse_in_CLIP_Fine-tuning_repo")
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_history(name: str) -> dict[str, Any] | None:
    path = METRICS_DIR / f"{name}_history.json"
    return load_json(path) if path.exists() else None


def bootstrap_ci(values: list[float], statistic=np.mean, n_boot: int = 10000, alpha: float = 0.05) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(statistic(sample)))
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return [float(lo), float(hi)]


def rankdata_no_ties(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def exact_permutation_correlation(x: list[float], y: list[float], method: str = "pearson") -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    assert len(x_arr) == len(y_arr)
    n = len(x_arr)

    if method == "spearman":
        x_use = rankdata_no_ties(x_arr.tolist())
        y_use = rankdata_no_ties(y_arr.tolist())
        corr_fn = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    else:
        x_use = x_arr
        y_use = y_arr
        corr_fn = lambda a, b: float(np.corrcoef(a, b)[0, 1])

    observed = corr_fn(x_use, y_use)
    permuted = []
    for perm in itertools.permutations(range(n)):
        permuted.append(corr_fn(x_use, y_use[list(perm)]))
    permuted = np.asarray(permuted, dtype=float)
    p_two = float((np.abs(permuted) >= abs(observed) - 1e-12).mean())
    return {"statistic": float(observed), "p_value": p_two}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted: dict[str, Any] = {}
    m = len(items)
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted[name] = {
            "raw_p": float(p),
            "holm_bonferroni_p": float(min(running_max, 1.0)),
            "reject_0.05": bool(min(running_max, 1.0) < alpha),
        }
    return adjusted


def mean_delta(history: dict[str, Any], key: str) -> float:
    b = history["baseline_metrics"][key]
    f = history["final_metrics"][key]
    return float((f - b) / b * 100.0)


def exact_group_mean_permutation(group_a: list[float], group_b: list[float]) -> dict[str, float]:
    values = np.asarray(group_a + group_b, dtype=float)
    n_a = len(group_a)
    observed = float(np.mean(group_a) - np.mean(group_b))
    diffs = []
    indices = range(len(values))
    for combo in itertools.combinations(indices, n_a):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(combo)] = True
        a = values[mask]
        b = values[~mask]
        diffs.append(float(np.mean(a) - np.mean(b)))
    diffs = np.asarray(diffs, dtype=float)
    p_two = float((np.abs(diffs) >= abs(observed) - 1e-12).mean())
    return {"difference_in_means": observed, "p_value": p_two}


def get_dataset_info(exp_name: str) -> tuple[str, int]:
    if "pets" in exp_name:
        return "pets", 37
    return "eurosat", 10


def get_lora_config_info(exp_name: str) -> tuple[int, list[str]]:
    if "r4" in exp_name.lower():
        rank = 4
    elif "r16" in exp_name.lower():
        rank = 16
    else:
        rank = 8

    if "qkvo" in exp_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]
    return rank, target_modules


def load_model_from_checkpoint(exp_name: str) -> torch.nn.Module:
    ckpt_path = CHECKPOINTS_DIR / exp_name / "best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    dataset_name, num_classes = get_dataset_info(exp_name)

    is_lora = ("lora" in exp_name.lower())
    if is_lora:
        rank, target_modules = get_lora_config_info(exp_name)
        model = create_lora_model(
            num_classes=num_classes,
            lora_r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
    else:
        model = CLIPClassifier(num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def load_eval_subset(dataset_name: str, subset_seed: int = 42, num_samples: int = 200):
    if dataset_name == "eurosat":
        _, test_dataset, num_classes, _ = load_eurosat(cache_dir=str(PROJECT_DIR / "data"))
    else:
        _, test_dataset, num_classes, _ = load_oxford_pets(cache_dir=str(PROJECT_DIR / "data"))
    subset, _ = create_fixed_eval_subset(test_dataset, num_samples=num_samples, num_classes=num_classes, seed=subset_seed)
    return get_dataloader(subset, batch_size=32, shuffle=False, num_workers=2)


@torch.no_grad()
def compute_text_features_for_dataset(model: torch.nn.Module, dataset_name: str) -> tuple[torch.Tensor, list[str], Any]:
    if dataset_name == "cifar100":
        dataset, _, class_names = load_cifar100(cache_dir=str(PROJECT_DIR / "data"))
        prompt_prefix = "a photo of a "
    elif dataset_name == "flowers102":
        dataset, _, class_names = load_flowers102(cache_dir=str(PROJECT_DIR / "data"))
        prompt_prefix = "a photo of a "
    else:
        raise ValueError(dataset_name)

    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = processor(text=[f"{prompt_prefix}{name}" for name in class_names], return_tensors="pt", padding=True).to(DEVICE)
    text_features = model.clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, class_names, dataset


@torch.no_grad()
def compute_image_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    vision_outputs = model.vision_model(pixel_values=images, output_attentions=False)
    pooled = vision_outputs.pooler_output
    projected = model.visual_projection(pooled)
    projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected


@torch.no_grad()
def evaluate_zero_shot_adapter_aware(exp_name: str, dataset_name: str) -> dict[str, Any]:
    model = load_model_from_checkpoint(exp_name)
    text_features, class_names, dataset = compute_text_features_for_dataset(model, dataset_name)
    loader = get_dataloader(dataset, batch_size=64, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        image_features = compute_image_features(model, images)
        logits = image_features @ text_features.T
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": float(correct / total), "num_classes": len(class_names), "n_samples": total}


def patch_to_patch_row_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Average row entropy over patch-to-patch attention.

    attn: (B, H, seq, seq)
    """
    patch_patch = attn[:, :, 1:, 1:]
    patch_patch = patch_patch / (patch_patch.sum(dim=-1, keepdim=True) + 1e-10)
    ent = -(patch_patch * torch.log2(patch_patch + 1e-10)).sum(dim=-1)
    return ent.mean(dim=(-1, -2))


@torch.no_grad()
def compute_rollout_patch_metrics(exp_name: str) -> dict[str, Any]:
    model = load_model_from_checkpoint(exp_name)
    dataset_name, _ = get_dataset_info(exp_name)
    loader = load_eval_subset(dataset_name, subset_seed=42, num_samples=200)
    rollout_entropy = []
    rollout_erf = []
    rollout_gini = []
    patch_entropy_layers = []

    for images, _ in loader:
        images = images.to(DEVICE)
        _, attentions = model(images, output_attentions=True)
        rollout = compute_attention_rollout(attentions)
        rollout_entropy.append(float(attention_entropy(rollout).mean().item() / math.log(2)))
        rollout_erf.append(float(erf_at_threshold(rollout, threshold=0.95).mean().item()))
        rollout_gini.append(float(gini_coefficient(rollout).mean().item()))
        per_layer_patch_entropy = []
        for layer_attn in attentions:
            per_layer_patch_entropy.append(float(patch_to_patch_row_entropy(layer_attn).mean().item()))
        patch_entropy_layers.append(per_layer_patch_entropy)

    patch_entropy_layers = np.asarray(patch_entropy_layers, dtype=float)
    return {
        "rollout_entropy_bits_mean": float(np.mean(rollout_entropy)),
        "rollout_entropy_bits_std": float(np.std(rollout_entropy)),
        "rollout_erf95_mean": float(np.mean(rollout_erf)),
        "rollout_gini_mean": float(np.mean(rollout_gini)),
        "patch_to_patch_entropy_bits_per_layer": patch_entropy_layers.mean(axis=0).tolist(),
        "patch_to_patch_entropy_bits_mean": float(patch_entropy_layers.mean()),
    }


@torch.no_grad()
def extract_hidden_states(model: torch.nn.Module, loader) -> list[np.ndarray]:
    layer_buffers: list[list[np.ndarray]] = []
    for images, _ in loader:
        images = images.to(DEVICE)
        outputs = model.vision_model(pixel_values=images, output_hidden_states=True, output_attentions=False)
        hidden_states = outputs.hidden_states
        if not layer_buffers:
            layer_buffers = [[] for _ in range(len(hidden_states))]
        for i, hs in enumerate(hidden_states):
            cls = hs[:, 0, :].detach().cpu().numpy()
            layer_buffers[i].append(cls)
    return [np.concatenate(buf, axis=0) for buf in layer_buffers]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(hsic / (xx * yy + 1e-12))


@torch.no_grad()
def compute_layerwise_cka(exp_name: str) -> dict[str, Any]:
    dataset_name, _ = get_dataset_info(exp_name)
    loader = load_eval_subset(dataset_name, subset_seed=42, num_samples=200)
    pretrained = get_pretrained_model().to(DEVICE)
    pretrained.eval()
    target = load_model_from_checkpoint(exp_name)
    x_layers = extract_hidden_states(pretrained, loader)
    y_layers = extract_hidden_states(target, loader)
    cka_vals = [linear_cka(x, y) for x, y in zip(x_layers, y_layers)]
    return {
        "layerwise_cka": cka_vals,
        "mean_cka": float(np.mean(cka_vals)),
        "min_cka": float(np.min(cka_vals)),
    }


@torch.no_grad()
def subset_sensitivity(exp_name: str, subset_seeds: list[int]) -> dict[str, Any]:
    model = load_model_from_checkpoint(exp_name)
    dataset_name, _ = get_dataset_info(exp_name)
    values = defaultdict(list)
    for seed in subset_seeds:
        loader = load_eval_subset(dataset_name, subset_seed=seed, num_samples=200)
        entropies = []
        rollout_entropies = []
        for images, _ in loader:
            images = images.to(DEVICE)
            _, attentions = model(images, output_attentions=True)
            per_layer = []
            for attn in attentions:
                cls_attn = attn[:, :, 0, 1:]
                cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
                per_layer.append(float((attention_entropy(cls_attn).mean().item()) / math.log(2)))
            entropies.append(float(np.mean(per_layer)))
            rollout = compute_attention_rollout(attentions)
            rollout_entropies.append(float(attention_entropy(rollout).mean().item() / math.log(2)))
        values["entropy_bits_mean"].append(float(np.mean(entropies)))
        values["rollout_entropy_bits_mean"].append(float(np.mean(rollout_entropies)))
    return {
        k: {
            "values": v,
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "cv": float(np.std(v) / (np.mean(v) + 1e-12)),
        }
        for k, v in values.items()
    }


def analyze_lr_correlation_corrected() -> dict[str, Any]:
    names = ["A1_lr_1e-06", "A1_lr_5e-06", "E2_full_ft_eurosat", "A1_lr_5e-05", "A1_lr_0.0001"]
    lrs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    x = [math.log10(v) for v in lrs]
    y = [mean_delta(load_history(name), "entropy_mean") for name in names]
    pearson = exact_permutation_correlation(x, y, method="pearson")
    spearman = exact_permutation_correlation(x, y, method="spearman")
    return {
        "x_log10_lr": x,
        "delta_entropy": y,
        "pearson_exact": pearson,
        "spearman_exact": spearman,
        "pearson_bootstrap_ci": bootstrap_ci(y, statistic=np.mean),
    }


def analyze_regularizers_corrected() -> dict[str, Any]:
    base = load_history("E2_full_ft_eurosat")
    reg_names = {
        "APR_0.01": "reg_apr_lambda0.01_eurosat",
        "APR_0.1": "reg_apr_lambda0.1_eurosat",
        "APR_1.0": "reg_apr_lambda1.0_eurosat",
        "EntFloor_0.01": "reg_entropy_floor_lambda0.01_eurosat",
        "EntFloor_0.1": "reg_entropy_floor_lambda0.1_eurosat",
    }
    raw_p = {}
    details = {}
    base_layers = np.asarray(base["final_metrics"]["entropy_per_layer"], dtype=float)
    for label, name in reg_names.items():
        hist = load_history(name)
        layers = np.asarray(hist["final_metrics"]["entropy_per_layer"], dtype=float)
        t_stat, p_val = stats.ttest_rel(layers, base_layers)
        raw_p[label] = float(p_val)
        details[label] = {
            "paired_t": float(t_stat),
            "raw_p": float(p_val),
            "mean_diff": float(np.mean(layers - base_layers)),
        }
    corrected = holm_bonferroni(raw_p)
    for label in details:
        details[label].update(corrected[label])
    return details


def analyze_multiseed_followup() -> dict[str, Any]:
    fullft_names = ["E2_full_ft_eurosat"] + sorted(
        p.name.replace("_history.json", "")
        for p in METRICS_DIR.glob("F1_fullft_eurosat_seed*_e*_history.json")
    )
    lora_names = ["E5_lora_r8_eurosat"] + sorted(
        p.name.replace("_history.json", "")
        for p in METRICS_DIR.glob("F1_lora_r8_eurosat_seed*_e*_history.json")
    )
    fullft_hist = [load_history(name) for name in fullft_names if load_history(name) is not None]
    lora_hist = [load_history(name) for name in lora_names if load_history(name) is not None]
    if len(fullft_hist) < 2 or len(lora_hist) < 2:
        return {
            "status": "pending",
            "available_fullft": len(fullft_hist),
            "available_lora": len(lora_hist),
            "fullft_runs": fullft_names,
            "lora_runs": lora_names,
        }

    metric_keys = ["entropy_mean", "erf95_mean", "gini_mean", "head_diversity_mean", "best_val_acc"]
    results = {
        "status": "complete",
        "n_fullft": len(fullft_hist),
        "n_lora": len(lora_hist),
        "fullft_runs": fullft_names,
        "lora_runs": lora_names,
        "metrics": {},
    }
    for key in metric_keys:
        if key == "best_val_acc":
            fullft_vals = [float(h["best_val_acc"]) for h in fullft_hist]
            lora_vals = [float(h["best_val_acc"]) for h in lora_hist]
        else:
            fullft_vals = [mean_delta(h, key) for h in fullft_hist]
            lora_vals = [mean_delta(h, key) for h in lora_hist]
        exact = exact_group_mean_permutation(fullft_vals, lora_vals)
        results["metrics"][key] = {
            "fullft_values": fullft_vals,
            "lora_values": lora_vals,
            "fullft_mean": float(np.mean(fullft_vals)),
            "lora_mean": float(np.mean(lora_vals)),
            "fullft_ci": bootstrap_ci(fullft_vals),
            "lora_ci": bootstrap_ci(lora_vals),
            "exact_permutation": exact,
        }
    return results


def main() -> None:
    followup = {
        "corrected_lr_correlation": analyze_lr_correlation_corrected(),
        "corrected_regularization": analyze_regularizers_corrected(),
        "adapter_aware_zero_shot": {},
        "rollout_patch_metrics": {},
        "layerwise_cka": {},
        "subset_sensitivity": {},
        "multiseed_followup": analyze_multiseed_followup(),
    }

    for exp_name in ["E2_full_ft_eurosat", "E5_lora_r8_eurosat"]:
        followup["rollout_patch_metrics"][exp_name] = compute_rollout_patch_metrics(exp_name)
        followup["layerwise_cka"][exp_name] = compute_layerwise_cka(exp_name)
        followup["subset_sensitivity"][exp_name] = subset_sensitivity(exp_name, [7, 21, 42, 123, 777])

    for exp_name in ["E2_full_ft_eurosat", "E5_lora_r8_eurosat", "E4_lora_r4_eurosat", "E6_lora_r16_eurosat"]:
        followup["adapter_aware_zero_shot"][exp_name] = {
            "cifar100": evaluate_zero_shot_adapter_aware(exp_name, "cifar100"),
            "flowers102": evaluate_zero_shot_adapter_aware(exp_name, "flowers102"),
        }

    save_json(METRICS_DIR / "followup_analysis.json", followup)
    print(json.dumps(followup, indent=2)[:4000])
    print(f"Saved follow-up analysis to {METRICS_DIR / 'followup_analysis.json'}")


if __name__ == "__main__":
    main()
