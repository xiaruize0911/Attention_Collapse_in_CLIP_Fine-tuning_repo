from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F


def attention_entropy(attn_vector: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    attn_vector = attn_vector.clamp_min(eps)
    return -(attn_vector * attn_vector.log()).sum(dim=-1)


def erf_at_threshold(attn_vector: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    num_patches = attn_vector.shape[-1]
    sorted_attn, _ = torch.sort(attn_vector, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_attn, dim=-1)
    mask = cumsum >= threshold
    k_star = mask.float().argmax(dim=-1) + 1
    return k_star.float() / num_patches


def gini_coefficient(attn_vector: torch.Tensor) -> torch.Tensor:
    if attn_vector.dim() == 1:
        attn_vector = attn_vector.unsqueeze(0)
    batch, num_items = attn_vector.shape
    sorted_attn, _ = torch.sort(attn_vector, dim=-1)
    index = torch.arange(1, num_items + 1, dtype=sorted_attn.dtype, device=sorted_attn.device)
    numerator = (2 * index.unsqueeze(0) * sorted_attn).sum(dim=-1)
    denominator = num_items * sorted_attn.sum(dim=-1).clamp_min(1e-10)
    gini = numerator / denominator - (num_items + 1) / num_items
    return gini.reshape(batch)


def head_diversity(attn_all_heads: torch.Tensor) -> float:
    _, num_heads, _ = attn_all_heads.shape
    mean_attn = attn_all_heads.mean(dim=0)
    similarities = []
    for i, j in combinations(range(num_heads), 2):
        sim = F.cosine_similarity(mean_attn[i].unsqueeze(0), mean_attn[j].unsqueeze(0)).item()
        similarities.append(sim)
    return 1.0 - float(np.mean(similarities)) if similarities else 0.0


def compute_attention_rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    device = attentions[0].device
    batch = attentions[0].shape[0]
    seq_len = attentions[0].shape[-1]
    result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch, -1, -1)
    identity = torch.eye(seq_len, device=device).unsqueeze(0)
    for attn in attentions:
        averaged = attn.mean(dim=1)
        with_residual = 0.5 * averaged + 0.5 * identity
        result = torch.bmm(with_residual, result)
    cls_rollout = result[:, 0, 1:]
    return cls_rollout / cls_rollout.sum(dim=-1, keepdim=True).clamp_min(1e-10)


def compute_patch_to_patch_entropy(attentions: tuple[torch.Tensor, ...]) -> float:
    entropies: list[float] = []
    for attn in attentions:
        patch_patch = attn[:, :, 1:, 1:]
        patch_patch = patch_patch / patch_patch.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        entropy = attention_entropy(patch_patch).mean().item()
        entropies.append(entropy)
    return float(np.mean(entropies)) if entropies else float("nan")


def compute_attention_summary(attentions: tuple[torch.Tensor, ...]) -> dict[str, float | list[float]]:
    entropy_per_layer = []
    erf_per_layer = []
    gini_per_layer = []
    head_div_per_layer = []

    for layer_attn in attentions:
        cls_attn = layer_attn[:, :, 0, 1:]
        cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        entropy_per_layer.append(float(attention_entropy(cls_attn).mean().item()))
        erf_per_layer.append(float(erf_at_threshold(cls_attn).mean().item()))
        batch, heads, patches = cls_attn.shape
        gini_per_layer.append(float(gini_coefficient(cls_attn.reshape(batch * heads, patches)).mean().item()))
        head_div_per_layer.append(float(head_diversity(cls_attn)))

    rollout = compute_attention_rollout(attentions)
    rollout_entropy = float(attention_entropy(rollout).mean().item())
    rollout_erf = float(erf_at_threshold(rollout).mean().item())
    rollout_gini = float(gini_coefficient(rollout).mean().item())
    patch_to_patch = compute_patch_to_patch_entropy(attentions)

    return {
        "entropy_per_layer": entropy_per_layer,
        "erf95_per_layer": erf_per_layer,
        "gini_per_layer": gini_per_layer,
        "head_diversity_per_layer": head_div_per_layer,
        "entropy_mean": float(np.mean(entropy_per_layer)),
        "erf95_mean": float(np.mean(erf_per_layer)),
        "gini_mean": float(np.mean(gini_per_layer)),
        "head_diversity_mean": float(np.mean(head_div_per_layer)),
        "rollout_entropy_mean": rollout_entropy,
        "rollout_erf95_mean": rollout_erf,
        "rollout_gini_mean": rollout_gini,
        "patch_to_patch_entropy_mean": patch_to_patch,
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = torch.linalg.norm(x.T @ y, ord="fro") ** 2
    x_norm = torch.linalg.norm(x.T @ x, ord="fro")
    y_norm = torch.linalg.norm(y.T @ y, ord="fro")
    denom = (x_norm * y_norm).clamp_min(1e-10)
    return float((hsic / denom).item())


def compute_layerwise_cka(
    model_hidden_states: tuple[torch.Tensor, ...],
    baseline_hidden_states: tuple[torch.Tensor, ...],
) -> dict[str, float | list[float]]:
    layer_scores = []
    for model_h, base_h in zip(model_hidden_states, baseline_hidden_states):
        model_flat = model_h.reshape(-1, model_h.shape[-1])
        base_flat = base_h.reshape(-1, base_h.shape[-1])
        layer_scores.append(linear_cka(model_flat, base_flat))
    return {
        "layerwise_cka": layer_scores,
        "mean_layerwise_cka": float(np.mean(layer_scores)),
    }


def relative_shift(current: float, baseline: float) -> float:
    baseline = float(baseline)
    if math.isclose(baseline, 0.0, abs_tol=1e-12):
        return float("nan")
    return float((float(current) - baseline) / baseline)
