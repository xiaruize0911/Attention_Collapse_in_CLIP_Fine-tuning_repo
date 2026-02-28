"""
metrics.py - Attention structural collapse metrics
Implements: Entropy, ERF@0.95, Gini coefficient, Head Diversity
"""
import torch
import numpy as np
from itertools import combinations


def attention_entropy(attn_vector, eps=1e-10):
    """
    Compute entropy of attention distribution.
    attn_vector: (..., N) attention weights (already softmaxed)
    Returns: (...,) entropy values
    """
    log_attn = torch.log(attn_vector + eps)
    entropy = -(attn_vector * log_attn).sum(dim=-1)
    return entropy


def erf_at_threshold(attn_vector, threshold=0.95):
    """
    Effective Receptive Field: fraction of patches needed to cover `threshold` mass.
    attn_vector: (..., N) 
    Returns: (...,) ERF values in [1/N, 1]
    """
    N = attn_vector.shape[-1]
    sorted_attn, _ = torch.sort(attn_vector, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_attn, dim=-1)
    # K* = first index where cumsum >= threshold
    mask = cumsum >= threshold
    # argmax on bool tensor returns first True
    k_star = mask.float().argmax(dim=-1) + 1  # 1-indexed
    erf = k_star.float() / N
    return erf


def gini_coefficient(attn_vector):
    """
    Gini coefficient of attention distribution.
    attn_vector: (B, N) or (N,)
    Returns: scalar or (B,) Gini values
    """
    if attn_vector.dim() == 1:
        attn_vector = attn_vector.unsqueeze(0)
    
    B, N = attn_vector.shape
    sorted_attn, _ = torch.sort(attn_vector, dim=-1)
    index = torch.arange(1, N + 1, dtype=torch.float32, device=attn_vector.device)
    # Gini = (2 * sum(i * x_i) / (N * sum(x_i))) - (N+1)/N
    numerator = (2 * index.unsqueeze(0) * sorted_attn).sum(dim=-1)
    denominator = N * sorted_attn.sum(dim=-1)
    gini = numerator / (denominator + 1e-10) - (N + 1) / N
    return gini.squeeze()


def head_diversity_cosine(cls_attn_all_heads):
    """
    Head diversity using cosine similarity.
    cls_attn_all_heads: (B, H, N) - CLS->patch attention for all heads
    Returns: scalar diversity value (1 - mean cosine similarity between head pairs)
    """
    B, H, N = cls_attn_all_heads.shape
    # Average over batch
    mean_attn = cls_attn_all_heads.mean(dim=0)  # (H, N)
    
    # Compute pairwise cosine similarity
    similarities = []
    for i, j in combinations(range(H), 2):
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_attn[i].unsqueeze(0), mean_attn[j].unsqueeze(0)
        )
        similarities.append(cos_sim.item())
    
    mean_sim = np.mean(similarities) if similarities else 0.0
    diversity = 1.0 - mean_sim
    return diversity


def compute_all_metrics(attentions, num_layers=12, num_heads=12, num_patches=49):
    """
    Compute all attention metrics from model output attentions.
    
    attentions: tuple of (B, heads, seq, seq) tensors, one per layer
    Returns: dict with all metrics
    """
    entropy_per_layer = []
    erf95_per_layer = []
    gini_per_layer = []
    head_div_per_layer = []
    
    for layer_idx in range(min(num_layers, len(attentions))):
        attn = attentions[layer_idx]  # (B, H, seq, seq)
        
        # Extract CLS -> patch attention
        cls_attn = attn[:, :, 0, 1:]  # (B, H, N_patches)
        # Renormalize
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Entropy: average over heads and batch
        ent = attention_entropy(cls_attn)  # (B, H)
        entropy_per_layer.append(ent.mean().item())
        
        # ERF@0.95: average over heads and batch
        erf = erf_at_threshold(cls_attn, threshold=0.95)  # (B, H)
        erf95_per_layer.append(erf.mean().item())
        
        # Gini: flatten heads, average
        B, H, N = cls_attn.shape
        gini_vals = gini_coefficient(cls_attn.reshape(B * H, N))  # (B*H,)
        gini_per_layer.append(gini_vals.mean().item())
        
        # Head diversity
        div = head_diversity_cosine(cls_attn)
        head_div_per_layer.append(div)
    
    metrics = {
        'entropy_per_layer': entropy_per_layer,
        'erf95_per_layer': erf95_per_layer,
        'gini_per_layer': gini_per_layer,
        'head_diversity_per_layer': head_div_per_layer,
        'entropy_mean': np.mean(entropy_per_layer),
        'erf95_mean': np.mean(erf95_per_layer),
        'gini_mean': np.mean(gini_per_layer),
        'head_diversity_mean': np.mean(head_div_per_layer),
    }
    return metrics


def compute_attention_rollout(attentions):
    """
    Attention Rollout (Abnar & Zuidema, 2020).
    attentions: list of (B, heads, seq, seq) tensors
    Returns: (B, N_patches) rollout attention from CLS to patches
    """
    device = attentions[0].device
    B = attentions[0].shape[0]
    seq_len = attentions[0].shape[-1]
    
    result = torch.eye(seq_len, device=device).unsqueeze(0).expand(B, -1, -1)
    
    for attn in attentions:
        attn_avg = attn.mean(dim=1)  # (B, seq, seq)
        attn_res = 0.5 * attn_avg + 0.5 * torch.eye(seq_len, device=device).unsqueeze(0)
        result = torch.bmm(attn_res, result)
    
    cls_rollout = result[:, 0, 1:]  # (B, N_patches)
    cls_rollout = cls_rollout / (cls_rollout.sum(dim=-1, keepdim=True) + 1e-10)
    return cls_rollout
