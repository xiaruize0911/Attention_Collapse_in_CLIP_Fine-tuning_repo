"""Label-free structural probes plus predictive evaluations.

Every probe in this file is computed from the same fixed set of unlabeled
target-domain images, so all of them are available to a practitioner during
fine-tuning without access to a transfer benchmark. That is the property the
paper tests: can a cheap internal measurement stand in for an expensive
held-out evaluation?
"""

from __future__ import annotations

import numpy as np
import torch

from src.metrics import attention_entropy, erf_at_threshold, gini_coefficient


# --------------------------------------------------------------------------
# attention structure
# --------------------------------------------------------------------------
@torch.no_grad()
def attention_summary(model, loader, device) -> dict:
    """CLS-to-patch entropy / ERF@0.95 / Gini, per layer, image-weighted."""
    model.eval()
    totals: dict[str, np.ndarray] = {}
    count = 0
    for images, _ in loader:
        images = images.to(device)
        _, attentions = model(images, output_attentions=True)
        batch = images.shape[0]
        ent, erf, gini = [], [], []
        for attn in attentions:
            cls_attn = attn[:, :, 0, 1:]
            cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-10)
            ent.append(attention_entropy(cls_attn).mean(dim=1))
            erf.append(erf_at_threshold(cls_attn, 0.95).mean(dim=1))
            b, h, n = cls_attn.shape
            gini.append(gini_coefficient(cls_attn.reshape(b * h, n)).reshape(b, h).mean(dim=1))
        stacked = {
            "entropy_per_layer": torch.stack(ent, dim=1),
            "erf95_per_layer": torch.stack(erf, dim=1),
            "gini_per_layer": torch.stack(gini, dim=1),
        }
        for key, value in stacked.items():
            summed = value.sum(dim=0).float().cpu().numpy()
            totals[key] = summed if key not in totals else totals[key] + summed
        count += batch

    out = {key: (value / count).tolist() for key, value in totals.items()}
    for key in list(out):
        out[key.replace("_per_layer", "_mean")] = float(np.mean(out[key]))
    return out


# --------------------------------------------------------------------------
# representation drift
# --------------------------------------------------------------------------
@torch.no_grad()
def representation_probe(model, loader, device) -> dict:
    """Per-layer CLS hidden states and final image embeddings on the probe set."""
    model.eval()
    hidden, embeddings = [], []
    for images, _ in loader:
        images = images.to(device)
        outputs = model.vision_model(pixel_values=images, output_hidden_states=True)
        states = torch.stack([h[:, 0, :] for h in outputs.hidden_states], dim=0)
        hidden.append(states.float().cpu())
        embeddings.append(model.visual_projection(outputs.pooler_output).float().cpu())
    return {
        "hidden": torch.cat(hidden, dim=1),      # (layers, images, width)
        "embeddings": torch.cat(embeddings, dim=0),  # (images, 512)
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two centered feature matrices (images x features)."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = (x.T @ y).norm(p="fro") ** 2
    denom = (x.T @ x).norm(p="fro") * (y.T @ y).norm(p="fro")
    if denom <= 0:
        return float("nan")
    return float(cross / denom)


def representation_drift(current: dict, reference: dict) -> dict:
    # Index 0 is the encoder input, whose CLS row is a constant parameter shared
    # by every image, so its CKA is undefined. Blocks 1..L are the informative ones.
    ckas = [linear_cka(reference["hidden"][i], current["hidden"][i])
            for i in range(1, reference["hidden"].shape[0])]
    ref_emb = torch.nn.functional.normalize(reference["embeddings"], dim=-1)
    cur_emb = torch.nn.functional.normalize(current["embeddings"], dim=-1)
    cosine = (ref_emb * cur_emb).sum(dim=-1)
    return {
        "cka_per_layer": ckas,
        "cka_mean": float(np.mean(ckas)),
        "cka_last": float(ckas[-1]),
        "embedding_cosine_mean": float(cosine.mean()),
        "embedding_drift": float(1.0 - cosine.mean()),
    }


# --------------------------------------------------------------------------
# weight drift (comparable across Full FT and LoRA)
# --------------------------------------------------------------------------
def _canonical(name: str) -> str:
    return name.replace("base_model.model.", "").replace("base_layer.", "")


def encoder_weight_snapshot(vision_module) -> dict[str, torch.Tensor]:
    """Base (non-adapter) encoder weights keyed by canonical name."""
    snapshot = {}
    for name, param in vision_module.named_parameters():
        if "lora_" in name:
            continue
        snapshot[_canonical(name)] = param.detach().float().cpu().clone()
    return snapshot


def _lora_deltas(vision_module) -> dict[str, torch.Tensor]:
    deltas = {}
    for name, module in vision_module.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or len(lora_a) == 0:
            continue
        adapter = next(iter(lora_a.keys()))
        a = module.lora_A[adapter].weight.detach().float()
        b = module.lora_B[adapter].weight.detach().float()
        scale = float(module.scaling[adapter])
        key = _canonical(name) + ".weight"
        deltas[key] = ((b @ a) * scale).cpu()
    return deltas


def weight_drift(vision_module, snapshot: dict[str, torch.Tensor]) -> dict:
    """Relative L2 norm of the effective change to the visual encoder.

    For LoRA the base weights are frozen, so the effective update is the
    scaled product of the adapter factors. Expressing both methods as a
    fraction of pretrained weight norm keeps the numbers comparable.
    """
    current = {_canonical(n): p.detach().float().cpu()
               for n, p in vision_module.named_parameters() if "lora_" not in n}
    deltas = _lora_deltas(vision_module)

    sq_delta = 0.0
    sq_base = 0.0
    for key, base in snapshot.items():
        cur = current.get(key)
        diff = torch.zeros_like(base) if cur is None else (cur - base)
        if key in deltas:
            diff = diff + deltas[key]
        sq_delta += float((diff ** 2).sum())
        sq_base += float((base ** 2).sum())
    return {
        "weight_drift_rel": float(np.sqrt(sq_delta) / np.sqrt(sq_base)),
        "weight_delta_l2": float(np.sqrt(sq_delta)),
    }


# --------------------------------------------------------------------------
# predictive evaluations
# --------------------------------------------------------------------------
@torch.no_grad()
def classifier_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images, output_attentions=False)
        correct += int((logits.argmax(dim=-1) == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


@torch.no_grad()
def text_features(model, class_names: list[str], processor, device,
                  template: str = "a photo of a {}.") -> torch.Tensor:
    inputs = processor(text=[template.format(name.replace("_", " ")) for name in class_names],
                       return_tensors="pt", padding=True).to(device)
    feats = model.clip_model.get_text_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def zero_shot_accuracy(model, loader, text_feats, device) -> float:
    """Adapter-aware zero-shot accuracy through the live image encoder."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        vision = model.vision_model(pixel_values=images)
        embeds = model.visual_projection(vision.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        correct += int(((embeds @ text_feats.T).argmax(dim=-1) == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)
