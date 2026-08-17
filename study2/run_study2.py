#!/usr/bin/env python3
"""Protocol-hardened replication of the matched-learning-rate attention study.

Differences from the original 80-run matrix, all of them deliberate:

* validation images are carved out of the training split, so the provided test
  split is read exactly once per run and never drives model selection;
* the attention probe set is disjoint from both training and validation images;
* every epoch records structural drift, weight drift, representation drift and
  a small transfer probe, so the temporal ordering of forgetting is testable;
* transfer is always measured through the live (adapter-active) image encoder;
* per-run records are written to disk so the aggregate statistics in the paper
  can be recomputed from raw output.

The whole matrix is sized to run on a single Apple-silicon laptop GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")           # transformers must not import TensorFlow
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.model import CLIPClassifier, create_lora_model  # noqa: E402
from study2 import data_splits, probes  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "study2" / "results"
CACHE_DIR = PROJECT_DIR / "study2" / "cache"
MODEL_NAME = "openai/clip-vit-base-patch32"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

DEFAULTS = {
    "eurosat": {"train_per_class": 160, "val_per_class": 30, "probe_per_class": 20},
    "pets": {"train_per_class": 40, "val_per_class": 10, "probe_per_class": 6},
}
METHODS = ("full_ft", "lora_r8", "lora_r8_frozen_proj", "linear_probe", "last_block")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def model_slug(model_name: str) -> str:
    """Empty for the default backbone, so existing run ids stay valid."""
    if model_name == MODEL_NAME:
        return ""
    return "_" + model_name.split("/")[-1].replace("clip-vit-", "").replace("-patch", "p")


def run_id(dataset: str, method: str, lr: float, seed: int, tag: str | None,
           model_name: str = MODEL_NAME) -> str:
    base = f"S2_{dataset}_{method}{model_slug(model_name)}_lr{format_lr(lr)}_seed{seed}"
    return f"{base}_{tag}" if tag else base


def build_model(method: str, num_classes: int, model_name: str = MODEL_NAME) -> nn.Module:
    if method == "full_ft":
        model = CLIPClassifier(model_name, num_classes=num_classes)
    elif method in ("lora_r8", "lora_r8_frozen_proj"):
        model = create_lora_model(model_name, num_classes=num_classes, lora_r=8,
                                  lora_alpha=16, lora_dropout=0.05,
                                  target_modules=["q_proj", "v_proj"])
        if method == "lora_r8_frozen_proj":
            for p in model.visual_projection.parameters():
                p.requires_grad = False
    elif method == "linear_probe":
        model = CLIPClassifier(model_name, num_classes=num_classes)
        for p in model.vision_model.parameters():
            p.requires_grad = False
        for p in model.visual_projection.parameters():
            p.requires_grad = False
    elif method == "last_block":
        model = CLIPClassifier(model_name, num_classes=num_classes)
        for p in model.vision_model.parameters():
            p.requires_grad = False
        for p in model.vision_model.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in model.vision_model.post_layernorm.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown method: {method}")
    return model


def parameter_counts(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def pretrained_transfer_cache(processor, transfer_sets: dict, batch_size: int,
                               model_name: str = MODEL_NAME) -> dict:
    """Zero-shot accuracy of unmodified CLIP on each transfer benchmark."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"pretrained_transfer{model_slug(model_name) or '_b32p32'}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    needed = [name for name, spec in transfer_sets.items()
              if f"{name}@{spec['size']}" not in cache]
    if needed:
        model = CLIPClassifier(model_name, num_classes=2).to(DEVICE).eval()
        for name in needed:
            spec = transfer_sets[name]
            feats = probes.text_features(model, spec["class_names"], processor, DEVICE)
            loader = data_splits.loader(spec["dataset"], batch_size, shuffle=False)
            cache[f"{name}@{spec['size']}"] = probes.zero_shot_accuracy(model, loader, feats, DEVICE)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        del model
        empty_cache()
    return cache


def relative_change(current: float, baseline: float) -> float:
    return 100.0 * (current - baseline) / baseline if baseline else float("nan")


def train_run(config: dict) -> dict:
    from transformers import CLIPProcessor

    model_name = config.get("model_name", MODEL_NAME)
    identifier = run_id(config["dataset"], config["method"], config["lr"],
                        config["seed"], config.get("tag"), model_name)
    out_path = OUTPUT_DIR / f"{identifier}.json"
    if out_path.exists() and not config.get("overwrite"):
        print(f"skip (done): {identifier}")
        return json.loads(out_path.read_text())

    started = time.time()
    set_seed(config["seed"])

    spec = DEFAULTS[config["dataset"]]
    splits = data_splits.build_target_splits(
        config["dataset"], spec["train_per_class"], spec["val_per_class"],
        spec["probe_per_class"])
    num_classes = splits["num_classes"]
    batch_size = config["batch_size"]

    train_loader = data_splits.loader(splits["train"], batch_size, shuffle=True,
                                       seed=config["seed"])
    val_loader = data_splits.loader(splits["val"], batch_size, shuffle=False)
    probe_loader = data_splits.loader(splits["probe"], batch_size, shuffle=False)
    test_loader = data_splits.loader(splits["test"], batch_size, shuffle=False)

    processor = CLIPProcessor.from_pretrained(model_name)
    transfer_sets = {
        "cifar100": data_splits.build_transfer_set("cifar100", None),
        "cifar10": data_splits.build_transfer_set("cifar10", 2000),
    }
    track_set = data_splits.build_transfer_set("cifar100", 1000)
    corruption_sets = data_splits.build_corruption_splits(config["dataset"], 1000)
    baseline_transfer = pretrained_transfer_cache(processor, transfer_sets, batch_size,
                                                  model_name)

    model = build_model(config["method"], num_classes, model_name).to(DEVICE)
    counts = parameter_counts(model)

    encoder_snapshot = probes.encoder_weight_snapshot(model.vision_model)
    baseline_attention = probes.attention_summary(model, probe_loader, DEVICE)
    baseline_representation = probes.representation_probe(model, probe_loader, DEVICE)
    track_feats = probes.text_features(model, track_set["class_names"], processor, DEVICE)
    track_loader = data_splits.loader(track_set["dataset"], batch_size, shuffle=False)
    baseline_track = probes.zero_shot_accuracy(model, track_loader, track_feats, DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps = max(1, len(train_loader) * config["epochs"])
    warmup = max(1, int(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.CrossEntropyLoss()

    history = []
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    diverged = False

    for epoch in range(config["epochs"]):
        model.train()
        losses, correct, seen = [], 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(images, output_attentions=False)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(dim=-1) == labels).sum())
            seen += int(labels.numel())
        train_loss = float(np.mean(losses))
        if not np.isfinite(train_loss):
            diverged = True

        val_acc = probes.classifier_accuracy(model, val_loader, DEVICE)
        attention = probes.attention_summary(model, probe_loader, DEVICE)
        representation = probes.representation_probe(model, probe_loader, DEVICE)
        drift = probes.representation_drift(representation, baseline_representation)
        weights = probes.weight_drift(model.vision_model, encoder_snapshot)
        track_acc = probes.zero_shot_accuracy(model, track_loader, track_feats, DEVICE)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": correct / max(seen, 1),
            "val_acc": val_acc,
            "entropy_mean": attention["entropy_mean"],
            "entropy_per_layer": attention["entropy_per_layer"],
            "erf95_mean": attention["erf95_mean"],
            "gini_mean": attention["gini_mean"],
            "delta_entropy_pct": relative_change(attention["entropy_mean"],
                                                 baseline_attention["entropy_mean"]),
            "delta_erf95_pct": relative_change(attention["erf95_mean"],
                                               baseline_attention["erf95_mean"]),
            "delta_gini_pct": relative_change(attention["gini_mean"],
                                              baseline_attention["gini_mean"]),
            "cka_mean": drift["cka_mean"],
            "embedding_drift": drift["embedding_drift"],
            "weight_drift_rel": weights["weight_drift_rel"],
            "transfer_track_acc": track_acc,
            "transfer_track_retention": track_acc / baseline_track if baseline_track else None,
        })
        print(f"  [{identifier}] epoch {epoch+1}/{config['epochs']} "
              f"loss={train_loss:.4f} val={val_acc:.4f} "
              f"dH={history[-1]['delta_entropy_pct']:+.2f}% "
              f"track={track_acc:.4f}", flush=True)

        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()},
            }

    # ---- single pass over held-out data, using the selected checkpoint only ----
    if best["state"] is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best["state"].items()})
    model.eval()

    final_attention = probes.attention_summary(model, probe_loader, DEVICE)
    final_representation = probes.representation_probe(model, probe_loader, DEVICE)
    final_drift = probes.representation_drift(final_representation, baseline_representation)
    final_weights = probes.weight_drift(model.vision_model, encoder_snapshot)

    test_acc = probes.classifier_accuracy(model, test_loader, DEVICE)
    transfer = {}
    for name, tspec in transfer_sets.items():
        feats = probes.text_features(model, tspec["class_names"], processor, DEVICE)
        loader = data_splits.loader(tspec["dataset"], batch_size, shuffle=False)
        accuracy = probes.zero_shot_accuracy(model, loader, feats, DEVICE)
        reference = baseline_transfer[f"{name}@{tspec['size']}"]
        transfer[name] = {
            "accuracy": accuracy,
            "pretrained": reference,
            "retention": accuracy / reference if reference else None,
            "n_images": tspec["size"],
        }

    corruption = {}
    for name, dataset in corruption_sets.items():
        loader = data_splits.loader(dataset, batch_size, shuffle=False)
        corruption[name] = probes.classifier_accuracy(model, loader, DEVICE)

    record = {
        "run_id": identifier,
        "config": {**config, "device": str(DEVICE), "num_classes": num_classes,
                    "model_name": model_name, "splits": splits["sizes"],
                    "steps_per_epoch": len(train_loader), "total_steps": total_steps},
        "parameters": counts,
        "diverged": diverged,
        "baseline": {
            "attention": baseline_attention,
            "transfer_track_acc": baseline_track,
            "transfer": baseline_transfer,
        },
        "history": history,
        "selected_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "test_acc": test_acc,
        "final": {
            "attention": final_attention,
            "delta_entropy_pct": relative_change(final_attention["entropy_mean"],
                                                 baseline_attention["entropy_mean"]),
            "delta_erf95_pct": relative_change(final_attention["erf95_mean"],
                                               baseline_attention["erf95_mean"]),
            "delta_gini_pct": relative_change(final_attention["gini_mean"],
                                              baseline_attention["gini_mean"]),
            "delta_entropy_per_layer_pct": [
                relative_change(c, b) for c, b in
                zip(final_attention["entropy_per_layer"],
                    baseline_attention["entropy_per_layer"])],
            "cka_per_layer": final_drift["cka_per_layer"],
            "cka_mean": final_drift["cka_mean"],
            "embedding_drift": final_drift["embedding_drift"],
            "weight_drift_rel": final_weights["weight_drift_rel"],
        },
        "transfer": transfer,
        "corruption_acc": corruption,
        "runtime_seconds": time.time() - started,
        "environment": {
            "torch": torch.__version__,
            "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=PROJECT_DIR, capture_output=True,
                                      text=True).stdout.strip(),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"  [{identifier}] test={test_acc:.4f} "
          f"cifar100={transfer['cifar100']['accuracy']:.4f} "
          f"dH={record['final']['delta_entropy_pct']:+.2f}% "
          f"({record['runtime_seconds']/60:.1f} min)", flush=True)

    del model, best
    empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["eurosat"],
                        choices=list(DEFAULTS))
    parser.add_argument("--methods", nargs="+", default=["full_ft", "lora_r8"],
                        choices=list(METHODS))
    parser.add_argument("--learning-rates", nargs="+", type=float,
                        default=[1e-5, 3e-5, 1e-4, 3e-4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"device={DEVICE} torch={torch.__version__}")
    total = (len(args.datasets) * len(args.methods) *
             len(args.learning_rates) * len(args.seeds))
    done = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            for lr in args.learning_rates:
                for method in args.methods:
                    config = {
                        "dataset": dataset, "method": method, "lr": lr, "seed": seed,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "weight_decay": args.weight_decay, "tag": args.tag,
                        "model_name": args.model, "overwrite": args.overwrite,
                    }
                    train_run(config)
                    done += 1
                    print(f"progress: {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
