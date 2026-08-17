#!/usr/bin/env python3
"""Can the drift signal be acted on? Encoder interpolation after fine-tuning.

A run is trained exactly as in Study B, then the visual encoder and projection
are interpolated back towards their pretrained values,
    theta(alpha) = (1-alpha) * theta_pretrained + alpha * theta_finetuned,
and each interpolant is evaluated on the target test split and on CIFAR-100.
The classifier head is kept at its fine-tuned value because it has no
pretrained counterpart; zero-shot transfer never uses it. For LoRA this
interpolation is exactly a rescaling of the adapter update.

The point is not to propose a new method -- weight interpolation is already
known from robust fine-tuning work -- but to check that the trade-off the drift
signals predict is real and controllable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn

from study2 import data_splits, probes
from study2.run_study2 import (CACHE_DIR, DEFAULTS, DEVICE, MODEL_NAME, OUTPUT_DIR,
                                build_model, empty_cache, format_lr, relative_change,
                                set_seed)


def interpolate(state_initial: dict, state_final: dict, alpha: float,
                skip_prefixes: tuple[str, ...] = ("classifier.",)) -> dict:
    blended = {}
    for key, final in state_final.items():
        initial = state_initial[key]
        if any(key.startswith(prefix) for prefix in skip_prefixes) or not final.is_floating_point():
            blended[key] = final
        else:
            blended[key] = initial * (1.0 - alpha) + final * alpha
    return blended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eurosat")
    parser.add_argument("--method", default="full_ft")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--transfer-images", type=int, default=5000)
    args = parser.parse_args()

    from transformers import CLIPProcessor

    identifier = (f"INT_{args.dataset}_{args.method}_lr{format_lr(args.lr)}"
                  f"_seed{args.seed}")
    out_path = OUTPUT_DIR.parent / "intervention" / f"{identifier}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    set_seed(args.seed)
    spec = DEFAULTS[args.dataset]
    splits = data_splits.build_target_splits(args.dataset, spec["train_per_class"],
                                             spec["val_per_class"], spec["probe_per_class"])
    train_loader = data_splits.loader(splits["train"], args.batch_size, shuffle=True,
                                       seed=args.seed)
    val_loader = data_splits.loader(splits["val"], args.batch_size, shuffle=False)
    probe_loader = data_splits.loader(splits["probe"], args.batch_size, shuffle=False)
    test_loader = data_splits.loader(splits["test"], args.batch_size, shuffle=False)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    transfer = data_splits.build_transfer_set("cifar100", args.transfer_images)
    transfer_loader = data_splits.loader(transfer["dataset"], args.batch_size, shuffle=False)

    model = build_model(args.method, splits["num_classes"]).to(DEVICE)
    state_initial = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    encoder_snapshot = probes.encoder_weight_snapshot(model.vision_model)
    baseline_attention = probes.attention_summary(model, probe_loader, DEVICE)
    baseline_representation = probes.representation_probe(model, probe_loader, DEVICE)
    text_feats = probes.text_features(model, transfer["class_names"], processor, DEVICE)
    baseline_transfer = probes.zero_shot_accuracy(model, transfer_loader, text_feats, DEVICE)
    print(f"pretrained CIFAR-100 on {transfer['size']} images: {baseline_transfer:.4f}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(1, int(0.05 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = nn.CrossEntropyLoss()
    best = {"val_acc": -1.0, "state": None, "epoch": -1}

    for epoch in range(args.epochs):
        model.train()
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
        val_acc = probes.classifier_accuracy(model, val_loader, DEVICE)
        print(f"  epoch {epoch+1}/{args.epochs} val={val_acc:.4f}", flush=True)
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch + 1,
                    "state": {k: v.detach().to("cpu", copy=True)
                              for k, v in model.state_dict().items()}}

    rows = []
    for alpha in args.alphas:
        blended = interpolate(state_initial, best["state"], alpha)
        model.load_state_dict({k: v.to(DEVICE) for k, v in blended.items()})
        model.eval()
        attention = probes.attention_summary(model, probe_loader, DEVICE)
        representation = probes.representation_probe(model, probe_loader, DEVICE)
        drift = probes.representation_drift(representation, baseline_representation)
        weights = probes.weight_drift(model.vision_model, encoder_snapshot)
        feats = probes.text_features(model, transfer["class_names"], processor, DEVICE)
        row = {
            "alpha": alpha,
            "test_acc": probes.classifier_accuracy(model, test_loader, DEVICE),
            "transfer_acc": probes.zero_shot_accuracy(model, transfer_loader, feats, DEVICE),
            "delta_entropy_pct": relative_change(attention["entropy_mean"],
                                                 baseline_attention["entropy_mean"]),
            "cka_mean": drift["cka_mean"],
            "embedding_drift": drift["embedding_drift"],
            "weight_drift_rel": weights["weight_drift_rel"],
        }
        row["transfer_retention"] = 100 * row["transfer_acc"] / baseline_transfer
        rows.append(row)
        print(f"  alpha={alpha:.2f} test={row['test_acc']:.4f} "
              f"transfer={row['transfer_acc']:.4f} dH={row['delta_entropy_pct']:+.2f}%",
              flush=True)

    record = {
        "run_id": identifier,
        "config": vars(args) | {"device": str(DEVICE), "model_name": MODEL_NAME,
                                 "splits": splits["sizes"]},
        "selected_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "pretrained_transfer_acc": baseline_transfer,
        "transfer_images": transfer["size"],
        "sweep": rows,
        "runtime_seconds": time.time() - started,
    }
    out_path.write_text(json.dumps(record, indent=2))
    print(f"wrote {out_path} ({record['runtime_seconds']/60:.1f} min)")
    del model
    empty_cache()


if __name__ == "__main__":
    main()
