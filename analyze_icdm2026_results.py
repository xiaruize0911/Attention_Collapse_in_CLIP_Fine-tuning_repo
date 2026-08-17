#!/usr/bin/env python3
"""Aggregate and verify the controlled ICDM 2026 experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_all_experiments as experiments
from run_icdm2026_experiments import (
    DATASETS,
    EPOCHS,
    LEARNING_RATES,
    METHODS,
    SEEDS,
    experiment_id,
)
from src.dataset import get_dataloader, load_cifar100
from src.model import CLIPClassifier, create_lora_model


PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
CHECKPOINTS_DIR = PROJECT_DIR / "outputs" / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "icdm2026"
ZERO_SHOT_CACHE = OUTPUT_DIR / "cifar100_zero_shot.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def percent_delta(history: dict, key: str) -> float:
    baseline = float(history["baseline_metrics"][key])
    final = float(history["final_metrics"][key])
    return (final - baseline) / baseline * 100.0


def expected_run_ids() -> list[str]:
    return [
        experiment_id(method, dataset, lr, seed, EPOCHS[dataset])
        for seed in SEEDS
        for dataset in DATASETS
        for lr in LEARNING_RATES
        for method in METHODS
    ]


def rebuild_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=experiments.DEVICE, weights_only=False)
    config = checkpoint["config"]
    state = checkpoint["model_state_dict"]
    num_classes = 37 if config["dataset"] == "pets" else 10
    if config["method"] == "lora":
        model = create_lora_model(
            num_classes=num_classes,
            lora_r=int(config.get("lora_r", 8)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        )
    else:
        model = CLIPClassifier(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(experiments.DEVICE).eval()


@torch.no_grad()
def cifar100_zero_shot(run_id: str) -> float:
    from transformers import CLIPProcessor

    checkpoint_path = CHECKPOINTS_DIR / run_id / "best_model.pth"
    model = rebuild_model(checkpoint_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset, _, class_names = load_cifar100(cache_dir=str(PROJECT_DIR / "data"))
    loader = get_dataloader(dataset, batch_size=64, shuffle=False, num_workers=2)

    text_inputs = processor(
        text=[f"a photo of a {name}" for name in class_names],
        return_tensors="pt",
        padding=True,
    ).to(experiments.DEVICE)
    text_features = model.clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(experiments.DEVICE)
        labels = labels.to(experiments.DEVICE)
        vision = model.vision_model(pixel_values=images, output_attentions=False)
        image_features = model.visual_projection(vision.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        predictions = (image_features @ text_features.T).argmax(dim=-1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())

    del model
    experiments.empty_device_cache()
    return correct / total


def collect_rows(refresh_zero_shot: bool) -> tuple[list[dict], list[str]]:
    cache = load_json(ZERO_SHOT_CACHE) if ZERO_SHOT_CACHE.exists() else {}
    rows = []
    missing = []
    for run_id in expected_run_ids():
        history_path = METRICS_DIR / f"{run_id}_history.json"
        if not history_path.exists():
            missing.append(run_id)
            continue
        history = load_json(history_path)
        config = history.get("config", {})
        # Older histories keep configuration only in checkpoints; the ID remains authoritative.
        pieces = run_id.split("_")
        method = "full_ft" if pieces[1] == "fullft" else "lora"
        dataset = "pets" if "_pets_" in run_id else "eurosat"
        lr = next(value for value in LEARNING_RATES if f"lr{value:.0e}".replace("+0", "").replace("-0", "-") in run_id)
        seed = next(value for value in SEEDS if f"seed{value}_" in run_id)

        if refresh_zero_shot and run_id not in cache:
            cache[run_id] = cifar100_zero_shot(run_id)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ZERO_SHOT_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

        rows.append({
            "experiment_id": run_id,
            "dataset": dataset,
            "method": method,
            "learning_rate": lr,
            "seed": seed,
            "epochs": EPOCHS[dataset],
            "best_validation_accuracy": float(history["best_val_acc"]),
            "delta_entropy_pct": percent_delta(history, "entropy_mean"),
            "delta_erf95_pct": percent_delta(history, "erf95_mean"),
            "delta_gini_pct": percent_delta(history, "gini_mean"),
            "delta_head_diversity_pct": percent_delta(history, "head_diversity_mean"),
            "cifar100_zero_shot": cache.get(run_id),
        })
    return rows, missing


def summarize(rows: list[dict], missing: list[str]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["learning_rate"])].append(row)

    summary_rows = []
    metric_keys = (
        "best_validation_accuracy",
        "delta_entropy_pct",
        "delta_erf95_pct",
        "delta_gini_pct",
        "delta_head_diversity_pct",
        "cifar100_zero_shot",
    )
    for (dataset, method, lr), bucket in sorted(grouped.items()):
        item = {"dataset": dataset, "method": method, "learning_rate": lr, "n": len(bucket)}
        for key in metric_keys:
            values = [float(row[key]) for row in bucket if row[key] is not None]
            item[f"mean_{key}"] = float(np.mean(values)) if values else None
            item[f"std_{key}"] = float(np.std(values)) if values else None
        summary_rows.append(item)

    return {
        "expected_runs": len(expected_run_ids()),
        "completed_runs": len(rows),
        "completion_rate": len(rows) / len(expected_run_ids()),
        "missing_experiment_ids": missing,
        "grouped_summary": summary_rows,
    }


def write_outputs(rows: list[dict], summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        with (OUTPUT_DIR / "run_summaries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-zero-shot", action="store_true")
    args = parser.parse_args()
    rows, missing = collect_rows(args.refresh_zero_shot)
    summary = summarize(rows, missing)
    write_outputs(rows, summary)
    print(json.dumps({key: summary[key] for key in ("expected_runs", "completed_runs", "completion_rate")}, indent=2))
    if missing:
        print(f"Missing {len(missing)} runs; resume with run_icdm2026_experiments.py")


if __name__ == "__main__":
    main()
