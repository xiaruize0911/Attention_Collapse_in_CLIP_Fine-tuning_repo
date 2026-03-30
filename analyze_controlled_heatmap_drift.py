#!/usr/bin/env python3
"""Aggregate run-level summaries for the controlled heatmap-drift study."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "controlled_heatmap_drift"
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
CHECKPOINTS_DIR = PROJECT_DIR / "outputs" / "checkpoints"
ZERO_SHOT_CACHE_PATH = OUTPUT_DIR / "zero_shot_cache.json"
EXPECTED_DATASETS = ("eurosat", "pets")
EXPECTED_METHODS = ("full_ft", "lora")
EXPECTED_LRS = (1e-6, 5e-6, 1e-5, 5e-5)
EXPECTED_SEEDS = (7, 11, 19, 42, 123)
EXPECTED_EPOCHS = {"eurosat": 20, "pets": 30}


def parse_experiment_name(exp_id: str) -> dict[str, object] | None:
    parts = exp_id.split("_")
    if not parts or parts[0] != "CHD":
        return None

    if parts[1] == "fullft":
        if len(parts) < 6:
            return None
        method = "full_ft"
        dataset = parts[2]
        lr_token = parts[3]
        seed_token = parts[4]
        epoch_token = parts[5]
    elif parts[1] == "lora":
        if len(parts) < 7:
            return None
        method = "lora"
        dataset = parts[3]
        lr_token = parts[4]
        seed_token = parts[5]
        epoch_token = parts[6]
    else:
        return None

    return {
        "experiment_id": exp_id,
        "method": method,
        "dataset": dataset,
        "lr": float(lr_token.replace("lr", "")),
        "seed": int(seed_token.replace("seed", "")),
        "epochs": int(epoch_token.replace("e", "")),
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_zero_shot_cache() -> dict[str, float | None]:
    if not ZERO_SHOT_CACHE_PATH.exists():
        return {}
    with open(ZERO_SHOT_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_zero_shot_cache(cache: dict[str, float | None]) -> None:
    ZERO_SHOT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ZERO_SHOT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def mean_delta(history: dict, key: str) -> float:
    baseline = history["baseline_metrics"][key]
    final = history["final_metrics"][key]
    return float((final - baseline) / baseline * 100.0)


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def expected_experiment_ids() -> list[str]:
    exp_ids: list[str] = []
    for seed in EXPECTED_SEEDS:
        for dataset in EXPECTED_DATASETS:
            epochs = EXPECTED_EPOCHS[dataset]
            for lr in EXPECTED_LRS:
                lr_label = f"{lr:.0e}".replace("+0", "").replace("-0", "-")
                exp_ids.append(f"CHD_fullft_{dataset}_lr{lr_label}_seed{seed}_e{epochs}")
                exp_ids.append(f"CHD_lora_r8_{dataset}_lr{lr_label}_seed{seed}_e{epochs}")
    return exp_ids


def evaluate_cifar100(exp_id: str) -> float | None:
    import run_all_experiments as rae

    ckpt_path = CHECKPOINTS_DIR / exp_id / "best_model.pth"
    if not ckpt_path.exists():
        return None
    results = rae.run_zero_shot_evaluation(str(ckpt_path))
    return results.get("cifar100")


def collect_rows(refresh_zero_shot: bool = False) -> list[dict[str, object]]:
    zero_shot_cache = load_zero_shot_cache() if refresh_zero_shot else {}
    rows = []
    for history_path in sorted(METRICS_DIR.glob("CHD*_history.json")):
        exp_id = history_path.stem.replace("_history", "")
        parsed = parse_experiment_name(exp_id)
        if parsed is None:
            continue

        history = load_json(history_path)
        row = dict(parsed)
        row.update({
            "best_val_accuracy": float(history.get("best_val_acc", 0.0)),
            "delta_entropy_pct": mean_delta(history, "entropy_mean"),
            "delta_erf_pct": mean_delta(history, "erf95_mean"),
            "delta_gini_pct": mean_delta(history, "gini_mean"),
            "delta_head_diversity_pct": mean_delta(history, "head_diversity_mean"),
            "baseline_entropy_mean": float(history["baseline_metrics"]["entropy_mean"]),
            "final_entropy_mean": float(history["final_metrics"]["entropy_mean"]),
            "baseline_erf_mean": float(history["baseline_metrics"]["erf95_mean"]),
            "final_erf_mean": float(history["final_metrics"]["erf95_mean"]),
            "baseline_entropy_per_layer": [float(v) for v in history["baseline_metrics"]["entropy_per_layer"]],
            "final_entropy_per_layer": [float(v) for v in history["final_metrics"]["entropy_per_layer"]],
            "delta_entropy_per_layer_pct": [
                float((final - base) / base * 100.0)
                for base, final in zip(
                    history["baseline_metrics"]["entropy_per_layer"],
                    history["final_metrics"]["entropy_per_layer"],
                )
            ],
        })

        if refresh_zero_shot:
            if exp_id not in zero_shot_cache:
                zero_shot_cache[exp_id] = evaluate_cifar100(exp_id)
                save_zero_shot_cache(zero_shot_cache)
            row["cifar100_zero_shot"] = zero_shot_cache.get(exp_id)
        else:
            row["cifar100_zero_shot"] = None

        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]), float(row["lr"]))].append(row)

    grouped_summary = []
    for (dataset, method, lr), bucket in sorted(grouped.items()):
        entropy = [float(item["delta_entropy_pct"]) for item in bucket]
        erf = [float(item["delta_erf_pct"]) for item in bucket]
        gini = [float(item["delta_gini_pct"]) for item in bucket]
        head_div = [float(item["delta_head_diversity_pct"]) for item in bucket]
        acc = [float(item["best_val_accuracy"]) for item in bucket]
        cifar = [float(item["cifar100_zero_shot"]) for item in bucket if item["cifar100_zero_shot"] is not None]
        per_layer = np.asarray([item["delta_entropy_per_layer_pct"] for item in bucket], dtype=float)
        grouped_summary.append({
            "dataset": dataset,
            "method": method,
            "lr": lr,
            "num_runs": len(bucket),
            "mean_delta_entropy_pct": float(np.mean(entropy)),
            "std_delta_entropy_pct": float(np.std(entropy)),
            "mean_delta_erf_pct": float(np.mean(erf)),
            "std_delta_erf_pct": float(np.std(erf)),
            "mean_delta_gini_pct": float(np.mean(gini)),
            "std_delta_gini_pct": float(np.std(gini)),
            "mean_delta_head_diversity_pct": float(np.mean(head_div)),
            "std_delta_head_diversity_pct": float(np.std(head_div)),
            "mean_best_val_accuracy": float(np.mean(acc)),
            "std_best_val_accuracy": float(np.std(acc)),
            "mean_cifar100_zero_shot": float(np.mean(cifar)) if cifar else None,
            "std_cifar100_zero_shot": float(np.std(cifar)) if cifar else None,
            "mean_delta_entropy_per_layer_pct": per_layer.mean(axis=0).tolist(),
            "std_delta_entropy_per_layer_pct": per_layer.std(axis=0).tolist(),
        })

    dataset_level = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        entropy = [float(row["delta_entropy_pct"]) for row in dataset_rows]
        cifar = [float(row["cifar100_zero_shot"]) for row in dataset_rows if row["cifar100_zero_shot"] is not None]
        acc = [float(row["best_val_accuracy"]) for row in dataset_rows]
        dataset_level.append({
            "dataset": dataset,
            "run_count": len(dataset_rows),
            "entropy_vs_accuracy_corr": safe_corr(entropy, acc),
            "entropy_vs_cifar100_corr": safe_corr(
                [float(row["delta_entropy_pct"]) for row in dataset_rows if row["cifar100_zero_shot"] is not None],
                cifar,
            ),
        })

    completed_ids = sorted(str(row["experiment_id"]) for row in rows)
    expected_ids = expected_experiment_ids()
    missing_ids = [exp_id for exp_id in expected_ids if exp_id not in completed_ids]

    return {
        "run_count": len(rows),
        "expected_run_count": len(expected_ids),
        "completion_rate": float(len(rows) / len(expected_ids)) if expected_ids else 0.0,
        "completed_experiment_ids": completed_ids,
        "missing_experiment_ids": missing_ids,
        "grouped_summary": grouped_summary,
        "dataset_level_correlations": dataset_level,
        "notes": [
            "Grouped summaries are run-level aggregates for the controlled heatmap-drift study.",
            "CIFAR-100 values are only populated when --refresh-zero-shot is used.",
            "Method, LR, and dataset should be analyzed jointly when writing the manuscript.",
        ],
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "dataset",
        "method",
        "lr",
        "seed",
        "epochs",
        "best_val_accuracy",
        "delta_entropy_pct",
        "delta_erf_pct",
        "delta_gini_pct",
        "delta_head_diversity_pct",
        "cifar100_zero_shot",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-zero-shot", action="store_true")
    args = parser.parse_args()

    rows = collect_rows(refresh_zero_shot=args.refresh_zero_shot)
    summary = summarize(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUTPUT_DIR / "run_summaries.csv")
    with open(OUTPUT_DIR / "run_summaries.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} run summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
