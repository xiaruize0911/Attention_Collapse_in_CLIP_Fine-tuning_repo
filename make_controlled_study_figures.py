#!/usr/bin/env python3
"""Generate paper-facing figures for the controlled heatmap-drift study."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_SUMMARY = PROJECT_DIR / "outputs" / "controlled_heatmap_drift" / "analysis_summary.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "controlled_heatmap_drift" / "figures"
DATASETS = ("eurosat", "pets")
METHODS = ("full_ft", "lora")
LRS = (1e-6, 5e-6, 1e-5, 5e-5)
METHOD_LABELS = {"full_ft": "Full FT", "lora": "LoRA r=8"}
DATASET_LABELS = {"eurosat": "EuroSAT", "pets": "Oxford-IIIT Pets"}
METHOD_COLORS = {"full_ft": "#b5483a", "lora": "#2f6b8a"}


def load_summary() -> dict:
    with open(INPUT_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric_panels(grouped: list[dict[str, object]]) -> None:
    grouped_map = {
        (row["dataset"], row["method"], float(row["lr"])): row
        for row in grouped
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex="col")
    metric_specs = [
        ("mean_best_val_accuracy", "std_best_val_accuracy", "Best Val Accuracy", None),
        ("mean_delta_entropy_pct", "std_delta_entropy_pct", "Delta Entropy (%)", 0.0),
        ("mean_cifar100_zero_shot", "std_cifar100_zero_shot", "CIFAR-100 Zero-Shot", None),
    ]

    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, (mean_key, std_key, title, hline) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                means = []
                stds = []
                for lr in LRS:
                    item = grouped_map[(dataset, method, lr)]
                    means.append(float(item[mean_key]))
                    std_value = item.get(std_key)
                    stds.append(float(std_value) if std_value is not None else 0.0)

                x = np.asarray(LRS, dtype=float)
                y = np.asarray(means, dtype=float)
                err = np.asarray(stds, dtype=float)
                ax.plot(x, y, marker="o", linewidth=2.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                ax.fill_between(x, y - err, y + err, alpha=0.14, color=METHOD_COLORS[method])

            if hline is not None:
                ax.axhline(hline, color="#777777", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_xticks(LRS)
            ax.set_xticklabels(["1e-6", "5e-6", "1e-5", "5e-5"], rotation=20)
            ax.grid(alpha=0.25, linestyle=":")
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS[dataset])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Controlled Study: Matched-LR Accuracy, Heatmap Drift, and Transfer", y=1.06, fontsize=14)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "controlled_lr_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer_heatmaps(grouped: list[dict[str, object]]) -> None:
    grouped_map = {
        (row["dataset"], row["method"], float(row["lr"])): row
        for row in grouped
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    all_values = []
    for dataset in DATASETS:
        for method in METHODS:
            for lr in LRS:
                item = grouped_map[(dataset, method, lr)]
                all_values.extend(float(v) for v in item["mean_delta_entropy_per_layer_pct"])

    vmax = max(abs(min(all_values)), abs(max(all_values)))
    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            matrix = []
            for lr in LRS:
                item = grouped_map[(dataset, method, lr)]
                matrix.append([float(v) for v in item["mean_delta_entropy_per_layer_pct"]])
            matrix = np.asarray(matrix, dtype=float)
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{DATASET_LABELS[dataset]} | {METHOD_LABELS[method]}")
            ax.set_xticks(range(12))
            ax.set_xticklabels([str(i) for i in range(1, 13)])
            ax.set_yticks(range(len(LRS)))
            ax.set_yticklabels(["1e-6", "5e-6", "1e-5", "5e-5"])
            ax.set_xlabel("Layer")
            ax.set_ylabel("Learning Rate")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Mean Delta Entropy per Layer (%)")
    fig.suptitle("Controlled Study: Per-Layer Entropy Drift Across the Matched LR Grid", fontsize=14)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "controlled_per_layer_entropy_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    grouped = summary["grouped_summary"]
    plot_metric_panels(grouped)
    plot_per_layer_heatmaps(grouped)
    print(f"Wrote controlled-study figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
