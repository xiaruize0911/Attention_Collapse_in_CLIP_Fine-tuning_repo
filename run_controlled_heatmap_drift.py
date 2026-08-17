#!/usr/bin/env python3
"""Run the matched-learning-rate CLIP heatmap-drift study.

This study keeps attention heatmaps central while separating adaptation method
from optimization scale. Full FT and LoRA are both evaluated on the same LR
grid, datasets, and seed set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"

DEFAULT_DATASETS = ("eurosat", "pets")
DEFAULT_METHODS = ("full_ft", "lora")
DEFAULT_LRS = (1e-6, 5e-6, 1e-5, 5e-5)
DEFAULT_EPOCHS = {"eurosat": 20, "pets": 30}


def history_exists(experiment_id: str) -> bool:
    return (METRICS_DIR / f"{experiment_id}_history.json").exists()


def maybe_run(experiment_id: str, runner) -> None:
    if history_exists(experiment_id):
        print(f"Skipping existing {experiment_id}")
        return
    print(f"Running {experiment_id}")
    runner()


def baseline_exists() -> bool:
    return (METRICS_DIR / "E1_baseline_stats.json").exists()


def ensure_baseline() -> None:
    import run_all_experiments as rae

    if baseline_exists():
        print("Skipping existing baseline analysis")
        return
    print("Running baseline analysis")
    rae.run_baseline_analysis()


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("-0", "-")


def run_study(
    datasets: list[str],
    methods: list[str],
    lrs: list[float],
    seeds: list[int],
    epochs_override: int | None = None,
) -> None:
    import run_all_experiments as rae

    ensure_baseline()

    for seed in seeds:
        print(f"\n=== Controlled heatmap drift seed {seed} ===")
        rae.SEED = seed
        rae.set_seed(seed)

        for dataset_name in datasets:
            num_epochs = epochs_override if epochs_override is not None else DEFAULT_EPOCHS[dataset_name]

            for lr in lrs:
                lr_label = format_lr(lr)

                if "full_ft" in methods:
                    exp_id = f"CHD_fullft_{dataset_name}_lr{lr_label}_seed{seed}_e{num_epochs}"
                    maybe_run(
                        exp_id,
                        lambda dataset_name=dataset_name, lr=lr, num_epochs=num_epochs, exp_id=exp_id:
                            rae.run_full_ft_experiment(
                                dataset_name=dataset_name,
                                lr=lr,
                                num_epochs=num_epochs,
                                experiment_id=exp_id,
                                attention_eval_mode="epoch",
                            ),
                    )

                if "lora" in methods:
                    exp_id = f"CHD_lora_r8_{dataset_name}_lr{lr_label}_seed{seed}_e{num_epochs}"
                    maybe_run(
                        exp_id,
                        lambda dataset_name=dataset_name, lr=lr, num_epochs=num_epochs, exp_id=exp_id:
                            rae.run_lora_experiment(
                                dataset_name=dataset_name,
                                lora_r=8,
                                lr=lr,
                                num_epochs=num_epochs,
                                experiment_id=exp_id,
                                attention_eval_mode="epoch",
                            ),
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DEFAULT_DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(DEFAULT_METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19, 42, 123])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_study(
        datasets=args.datasets,
        methods=args.methods,
        lrs=args.lrs,
        seeds=args.seeds,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
