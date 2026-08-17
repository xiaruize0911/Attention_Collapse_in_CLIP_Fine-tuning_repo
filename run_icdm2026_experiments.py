#!/usr/bin/env python3
"""Reproduce the controlled experiment matrix used in the ICDM 2026 paper.

The design matches learning rate, dataset, and seed across Full FT and LoRA.
Completed histories are skipped, so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_all_experiments as experiments


PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
DATASETS = ("eurosat", "pets")
METHODS = ("full_ft", "lora")
LEARNING_RATES = (1e-6, 5e-6, 1e-5, 5e-5)
SEEDS = (7, 11, 19, 42, 123)
EPOCHS = {"eurosat": 20, "pets": 30}


def format_learning_rate(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


def experiment_id(method: str, dataset: str, lr: float, seed: int, epochs: int) -> str:
    prefix = "CHD_fullft" if method == "full_ft" else "CHD_lora_r8"
    return f"{prefix}_{dataset}_lr{format_learning_rate(lr)}_seed{seed}_e{epochs}"


def is_complete(run_id: str) -> bool:
    return (METRICS_DIR / f"{run_id}_history.json").exists()


def run_matrix(
    datasets: list[str],
    methods: list[str],
    learning_rates: list[float],
    seeds: list[int],
    epochs_override: int | None,
    max_train_samples: int | None,
) -> None:
    print(f"Experiment device: {experiments.DEVICE}")
    for seed in seeds:
        experiments.SEED = seed
        experiments.set_seed(seed)
        for dataset in datasets:
            epochs = epochs_override if epochs_override is not None else EPOCHS[dataset]
            for lr in learning_rates:
                for method in methods:
                    run_id = experiment_id(method, dataset, lr, seed, epochs)
                    if max_train_samples is not None:
                        run_id = f"{run_id}_n{max_train_samples}"
                    if is_complete(run_id):
                        print(f"Skipping completed run: {run_id}")
                        continue

                    print(f"Running: {run_id}")
                    if method == "full_ft":
                        experiments.run_full_ft_experiment(
                            dataset,
                            lr=lr,
                            num_epochs=epochs,
                            experiment_id=run_id,
                            attention_eval_mode="epoch",
                            max_train_samples=max_train_samples,
                        )
                    else:
                        experiments.run_lora_experiment(
                            dataset,
                            lora_r=8,
                            lr=lr,
                            num_epochs=epochs,
                            experiment_id=run_id,
                            attention_eval_mode="epoch",
                            max_train_samples=max_train_samples,
                        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--learning-rates", nargs="+", type=float, default=list(LEARNING_RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the paper protocol for a smoke test (for example, --epochs 1).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Bound the training set for a smoke test; omit for the paper protocol.",
    )
    args = parser.parse_args()
    run_matrix(
        args.datasets,
        args.methods,
        args.learning_rates,
        args.seeds,
        args.epochs,
        args.max_train_samples,
    )


if __name__ == "__main__":
    main()
