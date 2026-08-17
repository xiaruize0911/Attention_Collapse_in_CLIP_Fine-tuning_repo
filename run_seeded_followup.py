#!/usr/bin/env python3
"""Focused multi-seed follow-up experiments for the paper revision.

Runs the key EuroSAT comparison across additional seeds:
- Full fine-tuning
- LoRA r=8

This script reuses the main training pipeline so the follow-up remains aligned
with the original protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"


def load_baseline_summary() -> dict:
    baseline_path = METRICS_DIR / "E1_baseline_stats.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Baseline metrics not found. Run E1 baseline analysis first so regularized "
            "follow-up jobs can reuse the stored entropy statistics."
        )
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    return baseline["summary"]


def maybe_run(experiment_id: str, runner) -> None:
    metrics_path = METRICS_DIR / f"{experiment_id}_history.json"
    if metrics_path.exists():
        print(f"Skipping existing {experiment_id}")
        return
    print(f"Running {experiment_id}")
    runner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["eurosat", "pets"], default=["eurosat"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-fullft", action="store_true")
    parser.add_argument("--run-lora", action="store_true")
    parser.add_argument("--run-regularized", action="store_true")
    parser.add_argument(
        "--regularizers",
        nargs="+",
        choices=["apr", "entropy_floor"],
        default=["apr", "entropy_floor"],
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1])
    args = parser.parse_args()

    import run_all_experiments as rae

    run_fullft = args.run_fullft or (not args.run_fullft and not args.run_lora and not args.run_regularized)
    run_lora = args.run_lora or (not args.run_fullft and not args.run_lora and not args.run_regularized)
    run_regularized = args.run_regularized

    baseline_summary = load_baseline_summary() if run_regularized else None
    default_epochs = {"eurosat": 20, "pets": 30}

    for seed in args.seeds:
        print(f"\n=== Follow-up seed {seed} ===")
        rae.SEED = seed
        rae.set_seed(seed)
        for dataset_name in args.datasets:
            num_epochs = args.epochs if args.epochs is not None else default_epochs[dataset_name]
            suffix = f"seed{seed}_e{num_epochs}"

            if run_fullft:
                exp_id = f"F1_fullft_{dataset_name}_{suffix}"
                maybe_run(
                    exp_id,
                    lambda dataset_name=dataset_name, num_epochs=num_epochs, exp_id=exp_id: rae.run_full_ft_experiment(
                        dataset_name,
                        lr=1e-5,
                        num_epochs=num_epochs,
                        experiment_id=exp_id,
                    ),
                )

            if run_lora:
                exp_id = f"F1_lora_r8_{dataset_name}_{suffix}"
                maybe_run(
                    exp_id,
                    lambda dataset_name=dataset_name, num_epochs=num_epochs, exp_id=exp_id: rae.run_lora_experiment(
                        dataset_name,
                        lora_r=8,
                        lr=1e-4,
                        num_epochs=num_epochs,
                        experiment_id=exp_id,
                    ),
                )

            if run_regularized:
                for regularizer in args.regularizers:
                    for lambda_reg in args.lambdas:
                        exp_id = f"F1_reg_{regularizer}_lambda{lambda_reg}_{dataset_name}_{suffix}"
                        maybe_run(
                            exp_id,
                            lambda dataset_name=dataset_name, num_epochs=num_epochs, regularizer=regularizer, lambda_reg=lambda_reg, exp_id=exp_id: rae.run_regularization_experiment(
                                regularizer,
                                lambda_reg,
                                baseline_summary,
                                dataset_name=dataset_name,
                                lr=1e-5,
                                num_epochs=num_epochs,
                                experiment_id=exp_id,
                            ),
                        )


if __name__ == "__main__":
    main()
