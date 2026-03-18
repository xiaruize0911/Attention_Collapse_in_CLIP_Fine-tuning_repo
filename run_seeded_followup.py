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
from pathlib import Path

import run_all_experiments as rae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--run-fullft", action="store_true")
    parser.add_argument("--run-lora", action="store_true")
    args = parser.parse_args()

    run_fullft = args.run_fullft or (not args.run_fullft and not args.run_lora)
    run_lora = args.run_lora or (not args.run_fullft and not args.run_lora)

    for seed in args.seeds:
        print(f"\n=== Follow-up seed {seed} ===")
        rae.SEED = seed
        rae.set_seed(seed)
        suffix = f"seed{seed}_e{args.epochs}"

        if run_fullft:
            exp_id = f"F1_fullft_eurosat_{suffix}"
            metrics_path = Path("outputs/metrics") / f"{exp_id}_history.json"
            if metrics_path.exists():
                print(f"Skipping existing {exp_id}")
            else:
                print(f"Running {exp_id}")
                rae.run_full_ft_experiment(
                    "eurosat",
                    lr=1e-5,
                    num_epochs=args.epochs,
                    experiment_id=exp_id,
                )

        if run_lora:
            exp_id = f"F1_lora_r8_eurosat_{suffix}"
            metrics_path = Path("outputs/metrics") / f"{exp_id}_history.json"
            if metrics_path.exists():
                print(f"Skipping existing {exp_id}")
            else:
                print(f"Running {exp_id}")
                rae.run_lora_experiment(
                    "eurosat",
                    lora_r=8,
                    lr=1e-4,
                    num_epochs=args.epochs,
                    experiment_id=exp_id,
                )


if __name__ == "__main__":
    main()
