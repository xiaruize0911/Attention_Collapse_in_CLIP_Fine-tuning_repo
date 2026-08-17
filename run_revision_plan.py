#!/usr/bin/env python3
"""Run the planned revision experiments for the CLIP attention-collapse paper.

This script executes the concrete plan in four phases:
1. Multi-seed replications for EuroSAT/Pets and regularized EuroSAT.
2. Finer APR sweep and APR-on-LoRA experiments.
3. Cross-backbone validation on ViT-B/16.
4. Follow-up analysis refresh.

All runs are skipped automatically if their history JSON already exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import run_all_experiments as rae

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"


def load_baseline_summary() -> dict:
    path = METRICS_DIR / "E1_baseline_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            "Baseline metrics missing. Run the baseline analysis first so regularized experiments "
            "have the pretrained entropy reference."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def history_exists(experiment_id: str) -> bool:
    return (METRICS_DIR / f"{experiment_id}_history.json").exists()


def maybe_run(experiment_id: str, runner: Callable[[], None]) -> None:
    if history_exists(experiment_id):
        print(f"Skipping existing {experiment_id}")
        return
    print(f"Running {experiment_id}")
    runner()


def phase1_multiseed() -> None:
    baseline_summary = load_baseline_summary()

    # EuroSAT completion runs
    for seed in [123]:
        rae.SEED = seed
        rae.set_seed(seed)
        maybe_run(
            f"F1_fullft_eurosat_seed{seed}_e20",
            lambda seed=seed: rae.run_full_ft_experiment(
                "eurosat", lr=1e-5, num_epochs=20,
                experiment_id=f"F1_fullft_eurosat_seed{seed}_e20",
            ),
        )
        maybe_run(
            f"F1_lora_r8_eurosat_seed{seed}_e20",
            lambda seed=seed: rae.run_lora_experiment(
                "eurosat", lora_r=8, lr=1e-4, num_epochs=20,
                experiment_id=f"F1_lora_r8_eurosat_seed{seed}_e20",
            ),
        )

    # Pets multi-seed comparisons
    for seed in [7, 123]:
        rae.SEED = seed
        rae.set_seed(seed)
        maybe_run(
            f"F1_fullft_pets_seed{seed}_e30",
            lambda seed=seed: rae.run_full_ft_experiment(
                "pets", lr=1e-5, num_epochs=30,
                experiment_id=f"F1_fullft_pets_seed{seed}_e30",
            ),
        )
        maybe_run(
            f"F1_lora_r8_pets_seed{seed}_e30",
            lambda seed=seed: rae.run_lora_experiment(
                "pets", lora_r=8, lr=1e-4, num_epochs=30,
                experiment_id=f"F1_lora_r8_pets_seed{seed}_e30",
            ),
        )

    # Seeded regularized EuroSAT runs for run-level comparison
    for seed in [7, 123]:
        rae.SEED = seed
        rae.set_seed(seed)
        maybe_run(
            f"F1_reg_apr_lambda0.1_eurosat_seed{seed}_e20",
            lambda seed=seed: rae.run_regularization_experiment(
                "apr", 0.1, baseline_summary,
                dataset_name="eurosat", lr=1e-5, num_epochs=20,
                experiment_id=f"F1_reg_apr_lambda0.1_eurosat_seed{seed}_e20",
            ),
        )
        maybe_run(
            f"F1_reg_entropy_floor_lambda0.1_eurosat_seed{seed}_e20",
            lambda seed=seed: rae.run_regularization_experiment(
                "entropy_floor", 0.1, baseline_summary,
                dataset_name="eurosat", lr=1e-5, num_epochs=20,
                experiment_id=f"F1_reg_entropy_floor_lambda0.1_eurosat_seed{seed}_e20",
            ),
        )


def phase2_regularization() -> None:
    baseline_summary = load_baseline_summary()
    rae.SEED = 42
    rae.set_seed(42)

    for lam in [0.005, 0.5]:
        maybe_run(
            f"reg_apr_lambda{lam}_eurosat",
            lambda lam=lam: rae.run_regularization_experiment(
                "apr", lam, baseline_summary,
                dataset_name="eurosat", lr=1e-5, num_epochs=20,
                experiment_id=f"reg_apr_lambda{lam}_eurosat",
            ),
        )

    for lam in [0.01, 0.1]:
        maybe_run(
            f"reg_lora_apr_lambda{lam}_eurosat",
            lambda lam=lam: rae.run_regularized_lora_experiment(
                "eurosat", "apr", lam, baseline_summary,
                lora_r=8, lr=1e-4, num_epochs=20,
                experiment_id=f"reg_lora_apr_lambda{lam}_eurosat",
            ),
        )


def phase3_backbone() -> None:
    rae.SEED = 42
    rae.set_seed(42)
    model_name = "openai/clip-vit-base-patch16"

    maybe_run(
        "B1_full_ft_eurosat_vitb16",
        lambda: rae.run_full_ft_experiment(
            "eurosat", lr=1e-5, num_epochs=20,
            experiment_id="B1_full_ft_eurosat_vitb16",
            model_name=model_name,
        ),
    )
    maybe_run(
        "B1_lora_r8_eurosat_vitb16",
        lambda: rae.run_lora_experiment(
            "eurosat", lora_r=8, lr=1e-4, num_epochs=20,
            experiment_id="B1_lora_r8_eurosat_vitb16",
            model_name=model_name,
        ),
    )


def phase4_analysis() -> None:
    import followup_analysis

    followup_analysis.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["phase1", "phase2", "phase3", "phase4", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    selected = set(args.phases)
    if "all" in selected:
        selected = {"phase1", "phase2", "phase3", "phase4"}

    if "phase1" in selected:
        phase1_multiseed()
    if "phase2" in selected:
        phase2_regularization()
    if "phase3" in selected:
        phase3_backbone()
    if "phase4" in selected:
        phase4_analysis()


if __name__ == "__main__":
    main()
