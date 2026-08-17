#!/usr/bin/env python3
"""Aggregate the protocol-hardened runs into the tables, statistics and figures.

Everything the paper reports about Study 2 is produced here from the per-run
JSON records in study2/results, so the numbers can be regenerated with a single
command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "study2" / "results"
OUT_DIR = PROJECT_DIR / "study2" / "analysis"
FIG_DIR = OUT_DIR / "figures"

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
CORE_METHODS = ("full_ft", "lora_r8")
METHOD_LABEL = {
    "full_ft": "Full FT",
    "lora_r8": "LoRA r=8",
    "lora_r8_frozen_proj": "LoRA (frozen proj.)",
    "linear_probe": "Linear probe",
    "last_block": "Last block",
}
DATASET_LABEL = {"eurosat": "EuroSAT", "pets": "Oxford-IIIT Pets"}
PREDICTORS = {
    "abs_delta_entropy_pct": r"$|\Delta H|$ (attention entropy)",
    "delta_entropy_pct": r"$\Delta H$ (signed)",
    "abs_delta_erf95_pct": r"$|\Delta$ERF@0.95$|$",
    "abs_delta_gini_pct": r"$|\Delta$Gini$|$",
    "abs_delta_entropy_last_layer_pct": r"$|\Delta H_{12}|$ (last block)",
    "delta_entropy_last_layer_pct": r"$\Delta H_{12}$ (signed)",
    "cka_mean": "CKA to pretrained (mean)",
    "cka_last": "CKA to pretrained (block 12)",
    "embedding_drift": "Embedding drift",
    "weight_drift_rel": "Relative weight change",
    "train_loss_final": "Final training loss",
    "test_acc": "Target-task accuracy",
    "log10_lr": "$\\log_{10}$ learning rate$^{*}$",
    "epoch1_transfer_retention_pct": "1k labelled probe$^{*}$",
}


# --------------------------------------------------------------------------
def load_runs(tag_filter: bool = True) -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        config = record["config"]
        if tag_filter and config.get("tag"):
            continue
        history = record["history"]
        first, last = history[0], history[-1]
        row = {
            "run_id": record["run_id"],
            "dataset": config["dataset"],
            "method": config["method"],
            "model_name": config.get("model_name", DEFAULT_MODEL),
            "lr": float(config["lr"]),
            "seed": int(config["seed"]),
            "epochs": int(config["epochs"]),
            "steps": int(config["total_steps"]),
            "trainable_params": record["parameters"]["trainable"],
            "selected_epoch": record["selected_epoch"],
            "val_acc": record["best_val_acc"],
            "test_acc": record["test_acc"],
            "diverged": record["diverged"],
            "delta_entropy_pct": record["final"]["delta_entropy_pct"],
            "delta_erf95_pct": record["final"]["delta_erf95_pct"],
            "delta_gini_pct": record["final"]["delta_gini_pct"],
            "delta_entropy_last_layer_pct": record["final"]["delta_entropy_per_layer_pct"][-1],
            "cka_mean": record["final"]["cka_mean"],
            "cka_last": record["final"]["cka_per_layer"][-1],
            "embedding_drift": record["final"]["embedding_drift"],
            "weight_drift_rel": record["final"]["weight_drift_rel"],
            "train_loss_final": last["train_loss"],
            "epoch1_delta_entropy_pct": first["delta_entropy_pct"],
            "epoch1_weight_drift_rel": first["weight_drift_rel"],
            "epoch1_embedding_drift": first["embedding_drift"],
            "epoch1_cka_mean": first["cka_mean"],
            "epoch1_transfer_retention": first["transfer_track_retention"],
            "baseline_entropy": record["baseline"]["attention"]["entropy_mean"],
        }
        for key in ("delta_entropy_pct", "delta_erf95_pct", "delta_gini_pct",
                    "delta_entropy_last_layer_pct"):
            row[f"abs_{key}"] = abs(row[key])
        row["log10_lr"] = float(np.log10(row["lr"]))
        row["epoch1_log10_lr"] = row["log10_lr"]
        track = first.get("transfer_track_retention")
        row["epoch1_transfer_retention_pct"] = 100 * track if track is not None else np.nan
        row["epoch1_epoch1_transfer_retention_pct"] = row["epoch1_transfer_retention_pct"]
        row["epoch1_abs_delta_entropy_pct"] = abs(row["epoch1_delta_entropy_pct"])
        row["epoch1_abs_delta_erf95_pct"] = abs(first["delta_erf95_pct"])
        row["epoch1_abs_delta_gini_pct"] = abs(first["delta_gini_pct"])
        row["epoch1_train_loss_final"] = first["train_loss"]
        baseline_layers = record["baseline"]["attention"]["entropy_per_layer"]
        epoch1_last = 100 * (first["entropy_per_layer"][-1] - baseline_layers[-1]) / baseline_layers[-1]
        row["epoch1_delta_entropy_last_layer_pct"] = epoch1_last
        row["epoch1_abs_delta_entropy_last_layer_pct"] = abs(epoch1_last)
        for name, spec in record["transfer"].items():
            row[f"{name}_acc"] = spec["accuracy"] * 100
            row[f"{name}_retention"] = spec["retention"] * 100
            row[f"{name}_pretrained"] = spec["pretrained"] * 100
        if record["corruption_acc"]:
            row["corruption_mean_acc"] = float(np.mean(list(record["corruption_acc"].values())))
            row["corruption_gap"] = record["test_acc"] - row["corruption_mean_acc"]
        row["delta_entropy_per_layer_pct"] = record["final"]["delta_entropy_per_layer_pct"]
        row["history"] = history
        rows.append(row)
    return pd.DataFrame(rows)


def mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "--"
    if len(values) == 1:
        return f"{values.iloc[0]:.2f}"
    return f"{values.mean():.2f}\\,$\\pm$\\,{values.std(ddof=1):.2f}"


# --------------------------------------------------------------------------
def latex_lr(lr: float) -> str:
    exponent = int(np.floor(np.log10(lr)))
    mantissa = lr / (10 ** exponent)
    if abs(mantissa - 1.0) < 1e-9:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\!\\cdot\\!10^{{{exponent}}}$"


def core_grid_table(df: pd.DataFrame) -> str:
    core = df[df.method.isin(CORE_METHODS)]
    lines = []
    for dataset in ("eurosat", "pets"):
        block = core[core.dataset == dataset]
        if block.empty:
            continue
        for method in CORE_METHODS:
            for lr in sorted(block.lr.unique()):
                cell = block[(block.method == method) & (block.lr == lr)]
                if cell.empty:
                    continue
                lines.append(" & ".join([
                    DATASET_LABEL[dataset],
                    METHOD_LABEL[method],
                    latex_lr(lr),
                    f"{len(cell)}",
                    mean_sd(cell.test_acc * 100),
                    mean_sd(cell.delta_entropy_pct),
                    mean_sd(cell.delta_entropy_last_layer_pct),
                    mean_sd(cell.cifar100_retention),
                ]) + r" \\")
    return "\n".join(lines)


def reference_rows_table(df: pd.DataFrame) -> str:
    """Compact single-column rows for the bracketing configurations."""
    short = {"eurosat": "ES", "pets": "Pets"}
    wanted = [
        ("eurosat", "linear_probe", None),
        ("eurosat", "last_block", None),
        ("eurosat", "lora_r8_frozen_proj", None),
        ("eurosat", "lora_r8", 1e-4),
        ("eurosat", "full_ft", 1e-4),
        ("pets", "linear_probe", None),
    ]
    lines = []
    for dataset, method, only_lr in wanted:
        block = df[(df.dataset == dataset) & (df.method == method)]
        for lr in sorted(block.lr.unique()):
            if only_lr is not None and not np.isclose(lr, only_lr):
                continue
            cell = block[block.lr == lr]
            if cell.empty:
                continue
            lines.append(" & ".join([
                f"{short[dataset]}, {METHOD_LABEL[method]}",
                latex_lr(lr),
                f"{cell.trainable_params.iloc[0]/1e6:.2f}",
                mean_sd(cell.test_acc * 100),
                mean_sd(cell.delta_entropy_pct),
                mean_sd(cell.cifar100_retention),
            ]) + r" \\")
    return "\n".join(lines)


NEEDS_LABELS = {"test_acc"}
PREDICTOR_TABLE_ORDER = (
    "cka_mean", "embedding_drift", "weight_drift_rel",
    "abs_delta_gini_pct", "abs_delta_entropy_pct", "abs_delta_entropy_last_layer_pct",
    "delta_entropy_pct", "log10_lr", "train_loss_final", "test_acc",
    "epoch1_transfer_retention_pct",
)


def predictor_rows_table(df: pd.DataFrame) -> str:
    final = {item["predictor"]: item for item in predictor_ranking(df, stage="final")}
    early = {item["predictor"]: item for item in predictor_ranking(df, stage="epoch1")}
    lines = []
    for key in PREDICTOR_TABLE_ORDER:
        item = final.get(key)
        if item is None:
            continue
        label = item["label"] + (r"$^{\dagger}$" if key in NEEDS_LABELS else "")
        early_value = early.get(key, {}).get("spearman_overall")
        lines.append(" & ".join([
            label,
            f"{item['spearman_overall']:+.3f}",
            f"{item['abs_spearman_within_dataset_mean']:.3f}",
            f"{item['abs_spearman_within_mean']:.3f}",
            f"{early_value:+.3f}" if early_value is not None else "--",
        ]) + r" \\")
    return "\n".join(lines)


def paired_method_tests(df: pd.DataFrame) -> list[dict]:
    """Full FT vs LoRA on identical (dataset, lr, seed) cells."""
    core = df[df.method.isin(CORE_METHODS)]
    out = []
    for dataset in sorted(core.dataset.unique()):
        block = core[core.dataset == dataset]
        pivot_keys = ["lr", "seed"]
        full = block[block.method == "full_ft"].set_index(pivot_keys)
        lora = block[block.method == "lora_r8"].set_index(pivot_keys)
        shared = full.index.intersection(lora.index)
        if len(shared) < 3:
            continue
        for metric in ("delta_entropy_pct", "cifar100_retention", "cifar10_retention",
                        "test_acc", "weight_drift_rel", "cka_mean"):
            a = full.loc[shared, metric].astype(float).to_numpy()
            b = lora.loc[shared, metric].astype(float).to_numpy()
            if metric == "test_acc":
                a, b = a * 100, b * 100
            diff = a - b
            wilcoxon = stats.wilcoxon(a, b) if len(shared) >= 6 else None
            ttest = stats.ttest_rel(a, b)
            out.append({
                "dataset": dataset,
                "metric": metric,
                "n_pairs": int(len(shared)),
                "full_ft_mean": float(np.mean(a)),
                "lora_mean": float(np.mean(b)),
                "mean_difference": float(np.mean(diff)),
                "cohens_dz": float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) else float("nan"),
                "t_statistic": float(ttest.statistic),
                "t_p_value": float(ttest.pvalue),
                "wilcoxon_p_value": float(wilcoxon.pvalue) if wilcoxon else None,
            })
    return out


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def dose_response(df: pd.DataFrame) -> list[dict]:
    core = df[df.method.isin(CORE_METHODS)]
    out = []
    for dataset in sorted(core.dataset.unique()):
        for method in CORE_METHODS:
            cell = core[(core.dataset == dataset) & (core.method == method)]
            if cell.lr.nunique() < 3:
                continue
            for metric in ("delta_entropy_pct", "cifar100_retention", "test_acc"):
                rho, p = stats.spearmanr(np.log10(cell.lr), cell[metric].astype(float))
                out.append({"dataset": dataset, "method": method, "metric": metric,
                            "n_runs": int(len(cell)), "spearman_rho": float(rho),
                            "p_value": float(p)})
    return out


def predictor_ranking(df: pd.DataFrame, target: str = "cifar100_retention",
                      stage: str = "final") -> list[dict]:
    """How well does each cheap internal signal rank the amount of forgetting?

    `overall` pools every core run. `within` averages the rank correlation
    computed inside each (dataset, method) group, which removes the possibility
    that a predictor only tracks the method label or the learning rate.
    """
    core = df[df.method.isin(CORE_METHODS)].copy()
    prefix = "" if stage == "final" else "epoch1_"
    out = []
    for key, label in PREDICTORS.items():
        column = f"{prefix}{key}"
        if column not in core.columns:
            # no epoch-1 counterpart exists for this signal (e.g. held-out accuracy)
            continue
        values = core[[column, target, "dataset", "method", "lr"]].dropna()
        if len(values) < 6:
            continue
        rho, p = stats.spearmanr(values[column], values[target])
        within = []
        for _, group in values.groupby(["dataset", "method"]):
            if len(group) >= 5 and group[column].nunique() > 2:
                within.append(stats.spearmanr(group[column], group[target]).statistic)
        per_dataset = []
        for _, group in values.groupby("dataset"):
            if len(group) >= 8 and group[column].nunique() > 2:
                per_dataset.append(stats.spearmanr(group[column], group[target]).statistic)
        out.append({
            "predictor": key,
            "label": label,
            "stage": stage,
            "n": int(len(values)),
            "spearman_overall": float(rho),
            "p_overall": float(p),
            "abs_spearman_overall": float(abs(rho)),
            "spearman_within_mean": float(np.mean(within)) if within else float("nan"),
            "abs_spearman_within_mean": float(np.mean(np.abs(within))) if within else float("nan"),
            "n_groups": len(within),
            "abs_spearman_within_dataset_mean": (float(np.mean(np.abs(per_dataset)))
                                                  if per_dataset else float("nan")),
            "n_datasets": len(per_dataset),
        })
    return sorted(out, key=lambda item: -item["abs_spearman_overall"])


def polarity_by_group(df_all: pd.DataFrame, target: str = "cifar100_retention") -> list[dict]:
    """Run-level sign of each signal's relation to forgetting, per backbone x method.

    The paper's claim is about the *polarity* of a signal, not its strength, so
    this reports the signed correlation inside each group on the EuroSAT runs
    that both backbones share. Run-level, to match the within-group statistics
    reported elsewhere.
    """
    block = df_all[(df_all.dataset == "eurosat") & df_all.method.isin(CORE_METHODS)]
    out = []
    for (model, method), group in block.groupby(["model_name", "method"]):
        entry = {"model": model, "method": method, "n": int(len(group)),
                  "n_rates": int(group.lr.nunique())}
        for key in ("delta_entropy_pct", "cka_mean", "weight_drift_rel",
                     "embedding_drift"):
            if group[key].nunique() > 2:
                rho, pvalue = stats.spearmanr(group[key], group[target])
                entry[key] = {"rho": float(rho), "p_value": float(pvalue)}
        out.append(entry)
    return out


def backbone_check(df_all: pd.DataFrame) -> dict:
    """Runs on any backbone other than the primary one, analysed separately."""
    other = df_all[df_all.model_name != DEFAULT_MODEL]
    if other.empty:
        return {}
    out = {}
    for model, block in other.groupby("model_name"):
        cells = (block.groupby(["dataset", "method", "lr"])
                 .agg(n=("test_acc", "size"), test_acc=("test_acc", "mean"),
                      delta_entropy_pct=("delta_entropy_pct", "mean"),
                      cifar100_retention=("cifar100_retention", "mean"),
                      cka_mean=("cka_mean", "mean"),
                      weight_drift_rel=("weight_drift_rel", "mean"))
                 .reset_index())
        signals = {}
        for key in ("cka_mean", "embedding_drift", "weight_drift_rel",
                     "abs_delta_entropy_pct", "delta_entropy_pct"):
            values = block[[key, "cifar100_retention"]].dropna()
            if len(values) >= 4 and values[key].nunique() > 2:
                rho, pvalue = stats.spearmanr(values[key], values.cifar100_retention)
                signals[key] = {"spearman_rho": float(rho), "p_value": float(pvalue),
                                 "n": int(len(values))}
        out[model] = {"n_runs": int(len(block)),
                      "cells": cells.to_dict("records"),
                      "predictors": signals}
    return out


def bootstrap_predictor_gap(df: pd.DataFrame, target: str = "cifar100_retention",
                            reference: str = "cka_mean", n_boot: int = 4000,
                            seed: int = 0) -> list[dict]:
    """Is the predictor ranking separable, or within resampling noise?

    Resamples runs with replacement and recomputes |rho| for each signal, giving
    a confidence interval per signal and, paired within each resample, the gap
    against the reference signal. The paired form is what answers "does CKA
    really beat this signal", because both are computed on the same runs.
    """
    core = df[df.method.isin(CORE_METHODS)]
    rng = np.random.RandomState(seed)
    signals = [key for key in PREDICTOR_TABLE_ORDER if key in core.columns]
    # Runs within a cell share a configuration, so resampling runs would treat
    # seeds as independent evidence and shrink every interval. Resample cells.
    values = (core.groupby(["dataset", "method", "lr"])[signals + [target]]
              .mean().reset_index()[signals + [target]].dropna())
    if len(values) < 8 or reference not in signals:
        return []
    indices = rng.randint(0, len(values), size=(n_boot, len(values)))

    draws = {key: np.empty(n_boot) for key in signals}
    for b in range(n_boot):
        sample = values.iloc[indices[b]]
        for key in signals:
            rho = stats.spearmanr(sample[key], sample[target]).statistic
            draws[key][b] = abs(rho) if np.isfinite(rho) else np.nan

    out = []
    for key in signals:
        gap = draws[reference] - draws[key]
        finite = draws[key][np.isfinite(draws[key])]
        out.append({
            "predictor": key,
            "label": PREDICTORS[key],
            "abs_rho": float(abs(stats.spearmanr(values[key], values[target]).statistic)),
            "ci_low": float(np.nanpercentile(finite, 2.5)),
            "ci_high": float(np.nanpercentile(finite, 97.5)),
            "gap_vs_reference_mean": float(np.nanmean(gap)),
            "gap_ci_low": float(np.nanpercentile(gap, 2.5)),
            "gap_ci_high": float(np.nanpercentile(gap, 97.5)),
            "reference_wins_fraction": float(np.nanmean(gap > 0)),
            "resampling_unit": "cell means",
            "n_units": int(len(values)),
        })
    return sorted(out, key=lambda item: -item["abs_rho"])


def iso_accuracy_comparison(df: pd.DataFrame, tolerance: float = 1.0) -> list[dict]:
    """Compare transfer at each method's best validation configuration."""
    core = df[df.method.isin(CORE_METHODS)]
    out = []
    for dataset in sorted(core.dataset.unique()):
        block = core[core.dataset == dataset]
        best = {}
        for method in CORE_METHODS:
            cell = block[block.method == method]
            if cell.empty:
                continue
            grouped = cell.groupby("lr").agg(val=("val_acc", "mean"))
            lr = grouped.val.idxmax()
            chosen = cell[cell.lr == lr]
            best[method] = {
                "lr": float(lr),
                "val_acc": float(chosen.val_acc.mean() * 100),
                "test_acc": float(chosen.test_acc.mean() * 100),
                "test_acc_sd": float(chosen.test_acc.std(ddof=1) * 100),
                "cifar100_retention": float(chosen.cifar100_retention.mean()),
                "cifar100_retention_sd": float(chosen.cifar100_retention.std(ddof=1)),
                "delta_entropy_pct": float(chosen.delta_entropy_pct.mean()),
                "n": int(len(chosen)),
            }
        if len(best) == 2:
            out.append({"dataset": dataset, "selection": "best validation accuracy",
                        **{f"{m}_{k}": v for m, spec in best.items() for k, v in spec.items()}})
    return out


LOWER_IS_BETTER = {
    "abs_delta_entropy_pct": True, "abs_delta_entropy_last_layer_pct": True,
    "abs_delta_erf95_pct": True,
    "abs_delta_gini_pct": True, "embedding_drift": True,
    "weight_drift_rel": True, "cka_mean": False,
}


def selection_utility(df: pd.DataFrame, margin: float = 2.0) -> list[dict]:
    """Decision-oriented test: pick a configuration using one signal only.

    Among the runs that come within `margin` points of the best target accuracy,
    choose the one the signal likes best and report how much of the achievable
    transfer retention that choice captures. The signal never sees the transfer
    benchmark, so this is the decision a practitioner could actually make.
    """
    core = df[df.method.isin(CORE_METHODS)]
    out = []
    for dataset in sorted(core.dataset.unique()):
        block = core[core.dataset == dataset]
        # the pool must be selectable without the test split, so gate on validation
        cutoff = block.val_acc.max() * 100 - margin
        candidates = block[block.val_acc * 100 >= cutoff]
        if len(candidates) < 3:
            continue
        oracle = float(candidates.cifar100_retention.max())
        worst = float(candidates.cifar100_retention.min())
        record = {
            "dataset": dataset, "margin_points": margin,
            "n_candidates": int(len(candidates)),
            "accuracy_cutoff": float(cutoff),
            "cutoff_basis": "validation accuracy",
            "oracle_retention": oracle, "worst_retention": worst,
            "random_choice_retention": float(candidates.cifar100_retention.mean()),
            "signals": {},
        }
        for signal, lower_better in LOWER_IS_BETTER.items():
            if signal not in candidates:
                continue
            index = (candidates[signal].idxmin() if lower_better
                     else candidates[signal].idxmax())
            chosen = candidates.loc[index]
            record["signals"][signal] = {
                "chosen_run": chosen.run_id,
                "retention": float(chosen.cifar100_retention),
                "test_acc": float(chosen.test_acc * 100),
                "regret": oracle - float(chosen.cifar100_retention),
                "fraction_of_range": ((float(chosen.cifar100_retention) - worst) /
                                       (oracle - worst)) if oracle > worst else float("nan"),
            }
        out.append(record)
    return out


def temporal_ordering(df: pd.DataFrame) -> dict:
    """Does structural drift precede measurable transfer loss?"""
    core = df[df.method.isin(CORE_METHODS)]
    rows = []
    for _, run in core.iterrows():
        history = run["history"]
        for entry in history:
            rows.append({
                "run_id": run["run_id"], "dataset": run["dataset"],
                "method": run["method"], "lr": run["lr"], "seed": run["seed"],
                "epoch": entry["epoch"],
                "delta_entropy_pct": entry["delta_entropy_pct"],
                "abs_delta_entropy_pct": abs(entry["delta_entropy_pct"]),
                "embedding_drift": entry["embedding_drift"],
                "transfer_retention": 100 * entry["transfer_track_retention"],
                "weight_drift_rel": entry["weight_drift_rel"],
                "cka_mean": entry["cka_mean"],
            })
    panel = pd.DataFrame(rows)
    early = panel[panel.epoch == 1]
    final_transfer = core.set_index("run_id")["cifar100_retention"]
    early = early.assign(final_retention=early.run_id.map(final_transfer)).dropna(
        subset=["final_retention"])
    result = {"n_runs": int(len(early))}
    for key in ("delta_entropy_pct", "weight_drift_rel", "cka_mean"):
        rho, p = stats.spearmanr(early[key], early.final_retention)
        result[f"epoch1_{key}_vs_final"] = {"spearman_rho": float(rho), "p_value": float(p)}
    # The paper argues runs within a cell are not independent, so the separation
    # test is reported over configurations, not runs.
    cells = early.groupby(["dataset", "method", "lr"]).mean(numeric_only=True).reset_index()
    cell_damaged = cells.final_retention < 50
    if cell_damaged.nunique() == 2:
        result["cell_level"] = {"n_damaged": int(cell_damaged.sum()),
                                 "n_healthy": int((~cell_damaged).sum())}
        for key in ("delta_entropy_pct", "abs_delta_entropy_pct", "weight_drift_rel",
                     "embedding_drift", "cka_mean"):
            if key not in cells:
                continue
            healthy = cells.loc[~cell_damaged, key]
            hurt = cells.loc[cell_damaged, key]
            raw = float(np.mean([1.0 * (d < h) + 0.5 * (d == h)
                                 for d in hurt for h in healthy]))
            result["cell_level"][key] = {"auc_oriented": max(raw, 1.0 - raw)}

    damaged = early.final_retention < 50
    if damaged.nunique() == 2:
        for key in ("delta_entropy_pct", "abs_delta_entropy_pct", "weight_drift_rel",
                     "embedding_drift", "cka_mean"):
            if key not in early.columns:
                continue
            healthy_values = early.loc[~damaged, key]
            damaged_values = early.loc[damaged, key]
            raw = float(np.mean([1.0 * (d < h) + 0.5 * (d == h)
                                 for d in damaged_values for h in healthy_values]))
            # orient the score so that 1.0 always means perfect separation
            result[f"epoch1_{key}_auc_for_damage"] = {
                "auc_oriented": max(raw, 1.0 - raw),
                "damaged_scores_lower": raw > 0.5,
                "n_damaged": int(damaged.sum()),
                "n_healthy": int((~damaged).sum()),
                "damaged_mean": float(damaged_values.mean()),
                "healthy_mean": float(healthy_values.mean()),
            }
    return result


def reference_table(df: pd.DataFrame) -> list[dict]:
    out = []
    for (dataset, method, lr), cell in df.groupby(["dataset", "method", "lr"]):
        out.append({
            "dataset": dataset, "method": method, "lr": float(lr), "n": int(len(cell)),
            "test_acc": float(cell.test_acc.mean() * 100),
            "test_acc_sd": float(cell.test_acc.std(ddof=1) * 100) if len(cell) > 1 else None,
            "delta_entropy_pct": float(cell.delta_entropy_pct.mean()),
            "delta_entropy_sd": float(cell.delta_entropy_pct.std(ddof=1)) if len(cell) > 1 else None,
            "cifar100_retention": float(cell.cifar100_retention.mean()),
            "cifar100_retention_sd": float(cell.cifar100_retention.std(ddof=1)) if len(cell) > 1 else None,
            "cifar10_retention": float(cell.cifar10_retention.mean()),
            "weight_drift_rel": float(cell.weight_drift_rel.mean()),
            "cka_mean": float(cell.cka_mean.mean()),
            "corruption_mean_acc": (float(cell.corruption_mean_acc.mean() * 100)
                                     if "corruption_mean_acc" in cell and cell.corruption_mean_acc.notna().any() else None),
            "trainable_params": int(cell.trainable_params.iloc[0]),
        })
    return sorted(out, key=lambda item: (item["dataset"], item["method"], item["lr"]))


# --------------------------------------------------------------------------
def digest(df: pd.DataFrame, summary: dict) -> str:
    """Human-readable digest of every number the paper needs."""
    lines = [f"# Study B digest ({len(df)} runs on the primary backbone)\n"]
    if summary.get("backbone_check"):
        for model, spec in summary["backbone_check"].items():
            lines.append(f"Plus {spec['n_runs']} runs on `{model}`, analysed separately "
                         "in the backbone section below.\n")

    lines.append("## Cells (mean over seeds)\n")
    header = ("| dataset | method | lr | n | test acc | dH % | dH L12 % | "
              "C100 ret % | C10 ret % | w-drift | emb-drift | CKA | corrupt acc |")
    lines += [header, "|" + "---|" * 13]
    for (dataset, method, lr), cell in df.groupby(["dataset", "method", "lr"]):
        corrupt = (f"{cell.corruption_mean_acc.mean()*100:.1f}"
                   if "corruption_mean_acc" in cell and cell.corruption_mean_acc.notna().any() else "--")
        lines.append("| " + " | ".join([
            dataset, METHOD_LABEL.get(method, method), f"{lr:.0e}", str(len(cell)),
            f"{cell.test_acc.mean()*100:.2f}±{cell.test_acc.std(ddof=1)*100 if len(cell)>1 else 0:.2f}",
            f"{cell.delta_entropy_pct.mean():+.2f}±{cell.delta_entropy_pct.std(ddof=1) if len(cell)>1 else 0:.2f}",
            f"{cell.delta_entropy_last_layer_pct.mean():+.2f}",
            f"{cell.cifar100_retention.mean():.1f}±{cell.cifar100_retention.std(ddof=1) if len(cell)>1 else 0:.1f}",
            f"{cell.cifar10_retention.mean():.1f}",
            f"{cell.weight_drift_rel.mean():.4f}",
            f"{cell.embedding_drift.mean():.3f}",
            f"{cell.cka_mean.mean():.3f}", corrupt]) + " |")

    lines.append("\n## Paired Full FT vs LoRA (identical dataset, lr, seed)\n")
    lines += ["| dataset | metric | n | Full FT | LoRA | diff | dz | p (t) | p Holm | p (Wilcoxon) |",
              "|" + "---|" * 10]
    for test in summary["paired_method_tests"]:
        lines.append("| " + " | ".join([
            test["dataset"], test["metric"], str(test["n_pairs"]),
            f"{test['full_ft_mean']:.3f}", f"{test['lora_mean']:.3f}",
            f"{test['mean_difference']:+.3f}", f"{test['cohens_dz']:+.2f}",
            f"{test['t_p_value']:.2g}", f"{test.get('t_p_holm', float('nan')):.2g}",
            f"{test['wilcoxon_p_value']:.2g}" if test["wilcoxon_p_value"] else "--"]) + " |")

    lines.append("\n## Dose-response (Spearman vs log10 lr)\n")
    lines += ["| dataset | method | metric | n | rho | p |", "|" + "---|" * 6]
    for item in summary["dose_response"]:
        lines.append("| " + " | ".join([
            item["dataset"], METHOD_LABEL.get(item["method"], item["method"]), item["metric"],
            str(item["n_runs"]), f"{item['spearman_rho']:+.3f}", f"{item['p_value']:.2g}"]) + " |")

    for stage in ("final", "epoch1"):
        lines.append(f"\n## Predictors of CIFAR-100 retention ({stage})\n")
        lines += ["| predictor | n | rho overall | p | mean |rho| within dataset | "
                  "mean |rho| within dataset x method |",
                  "|" + "---|" * 6]
        for item in summary[f"predictor_ranking_{stage}"]:
            lines.append("| " + " | ".join([
                item["predictor"], str(item["n"]), f"{item['spearman_overall']:+.3f}",
                f"{item['p_overall']:.2g}",
                f"{item.get('abs_spearman_within_dataset_mean', float('nan')):.3f}",
                f"{item['abs_spearman_within_mean']:.3f}"]) + " |")

    lines.append("\n## Bootstrap 95% CI on |rho| and paired gap vs CKA\n")
    lines += ["| predictor | |rho| | 95% CI | gap vs CKA | gap CI | CKA wins |",
              "|" + "---|" * 6]
    for item in summary.get("bootstrap_predictor_gap", []):
        lines.append("| " + " | ".join([
            item["predictor"], f"{item['abs_rho']:.3f}",
            f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
            f"{item['gap_vs_reference_mean']:+.3f}",
            f"[{item['gap_ci_low']:+.3f}, {item['gap_ci_high']:+.3f}]",
            f"{item['reference_wins_fraction']:.3f}"]) + " |")

    lines.append("\n## Best-validation operating point per method\n")
    for item in summary["iso_accuracy"]:
        lines.append(f"- **{item['dataset']}**: "
                     f"Full FT lr={item['full_ft_lr']:.0e} test={item['full_ft_test_acc']:.2f}% "
                     f"C100={item['full_ft_cifar100_retention']:.1f}% "
                     f"dH={item['full_ft_delta_entropy_pct']:+.2f}% | "
                     f"LoRA lr={item['lora_r8_lr']:.0e} test={item['lora_r8_test_acc']:.2f}% "
                     f"C100={item['lora_r8_cifar100_retention']:.1f}% "
                     f"dH={item['lora_r8_delta_entropy_pct']:+.2f}%")

    lines.append("\n## Signal-based configuration choice\n")
    for item in summary.get("selection_utility", []):
        lines.append(f"- **{item['dataset']}**: {item['n_candidates']} candidates above "
                     f"{item['accuracy_cutoff']:.2f}% test accuracy; retention range "
                     f"{item['worst_retention']:.1f}--{item['oracle_retention']:.1f}%, "
                     f"random pick {item['random_choice_retention']:.1f}%")
        for signal, spec in item["signals"].items():
            lines.append(f"    - {signal}: picks {spec['chosen_run']} -> "
                         f"retention {spec['retention']:.1f}% "
                         f"(regret {spec['regret']:.1f}, "
                         f"range fraction {spec['fraction_of_range']:.2f})")

    lines.append("\n## Accuracy / retention frontier (EuroSAT, all configurations)\n")
    frontier = (df[df.dataset == "eurosat"]
                .groupby(["method", "lr"])
                .agg(test=("test_acc", "mean"), ret=("cifar100_retention", "mean"),
                     wdrift=("weight_drift_rel", "mean"), n=("test_acc", "size"))
                .reset_index().sort_values("test"))
    lines += ["| method | lr | n | test acc | C100 ret | w-drift |", "|" + "---|" * 6]
    for _, row in frontier.iterrows():
        lines.append(f"| {METHOD_LABEL.get(row['method'], row['method'])} | {row['lr']:.0e} | "
                     f"{int(row['n'])} | {row['test']*100:.2f} | {row['ret']:.1f} | "
                     f"{row['wdrift']:.4f} |")

    lines.append("\n### Pareto-optimal configurations (accuracy vs retention)\n")
    for dataset in sorted(df.dataset.unique()):
        cells = (df[df.dataset == dataset].groupby(["method", "lr"])
                 .agg(test=("test_acc", "mean"), ret=("cifar100_retention", "mean"))
                 .reset_index())
        keep = []
        for _, row in cells.sort_values("test", ascending=False).iterrows():
            if all(row["ret"] > other["ret"] for other in keep):
                keep.append(row)
        lines.append(f"- **{dataset}**: " + "; ".join(
            f"{METHOD_LABEL.get(r['method'], r['method'])} @{r['lr']:.0e} "
            f"({r['test']*100:.1f}/{r['ret']:.1f})" for r in keep))
        dominated = [r for _, r in cells.iterrows()
                     if not any(r["method"] == k["method"] and r["lr"] == k["lr"] for k in keep)]
        lines.append(f"  dominated: " + "; ".join(
            f"{METHOD_LABEL.get(r['method'], r['method'])} @{r['lr']:.0e}" for r in dominated))

    lines.append("\n## Selection choice, sensitivity to the accuracy band\n")
    lines += ["| dataset | margin | candidates | oracle | random | CKA | emb. drift | w-drift | |dH| |",
              "|" + "---|" * 9]
    for margin, records in summary.get("selection_utility_margins", {}).items():
        for item in records:
            row = [item["dataset"], margin, str(item["n_candidates"]),
                   f"{item['oracle_retention']:.1f}", f"{item['random_choice_retention']:.1f}"]
            for signal in ("cka_mean", "embedding_drift", "weight_drift_rel",
                            "abs_delta_entropy_pct"):
                spec = item["signals"].get(signal)
                row.append(f"{spec['retention']:.1f}" if spec else "--")
            lines.append("| " + " | ".join(row) + " |")

    for model, spec in (summary.get("backbone_check") or {}).items():
        lines.append(f"\n## Second backbone: `{model}` ({spec['n_runs']} runs)\n")
        lines += ["| dataset | method | lr | n | test acc | dH % | C100 ret % | CKA |",
                  "|" + "---|" * 8]
        for cell in spec["cells"]:
            lines.append("| " + " | ".join([
                cell["dataset"], METHOD_LABEL.get(cell["method"], cell["method"]),
                f"{cell['lr']:.0e}", str(int(cell["n"])),
                f"{cell['test_acc']*100:.2f}", f"{cell['delta_entropy_pct']:+.2f}",
                f"{cell['cifar100_retention']:.1f}", f"{cell['cka_mean']:.3f}"]) + " |")
        lines.append("\nPredictors of retention on this backbone:\n")
        lines += ["| predictor | n | rho | p |", "|" + "---|" * 4]
        for key, value in spec["predictors"].items():
            lines.append(f"| {key} | {value['n']} | {value['spearman_rho']:+.3f} | "
                         f"{value['p_value']:.2g} |")

    lines.append("\n## Polarity by backbone x method (EuroSAT, run level)\n")
    lines += ["| backbone | method | n | rates | signed dH rho | CKA rho | w-drift rho |",
              "|" + "---|" * 7]
    for item in summary.get("polarity_by_group", []):
        lines.append("| " + " | ".join([
            item["model"].split("/")[-1], item["method"], str(item["n"]),
            str(item["n_rates"]),
            f"{item['delta_entropy_pct']['rho']:+.3f}" if "delta_entropy_pct" in item else "--",
            f"{item['cka_mean']['rho']:+.3f}" if "cka_mean" in item else "--",
            f"{item['weight_drift_rel']['rho']:+.3f}" if "weight_drift_rel" in item else "--",
        ]) + " |")

    lines.append("\n## Early warning (epoch 1 signal vs final retention)\n")
    lines.append("```json")
    lines.append(json.dumps(summary["temporal_ordering"], indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    core = df[df.method.isin(CORE_METHODS)]
    colours = {"full_ft": "#c0392b", "lora_r8": "#2471a3"}

    # ---- Figure 1: dose-response, one row, both datasets overlaid ---------
    datasets = [d for d in ("eurosat", "pets") if d in set(core.dataset)]
    styles = {"eurosat": "-", "pets": "--"}
    if datasets:
        metrics = [("test_acc", "Target test accuracy (%)", 100),
                    ("delta_entropy_pct", r"$\Delta$ attention entropy (%)", 1),
                    ("cifar100_retention", "CIFAR-100 retention (%)", 1)]
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.5))
        for ax, (metric, label, scale) in zip(axes, metrics):
            for dataset in datasets:
                for method in CORE_METHODS:
                    cell = core[(core.dataset == dataset) & (core.method == method)]
                    if cell.empty:
                        continue
                    grouped = cell.groupby("lr")[metric].agg(["mean", "std"])
                    x = grouped.index.to_numpy()
                    mean = grouped["mean"].to_numpy() * scale
                    sd = np.nan_to_num(grouped["std"].to_numpy()) * scale
                    ax.plot(x, mean, styles[dataset], color=colours[method],
                            marker="o" if dataset == "eurosat" else "^",
                            markersize=3.5, linewidth=1.5,
                            label=f"{METHOD_LABEL[method]}, {DATASET_LABEL[dataset]}")
                    ax.fill_between(x, mean - sd, mean + sd, color=colours[method], alpha=0.14)
            probe = df[df.method == "linear_probe"]
            if not probe.empty:
                for dataset in datasets:
                    values = probe[probe.dataset == dataset][metric]
                    if not values.empty:
                        ax.axhline(values.mean() * scale, color="#616a6b",
                                   linestyle=":", linewidth=1.0)
            ax.set_xscale("log")
            if metric == "delta_entropy_pct":
                ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
            ax.set_xlabel("learning rate")
            ax.set_title(label, fontsize=9.5)
            ax.grid(alpha=0.25, linewidth=0.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        fig.savefig(FIG_DIR / "study2_dose_response.png", dpi=220)
        plt.close(fig)

    # ---- Figure 2: which cheap signal predicts forgetting? ------------------
    final_rank = predictor_ranking(df, stage="final")
    early_rank = {item["predictor"]: item for item in predictor_ranking(df, stage="epoch1")}
    if final_rank:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax = axes[0]
        order = sorted(final_rank, key=lambda item: item["abs_spearman_overall"])
        y = np.arange(len(order))
        ax.barh(y + 0.19, [item["abs_spearman_overall"] for item in order], height=0.36,
                color="#2471a3", label="end of training")
        ax.barh(y - 0.19, [abs(early_rank.get(item["predictor"], {}).get("spearman_overall", np.nan))
                            for item in order], height=0.36, color="#8ab6d6",
                label="after one epoch")
        ax.set_yticks(y)
        ax.set_yticklabels([item["label"] for item in order], fontsize=8)
        ax.set_xlabel(r"$|\rho|$ with CIFAR-100 retention")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)

        ax = axes[1]
        markers = {"eurosat": "o", "pets": "^"}
        for method in CORE_METHODS:
            for dataset in datasets:
                cell = core[(core.method == method) & (core.dataset == dataset)]
                if cell.empty:
                    continue
                ax.scatter(cell.delta_entropy_pct, cell.cifar100_retention, s=24,
                           color=colours[method], marker=markers[dataset], alpha=0.8,
                           edgecolors="none",
                           label=f"{METHOD_LABEL[method]}, {DATASET_LABEL[dataset]}")
        values = core[["delta_entropy_pct", "cifar100_retention"]].dropna()
        if len(values) > 5:
            rho, _ = stats.spearmanr(values.delta_entropy_pct, values.cifar100_retention)
            ax.set_title(rf"all core runs: $\rho={rho:.2f}$", fontsize=9)
        ax.set_xlabel(r"$\Delta$ attention entropy (%)")
        ax.set_ylabel("CIFAR-100 retention (%)")
        ax.axvline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "study2_predictors.png", dpi=220)
        plt.close(fig)

    # ---- Figure 2b: single-column bar chart for the paper ------------------
    if final_rank:
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        order = sorted(final_rank, key=lambda item: item["abs_spearman_overall"])
        y = np.arange(len(order))
        ax.barh(y + 0.19, [item["abs_spearman_overall"] for item in order], height=0.36,
                color="#2471a3", label="end of training")
        ax.barh(y - 0.19, [abs(early_rank.get(item["predictor"], {}).get("spearman_overall", np.nan))
                            for item in order], height=0.36, color="#8ab6d6",
                label="after one epoch")
        ax.set_yticks(y)
        ax.set_yticklabels([item["label"] for item in order], fontsize=6.2)
        ax.set_xlabel(r"$|\rho|$ with CIFAR-100 retention", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "study2_predictor_bars.png", dpi=300)
        plt.close(fig)

    # ---- Figure 3: per-layer drift ------------------------------------------
    if datasets:
        fig, axes = plt.subplots(len(datasets), 2, figsize=(8.0, 2.5 * len(datasets)),
                                  squeeze=False)
        # clip the scale at the 90th percentile so mid-range structure stays visible
        stacked_values = np.abs(np.concatenate(
            [np.asarray(v) for v in core.delta_entropy_per_layer_pct]))
        limit = max(1e-6, float(np.percentile(stacked_values, 90)))
        for r, dataset in enumerate(datasets):
            for c, method in enumerate(CORE_METHODS):
                ax = axes[r][c]
                cell = core[(core.dataset == dataset) & (core.method == method)]
                lrs = sorted(cell.lr.unique())
                if not lrs:
                    ax.axis("off")
                    continue
                grid = np.array([[np.mean([run[i] for run in
                                            cell[cell.lr == lr].delta_entropy_per_layer_pct])
                                   for i in range(12)] for lr in lrs])
                image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
                ax.set_yticks(range(len(lrs)))
                ax.set_yticklabels([f"{lr:.0e}" for lr in lrs], fontsize=7)
                ax.set_xticks(range(0, 12, 2))
                ax.set_xticklabels(range(1, 13, 2), fontsize=7)
                if r == 0:
                    ax.set_title(METHOD_LABEL[method], fontsize=10)
                if r == len(datasets) - 1:
                    ax.set_xlabel("transformer block")
                if c == 0:
                    ax.set_ylabel(f"{DATASET_LABEL[dataset]}\nlearning rate", fontsize=8.5)
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, extend="both",
                      label=r"$\Delta$ entropy (%)")
        fig.savefig(FIG_DIR / "study2_layer_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # single-column version for the paper: EuroSAT only, two panels
        fig, axes = plt.subplots(1, 2, figsize=(4.0, 1.55))
        for ax, method in zip(axes, CORE_METHODS):
            cell = core[(core.dataset == "eurosat") & (core.method == method)]
            lrs = sorted(cell.lr.unique())
            if not lrs:
                ax.axis("off")
                continue
            grid = np.array([[np.mean([run[i] for run in
                                        cell[cell.lr == lr].delta_entropy_per_layer_pct])
                               for i in range(12)] for lr in lrs])
            image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_yticks(range(len(lrs)))
            if ax is axes[0]:
                ax.set_yticklabels([f"{lr:.0e}" for lr in lrs], fontsize=6)
            else:
                ax.set_yticklabels([])
            ax.set_xticks([0, 5, 11])
            ax.set_xticklabels([1, 6, 12], fontsize=6)
            ax.set_title(METHOD_LABEL[method], fontsize=7.5)
            ax.set_xlabel("block", fontsize=7)
        axes[0].set_ylabel("learning rate", fontsize=7)
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, extend="both")
        bar.ax.tick_params(labelsize=6)
        bar.set_label(r"$\Delta H$ (%)", fontsize=7)
        fig.savefig(FIG_DIR / "study2_layer_heatmap_compact.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- Figure 4: temporal ordering ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for ax, dataset in zip(axes, datasets + [datasets[0]] * (2 - len(datasets))):
        for method in CORE_METHODS:
            for lr in sorted(core.lr.unique()):
                cell = core[(core.dataset == dataset) & (core.method == method) & (core.lr == lr)]
                if cell.empty:
                    continue
                epochs = [entry["epoch"] for entry in cell.history.iloc[0]]
                drift = np.mean([[entry["delta_entropy_pct"] for entry in run]
                                  for run in cell.history], axis=0)
                retention = np.mean([[100 * entry["transfer_track_retention"] for entry in run]
                                      for run in cell.history], axis=0)
                ax.plot(drift, retention, "-o", markersize=3, linewidth=1.0,
                        color=colours[method], alpha=0.35 + 0.65 * (lr / max(core.lr)))
        ax.set_xlabel(r"$\Delta$ attention entropy (%)")
        ax.set_title(DATASET_LABEL[dataset], fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("CIFAR-100 retention (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study2_trajectories.png", dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    df_all = load_runs()
    if df_all.empty:
        print("no runs found")
        return
    df = df_all[df_all.model_name == DEFAULT_MODEL].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    flat_all = df_all.drop(columns=["history", "delta_entropy_per_layer_pct"])
    flat_all.to_csv(OUT_DIR / "run_level_all_backbones.csv", index=False)
    # `run_level.csv` is the primary backbone only: a naive groupby over the mixed
    # file would silently average two architectures together.
    flat_all[flat_all.model_name == DEFAULT_MODEL].to_csv(
        OUT_DIR / "run_level.csv", index=False)

    method_tests = paired_method_tests(df)
    if method_tests:
        adjusted = holm([test["t_p_value"] for test in method_tests])
        for test, value in zip(method_tests, adjusted):
            test["t_p_holm"] = value

    summary = {
        "n_runs": int(len(df)),
        "runs_by_cell": (df.groupby(["dataset", "method", "lr"]).size()
                          .rename("n").reset_index().to_dict("records")),
        "cells": reference_table(df),
        "paired_method_tests": method_tests,
        "dose_response": dose_response(df),
        "predictor_ranking_final": predictor_ranking(df, stage="final"),
        "predictor_ranking_epoch1": predictor_ranking(df, stage="epoch1"),
        "bootstrap_predictor_gap": bootstrap_predictor_gap(df),
        "backbone_check": backbone_check(df_all),
        "polarity_by_group": polarity_by_group(df_all),
        "iso_accuracy": iso_accuracy_comparison(df),
        "selection_utility": selection_utility(df),
        "selection_utility_margins": {
            f"{margin:g}": selection_utility(df, margin=margin)
            for margin in (1.0, 2.0, 3.0, 5.0)
        },
        "temporal_ordering": temporal_ordering(df),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT_DIR / "core_grid_rows.tex").write_text(core_grid_table(df))
    (OUT_DIR / "digest.md").write_text(digest(df, summary))
    (OUT_DIR / "reference_rows.tex").write_text(reference_rows_table(df))
    (OUT_DIR / "predictor_rows.tex").write_text(predictor_rows_table(df))

    if not args.no_figures:
        make_figures(df)

    print(json.dumps({
        "runs": summary["n_runs"],
        "cells": len(summary["cells"]),
        "top_predictors": [(item["predictor"], round(item["spearman_overall"], 3))
                            for item in summary["predictor_ranking_final"][:4]],
        "temporal": summary["temporal_ordering"].get("epoch1_delta_entropy_pct_vs_final"),
    }, indent=2))


if __name__ == "__main__":
    main()
