#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PAPER_DIR.parent
METRICS_DIR = REPO_DIR / "outputs" / "metrics"
SOURCE_FIGURES_DIR = REPO_DIR / "outputs" / "figures"
TARGET_FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

FIGURES_TO_COPY = [
    "fig2_baseline_structure.png",
    "fig5_fullft_vs_lora.png",
    "fig6_lr_sweep.png",
    "fig8_regularization.png",
    "fig10_per_layer_delta_heatmap.png",
    "fig11_statistical_summary.png",
    "fig12_training_dynamics.png",
    "fig15_zeroshot_extended.png",
]

MAIN_EXPERIMENTS = [
    ("E2_full_ft_eurosat", "Full FT", "EuroSAT"),
    ("E3_full_ft_pets", "Full FT", "Oxford-IIIT Pets"),
    ("E4_lora_r4_eurosat", "LoRA r=4", "EuroSAT"),
    ("E5_lora_r8_eurosat", "LoRA r=8", "EuroSAT"),
    ("E6_lora_r16_eurosat", "LoRA r=16", "EuroSAT"),
    ("E7_lora_r8_pets", "LoRA r=8", "Oxford-IIIT Pets"),
]

LR_SWEEP = [
    ("A1_lr_1e-06", "1e-6"),
    ("A1_lr_5e-06", "5e-6"),
    ("E2_full_ft_eurosat", "1e-5"),
    ("A1_lr_5e-05", "5e-5"),
    ("A1_lr_0.0001", "1e-4"),
]

REGULARIZATION = [
    ("E2_full_ft_eurosat", "No regularizer"),
    ("reg_apr_lambda0.01_eurosat", "APR $\\lambda=0.01$"),
    ("reg_apr_lambda0.1_eurosat", "APR $\\lambda=0.1$"),
    ("reg_apr_lambda1.0_eurosat", "APR $\\lambda=1.0$"),
    ("reg_entropy_floor_lambda0.01_eurosat", "Entropy floor $\\lambda=0.01$"),
    ("reg_entropy_floor_lambda0.1_eurosat", "Entropy floor $\\lambda=0.1$"),
    ("R3_wd_0.0", "Weight decay 0.0"),
    ("R3_wd_0.1", "Weight decay 0.1"),
]

ZS_FILE_MAP = {
    "baseline": "extended_zs_baseline.json",
    "E2_full_ft_eurosat": "extended_zs_E2_full_ft_eurosat.json",
    "E3_full_ft_pets": "extended_zs_E3_full_ft_pets.json",
    "E4_lora_r4_eurosat": "extended_zs_E4_lora_r4.json",
    "E5_lora_r8_eurosat": "extended_zs_E5_lora_r8.json",
    "E6_lora_r16_eurosat": "extended_zs_E6_lora_r16.json",
    "A1_lr_5e-05": "extended_zs_A1_lr_5e-5.json",
    "A1_lr_0.0001": "extended_zs_A1_lr_1e-4.json",
    "reg_apr_lambda0.01_eurosat": "extended_zs_APR_0.01.json",
}

LATEX_ROW_END = " \\\\"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(name: str) -> dict[str, Any]:
    return load_json(METRICS_DIR / f"{name}_history.json")


def fmt_pct(value: float | None, decimals: int = 2, scale: float = 100.0) -> str:
    if value is None:
        return "---"
    return f"{value * scale:.{decimals}f}"


def fmt_signed(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "---"
    return f"{value:+.{decimals}f}"


def fmt_num(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "---"
    return f"{value:.{decimals}f}"


def latex_escape(text: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def metric_change(history: dict[str, Any], key: str) -> float:
    baseline = history["baseline_metrics"][key]
    final = history["final_metrics"][key]
    return (final - baseline) / baseline * 100.0


def read_zero_shot(key: str) -> dict[str, Any] | None:
    filename = ZS_FILE_MAP.get(key)
    if not filename:
        return None
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    return load_json(path)


def read_adapter_aware_zero_shot(key: str) -> dict[str, Any] | None:
    followup = read_followup_analysis()
    adapter = followup.get("adapter_aware_zero_shot", {})
    entry = adapter.get(key)
    if not entry:
        return None
    return {
        "cifar100": entry.get("cifar100", {}).get("accuracy"),
        "flowers102": entry.get("flowers102", {}).get("accuracy"),
    }


def write_table(filename: str, content: str) -> None:
    (TABLES_DIR / filename).write_text(content, encoding="utf-8")


def read_followup_analysis() -> dict[str, Any]:
    path = METRICS_DIR / "followup_analysis.json"
    return load_json(path) if path.exists() else {}


def build_main_results_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    body = []
    for exp_id, method, dataset in MAIN_EXPERIMENTS:
        hist = load_history(exp_id)
        zs = read_adapter_aware_zero_shot(exp_id) or read_zero_shot(exp_id) or {}
        row = {
            "experiment": exp_id,
            "method": method,
            "dataset": dataset,
            "best_val_acc": hist["best_val_acc"],
            "delta_entropy": metric_change(hist, "entropy_mean"),
            "delta_erf": metric_change(hist, "erf95_mean"),
            "delta_gini": metric_change(hist, "gini_mean"),
            "delta_head_div": metric_change(hist, "head_diversity_mean"),
            "zs_cifar100": zs.get("cifar100"),
            "zs_flowers102": zs.get("flowers102"),
        }
        rows.append(row)
        body.append(
            f"{method} & {dataset} & {fmt_pct(row['best_val_acc'])} & {fmt_signed(row['delta_entropy'])} & {fmt_signed(row['delta_erf'])} & {fmt_signed(row['delta_gini'])} & {fmt_signed(row['delta_head_div'])} & {fmt_pct(row['zs_cifar100'])} & {fmt_pct(row['zs_flowers102'])} \\\\"
        )

    table = """\\begin{table*}[t]
\\centering
\\caption{Main experiment results. Positive $\\Delta$Entropy and $\\Delta$ERF indicate broader attention support relative to the corresponding pretrained baseline; positive $\\Delta$Gini indicates more concentrated attention.}
\\label{tab:main-results}
\\small
\\begin{tabular}{llrrrrrrr}
\\toprule
Method & Dataset & Task Acc. (\\%) & $\\Delta$Entropy (\\%) & $\\Delta$ERF (\\%) & $\\Delta$Gini (\\%) & $\\Delta$HeadDiv (\\%) & CIFAR-100 ZS (\\%) & Flowers102 ZS (\\%) \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    write_table("main_results.tex", table)
    return rows


def build_lr_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    body = []
    for exp_id, lr in LR_SWEEP:
        hist = load_history(exp_id)
        zs = read_zero_shot(exp_id) or {}
        row = {
            "lr": lr,
            "best_val_acc": hist["best_val_acc"],
            "delta_entropy": metric_change(hist, "entropy_mean"),
            "delta_erf": metric_change(hist, "erf95_mean"),
            "delta_gini": metric_change(hist, "gini_mean"),
            "zs_cifar100": zs.get("cifar100"),
            "zs_flowers102": zs.get("flowers102"),
        }
        rows.append(row)
        body.append(
            f"{lr} & {fmt_pct(row['best_val_acc'])} & {fmt_signed(row['delta_entropy'])} & {fmt_signed(row['delta_erf'])} & {fmt_signed(row['delta_gini'])} & {fmt_pct(row['zs_cifar100'])} & {fmt_pct(row['zs_flowers102'])} \\\\"
        )

    table = """\\begin{table}[t]
\\centering
\\caption{Learning-rate sweep for EuroSAT full fine-tuning. Entropy decreases monotonically as the learning rate increases across the tested range.}
\\label{tab:lr-sweep}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
LR & Task Acc. (\\%) & $\\Delta$Entropy (\\%) & $\\Delta$ERF (\\%) & $\\Delta$Gini (\\%) & CIFAR-100 ZS (\\%) & Flowers102 ZS (\\%) \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("lr_sweep.tex", table)
    return rows


def build_regularization_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    body = []
    for exp_id, label in REGULARIZATION:
        hist = load_history(exp_id)
        row = {
            "label": label,
            "best_val_acc": hist["best_val_acc"],
            "delta_entropy": metric_change(hist, "entropy_mean"),
            "delta_erf": metric_change(hist, "erf95_mean"),
            "delta_gini": metric_change(hist, "gini_mean"),
        }
        rows.append(row)
        body.append(
            f"{label} & {fmt_pct(row['best_val_acc'])} & {fmt_signed(row['delta_entropy'])} & {fmt_signed(row['delta_erf'])} & {fmt_signed(row['delta_gini'])} \\\\"
        )

    table = """\\begin{table}[t]
\\centering
\\caption{Regularization study on EuroSAT full fine-tuning. Among individually tested regularizers, entropy floor with $\\lambda=0.1$ is the only setting with a paired per-layer entropy difference reaching $p<0.05$ in the current analysis.}
\\label{tab:regularization}
\\small
\\begin{tabular}{lrrrr}
\\toprule
Setting & Task Acc. (\\%) & $\\Delta$Entropy (\\%) & $\\Delta$ERF (\\%) & $\\Delta$Gini (\\%) \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("regularization.tex", table)
    return rows


def build_stat_tests_table() -> dict[str, Any]:
    stats_data = read_followup_analysis()
    corr = stats_data["corrected_lr_correlation"]
    reg = stats_data["corrected_regularization"]

    body = [
        f"LR vs. $\\Delta$Entropy & Exact permutation Pearson $r$ & {fmt_num(corr['pearson_exact']['statistic'])} & {corr['pearson_exact']['p_value']:.4f} & $n=5$ learning rates \\\\ ",
        f"LR vs. $\\Delta$Entropy & Exact permutation Spearman $\\rho$ & {fmt_num(corr['spearman_exact']['statistic'])} & {corr['spearman_exact']['p_value']:.4f} & Small-$n$ exact test \\\\ ",
        f"APR $\\lambda=0.01$ vs. baseline & Paired per-layer t-test & {fmt_num(reg['APR_0.01']['paired_t'])} & {reg['APR_0.01']['holm_bonferroni_p']:.4f} & Holm--Bonferroni adjusted \\\\ ",
        f"Entropy floor $\\lambda=0.1$ vs. baseline & Paired per-layer t-test & {fmt_num(reg['EntFloor_0.1']['paired_t'])} & {reg['EntFloor_0.1']['holm_bonferroni_p']:.4f} & No regularizer survives Holm correction \\\\ ",
    ]

    table = """\\begin{table}[t]
\\centering
\\caption{Corrected inferential statistics derived from \\texttt{outputs/metrics/followup\_analysis.json}. Exact permutation tests are used for the five-point learning-rate analysis, and Holm--Bonferroni correction is applied across the regularizer family.}
\\label{tab:stats}
\\small
\\begin{tabular}{llrrl}
\\toprule
Comparison & Test & Statistic & $p$-value & Note \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("stat_tests.tex", table)
    return stats_data


def build_zero_shot_table() -> list[dict[str, Any]]:
    followup = read_followup_analysis()
    adapter = followup.get("adapter_aware_zero_shot", {})
    rows = []
    body = []
    baseline = read_zero_shot("baseline") or {}
    keys = [
        ("baseline", "Pretrained baseline", baseline.get("cifar100"), baseline.get("flowers102")),
        ("E2_full_ft_eurosat", "Full FT EuroSAT", adapter.get("E2_full_ft_eurosat", {}).get("cifar100", {}).get("accuracy"), adapter.get("E2_full_ft_eurosat", {}).get("flowers102", {}).get("accuracy")),
        ("E4_lora_r4_eurosat", "LoRA r=4 EuroSAT", adapter.get("E4_lora_r4_eurosat", {}).get("cifar100", {}).get("accuracy"), adapter.get("E4_lora_r4_eurosat", {}).get("flowers102", {}).get("accuracy")),
        ("E5_lora_r8_eurosat", "LoRA r=8 EuroSAT", adapter.get("E5_lora_r8_eurosat", {}).get("cifar100", {}).get("accuracy"), adapter.get("E5_lora_r8_eurosat", {}).get("flowers102", {}).get("accuracy")),
        ("E6_lora_r16_eurosat", "LoRA r=16 EuroSAT", adapter.get("E6_lora_r16_eurosat", {}).get("cifar100", {}).get("accuracy"), adapter.get("E6_lora_r16_eurosat", {}).get("flowers102", {}).get("accuracy")),
    ]
    for key, label, cifar100, flowers102 in keys:
        row = {
            "label": label,
            "cifar100": cifar100,
            "flowers102": flowers102,
        }
        rows.append(row)
        body.append(f"{label} & {fmt_pct(row['cifar100'])} & {fmt_pct(row['flowers102'])} \\\\")

    table = """\\begin{table}[t]
\\centering
\\caption{Zero-shot results using an adapter-aware image-encoding path for LoRA checkpoints. The pretrained baseline is retained from the original CLIP evaluation path, while adapted checkpoints are re-evaluated with their own image branch active.}
\\label{tab:zeroshot}
\\small
\\begin{tabular}{lrr}
\\toprule
Model & CIFAR-100 (\\%) & Flowers102 (\\%) \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("zero_shot.tex", table)
    return rows


def build_auxiliary_validation_table() -> dict[str, Any]:
    followup = read_followup_analysis()
    rollout = followup.get("rollout_patch_metrics", {})
    cka = followup.get("layerwise_cka", {})
    sensitivity = followup.get("subset_sensitivity", {})

    rows = []
    for exp_name, label in [
        ("E2_full_ft_eurosat", "Full FT EuroSAT"),
        ("E5_lora_r8_eurosat", "LoRA r=8 EuroSAT"),
    ]:
        rows.append(
            f"{label} & {fmt_num(rollout[exp_name]['rollout_entropy_bits_mean'])} & {fmt_num(rollout[exp_name]['rollout_erf95_mean'])} & {fmt_num(rollout[exp_name]['rollout_gini_mean'])} & {fmt_num(rollout[exp_name]['patch_to_patch_entropy_bits_mean'])} & {fmt_num(cka[exp_name]['mean_cka'])} & {fmt_num(sensitivity[exp_name]['entropy_bits_mean']['cv'], 4)} \\\\" 
        )

    table = """\\begin{table*}[t]
\\centering
\\caption{Auxiliary validation analyses beyond CLS-to-patch entropy. Rollout and patch-to-patch metrics indicate broader attention support for LoRA than for Full FT on EuroSAT, while layerwise CKA shows higher representational similarity to the pretrained encoder. The subset-sensitivity coefficient of variation (CV) is low for both methods, indicating that the reported metrics are stable across five balanced evaluation subsets.}
\\label{tab:auxiliary-validation}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
Model & Rollout Entropy (bits) & Rollout ERF@0.95 & Rollout Gini & Patch-to-Patch Entropy (bits) & Mean Layerwise CKA & Entropy CV \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    write_table("auxiliary_validation.tex", table)
    return followup


def build_multiseed_table() -> dict[str, Any]:
    followup = read_followup_analysis()
    multiseed = followup.get("multiseed_followup", {})
    comparisons = multiseed.get("comparisons", {})
    selected = [
        ("fullft_vs_lora_eurosat", "Full FT vs. LoRA r=8 (EuroSAT)"),
        ("fullft_vs_lora_pets", "Full FT vs. LoRA r=8 (Pets)"),
        ("fullft_vs_apr_eurosat", "Full FT vs. APR $\\lambda=0.1$ (EuroSAT)"),
        ("fullft_vs_entropy_floor_eurosat", "Full FT vs. Entropy floor $\\lambda=0.1$ (EuroSAT)"),
    ]
    body = []
    for key, label in selected:
        entry = comparisons.get(key, {})
        safe_label = latex_escape(label)
        if entry.get("status") != "complete":
            body.append(f"{safe_label} & pending & --- & --- & --- & ---" + LATEX_ROW_END)
            continue
        entropy = entry["metrics"]["entropy_mean"]
        body.append(
            f"{safe_label} & {entry['n_group_a']} vs {entry['n_group_b']} & "
            f"{fmt_signed(entropy['group_a_mean'])} vs {fmt_signed(entropy['group_b_mean'])} & "
            f"{fmt_num(entropy['welch_t'])} & "
            f"{fmt_num(entropy['welch_p'], 4)} & "
            f"{fmt_num(entropy['cohens_d'])}" + LATEX_ROW_END)

    table = """\\begin{table*}[t]
\\centering
\\caption{Run-level multi-seed comparisons on mean $\\Delta$entropy. Values are reported as percentage change relative to the corresponding pretrained baseline, aggregated across completed seed runs.}
\\label{tab:multiseed}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Comparison & Seeds & Mean $\\Delta$Entropy (\\%) & Welch $t$ & Welch $p$ & Cohen's $d$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    write_table("multiseed_results.tex", table)
    return multiseed


def build_backbone_table() -> dict[str, Any]:
    followup = read_followup_analysis()
    backbone = followup.get("backbone_comparison", {})
    selected = [
        ("full_ft_eurosat", "Full FT on EuroSAT"),
        ("lora_r8_eurosat", "LoRA r=8 on EuroSAT"),
    ]
    body = []
    for key, label in selected:
        entry = backbone.get(key, {})
        safe_label = latex_escape(label)
        if entry.get("status") != "complete":
            body.append(f"{safe_label} & pending & --- & --- & --- & ---" + LATEX_ROW_END)
            continue
        metrics = entry["metrics"]
        body.append(
            f"{safe_label} & "
            f"{fmt_signed(metrics['entropy_mean']['patch32_delta'])} / {fmt_signed(metrics['entropy_mean']['patch16_delta'])} & "
            f"{fmt_signed(metrics['erf95_mean']['patch32_delta'])} / {fmt_signed(metrics['erf95_mean']['patch16_delta'])} & "
            f"{fmt_signed(metrics['gini_mean']['patch32_delta'])} / {fmt_signed(metrics['gini_mean']['patch16_delta'])} & "
            f"{fmt_pct(metrics['best_val_acc']['patch32'])} / {fmt_pct(metrics['best_val_acc']['patch16'])} & "
            f"{fmt_signed(metrics['entropy_mean']['delta_gap'])}" + LATEX_ROW_END)

    table = """\\begin{table}[t]
\\centering
\\caption{Backbone comparison between CLIP ViT-B/32 and ViT-B/16 on EuroSAT. Metric cells show B/32 / B/16 values.}
\\label{tab:backbone-comparison}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Method & $\\Delta$Entropy (\\%) & $\\Delta$ERF (\\%) & $\\Delta$Gini (\\%) & Task Acc. (\\%) & Entropy Gap \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("backbone_comparison.tex", table)
    return backbone


def build_cka_entropy_table() -> dict[str, Any]:
    followup = read_followup_analysis()
    corr = followup.get("cka_entropy_correlation", {})
    run_level = corr.get("run_level", {})
    per_layer = corr.get("per_layer", {})
    complete_layers = [v for v in per_layer.values() if v.get("status") == "complete"]
    strongest_layer = None
    if complete_layers:
        strongest_layer = max(complete_layers, key=lambda item: abs(item["pearson"]["statistic"]))

    rows = []
    if run_level.get("status") == "complete":
        rows.append(
            f"Run-level mean CKA vs. mean $\\Delta$Entropy & {run_level['n']} & "
            f"{fmt_num(run_level['pearson']['statistic'])} & {run_level['pearson']['p_value']:.4f} & "
            f"{fmt_num(run_level['spearman']['statistic'])} & {run_level['spearman']['p_value']:.4f}" + LATEX_ROW_END)
    else:
        rows.append("Run-level mean CKA vs. mean $\\Delta$Entropy & pending & --- & --- & --- & ---" + LATEX_ROW_END)

    if strongest_layer is not None:
        layer_name = next(k for k, v in per_layer.items() if v is strongest_layer)
        rows.append(
            f"Strongest per-layer association ({latex_escape(layer_name.replace('_', ' '))}) & {strongest_layer['n']} & "
            f"{fmt_num(strongest_layer['pearson']['statistic'])} & {strongest_layer['pearson']['p_value']:.4f} & "
            f"{fmt_num(strongest_layer['spearman']['statistic'])} & {strongest_layer['spearman']['p_value']:.4f}" + LATEX_ROW_END)
    else:
        rows.append("Strongest per-layer association & pending & --- & --- & --- & ---" + LATEX_ROW_END)

    table = """\\begin{table}[t]
\\centering
\\caption{Correlation between representational fidelity (CKA to the pretrained model) and structural change ($\\Delta$entropy).}
\\label{tab:cka-entropy}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Analysis & $n$ & Pearson $r$ & Pearson $p$ & Spearman $\\rho$ & Spearman $p$ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    write_table("cka_entropy.tex", table)
    return corr


def copy_figures() -> None:
    for name in FIGURES_TO_COPY:
        src = SOURCE_FIGURES_DIR / name
        dst = TARGET_FIGURES_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing figure: {src}")
        shutil.copy2(src, dst)


def main() -> None:
    TARGET_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    copy_figures()
    main_rows = build_main_results_table()
    lr_rows = build_lr_table()
    reg_rows = build_regularization_table()
    stats_data = build_stat_tests_table()
    zs_rows = build_zero_shot_table()
    followup_data = build_auxiliary_validation_table()
    multiseed_data = build_multiseed_table()
    backbone_data = build_backbone_table()
    cka_entropy_data = build_cka_entropy_table()

    summary = {
        "source_repo": str(REPO_DIR),
        "metrics_dir": str(METRICS_DIR),
        "figures_dir": str(SOURCE_FIGURES_DIR),
        "copied_figures": FIGURES_TO_COPY,
        "main_results": main_rows,
        "lr_sweep": lr_rows,
        "regularization": reg_rows,
        "zero_shot": zs_rows,
        "statistical_tests": stats_data,
        "followup_analysis": followup_data,
        "multiseed_followup": multiseed_data,
        "backbone_comparison": backbone_data,
        "cka_entropy_correlation": cka_entropy_data,
        "zero_shot_caveat": (
            "Adapter-aware follow-up evaluation is now available in outputs/metrics/followup_analysis.json; "
            "the paper uses those values for LoRA zero-shot results."
        ),
    }
    (PAPER_DIR / "paper_assets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote tables to {TABLES_DIR}")
    print(f"Copied figures to {TARGET_FIGURES_DIR}")
    print(f"Wrote summary JSON to {PAPER_DIR / 'paper_assets.json'}")


if __name__ == "__main__":
    main()
