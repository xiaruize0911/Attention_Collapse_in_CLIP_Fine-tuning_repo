from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import ArtifactPaths, save_json
from .config import RevisionConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_run_rows(output_root: Path) -> list[dict[str, Any]]:
    artifacts = ArtifactPaths.from_root(output_root)
    json_path = artifacts.metrics / "run_summaries.json"
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def _to_float(values: list[Any]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    arr = _to_float(values)
    mean = float(arr.mean()) if len(arr) else float("nan")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci_half = 1.96 * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci_low": mean - ci_half, "ci_high": mean + ci_half}


def cohens_d(a: list[float], b: list[float]) -> float:
    arr_a = _to_float(a)
    arr_b = _to_float(b)
    if len(arr_a) < 2 or len(arr_b) < 2:
        return float("nan")
    var_a = arr_a.var(ddof=1)
    var_b = arr_b.var(ddof=1)
    pooled = (((len(arr_a) - 1) * var_a) + ((len(arr_b) - 1) * var_b)) / (len(arr_a) + len(arr_b) - 2)
    if pooled <= 0:
        return float("nan")
    return float((arr_a.mean() - arr_b.mean()) / math.sqrt(pooled))


def simple_t_stat(a: list[float], b: list[float]) -> float:
    arr_a = _to_float(a)
    arr_b = _to_float(b)
    if len(arr_a) < 2 or len(arr_b) < 2:
        return float("nan")
    var_term = arr_a.var(ddof=1) / len(arr_a) + arr_b.var(ddof=1) / len(arr_b)
    if var_term <= 0:
        return float("nan")
    return float((arr_a.mean() - arr_b.mean()) / math.sqrt(var_term))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def pearsonr(x: list[float], y: list[float]) -> float:
    arr_x = _to_float(x)
    arr_y = _to_float(y)
    if len(arr_x) < 2:
        return float("nan")
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def spearmanr(x: list[float], y: list[float]) -> float:
    arr_x = rankdata(_to_float(x))
    arr_y = rankdata(_to_float(y))
    return pearsonr(arr_x.tolist(), arr_y.tolist())


def one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    keys = sorted(set(values))
    mapping = {key: idx for idx, key in enumerate(keys)}
    matrix = np.zeros((len(values), len(keys) - 1), dtype=float)
    for row_idx, value in enumerate(values):
        col_idx = mapping[value]
        if col_idx > 0:
            matrix[row_idx, col_idx - 1] = 1.0
    return matrix, keys


def fit_retention_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "pending"}

    continuous_names = ["mean_entropy_shift", "mean_erf95_shift", "mean_layerwise_cka"]
    continuous = np.column_stack([_to_float([row[name] for row in rows]) for name in continuous_names])
    cont_means = continuous.mean(axis=0, keepdims=True)
    cont_stds = continuous.std(axis=0, ddof=1, keepdims=True)
    cont_stds[cont_stds == 0] = 1.0
    continuous = (continuous - cont_means) / cont_stds

    method_matrix, method_levels = one_hot([str(row["method"]) for row in rows])
    dataset_matrix, dataset_levels = one_hot([str(row["source_dataset"]) for row in rows])
    backbone_matrix, backbone_levels = one_hot([str(row["backbone"]) for row in rows])
    intercept = np.ones((len(rows), 1), dtype=float)
    x = np.concatenate([intercept, continuous, method_matrix, dataset_matrix, backbone_matrix], axis=1)
    y = _to_float([row["transfer_retention_score"] for row in rows])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    names = ["intercept", *continuous_names]
    names.extend([f"method[{level}]" for level in method_levels[1:]])
    names.extend([f"dataset[{level}]" for level in dataset_levels[1:]])
    names.extend([f"backbone[{level}]" for level in backbone_levels[1:]])
    coefficients = []
    for idx, name in enumerate(names):
        coefficients.append(
            {
                "name": name,
                "coef": float(beta[idx]),
                "ci_low": float(ci_low[idx]),
                "ci_high": float(ci_high[idx]),
            }
        )
    return {"status": "complete", "coefficients": coefficients, "n": len(rows)}


def group_rows(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def compare_groups(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    a = [float(row[metric]) for row in rows_a]
    b = [float(row[metric]) for row in rows_b]
    return {
        "metric": metric,
        "group_a": mean_sd_ci(a),
        "group_b": mean_sd_ci(b),
        "difference_in_means": float(np.mean(a) - np.mean(b)) if a and b else float("nan"),
        "t_stat": simple_t_stat(a, b),
        "cohens_d": cohens_d(a, b),
    }


def build_analysis(output_root: Path) -> dict[str, Any]:
    rows = load_run_rows(output_root)
    main_rows = group_rows(
        rows,
        lambda row: row["group"] == "main"
        and row["backbone"] == "openai/clip-vit-base-patch32"
        and row["method"] in {"full_ft", "lora"},
    )
    ft_rows = [row for row in main_rows if row["method"] == "full_ft"]
    lora_rows = [row for row in main_rows if row["method"] == "lora"]

    comparisons = {
        metric: compare_groups(ft_rows, lora_rows, metric)
        for metric in ["mean_entropy_shift", "mean_erf95_shift", "mean_layerwise_cka", "transfer_retention_score"]
    }

    lr_rows = sorted(
        group_rows(
            rows,
            lambda row: row["group"] == "lr_sweep"
            and row["source_dataset"] == "eurosat"
            and row["backbone"] == "openai/clip-vit-base-patch32",
        ),
        key=lambda row: float(row["lr"]),
    )
    lr_x = [math.log10(float(row["lr"])) for row in lr_rows]
    lr_entropy = [float(row["mean_entropy_shift"]) for row in lr_rows]
    lr_erf = [float(row["mean_erf95_shift"]) for row in lr_rows]
    lr_retention = [float(row["transfer_retention_score"]) for row in lr_rows]
    slope = intercept = float("nan")
    if lr_rows:
        slope, intercept = np.polyfit(lr_x, lr_entropy, 1)
    retention_model = fit_retention_model(
        [row for row in rows if row["method"] in {"full_ft", "lora"} and row["group"] in {"main", "backbone_confirmation", "appendix", "lr_sweep"}]
    )

    analysis = {
        "run_count": len(rows),
        "ft_vs_lora_b32": comparisons,
        "lr_sweep_eurosat_b32": {
            "n": len(lr_rows),
            "entropy_regression_slope": float(slope),
            "entropy_regression_intercept": float(intercept),
            "pearson_entropy_vs_log10_lr": pearsonr(lr_x, lr_entropy) if lr_rows else float("nan"),
            "spearman_entropy_vs_log10_lr": spearmanr(lr_x, lr_entropy) if lr_rows else float("nan"),
            "pearson_erf_vs_log10_lr": pearsonr(lr_x, lr_erf) if lr_rows else float("nan"),
            "pearson_retention_vs_log10_lr": pearsonr(lr_x, lr_retention) if lr_rows else float("nan"),
        },
        "retention_model": retention_model,
    }
    artifacts = ArtifactPaths.from_root(output_root)
    save_json(artifacts.metrics / "analysis_summary.json", analysis)
    write_analysis_csv(artifacts.metrics / "analysis_summary.csv", analysis)
    create_placeholder_figures(artifacts.figures, analysis)
    return analysis


def write_analysis_csv(path: Path, analysis: dict[str, Any]) -> None:
    rows = []
    for metric, payload in analysis.get("ft_vs_lora_b32", {}).items():
        rows.append(
            {
                "section": "ft_vs_lora_b32",
                "metric": metric,
                "group_a_mean": payload["group_a"]["mean"],
                "group_b_mean": payload["group_b"]["mean"],
                "difference_in_means": payload["difference_in_means"],
                "t_stat": payload["t_stat"],
                "cohens_d": payload["cohens_d"],
            }
        )
    if analysis.get("lr_sweep_eurosat_b32"):
        payload = analysis["lr_sweep_eurosat_b32"]
        rows.append(
            {
                "section": "lr_sweep_eurosat_b32",
                "metric": "mean_entropy_shift",
                "group_a_mean": payload["entropy_regression_slope"],
                "group_b_mean": payload["pearson_entropy_vs_log10_lr"],
                "difference_in_means": payload["spearman_entropy_vs_log10_lr"],
                "t_stat": payload["pearson_erf_vs_log10_lr"],
                "cohens_d": payload["pearson_retention_vs_log10_lr"],
            }
        )
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_placeholder_figures(figures_dir: Path, analysis: dict[str, Any]) -> None:
    figure_specs = {
        "fig_overview_revision.png": [
            "Overview figure",
            f"Runs indexed: {analysis.get('run_count', 0)}",
            "FT vs LoRA across EuroSAT / Pets / Cars",
        ],
        "fig_lr_sweep_revision.png": [
            "Learning-rate figure",
            f"Slope: {analysis.get('lr_sweep_eurosat_b32', {}).get('entropy_regression_slope', float('nan')):.4f}",
            "EuroSAT B/32 Full FT",
        ],
        "fig_backbone_revision.png": [
            "Backbone confirmation",
            "Compare B/32 vs B/16 FT-LoRA gaps",
            "Generated from revision_v2 outputs",
        ],
        "fig_prediction_revision.png": [
            "Prediction figure",
            "Structural preservation vs retention",
            "Generated from revision_v2 outputs",
        ],
        "fig_appendix_attention_revision.png": [
            "Appendix attention profile",
            "EuroSAT clearest comparison",
            "Generated from revision_v2 outputs",
        ],
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in figure_specs.items():
        image = Image.new("RGB", (1200, 700), color=(250, 247, 240))
        draw = ImageDraw.Draw(image)
        y = 80
        for line in lines:
            draw.text((80, y), line, fill=(30, 30, 30))
            y += 80
        image.save(figures_dir / name)


def build_revision_tables(output_root: Path) -> dict[str, str]:
    artifacts = ArtifactPaths.from_root(output_root)
    rows = load_run_rows(output_root)
    analysis = json.loads((artifacts.metrics / "analysis_summary.json").read_text(encoding="utf-8")) if (artifacts.metrics / "analysis_summary.json").exists() else {}

    def fmt_pct(value: float | None) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "---"
        return f"{100.0 * float(value):.2f}"

    def fmt_num(value: float | None) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "---"
        return f"{float(value):.3f}"

    def write_table(name: str, content: str) -> str:
        path = artifacts.tables / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    main_rows = [
        row for row in rows
        if row["group"] == "main"
        and row["backbone"] == "openai/clip-vit-base-patch32"
        and row["method"] in {"full_ft", "lora"}
    ]
    main_body = []
    for row in sorted(main_rows, key=lambda item: (item["source_dataset"], item["method"], int(item["seed"]))):
        main_body.append(
            f"{row['method']} & {row['source_dataset']} & {row['seed']} & "
            f"{fmt_pct(row['in_domain_test_accuracy'])} & {fmt_num(row['mean_entropy_shift'])} & "
            f"{fmt_num(row['mean_erf95_shift'])} & {fmt_num(row['mean_layerwise_cka'])} & "
            f"{fmt_pct(row['transfer_retention_score'])} \\\\"
        )
    write_table(
        "main_results.tex",
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Primary B/32 results at the run level.}\n"
        "\\label{tab:main-results}\n\\small\n"
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        "Method & Dataset & Seed & In-domain Acc. (\\%) & $\\Delta$Entropy & $\\Delta$ERF@0.95 & Mean CKA & Retention (\\%) \\\\\n\\midrule\n"
        + "\n".join(main_body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
    )

    lr_rows = [
        row for row in rows
        if row["group"] == "lr_sweep"
        and row["source_dataset"] == "eurosat"
        and row["backbone"] == "openai/clip-vit-base-patch32"
    ]
    lr_body = []
    for row in sorted(lr_rows, key=lambda item: (float(item["lr"]), int(item["seed"]))):
        lr_body.append(
            f"{row['lr']:.0e} & {row['seed']} & {fmt_pct(row['in_domain_test_accuracy'])} & "
            f"{fmt_num(row['mean_entropy_shift'])} & {fmt_num(row['mean_erf95_shift'])} & {fmt_pct(row['transfer_retention_score'])} \\\\"
        )
    write_table(
        "lr_sweep.tex",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{EuroSAT B/32 full fine-tuning learning-rate sweep.}\n"
        "\\label{tab:lr-sweep}\n\\small\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "LR & Seed & In-domain Acc. (\\%) & $\\Delta$Entropy & $\\Delta$ERF@0.95 & Retention (\\%) \\\\\n\\midrule\n"
        + "\n".join(lr_body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )

    coeff_rows = []
    for coeff in analysis.get("retention_model", {}).get("coefficients", []):
        coeff_rows.append(
            f"{coeff['name']} & {fmt_num(coeff['coef'])} & {fmt_num(coeff['ci_low'])} & {fmt_num(coeff['ci_high'])} \\\\"
        )
    write_table(
        "prediction_results.tex",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Retention model coefficients with approximate 95\\% confidence intervals.}\n"
        "\\label{tab:prediction-results}\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Predictor & Coef. & CI Low & CI High \\\\\n\\midrule\n"
        + "\n".join(coeff_rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )

    stats_rows = []
    for metric, payload in analysis.get("ft_vs_lora_b32", {}).items():
        stats_rows.append(
            f"{metric} & {fmt_num(payload['group_a']['mean'])} & {fmt_num(payload['group_b']['mean'])} & "
            f"{fmt_num(payload['difference_in_means'])} & {fmt_num(payload['t_stat'])} & {fmt_num(payload['cohens_d'])} \\\\"
        )
    write_table(
        "stat_tests.tex",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Primary B/32 FT vs LoRA comparisons on run-level metrics.}\n"
        "\\label{tab:stats}\n\\small\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Metric & Full FT Mean & LoRA Mean & Diff. & t-stat & Cohen's d \\\\\n\\midrule\n"
        + "\n".join(stats_rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )

    appendix_rows = [
        row for row in rows
        if row["group"] == "appendix"
        and row["source_dataset"] == "eurosat"
    ]
    appendix_body = []
    for row in sorted(appendix_rows, key=lambda item: int(item["seed"])):
        appendix_body.append(
            f"{row['seed']} & {fmt_pct(row['in_domain_test_accuracy'])} & {fmt_num(row['mean_entropy_shift'])} & {fmt_pct(row['transfer_retention_score'])} \\\\"
        )
    write_table(
        "appendix_regularization.tex",
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Appendix-only entropy-floor experiment on EuroSAT B/32 full fine-tuning.}\n"
        "\\label{tab:appendix-regularization}\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Seed & In-domain Acc. (\\%) & $\\Delta$Entropy & Retention (\\%) \\\\\n\\midrule\n"
        + "\n".join(appendix_body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
    )

    return {
        "main_results": str(artifacts.tables / "main_results.tex"),
        "lr_sweep": str(artifacts.tables / "lr_sweep.tex"),
        "prediction_results": str(artifacts.tables / "prediction_results.tex"),
        "stat_tests": str(artifacts.tables / "stat_tests.tex"),
        "appendix_regularization": str(artifacts.tables / "appendix_regularization.tex"),
    }


def analyze(config: RevisionConfig) -> dict[str, Any]:
    output_root = config.resolved_output_root(PROJECT_ROOT)
    analysis = build_analysis(output_root)
    tables = build_revision_tables(output_root)
    save_json(ArtifactPaths.from_root(output_root).metrics / "paper_assets.json", {"analysis": analysis, "tables": tables})
    return {"analysis": analysis, "tables": tables}
