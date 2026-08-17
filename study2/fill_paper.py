#!/usr/bin/env python3
"""Render the paper from its template so every number comes from the data.

The template (icdm2026_teen_submission/main_template.tex) contains directives
that this script replaces:

    @cell(dataset,method,lr,metric[,fmt])@    mean over seeds of one grid cell
    @sd(dataset,method,lr,metric[,fmt])@      across-seed standard deviation
    @ms(dataset,method,lr,metric[,fmt])@      "mean$\\pm$sd"
    @agg(dataset,method,metric[,fmt])@        mean over every learning rate
    @pred(predictor,field[,fmt])@             predictor-ranking entry
    @pred1(predictor,field[,fmt])@            same, measured after one epoch
    @stat(dotted.path[,fmt])@                 any value in summary.json
    @nruns()@ @ncells()@ @rows()@ @refrows()@

Fractional metrics (accuracies) are converted to percentages automatically.
Unresolvable directives are reported and left in place, so a stale template
fails loudly instead of silently shipping a wrong number.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_DIR / "study2" / "analysis"
PAPER_DIR = PROJECT_DIR / "icdm2026_teen_submission"
TEMPLATE = PAPER_DIR / "main_template.tex"
OUTPUT = PAPER_DIR / "main.tex"
FIGURES = ("study2_dose_response", "study2_predictors", "study2_layer_heatmap",
           "study2_layer_heatmap_compact", "study2_predictor_bars",
           "study2_intervention")

FRACTIONAL = {"test_acc", "val_acc", "corruption_mean_acc", "corruption_gap"}
DATASET_KEY = {"eurosat": "eurosat", "pets": "pets"}
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(ANALYSIS_DIR / "run_level.csv")
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    for column in FRACTIONAL:
        if column in df:
            df[column] = df[column] * 100
    return df, summary


def _cell(df: pd.DataFrame, dataset: str, method: str, lr: str, metric: str) -> pd.Series:
    # never mix backbones: every @cell@ refers to the primary one
    if "model_name" in df:
        df = df[df.model_name == DEFAULT_MODEL]
    subset = df[(df.dataset == DATASET_KEY[dataset]) & (df.method == method)]
    if lr not in ("any", "*"):
        subset = subset[np.isclose(subset.lr, float(lr))]
    if metric not in subset:
        raise KeyError(f"unknown metric {metric}")
    return subset[metric].dropna()


def _fmt(value: float, fmt: str) -> str:
    if fmt == "sci":                      # p-values as LaTeX scientific notation
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        if exponent >= -2:
            return f"{value:.3f}".rstrip("0")
        mantissa = value / (10 ** exponent)
        if abs(mantissa - 1.0) < 0.05:
            return f"10^{{{exponent}}}"
        return rf"{mantissa:.1f}\!\cdot\!10^{{{exponent}}}"
    if fmt == "lr":
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-9:
            return f"10^{{{exponent}}}"
        return rf"{mantissa:g}\!\cdot\!10^{{{exponent}}}"
    if fmt == "abs2":
        return format(abs(value), ".2f")
    if fmt == "abs1":
        return format(abs(value), ".1f")
    if fmt == "abs3":
        return format(abs(value), ".3f")
    return format(value, fmt)


def render(text: str, df: pd.DataFrame, summary: dict) -> tuple[str, list[str]]:
    problems: list[str] = []

    def resolve(match: re.Match) -> str:
        name, args = match.group(1), match.group(2)
        parts = [part.strip() for part in args.split(",")] if args.strip() else []
        try:
            if name in ("cell", "sd", "ms"):
                dataset, method, lr, metric = parts[:4]
                fmt = parts[4] if len(parts) > 4 else ".2f"
                values = _cell(df, dataset, method, lr, metric)
                if values.empty:
                    raise KeyError("empty cell")
                if name == "cell":
                    return _fmt(values.mean(), fmt)
                sd = values.std(ddof=1) if len(values) > 1 else 0.0
                if name == "sd":
                    return _fmt(sd, fmt)
                return rf"{_fmt(values.mean(), fmt)}\,$\pm$\,{_fmt(sd, fmt)}"
            if name == "agg":
                dataset, method, metric = parts[:3]
                fmt = parts[3] if len(parts) > 3 else ".2f"
                values = _cell(df, dataset, method, "any", metric)
                return _fmt(values.mean(), fmt)
            if name in ("pred", "pred1"):
                predictor, field = parts[0], parts[1]
                fmt = parts[2] if len(parts) > 2 else ".3f"
                key = "predictor_ranking_final" if name == "pred" else "predictor_ranking_epoch1"
                entry = next(item for item in summary[key] if item["predictor"] == predictor)
                return _fmt(entry[field], fmt) if isinstance(entry[field], (int, float)) else str(entry[field])
            if name == "dose":
                dataset, method, metric = parts[:3]
                field = parts[3] if len(parts) > 3 else "spearman_rho"
                fmt = parts[4] if len(parts) > 4 else ".3f"
                entry = next(item for item in summary["dose_response"]
                             if item["dataset"] == DATASET_KEY[dataset]
                             and item["method"] == method and item["metric"] == metric)
                return _fmt(entry[field], fmt)
            if name == "dosebest":
                metric = parts[0]
                fmt = parts[1] if len(parts) > 1 else ".3f"
                values = [item["spearman_rho"] for item in summary["dose_response"]
                          if item["metric"] == metric]
                return _fmt(min(values) if np.mean(values) < 0 else max(values), fmt)
            if name == "doseworst":
                metric = parts[0]
                fmt = parts[1] if len(parts) > 1 else ".3f"
                values = [item["spearman_rho"] for item in summary["dose_response"]
                          if item["metric"] == metric]
                return _fmt(max(values) if np.mean(values) < 0 else min(values), fmt)
            if name == "paired":
                dataset, metric = parts[:2]
                field = parts[2] if len(parts) > 2 else "mean_difference"
                fmt = parts[3] if len(parts) > 3 else ".2f"
                entry = next(item for item in summary["paired_method_tests"]
                             if item["dataset"] == DATASET_KEY[dataset]
                             and item["metric"] == metric)
                return _fmt(entry[field], fmt)
            if name == "iso":
                dataset, field = parts[0], parts[1]
                fmt = parts[2] if len(parts) > 2 else ".2f"
                entry = next(item for item in summary["iso_accuracy"]
                             if item["dataset"] == DATASET_KEY[dataset])
                value = entry[field]
                if field.endswith("_lr"):
                    fmt = "lr"
                return _fmt(value, fmt) if isinstance(value, (int, float)) else str(value)
            if name == "sel":
                dataset, signal, field = parts[:3]
                fmt = parts[3] if len(parts) > 3 else ".1f"
                entry = next(item for item in summary["selection_utility"]
                             if item["dataset"] == DATASET_KEY[dataset])
                value = entry["signals"][signal][field] if signal != "-" else entry[field]
                return _fmt(value, fmt) if isinstance(value, (int, float)) else str(value)
            if name == "selm":
                margin, dataset, signal, field = parts[:4]
                fmt = parts[4] if len(parts) > 4 else ".1f"
                records = summary["selection_utility_margins"][margin]
                entry = next(item for item in records
                             if item["dataset"] == DATASET_KEY[dataset])
                value = (entry["signals"][signal][field] if signal != "-"
                         else entry[field])
                return _fmt(value, fmt) if isinstance(value, (int, float)) else str(value)
            if name == "boot":
                predictor, field = parts[0], parts[1]
                fmt = parts[2] if len(parts) > 2 else ".3f"
                entry = next(item for item in summary["bootstrap_predictor_gap"]
                             if item["predictor"] == predictor)
                return _fmt(entry[field], fmt)
            if name == "pol":
                model, method, signal = parts[0], parts[1], parts[2]
                field = parts[3] if len(parts) > 3 else "rho"
                fmt = parts[4] if len(parts) > 4 else "+.2f"
                entry = next(item for item in summary["polarity_by_group"]
                             if model in item["model"] and item["method"] == method)
                return _fmt(entry[signal][field], fmt)
            if name == "polrange":
                signal, field = parts[0], parts[1]
                fmt = parts[2] if len(parts) > 2 else "+.2f"
                values = [item[signal]["rho"] for item in summary["polarity_by_group"]
                          if signal in item]
                return _fmt(min(values) if field == "min" else max(values), fmt)
            if name == "stat":
                node = summary
                for step in parts[0].split("."):
                    node = node[int(step)] if step.isdigit() else node[step]
                fmt = parts[1] if len(parts) > 1 else ".3f"
                return _fmt(node, fmt) if isinstance(node, (int, float)) else str(node)
            if name == "nruns":
                if "model_name" in df:
                    return str(int((df.model_name == DEFAULT_MODEL).sum()))
                return str(len(df))
            if name == "bbruns":
                total = sum(spec["n_runs"] for spec in
                            (summary.get("backbone_check") or {}).values())
                return str(total)
            if name == "bb":
                model, key, field = parts[0], parts[1], parts[2]
                fmt = parts[3] if len(parts) > 3 else ".3f"
                spec = next(value for name_, value in summary["backbone_check"].items()
                            if model in name_)
                return _fmt(spec["predictors"][key][field], fmt)
            if name == "ncells":
                return str(len(summary["cells"]))
            if name == "rows":
                return (ANALYSIS_DIR / "core_grid_rows.tex").read_text().strip()
            if name == "refrows":
                return (ANALYSIS_DIR / "reference_rows.tex").read_text().strip()
            if name == "predrows":
                return (ANALYSIS_DIR / "predictor_rows.tex").read_text().strip()
        except Exception as error:                                  # noqa: BLE001
            problems.append(f"{match.group(0)} -> {type(error).__name__}: {error}")
            return match.group(0)
        problems.append(f"{match.group(0)} -> unknown directive")
        return match.group(0)

    return re.sub(r"@(\w+)\(([^)]*)\)@", resolve, text), problems


def audit(template: str, df: pd.DataFrame, summary: dict) -> str:
    """Every directive in document order with its resolved value.

    Used for the final read-through: each printed number can be checked against
    the sentence that surrounds it, in one place, without trusting the prose.
    """
    lines = ["# Paper numbers, in document order\n",
             "Each row is a directive from `main_template.tex` and the value it",
             "resolved to. Cross-check against `digest.md`.\n",
             "| # | context | directive | value |", "|---|---|---|---|"]
    count = 0
    for match in re.finditer(r"@(\w+)\(([^)]*)\)@", template):
        count += 1
        rendered, problems = render(match.group(0), df, summary)
        before = template[max(0, match.start() - 60):match.start()]
        context = re.sub(r"\s+", " ", before).strip()[-52:].replace("|", r"\|")
        value = "UNRESOLVED" if problems else rendered
        lines.append(f"| {count} | ...{context} | `{match.group(0)}` | {value} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true",
                        help="replace unresolved directives with ?? so the document "
                             "still compiles while runs are pending")
    parser.add_argument("--audit", action="store_true",
                        help="also write study2/analysis/paper_numbers.md")
    args = parser.parse_args()

    df, summary = load()
    text, problems = render(TEMPLATE.read_text(), df, summary)
    if args.draft and problems:
        text = re.sub(r"@\w+\([^)]*\)@", "??", text)
        print("DRAFT BUILD: unresolved directives replaced with ?? -- not submittable")
    OUTPUT.write_text(text)

    figure_dir = PAPER_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name in FIGURES:
        source = ANALYSIS_DIR / "figures" / f"{name}.png"
        if source.exists():
            shutil.copy2(source, figure_dir / f"{name}.png")

    if args.audit:
        (ANALYSIS_DIR / "paper_numbers.md").write_text(
            audit(TEMPLATE.read_text(), df, summary))
        print(f"wrote {(ANALYSIS_DIR / 'paper_numbers.md').relative_to(PROJECT_DIR)}")

    leftovers = re.findall(r"PLACEHOLDER[-\w]*", text)
    print(f"wrote {OUTPUT.relative_to(PROJECT_DIR)} from {len(df)} runs")
    if problems:
        print(f"{len(problems)} unresolved directive(s):")
        for problem in problems:
            print("  -", problem)
    if leftovers:
        print(f"{len(leftovers)} placeholder(s) left: {sorted(set(leftovers))}")
    if not problems and not leftovers:
        print("all directives resolved")


if __name__ == "__main__":
    main()
