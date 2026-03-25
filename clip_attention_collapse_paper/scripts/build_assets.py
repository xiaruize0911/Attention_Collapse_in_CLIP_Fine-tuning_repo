#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import sys

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PAPER_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from revision_v2.analysis import analyze
from revision_v2.artifacts import ArtifactPaths
from revision_v2.config import RevisionConfig


TARGET_FIGURES_DIR = PAPER_DIR / "figures"
TARGET_TABLES_DIR = PAPER_DIR / "tables"
REVISION_ROOT = REPO_DIR / "outputs" / "revision_v2"

FIGURES_TO_COPY = [
    "fig_overview_revision.png",
    "fig_lr_sweep_revision.png",
    "fig_backbone_revision.png",
    "fig_prediction_revision.png",
    "fig_appendix_attention_revision.png",
]

TABLES_TO_COPY = [
    "main_results.tex",
    "lr_sweep.tex",
    "prediction_results.tex",
    "stat_tests.tex",
    "appendix_regularization.tex",
]


def copy_assets() -> None:
    TARGET_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactPaths.from_root(REVISION_ROOT)
    for name in FIGURES_TO_COPY:
        src = artifacts.figures / name
        if not src.exists():
            raise FileNotFoundError(f"Missing revision figure: {src}")
        shutil.copy2(src, TARGET_FIGURES_DIR / name)
    for name in TABLES_TO_COPY:
        src = artifacts.tables / name
        if not src.exists():
            raise FileNotFoundError(f"Missing revision table: {src}")
        shutil.copy2(src, TARGET_TABLES_DIR / name)


def main() -> None:
    analyze(RevisionConfig())
    copy_assets()


if __name__ == "__main__":
    main()
