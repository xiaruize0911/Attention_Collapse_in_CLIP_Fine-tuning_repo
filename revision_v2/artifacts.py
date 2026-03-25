from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    manifests: Path
    runs: Path
    metrics: Path
    tables: Path
    figures: Path

    @classmethod
    def from_root(cls, root: Path) -> "ArtifactPaths":
        paths = cls(
            root=root,
            manifests=root / "manifests",
            runs=root / "runs",
            metrics=root / "metrics",
            tables=root / "tables",
            figures=root / "figures",
        )
        for path in paths.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def run_dir(self, run_name: str) -> Path:
        path = self.runs / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
