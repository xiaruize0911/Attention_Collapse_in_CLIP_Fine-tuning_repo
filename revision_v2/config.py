from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OUTPUT_ROOT = Path("outputs") / "revision_v2"
DEFAULT_DATA_ROOT = Path("data") / "revision_v2"
DEFAULT_SPLIT_SEED = 20260325

SOURCE_DATASETS = ("eurosat", "pets", "cars")
TRANSFER_DATASETS = ("cifar100", "dtd", "caltech101")
BACKBONES = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
)


@dataclass
class RevisionConfig:
    backbone: str = "openai/clip-vit-base-patch32"
    source_dataset: str = "eurosat"
    method: str = "full_ft"
    seed: int = 42
    lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 32
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    data_root: str = str(DEFAULT_DATA_ROOT)
    manifest_seed: int = DEFAULT_SPLIT_SEED
    num_workers: int = 4
    use_amp: bool = True
    pilot_fraction: float = 0.1
    transfer_batch_size: int = 64
    fixed_lora_rank: int = 8
    fixed_lora_alpha: int = 16
    fixed_lora_dropout: float = 0.0
    fixed_lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    cars_root: str = ""
    entropy_floor_lambda: float = 0.1
    train_flip_probability: float = 0.5
    log_every_steps: int = 20
    max_metric_batches: int = 8

    def resolved_output_root(self, project_root: Path) -> Path:
        return _resolve_path(project_root, self.output_root)

    def resolved_data_root(self, project_root: Path) -> Path:
        return _resolve_path(project_root, self.data_root)

    def resolved_cars_root(self, project_root: Path) -> Path:
        raw = self.cars_root or os.environ.get("STANFORD_CARS_ROOT", "")
        if not raw:
            raise ValueError(
                "Stanford Cars requires a dataset root. Set `cars_root` in the config "
                "or export STANFORD_CARS_ROOT."
            )
        return _resolve_path(project_root, raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_config(path: str | Path | None = None) -> RevisionConfig:
    if path is None:
        return RevisionConfig()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    elif cfg_path.suffix == ".json":
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {cfg_path.suffix}")

    return RevisionConfig(**raw)
