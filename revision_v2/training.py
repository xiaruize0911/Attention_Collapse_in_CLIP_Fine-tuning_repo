from __future__ import annotations

import copy
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .artifacts import ArtifactPaths, save_json, utc_timestamp
from .config import BACKBONES, SOURCE_DATASETS, TRANSFER_DATASETS, RevisionConfig
from .data import DatasetBundle, TransferDatasetBundle, load_source_dataset_bundle, load_transfer_dataset_bundle, maybe_limit_dataset
from .metrics import compute_attention_summary, compute_layerwise_cka, relative_shift
from .modeling import build_model, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOURCE_PROMPTS = {
    "eurosat": "a satellite photo of {label}.",
    "pets": "a photo of a {label}.",
    "cars": "a photo of a {label}.",
    "cifar100": "a photo of a {label}.",
    "dtd": "a photo of a {label} texture.",
    "caltech101": "a photo of a {label}.",
}


@dataclass
class RunResult:
    run_name: str
    summary_path: Path
    checkpoint_path: Path | None


class EntropyFloorRegularizer(nn.Module):
    def __init__(self, baseline_entropy: list[float], lambda_reg: float):
        super().__init__()
        self.register_buffer("baseline_entropy", torch.tensor(baseline_entropy, dtype=torch.float32))
        self.lambda_reg = lambda_reg

    def forward(self, attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        penalties = []
        for layer_idx, layer_attn in enumerate(attentions):
            cls_attn = layer_attn[:, :, 0, 1:]
            cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            entropy = -(cls_attn * cls_attn.clamp_min(1e-10).log()).sum(dim=-1).mean()
            penalties.append(torch.relu(self.baseline_entropy[layer_idx] - entropy))
        return self.lambda_reg * torch.stack(penalties).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def sanitize_label(label: str) -> str:
    return str(label).replace("_", " ").replace("-", " ")


def build_zero_shot_text_embeddings(model, processor, labels: list[str], dataset_name: str, device: torch.device) -> torch.Tensor:
    prompt_template = SOURCE_PROMPTS.get(dataset_name, "a photo of a {label}.")
    prompts = [prompt_template.format(label=sanitize_label(label)) for label in labels]
    encoded = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.clip_model.get_text_features(**encoded)
    return text_features


def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, output_attentions=False, output_hidden_states=False)
            predictions = outputs["logits"].argmax(dim=-1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
    return correct / max(total, 1)


def evaluate_zero_shot(model, processor, bundle: TransferDatasetBundle, dataset_name: str, batch_size: int, num_workers: int, device: torch.device) -> float:
    loader = make_loader(bundle.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_embeds = build_zero_shot_text_embeddings(model, processor, bundle.class_names, dataset_name, device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model.get_zero_shot_logits(images, text_embeds)
            predictions = logits.argmax(dim=-1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
    return correct / max(total, 1)


def collect_structural_metrics(
    model,
    baseline_model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    model.eval()
    baseline_model.eval()
    attention_records: list[dict[str, Any]] = []
    cka_records: list[list[float]] = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            outputs = model(images, output_attentions=True, output_hidden_states=True)
            baseline_outputs = baseline_model(images, output_attentions=True, output_hidden_states=True)
            attention_records.append(compute_attention_summary(outputs["attentions"]))
            cka = compute_layerwise_cka(outputs["hidden_states"], baseline_outputs["hidden_states"])
            cka_records.append(list(cka["layerwise_cka"]))

    if not attention_records:
        raise RuntimeError("No batches available for structural metrics.")

    aggregated: dict[str, Any] = {}
    keys = attention_records[0].keys()
    for key in keys:
        value = attention_records[0][key]
        if isinstance(value, list):
            aggregated[key] = [
                float(np.mean([record[key][i] for record in attention_records]))
                for i in range(len(value))
            ]
        else:
            aggregated[key] = float(np.mean([record[key] for record in attention_records]))

    aggregated["layerwise_cka"] = [
        float(np.mean([record[i] for record in cka_records]))
        for i in range(len(cka_records[0]))
    ]
    aggregated["mean_layerwise_cka"] = float(np.mean(aggregated["layerwise_cka"]))
    return aggregated


def build_run_name(group: str, backbone: str, dataset: str, method: str, seed: int, lr: float, extra: str = "") -> str:
    backbone_tag = "b32" if backbone.endswith("patch32") else "b16"
    method_tag = {"full_ft": "ft", "lora": "lora", "entropy_floor": "entfloor", "pretrained": "pretrained"}[method]
    lr_tag = f"lr{lr:.0e}".replace("+0", "").replace("-0", "-")
    pieces = [group, backbone_tag, dataset, method_tag, lr_tag, f"seed{seed}"]
    if extra:
        pieces.append(extra)
    return "_".join(pieces)


def prepare_source_and_transfer(
    config: RevisionConfig,
    processor,
    *,
    source_dataset: str,
) -> tuple[DatasetBundle, dict[str, TransferDatasetBundle]]:
    output_root = config.resolved_output_root(PROJECT_ROOT)
    data_root = config.resolved_data_root(PROJECT_ROOT)
    artifacts = ArtifactPaths.from_root(output_root)
    cars_root = None
    if source_dataset == "cars":
        cars_root = config.resolved_cars_root(PROJECT_ROOT)

    source_bundle = load_source_dataset_bundle(
        source_dataset,
        data_root,
        artifacts.manifests,
        config.manifest_seed,
        processor,
        cars_root=cars_root,
        train_flip_probability=config.train_flip_probability,
    )
    transfer_bundles = {
        name: load_transfer_dataset_bundle(name, data_root, processor)
        for name in TRANSFER_DATASETS
    }
    return source_bundle, transfer_bundles


def prepare_data(config: RevisionConfig) -> dict[str, Any]:
    output_root = config.resolved_output_root(PROJECT_ROOT)
    artifacts = ArtifactPaths.from_root(output_root)
    manifest_records = []
    for backbone in BACKBONES:
        built = build_model(backbone, num_classes=2, method="pretrained")
        processor = built.processor
        for dataset_name in SOURCE_DATASETS:
            cars_root = config.resolved_cars_root(PROJECT_ROOT) if dataset_name == "cars" else None
            bundle = load_source_dataset_bundle(
                dataset_name,
                config.resolved_data_root(PROJECT_ROOT),
                artifacts.manifests,
                config.manifest_seed,
                processor,
                cars_root=cars_root,
                train_flip_probability=config.train_flip_probability,
            )
            manifest_records.append(
                {
                    "backbone": backbone,
                    "dataset": dataset_name,
                    "manifest_path": str(bundle.manifest_path),
                    "train_size": len(bundle.train_dataset),
                    "val_size": len(bundle.val_dataset),
                    "test_size": len(bundle.test_dataset),
                }
            )
    payload = {"prepared_at": utc_timestamp(), "manifests": manifest_records}
    save_json(artifacts.metrics / "data_preparation.json", payload)
    return payload


def _autocast_context(use_amp: bool):
    if use_amp and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _run_training(
    config: RevisionConfig,
    *,
    group: str,
    backbone: str,
    source_dataset: str,
    method: str,
    seed: int,
    lr: float,
    use_pilot_fraction: bool = False,
    entropy_floor_lambda: float | None = None,
) -> RunResult:
    set_seed(seed)
    output_root = config.resolved_output_root(PROJECT_ROOT)
    artifacts = ArtifactPaths.from_root(output_root)
    run_name = build_run_name(group, backbone, source_dataset, method, seed, lr)
    run_dir = artifacts.run_dir(run_name)

    built_model = build_model(
        backbone,
        num_classes=2,  # placeholder until the dataset is loaded below
        method="pretrained" if method == "pretrained" else method,
        lora_targets=config.fixed_lora_targets,
        lora_rank=config.fixed_lora_rank,
        lora_alpha=config.fixed_lora_alpha,
        lora_dropout=config.fixed_lora_dropout,
    )
    processor = built_model.processor
    source_bundle, transfer_bundles = prepare_source_and_transfer(config, processor, source_dataset=source_dataset)
    num_classes = len(source_bundle.class_names)

    built_model = build_model(
        backbone,
        num_classes=num_classes,
        method="pretrained" if method == "pretrained" else ("lora" if method == "lora" else "full_ft"),
        lora_targets=config.fixed_lora_targets,
        lora_rank=config.fixed_lora_rank,
        lora_alpha=config.fixed_lora_alpha,
        lora_dropout=config.fixed_lora_dropout,
    )
    model = built_model.model.to(DEVICE)
    processor = built_model.processor

    baseline_built = build_model(backbone, num_classes=num_classes, method="pretrained")
    baseline_model = baseline_built.model.to(DEVICE)

    train_dataset = source_bundle.train_dataset
    if use_pilot_fraction:
        train_dataset = maybe_limit_dataset(train_dataset, config.pilot_fraction, seed)

    train_loader = make_loader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = make_loader(source_bundle.val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = make_loader(source_bundle.test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    baseline_metrics = collect_structural_metrics(
        baseline_model,
        baseline_model,
        test_loader,
        DEVICE,
        max_batches=config.max_metric_batches,
    )

    if method == "pretrained":
        in_domain_acc = evaluate_zero_shot(
            baseline_model,
            processor,
            TransferDatasetBundle(dataset=source_bundle.test_dataset, class_names=source_bundle.class_names),
            source_dataset,
            batch_size=config.transfer_batch_size,
            num_workers=config.num_workers,
            device=DEVICE,
        )
        transfer_scores = {
            name: evaluate_zero_shot(
                baseline_model,
                processor,
                bundle,
                name,
                batch_size=config.transfer_batch_size,
                num_workers=config.num_workers,
                device=DEVICE,
            )
            for name, bundle in transfer_bundles.items()
        }
        summary = build_summary(
            config=config,
            run_name=run_name,
            group=group,
            backbone=backbone,
            source_dataset=source_dataset,
            method=method,
            seed=seed,
            lr=0.0,
            manifest_path=source_bundle.manifest_path,
            checkpoint_path=None,
            in_domain_acc=in_domain_acc,
            baseline_metrics=baseline_metrics,
            final_metrics=baseline_metrics,
            transfer_scores=transfer_scores,
            pretrained_transfer_scores=transfer_scores,
            parameter_counts=count_parameters(model),
            epochs_completed=0,
            best_val_loss=None,
        )
        summary_path = run_dir / "summary.json"
        save_json(summary_path, summary)
        return RunResult(run_name=run_name, summary_path=summary_path, checkpoint_path=None)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and torch.cuda.is_available())
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_rows = []
    regularizer = None
    if entropy_floor_lambda is not None:
        regularizer = EntropyFloorRegularizer(baseline_metrics["entropy_per_layer"], entropy_floor_lambda).to(DEVICE)

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_items = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with _autocast_context(config.use_amp):
                outputs = model(images, output_attentions=regularizer is not None, output_hidden_states=False)
                loss = criterion(outputs["logits"], labels)
                if regularizer is not None and outputs["attentions"] is not None:
                    loss = loss + regularizer(outputs["attentions"])
                loss = loss / max(config.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % max(config.grad_accum_steps, 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item()) * labels.size(0) * max(config.grad_accum_steps, 1)
            epoch_items += labels.size(0)

        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, criterion)
        train_acc = evaluate_accuracy(model, train_loader, DEVICE)
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)
        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(epoch_items, 1),
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = run_dir / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
            "source_dataset": source_dataset,
            "method": method,
            "backbone": backbone,
        },
        checkpoint_path,
    )

    final_metrics = collect_structural_metrics(
        model,
        baseline_model,
        test_loader,
        DEVICE,
        max_batches=config.max_metric_batches,
    )
    in_domain_acc = evaluate_accuracy(model, test_loader, DEVICE)
    pretrained_transfer_scores = {
        name: evaluate_zero_shot(
            baseline_model,
            processor,
            bundle,
            name,
            batch_size=config.transfer_batch_size,
            num_workers=config.num_workers,
            device=DEVICE,
        )
        for name, bundle in transfer_bundles.items()
    }
    transfer_scores = {
        name: evaluate_zero_shot(
            model,
            processor,
            bundle,
            name,
            batch_size=config.transfer_batch_size,
            num_workers=config.num_workers,
            device=DEVICE,
        )
        for name, bundle in transfer_bundles.items()
    }

    summary = build_summary(
        config=config,
        run_name=run_name,
        group=group,
        backbone=backbone,
        source_dataset=source_dataset,
        method=method,
        seed=seed,
        lr=lr,
        manifest_path=source_bundle.manifest_path,
        checkpoint_path=checkpoint_path,
        in_domain_acc=in_domain_acc,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        transfer_scores=transfer_scores,
        pretrained_transfer_scores=pretrained_transfer_scores,
        parameter_counts=count_parameters(model),
        epochs_completed=len(history_rows),
        best_val_loss=best_val_loss,
    )
    summary_path = run_dir / "summary.json"
    save_json(summary_path, summary)
    write_history(run_dir / "history.csv", history_rows)
    return RunResult(run_name=run_name, summary_path=summary_path, checkpoint_path=checkpoint_path)


def evaluate_loss(model, loader: DataLoader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images, output_attentions=False, output_hidden_states=False)
            loss = criterion(outputs["logits"], labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_items += labels.size(0)
    return total_loss / max(total_items, 1)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    config: RevisionConfig,
    run_name: str,
    group: str,
    backbone: str,
    source_dataset: str,
    method: str,
    seed: int,
    lr: float,
    manifest_path: Path,
    checkpoint_path: Path | None,
    in_domain_acc: float,
    baseline_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    transfer_scores: dict[str, float],
    pretrained_transfer_scores: dict[str, float],
    parameter_counts: dict[str, int],
    epochs_completed: int,
    best_val_loss: float | None,
) -> dict[str, Any]:
    retention_components = {}
    for dataset_name, score in transfer_scores.items():
        baseline_score = pretrained_transfer_scores[dataset_name]
        retention_components[dataset_name] = score / baseline_score if baseline_score else float("nan")

    summary = {
        "timestamp": utc_timestamp(),
        "run_name": run_name,
        "group": group,
        "backbone": backbone,
        "source_dataset": source_dataset,
        "method": method,
        "seed": seed,
        "lr": lr,
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "parameter_counts": parameter_counts,
        "in_domain_test_accuracy": in_domain_acc,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "delta_metrics": {
            "entropy_shift": relative_shift(final_metrics["entropy_mean"], baseline_metrics["entropy_mean"]),
            "erf95_shift": relative_shift(final_metrics["erf95_mean"], baseline_metrics["erf95_mean"]),
            "gini_shift": relative_shift(final_metrics["gini_mean"], baseline_metrics["gini_mean"]),
            "head_diversity_shift": relative_shift(final_metrics["head_diversity_mean"], baseline_metrics["head_diversity_mean"]),
            "rollout_entropy_shift": relative_shift(final_metrics["rollout_entropy_mean"], baseline_metrics["rollout_entropy_mean"]),
            "rollout_erf95_shift": relative_shift(final_metrics["rollout_erf95_mean"], baseline_metrics["rollout_erf95_mean"]),
            "rollout_gini_shift": relative_shift(final_metrics["rollout_gini_mean"], baseline_metrics["rollout_gini_mean"]),
            "patch_to_patch_entropy_shift": relative_shift(final_metrics["patch_to_patch_entropy_mean"], baseline_metrics["patch_to_patch_entropy_mean"]),
            "mean_layerwise_cka": final_metrics["mean_layerwise_cka"],
        },
        "zero_shot_accuracy": transfer_scores,
        "pretrained_zero_shot_accuracy": pretrained_transfer_scores,
        "transfer_retention_components": retention_components,
        "transfer_retention_score": float(np.mean(list(retention_components.values()))),
        "summary_row": {
            "backbone": backbone,
            "source_dataset": source_dataset,
            "method": method,
            "seed": seed,
            "in_domain_test_accuracy": in_domain_acc,
            "mean_entropy_shift": relative_shift(final_metrics["entropy_mean"], baseline_metrics["entropy_mean"]),
            "mean_erf95_shift": relative_shift(final_metrics["erf95_mean"], baseline_metrics["erf95_mean"]),
            "mean_gini_shift": relative_shift(final_metrics["gini_mean"], baseline_metrics["gini_mean"]),
            "mean_head_diversity_shift": relative_shift(final_metrics["head_diversity_mean"], baseline_metrics["head_diversity_mean"]),
            "mean_rollout_entropy": final_metrics["rollout_entropy_mean"],
            "mean_rollout_erf95": final_metrics["rollout_erf95_mean"],
            "mean_rollout_gini": final_metrics["rollout_gini_mean"],
            "mean_patch_to_patch_entropy": final_metrics["patch_to_patch_entropy_mean"],
            "mean_layerwise_cka": final_metrics["mean_layerwise_cka"],
            "zero_shot_cifar100": transfer_scores["cifar100"],
            "zero_shot_dtd": transfer_scores["dtd"],
            "zero_shot_caltech101": transfer_scores["caltech101"],
            "transfer_retention_score": float(np.mean(list(retention_components.values()))),
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "timestamp": utc_timestamp(),
        },
        "config": config.to_dict(),
    }
    return summary


def write_run_summary_index(output_root: Path) -> Path:
    artifacts = ArtifactPaths.from_root(output_root)
    rows = []
    for summary_path in sorted(artifacts.runs.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = payload["summary_row"]
        row["run_name"] = payload["run_name"]
        row["group"] = payload["group"]
        row["lr"] = payload["lr"]
        rows.append(row)

    csv_path = artifacts.metrics / "run_summaries.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json(artifacts.metrics / "run_summaries.json", {"rows": rows, "updated_at": utc_timestamp()})
    return csv_path


def run_pilot(config: RevisionConfig) -> list[RunResult]:
    results = []
    for method, lr in [("pretrained", 0.0), ("full_ft", 1e-5), ("lora", 5e-5)]:
        results.append(
            _run_training(
                config,
                group="pilot",
                backbone="openai/clip-vit-base-patch32",
                source_dataset="eurosat",
                method=method,
                seed=config.seed,
                lr=lr,
                use_pilot_fraction=(method != "pretrained"),
            )
        )
    write_run_summary_index(config.resolved_output_root(PROJECT_ROOT))
    return results


def run_main(config: RevisionConfig) -> list[RunResult]:
    results = []
    seeds = [1, 2, 3, 4, 5]
    for source_dataset in SOURCE_DATASETS:
        results.append(
            _run_training(
                config,
                group="main",
                backbone="openai/clip-vit-base-patch32",
                source_dataset=source_dataset,
                method="pretrained",
                seed=0,
                lr=0.0,
            )
        )
        for method, lr in [("full_ft", 1e-5), ("lora", 5e-5)]:
            for seed in seeds:
                results.append(
                    _run_training(
                        config,
                        group="main",
                        backbone="openai/clip-vit-base-patch32",
                        source_dataset=source_dataset,
                        method=method,
                        seed=seed,
                        lr=lr,
                    )
                )
    write_run_summary_index(config.resolved_output_root(PROJECT_ROOT))
    return results


def run_lr_sweep(config: RevisionConfig) -> list[RunResult]:
    results = []
    for lr in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:
        for seed in [1, 2, 3]:
            results.append(
                _run_training(
                    config,
                    group="lr_sweep",
                    backbone="openai/clip-vit-base-patch32",
                    source_dataset="eurosat",
                    method="full_ft",
                    seed=seed,
                    lr=lr,
                )
            )
    write_run_summary_index(config.resolved_output_root(PROJECT_ROOT))
    return results


def run_backbone_confirmation(config: RevisionConfig) -> list[RunResult]:
    results = []
    for source_dataset in SOURCE_DATASETS:
        results.append(
            _run_training(
                config,
                group="backbone_confirmation",
                backbone="openai/clip-vit-base-patch16",
                source_dataset=source_dataset,
                method="pretrained",
                seed=0,
                lr=0.0,
            )
        )
        for method, lr in [("full_ft", 1e-5), ("lora", 5e-5)]:
            for seed in [1, 2, 3]:
                results.append(
                    _run_training(
                        config,
                        group="backbone_confirmation",
                        backbone="openai/clip-vit-base-patch16",
                        source_dataset=source_dataset,
                        method=method,
                        seed=seed,
                        lr=lr,
                    )
                )
    write_run_summary_index(config.resolved_output_root(PROJECT_ROOT))
    return results


def run_appendix(config: RevisionConfig) -> list[RunResult]:
    results = []
    for seed in [1, 2, 3]:
        results.append(
            _run_training(
                config,
                group="appendix",
                backbone="openai/clip-vit-base-patch32",
                source_dataset="eurosat",
                method="entropy_floor",
                seed=seed,
                lr=1e-5,
                entropy_floor_lambda=config.entropy_floor_lambda,
            )
        )
    write_run_summary_index(config.resolved_output_root(PROJECT_ROOT))
    return results
