from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF

from .artifacts import save_json


class ProcessorTransform:
    """Apply checkpoint-matched CLIP preprocessing to a PIL image."""

    def __init__(self, processor, train: bool = False, flip_probability: float = 0.5):
        self.processor = processor
        self.train = train
        self.flip_probability = flip_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.train and self.flip_probability > 0.0 and random.random() < self.flip_probability:
            image = TF.hflip(image)
        encoded = self.processor(images=image, return_tensors="pt")
        return encoded["pixel_values"][0]


class LabelMappedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        label_transform: Callable | None = None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.label_transform = label_transform or _default_label_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        label = self.label_transform(label)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class ManifestSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    class_names: list[str]
    manifest_path: Path


@dataclass
class TransferDatasetBundle:
    dataset: Dataset
    class_names: list[str]


def _default_label_transform(label):
    if isinstance(label, torch.Tensor):
        label = label.item()
    if isinstance(label, (list, tuple)):
        label = label[0]
    return int(label)


def _extract_targets(dataset: Dataset) -> list[int]:
    targets: list[int] = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        targets.append(_default_label_transform(label))
    return targets


def stratified_split_indices(
    targets: list[int],
    splits: dict[str, float],
    seed: int,
) -> dict[str, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        grouped[label].append(index)

    rng = random.Random(seed)
    result = {name: [] for name in splits}
    split_names = list(splits.keys())
    split_fracs = [splits[name] for name in split_names]

    for label, indices in grouped.items():
        indices = list(indices)
        rng.shuffle(indices)
        counts = []
        remaining = len(indices)
        for frac in split_fracs[:-1]:
            count = int(round(len(indices) * frac))
            count = min(count, remaining)
            counts.append(count)
            remaining -= count
        counts.append(remaining)

        cursor = 0
        for split_name, count in zip(split_names, counts):
            result[split_name].extend(indices[cursor: cursor + count])
            cursor += count

    for split_name in result:
        result[split_name].sort()
    return result


def subset_from_manifest(dataset: Dataset, indices: list[int]) -> ManifestSubset:
    return ManifestSubset(dataset, indices)


def build_or_load_source_manifest(
    dataset_name: str,
    root: Path,
    manifest_dir: Path,
    split_seed: int,
) -> tuple[dict[str, list[int]], Path]:
    manifest_path = manifest_dir / f"{dataset_name}_split_seed{split_seed}.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {k: list(v) for k, v in payload["indices"].items()}, manifest_path

    if dataset_name == "eurosat":
        dataset = datasets.EuroSAT(root=str(root), download=True)
        targets = _extract_targets(dataset)
        split_map = stratified_split_indices(targets, {"train": 0.8, "val": 0.1, "test": 0.1}, split_seed)
    elif dataset_name == "pets":
        trainval = datasets.OxfordIIITPet(
            root=str(root),
            split="trainval",
            target_types="category",
            download=True,
        )
        targets = _extract_targets(trainval)
        tv_split = stratified_split_indices(targets, {"train": 0.9, "val": 0.1}, split_seed)
        test = datasets.OxfordIIITPet(
            root=str(root),
            split="test",
            target_types="category",
            download=True,
        )
        split_map = {
            "train": tv_split["train"],
            "val": tv_split["val"],
            "test": list(range(len(test))),
        }
    elif dataset_name == "cars":
        raise ValueError("Stanford Cars manifest requires `cars_root` and should be created via `build_cars_manifest`.")
    else:
        raise ValueError(f"Unsupported source dataset: {dataset_name}")

    save_json(
        manifest_path,
        {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "indices": split_map,
        },
    )
    return split_map, manifest_path


def build_cars_manifest(cars_root: Path, manifest_dir: Path, split_seed: int) -> tuple[dict[str, list[int]], Path]:
    manifest_path = manifest_dir / f"cars_split_seed{split_seed}.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {k: list(v) for k, v in payload["indices"].items()}, manifest_path

    if not cars_root.exists():
        raise FileNotFoundError(
            f"Stanford Cars root not found: {cars_root}. "
            "Set STANFORD_CARS_ROOT or `cars_root` in the config."
        )

    train_ds = datasets.StanfordCars(root=str(cars_root), split="train", download=False)
    test_ds = datasets.StanfordCars(root=str(cars_root), split="test", download=False)
    targets = _extract_targets(train_ds)
    tv_split = stratified_split_indices(targets, {"train": 0.9, "val": 0.1}, split_seed)
    split_map = {
        "train": tv_split["train"],
        "val": tv_split["val"],
        "test": list(range(len(test_ds))),
    }
    save_json(
        manifest_path,
        {
            "dataset": "cars",
            "split_seed": split_seed,
            "indices": split_map,
        },
    )
    return split_map, manifest_path


def _class_names_from_dataset(dataset: Dataset) -> list[str]:
    for attr in ("classes", "categories"):
        value = getattr(dataset, attr, None)
        if value:
            return [str(v) for v in value]
    targets = _extract_targets(dataset)
    return [str(i) for i in sorted(set(targets))]


def load_source_dataset_bundle(
    dataset_name: str,
    data_root: Path,
    manifest_dir: Path,
    split_seed: int,
    processor,
    cars_root: Path | None = None,
    train_flip_probability: float = 0.5,
) -> DatasetBundle:
    train_transform = ProcessorTransform(processor, train=True, flip_probability=train_flip_probability)
    eval_transform = ProcessorTransform(processor, train=False)

    if dataset_name == "eurosat":
        raw = datasets.EuroSAT(root=str(data_root), download=True)
        manifest, manifest_path = build_or_load_source_manifest(dataset_name, data_root, manifest_dir, split_seed)
        mapped_train = LabelMappedDataset(raw, transform=train_transform)
        mapped_eval = LabelMappedDataset(raw, transform=eval_transform)
        return DatasetBundle(
            train_dataset=subset_from_manifest(mapped_train, manifest["train"]),
            val_dataset=subset_from_manifest(mapped_eval, manifest["val"]),
            test_dataset=subset_from_manifest(mapped_eval, manifest["test"]),
            class_names=_class_names_from_dataset(raw),
            manifest_path=manifest_path,
        )

    if dataset_name == "pets":
        trainval = datasets.OxfordIIITPet(
            root=str(data_root),
            split="trainval",
            target_types="category",
            download=True,
        )
        test = datasets.OxfordIIITPet(
            root=str(data_root),
            split="test",
            target_types="category",
            download=True,
        )
        manifest, manifest_path = build_or_load_source_manifest(dataset_name, data_root, manifest_dir, split_seed)
        mapped_train = LabelMappedDataset(trainval, transform=train_transform)
        mapped_train_eval = LabelMappedDataset(trainval, transform=eval_transform)
        mapped_test = LabelMappedDataset(test, transform=eval_transform)
        return DatasetBundle(
            train_dataset=subset_from_manifest(mapped_train, manifest["train"]),
            val_dataset=subset_from_manifest(mapped_train_eval, manifest["val"]),
            test_dataset=subset_from_manifest(mapped_test, manifest["test"]),
            class_names=_class_names_from_dataset(trainval),
            manifest_path=manifest_path,
        )

    if dataset_name == "cars":
        if cars_root is None:
            raise ValueError("Stanford Cars requires `cars_root`.")
        train_ds = datasets.StanfordCars(root=str(cars_root), split="train", download=False)
        test_ds = datasets.StanfordCars(root=str(cars_root), split="test", download=False)
        manifest, manifest_path = build_cars_manifest(cars_root, manifest_dir, split_seed)
        mapped_train = LabelMappedDataset(train_ds, transform=train_transform)
        mapped_train_eval = LabelMappedDataset(train_ds, transform=eval_transform)
        mapped_test = LabelMappedDataset(test_ds, transform=eval_transform)
        return DatasetBundle(
            train_dataset=subset_from_manifest(mapped_train, manifest["train"]),
            val_dataset=subset_from_manifest(mapped_train_eval, manifest["val"]),
            test_dataset=subset_from_manifest(mapped_test, manifest["test"]),
            class_names=_class_names_from_dataset(train_ds),
            manifest_path=manifest_path,
        )

    raise ValueError(f"Unsupported source dataset: {dataset_name}")


def load_transfer_dataset_bundle(dataset_name: str, data_root: Path, processor) -> TransferDatasetBundle:
    transform = ProcessorTransform(processor, train=False)

    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)
        class_names = [str(v) for v in dataset.classes]
    elif dataset_name == "dtd":
        dataset = datasets.DTD(root=str(data_root), split="test", partition=1, download=True, transform=transform)
        class_names = [str(v) for v in dataset.classes]
    elif dataset_name == "caltech101":
        raw = datasets.Caltech101(root=str(data_root), target_type="category", download=True)
        dataset = LabelMappedDataset(raw, transform=transform)
        class_names = [str(v) for v in getattr(raw, "categories", [str(i) for i in range(101)])]
    else:
        raise ValueError(f"Unsupported transfer dataset: {dataset_name}")

    return TransferDatasetBundle(dataset=dataset, class_names=class_names)


def maybe_limit_dataset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 1.0:
        return dataset
    n_items = max(1, int(len(dataset) * fraction))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, sorted(indices[:n_items]))
