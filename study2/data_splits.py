"""Deterministic, leakage-free splits for the protocol-hardened replication.

Design rules
------------
1. The provided test split is never used for training, model selection, or
   attention measurement. It is read exactly once per run, at the end.
2. Validation images are carved out of the provided train split.
3. The attention/feature probe set is carved out of the train split as well and
   is disjoint from both training and validation images. Only pixels are used,
   never labels, so the probe is something a practitioner could always compute.
4. Every split is chosen with a fixed data seed that is independent of the
   training seed, so all runs of a cell see identical images.
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset

from src.dataset import HFImageDataset, get_clip_transform

DATA_SEED = 0

DATASET_SPECS = {
    "eurosat": {
        "hf_name": "tanganke/eurosat",
        "image_key": "image",
        "label_key": "label",
        "num_classes": 10,
        "corruption_splits": (
            "contrast",
            "gaussian_noise",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "spatter",
        ),
    },
    "pets": {
        "hf_name": "timm/oxford-iiit-pet",
        "image_key": "image",
        "label_key": "label",
        "num_classes": 37,
        "corruption_splits": (),
    },
}


def _stratified_take(labels: np.ndarray, per_class: int, rng: np.random.RandomState,
                     forbidden: set[int]) -> list[int]:
    """Take `per_class` indices for every class, skipping already used indices."""
    picked: list[int] = []
    for cls in sorted(set(labels.tolist())):
        pool = [int(i) for i in np.flatnonzero(labels == cls) if int(i) not in forbidden]
        rng.shuffle(pool)
        picked.extend(pool[:per_class])
    return picked


def build_target_splits(dataset_name: str, train_per_class: int, val_per_class: int,
                        probe_per_class: int, cache_dir: str = "./data") -> dict:
    spec = DATASET_SPECS[dataset_name]
    hf = load_dataset(spec["hf_name"], cache_dir=cache_dir)
    train_labels = np.asarray(hf["train"][spec["label_key"]])

    rng = np.random.RandomState(DATA_SEED)
    used: set[int] = set()

    val_idx = _stratified_take(train_labels, val_per_class, rng, used)
    used.update(val_idx)
    probe_idx = _stratified_take(train_labels, probe_per_class, rng, used)
    used.update(probe_idx)
    train_idx = _stratified_take(train_labels, train_per_class, rng, used)

    train_tf = get_clip_transform(is_train=True)
    eval_tf = get_clip_transform(is_train=False)

    def wrap(split, transform):
        return HFImageDataset(split, transform=transform,
                              image_key=spec["image_key"], label_key=spec["label_key"])

    splits = {
        "train": Subset(wrap(hf["train"], train_tf), sorted(train_idx)),
        "val": Subset(wrap(hf["train"], eval_tf), sorted(val_idx)),
        "probe": Subset(wrap(hf["train"], eval_tf), sorted(probe_idx)),
        "test": wrap(hf["test"], eval_tf),
        "num_classes": spec["num_classes"],
        "class_names": hf["train"].features[spec["label_key"]].names,
        "sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "probe": len(probe_idx),
            "test": hf["test"].num_rows,
        },
        "indices": {
            "train": sorted(int(i) for i in train_idx),
            "val": sorted(int(i) for i in val_idx),
            "probe": sorted(int(i) for i in probe_idx),
        },
    }
    return splits


def build_corruption_splits(dataset_name: str, per_split: int,
                            cache_dir: str = "./data") -> dict:
    """Balanced fixed subsets of the corrupted target-domain test splits."""
    spec = DATASET_SPECS[dataset_name]
    if not spec["corruption_splits"]:
        return {}
    hf = load_dataset(spec["hf_name"], cache_dir=cache_dir)
    eval_tf = get_clip_transform(is_train=False)
    out = {}
    for split in spec["corruption_splits"]:
        labels = np.asarray(hf[split][spec["label_key"]])
        rng = np.random.RandomState(DATA_SEED)
        per_class = max(1, per_split // len(set(labels.tolist())))
        idx = _stratified_take(labels, per_class, rng, set())
        base = HFImageDataset(hf[split], transform=eval_tf,
                             image_key=spec["image_key"], label_key=spec["label_key"])
        out[split] = Subset(base, sorted(idx))
    return out


def build_transfer_set(name: str, max_images: int | None, cache_dir: str = "./data") -> dict:
    """Zero-shot transfer benchmark, optionally subsampled in a balanced way."""
    configs = {
        "cifar100": ("uoft-cs/cifar100", "img", "fine_label"),
        "cifar10": ("uoft-cs/cifar10", "img", "label"),
    }
    hf_name, image_key, label_key = configs[name]
    hf = load_dataset(hf_name, cache_dir=cache_dir)["test"]
    eval_tf = get_clip_transform(is_train=False)
    base = HFImageDataset(hf, transform=eval_tf, image_key=image_key, label_key=label_key)
    class_names = hf.features[label_key].names

    if max_images is not None and max_images < hf.num_rows:
        labels = np.asarray(hf[label_key])
        rng = np.random.RandomState(DATA_SEED)
        per_class = max(1, max_images // len(set(labels.tolist())))
        idx = _stratified_take(labels, per_class, rng, set())
        dataset = Subset(base, sorted(idx))
    else:
        dataset = base
    return {"dataset": dataset, "class_names": class_names, "size": len(dataset)}


def loader(dataset, batch_size: int, shuffle: bool, seed: int | None = None,
           num_workers: int = 0) -> DataLoader:
    import torch

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(0 if seed is None else seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator,
                      persistent_workers=False, drop_last=False)
