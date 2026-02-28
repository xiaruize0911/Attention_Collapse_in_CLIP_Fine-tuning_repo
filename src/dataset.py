"""
dataset.py - Data loading for EuroSAT, Oxford-IIIT Pets, and evaluation datasets.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import json
from pathlib import Path


# CLIP's normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_clip_transform(is_train=True):
    """Get CLIP-compatible image transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


class HFImageDataset(Dataset):
    """Wrapper for HuggingFace image datasets."""
    
    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]
        label = item[self.label_key]
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_eurosat(cache_dir="./data"):
    """Load EuroSAT dataset."""
    dataset = load_dataset("tanganke/eurosat", cache_dir=cache_dir)
    
    train_transform = get_clip_transform(is_train=True)
    test_transform = get_clip_transform(is_train=False)
    
    train_dataset = HFImageDataset(dataset["train"], transform=train_transform)
    test_dataset = HFImageDataset(dataset["test"] if "test" in dataset else dataset["train"], 
                                   transform=test_transform)
    
    num_classes = len(set(dataset["train"]["label"]))
    class_names = dataset["train"].features["label"].names if hasattr(dataset["train"].features["label"], 'names') else [str(i) for i in range(num_classes)]
    
    return train_dataset, test_dataset, num_classes, class_names


def load_oxford_pets(cache_dir="./data"):
    """Load Oxford-IIIT Pets dataset."""
    dataset = load_dataset("timm/oxford-iiit-pet", cache_dir=cache_dir)
    
    train_transform = get_clip_transform(is_train=True)
    test_transform = get_clip_transform(is_train=False)
    
    train_dataset = HFImageDataset(dataset["train"], transform=train_transform)
    test_dataset = HFImageDataset(dataset["test"], transform=test_transform)
    
    num_classes = len(set(dataset["train"]["label"]))
    class_names = dataset["train"].features["label"].names if hasattr(dataset["train"].features["label"], 'names') else [str(i) for i in range(num_classes)]
    
    return train_dataset, test_dataset, num_classes, class_names


def load_cifar100(cache_dir="./data"):
    """Load CIFAR-100 for zero-shot evaluation."""
    dataset = load_dataset("uoft-cs/cifar100", cache_dir=cache_dir)
    
    test_transform = get_clip_transform(is_train=False)
    test_dataset = HFImageDataset(dataset["test"], transform=test_transform,
                                   image_key="img", label_key="fine_label")
    
    num_classes = 100
    class_names = dataset["test"].features["fine_label"].names
    
    return test_dataset, num_classes, class_names


def load_flowers102(cache_dir="./data"):
    """Load Flowers102 for zero-shot evaluation."""
    dataset = load_dataset("nelorth/oxford-flowers", cache_dir=cache_dir)
    
    test_transform = get_clip_transform(is_train=False)
    test_dataset = HFImageDataset(dataset["test"], transform=test_transform)
    
    num_classes = 102
    class_names = dataset["test"].features["label"].names if hasattr(dataset["test"].features["label"], 'names') else [str(i) for i in range(102)]
    
    return test_dataset, num_classes, class_names


def create_fixed_eval_subset(dataset, num_samples=200, num_classes=10, seed=42):
    """Create a fixed evaluation subset, balanced across classes."""
    rng = np.random.RandomState(seed)
    
    # Group by class
    class_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label if isinstance(label, int) else label.item()
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Sample equally from each class
    samples_per_class = num_samples // len(class_indices)
    selected_indices = []
    
    for cls_id in sorted(class_indices.keys()):
        indices = class_indices[cls_id]
        n = min(samples_per_class, len(indices))
        selected = rng.choice(indices, size=n, replace=False)
        selected_indices.extend(selected.tolist())
    
    # If we need more, sample from remaining
    if len(selected_indices) < num_samples:
        remaining = list(set(range(len(dataset))) - set(selected_indices))
        extra = rng.choice(remaining, size=num_samples - len(selected_indices), replace=False)
        selected_indices.extend(extra.tolist())
    
    return Subset(dataset, selected_indices[:num_samples]), selected_indices[:num_samples]


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=4):
    """Create a DataLoader."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
