from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 102


def build_transform(split: str, color_jitter: bool = True) -> Callable:
    split = split.lower()
    if split == "train":
        augments: list[Callable] = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        if color_jitter:
            augments.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                )
            )
        augments.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return transforms.Compose(augments)

    if split in {"val", "test"}:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported split: {split}")


def get_flowers102_dataset(
    data_dir: str | Path,
    split: str,
    download: bool = False,
    color_jitter: bool = True,
) -> datasets.Flowers102:
    return datasets.Flowers102(
        root=str(data_dir),
        split=split,
        transform=build_transform(split, color_jitter=color_jitter),
        download=download,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception:
        pass


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    download: bool = False,
    color_jitter: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = get_flowers102_dataset(
        data_dir, "train", download=download, color_jitter=color_jitter
    )
    val_dataset = get_flowers102_dataset(data_dir, "val", download=download)
    test_dataset = get_flowers102_dataset(data_dir, "test", download=download)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def dataset_summary(data_dir: str | Path, download: bool = False) -> dict[str, int]:
    summary: dict[str, int] = {}
    for split in ("train", "val", "test"):
        dataset = datasets.Flowers102(
            root=str(data_dir),
            split=split,
            transform=None,
            download=download,
        )
        labels = list(getattr(dataset, "_labels", []))
        summary[f"{split}_samples"] = len(dataset)
        summary[f"{split}_classes"] = len(set(labels)) if len(labels) > 0 else NUM_CLASSES
    summary["num_classes"] = NUM_CLASSES
    return summary
