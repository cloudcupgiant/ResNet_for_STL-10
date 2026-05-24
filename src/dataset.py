"""Dataset splitting and DataLoader construction for STL-10 ImageFolder data."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from PIL import ImageFile
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.utils import ensure_dir, save_json

# Some downloaded image datasets can contain truncated files; PIL can still decode
# the readable prefix, which is usually preferable to failing a long training run.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Channel statistics for STL-10 RGB images, used by both train and eval transforms.
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2603, 0.2566, 0.2713]


def build_train_transform(
    augmentation: str,
    input_size: int = 96,
    mean: Sequence[float] = STL10_MEAN,
    std: Sequence[float] = STL10_STD,
    random_erasing_p: float = 0.25,
) -> transforms.Compose:
    """Build the training transform pipeline for weak or strong augmentation."""
    if augmentation == "weak":
        # Weak augmentation keeps the image distribution close to evaluation.
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif augmentation == "strong":
        # Strong augmentation adds spatial, color, and erasing noise to reduce overfitting.
        transform_list = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomCrop(input_size, padding=12),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=random_erasing_p, value="random"),
        ]
    else:
        raise ValueError(f"Unsupported augmentation: {augmentation}")
    return transforms.Compose(transform_list)


def build_eval_transform(
    input_size: int = 96,
    mean: Sequence[float] = STL10_MEAN,
    std: Sequence[float] = STL10_STD,
) -> transforms.Compose:
    """Use deterministic preprocessing for validation, test, and Grad-CAM images."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def stratified_train_valid_indices(
    targets: Sequence[int],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split indices while preserving class proportions as closely as possible."""
    if valid_ratio == 0:
        return list(range(len(targets))), []
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be 0 or between 0 and 1.")

    try:
        from sklearn.model_selection import train_test_split

        indices = list(range(len(targets)))
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=valid_ratio,
            random_state=seed,
            stratify=list(targets),
        )
        return sorted(train_idx), sorted(valid_idx)
    except Exception:
        # Fall back to a local stratified split when scikit-learn is unavailable.
        grouped: dict[int, list[int]] = {}
        for index, target in enumerate(targets):
            grouped.setdefault(int(target), []).append(index)

        rng = random.Random(seed)
        train_idx: list[int] = []
        valid_idx: list[int] = []
        for _, class_indices in grouped.items():
            rng.shuffle(class_indices)
            valid_count = max(1, round(len(class_indices) * valid_ratio))
            valid_idx.extend(class_indices[:valid_count])
            train_idx.extend(class_indices[valid_count:])
        return sorted(train_idx), sorted(valid_idx)


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    valid_ratio: float = 0.2,
    augmentation: str = "weak",
    input_size: int = 96,
    seed: int = 42,
    num_workers: int = 4,
    output_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, int], dict[str, list[int]]]:
    """Create train/validation loaders and optionally write split metadata."""
    data_dir = Path(data_dir)
    train_root = data_dir / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_root}")

    train_transform = build_train_transform(augmentation=augmentation, input_size=input_size)
    eval_transform = build_eval_transform(input_size=input_size)

    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    valid_dataset = datasets.ImageFolder(train_root, transform=eval_transform)
    if train_dataset.class_to_idx != valid_dataset.class_to_idx:
        raise RuntimeError("Train and validation class mappings differ.")

    # Use two dataset objects over the same files so validation keeps deterministic transforms.
    targets = [sample[1] for sample in train_dataset.samples]
    train_idx, valid_idx = stratified_train_valid_indices(targets, valid_ratio, seed)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    valid_loader = None
    if valid_idx:
        valid_loader = DataLoader(
            Subset(valid_dataset, valid_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    split_info = {"train_indices": train_idx, "valid_indices": valid_idx}
    if output_dir is not None:
        # Persisting indices lets evaluation utilities reproduce the original split later.
        out = ensure_dir(output_dir)
        save_json(train_dataset.class_to_idx, out / "class_to_idx.json")
        save_json(
            {
                "seed": seed,
                "valid_ratio": valid_ratio,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                **split_info,
            },
            out / "split_indices.json",
        )

    return train_loader, valid_loader, train_dataset.class_to_idx, split_info


def build_test_loader(
    data_dir: str | Path,
    class_to_idx: dict[str, int],
    batch_size: int = 64,
    input_size: int = 96,
    num_workers: int = 4,
) -> tuple[DataLoader, datasets.ImageFolder]:
    """Build the test loader and verify it matches the checkpoint class mapping."""
    data_dir = Path(data_dir)
    test_root = data_dir / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    dataset = datasets.ImageFolder(test_root, transform=build_eval_transform(input_size=input_size))
    if dataset.class_to_idx != class_to_idx:
        raise RuntimeError(
            f"Test class mapping differs from checkpoint mapping: {dataset.class_to_idx} != {class_to_idx}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, dataset
