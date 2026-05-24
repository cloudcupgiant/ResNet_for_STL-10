"""Shared filesystem, reproducibility, checkpoint, and device helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def ensure_dir(path: str | Path) -> Path:
    """Create a directory tree if needed and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs used by training scripts."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cuDNN autotuning enabled for performance; exact bitwise determinism is not required here.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    """Resolve an explicit device string or choose CUDA when it is available."""
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file, creating parent directories first."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate from the first optimizer parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    best_valid_acc: float,
    config: dict[str, Any],
    class_to_idx: dict[str, int],
    model_kwargs: dict[str, Any],
) -> None:
    """Save training state plus metadata needed to recreate the model and data mapping."""
    import torch

    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_valid_acc": best_valid_acc,
            "config": config,
            "class_to_idx": class_to_idx,
            "model_kwargs": model_kwargs,
        },
        path,
    )


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load checkpoints across PyTorch versions with and without ``weights_only``."""
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    """Convert an ImageFolder class-to-index mapping into index-ordered class names."""
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return [idx_to_class[idx] for idx in range(len(idx_to_class))]


def worker_count(default: int = 4) -> int:
    """Cap DataLoader workers by available CPU cores and the project default."""
    cpu_count = os.cpu_count() or default
    return max(0, min(default, cpu_count))
