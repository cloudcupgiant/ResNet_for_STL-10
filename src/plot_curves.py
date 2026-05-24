"""Plot loss and accuracy curves from a training log CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def safe_file_stem(value: str) -> str:
    """Sanitize experiment names before using them in output filenames."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem.strip("_") or "experiment"


def resolve_experiment_name(log_csv: Path, output_dir: Path, experiment_name: str | None) -> str:
    """Resolve a stable plot prefix from CLI input, config, or the log directory name."""
    if experiment_name:
        return safe_file_stem(experiment_name)

    config_path = output_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            name = config.get("experiment_name")
            if name:
                return safe_file_stem(str(name))
        except (OSError, json.JSONDecodeError):
            # Plotting should still work even if the optional config file is incomplete.
            pass

    return safe_file_stem(log_csv.parent.name)


def save_curve(output_dir: Path, experiment_name: str, legacy_filename: str) -> None:
    """Save both legacy filenames and experiment-prefixed filenames."""
    plt.savefig(output_dir / legacy_filename, dpi=200)
    plt.savefig(output_dir / f"{experiment_name}_{legacy_filename}", dpi=200)


def plot_curves(
    log_csv: str | Path,
    output_dir: str | Path | None = None,
    experiment_name: str | None = None,
) -> None:
    """Read a training log and save loss/accuracy curve PNGs."""
    log_csv = Path(log_csv)
    if output_dir is None:
        output_dir = log_csv.parent
    output_dir = ensure_dir(output_dir)
    experiment_name = resolve_experiment_name(log_csv, output_dir, experiment_name)

    history = pd.read_csv(log_csv)
    epochs = history["epoch"]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["valid_loss"], label="valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_curve(output_dir, experiment_name, "loss_curve.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_acc"], label="train accuracy")
    plt.plot(epochs, history["valid_acc"], label="valid accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_curve(output_dir, experiment_name, "accuracy_curve.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.csv.")
    parser.add_argument("log_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()
    plot_curves(args.log_csv, args.output_dir, args.experiment_name)


if __name__ == "__main__":
    main()
