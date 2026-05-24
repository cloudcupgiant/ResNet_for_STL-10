"""Summarize experiment output directories into CSV, Markdown, and JSON reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, load_json, save_json


ARCHIVED_MODELS = {"mobilenet_v2"}


def numeric_series(history: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric log column, or an empty series when the column is absent."""
    if column not in history.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(history[column], errors="coerce").dropna()


def column_sum(history: pd.DataFrame, column: str) -> float | str:
    """Sum an optional numeric column, returning an empty string for old logs."""
    values = numeric_series(history, column)
    if values.empty:
        return ""
    return float(values.sum())


def column_mean(history: pd.DataFrame, column: str) -> float | str:
    """Average an optional numeric column, returning an empty string for old logs."""
    values = numeric_series(history, column)
    if values.empty:
        return ""
    return float(values.mean())


def summarize(outputs_dir: str | Path = "outputs") -> list[dict[str, object]]:
    """Collect best/last metrics and configuration values from experiment folders."""
    outputs_dir = Path(outputs_dir)
    rows: list[dict[str, object]] = []
    for log_path in sorted(outputs_dir.glob("exp*/train_log.csv")):
        exp_dir = log_path.parent
        history = pd.read_csv(log_path)
        if history.empty:
            continue
        best_row = history.loc[history["valid_acc"].idxmax()]
        last_row = history.iloc[-1]
        config_path = exp_dir / "config.json"
        config = load_json(config_path) if config_path.exists() else {}
        model_name = config.get("model", config.get("model_name", "small_resnet18"))
        if model_name in ARCHIVED_MODELS:
            continue
        # EfficientNet keeps native SiLU feature activations; only the SE gate may vary.
        activation = "native_silu" if model_name == "efficientnet_b0" else config.get("activation", "")
        se_activation = config.get("se_activation", "") if model_name == "efficientnet_b0" else ""
        rows.append(
            {
                "experiment": exp_dir.name,
                "best_epoch": int(best_row["epoch"]),
                "best_valid_acc": float(best_row["valid_acc"]),
                "best_valid_loss": float(best_row["valid_loss"]),
                "last_train_acc": float(last_row["train_acc"]),
                "last_valid_acc": float(last_row["valid_acc"]),
                "last_valid_loss": float(last_row["valid_loss"]),
                "total_train_seconds": column_sum(history, "train_seconds"),
                "total_valid_seconds": column_sum(history, "valid_seconds"),
                "total_valid_infer_seconds": column_sum(history, "valid_infer_seconds"),
                "avg_epoch_seconds": column_mean(history, "epoch_seconds"),
                "avg_valid_infer_ms_per_sample": column_mean(history, "valid_infer_ms_per_sample"),
                "train_valid_gap": float(last_row["train_acc"]) - float(last_row["valid_acc"]),
                "model": model_name,
                "augmentation": config.get("augmentation", ""),
                "activation": activation,
                "se_activation": se_activation,
                "optimizer": config.get("optimizer", ""),
                "lr": config.get("lr", ""),
                "scheduler": config.get("scheduler", ""),
                "warmup_epochs": config.get("warmup_epochs", 0),
                "warmup_start_factor": config.get("warmup_start_factor", ""),
                "step_size": config.get("step_size", ""),
                "gamma": config.get("gamma", ""),
                "eta_min": config.get("eta_min", ""),
                "restart_t0": config.get("restart_t0", ""),
                "restart_t_mult": config.get("restart_t_mult", ""),
                "seed": config.get("seed", ""),
                "random_seed": config.get("random_seed", False),
                "l1_lambda": config.get("l1_lambda", 0.0),
                "dropout": config.get("dropout", ""),
                "weight_decay": config.get("weight_decay", ""),
                "label_smoothing": config.get("label_smoothing", ""),
                "checkpoint": str(exp_dir / "best_model.pth"),
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]], output_dir: str | Path) -> None:
    """Write the experiment summary in machine-readable and report-friendly forms."""
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("No experiment logs found.\n", encoding="utf-8")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = list(rows[0].keys())
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in header) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = max(rows, key=lambda row: float(row["best_valid_acc"]))
    save_json(best, output_dir / "best_experiment.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize STL-10 experiment logs.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(args.outputs_dir)
    write_summary(rows, args.outputs_dir)
    if rows:
        best = max(rows, key=lambda row: float(row["best_valid_acc"]))
        print(f"best_experiment={best['experiment']} best_valid_acc={best['best_valid_acc']:.6f}")
    else:
        print("No experiment logs found.")


if __name__ == "__main__":
    main()
