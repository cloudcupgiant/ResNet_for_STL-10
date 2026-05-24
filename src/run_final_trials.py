"""Run final full-training trials, evaluate test accuracy, and export Grad-CAM samples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final full-train trials, evaluate on test, and generate Grad-CAM samples."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/final_full_train_3seeds.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("STL10"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/final_full_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/final_test_full"))
    parser.add_argument("--gradcam-dir", type=Path, default=Path("outputs/gradcam/final_full_train"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-wrong", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser.parse_args()


def append_args(command: list[str], values: dict[str, object]) -> None:
    """Append argparse-style flags from a dictionary of experiment options."""
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a subprocess while capturing stdout and stderr in one log file."""
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def train_is_complete(output_dir: Path, epochs: int) -> bool:
    """Decide whether a training directory already contains the requested final epoch."""
    log_path = output_dir / "train_log.csv"
    checkpoint_path = output_dir / "last_model.pth"
    if not log_path.exists() or not checkpoint_path.exists():
        return False
    with log_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and int(rows[-1]["epoch"]) >= epochs


def read_train_config(output_dir: Path) -> dict[str, object]:
    """Load the config persisted by src.train."""
    return json.loads((output_dir / "config.json").read_text(encoding="utf-8"))


def read_test_accuracy(eval_output_dir: Path) -> float:
    """Read the scalar test accuracy produced by src.evaluate."""
    metrics = json.loads((eval_output_dir / "metrics.json").read_text(encoding="utf-8"))
    return float(metrics["test_accuracy"])


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write per-run and per-model aggregate final-test summaries."""
    ensure_dir(output_dir)
    csv_path = output_dir / "final_test_summary.csv"
    fieldnames = [
        "model",
        "run",
        "experiment",
        "seed",
        "optimizer",
        "lr",
        "scheduler",
        "test_accuracy",
        "checkpoint",
        "eval_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    aggregate_rows = []
    for model, model_rows in sorted(grouped.items()):
        accuracies = [float(row["test_accuracy"]) for row in model_rows]
        best = max(model_rows, key=lambda row: float(row["test_accuracy"]))
        # Report mean/std across seeds and keep the best checkpoint for visualization.
        aggregate_rows.append(
            {
                "model": model,
                "runs": len(model_rows),
                "mean_test_accuracy": statistics.mean(accuracies),
                "std_test_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "best_test_accuracy": float(best["test_accuracy"]),
                "best_experiment": best["experiment"],
                "best_checkpoint": best["checkpoint"],
                "best_eval_dir": best["eval_dir"],
            }
        )

    save_json({"runs": rows, "aggregate": aggregate_rows}, output_dir / "final_test_summary.json")

    md_lines = [
        "| 模型 | runs | mean test acc | std | best test acc | best run |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        md_lines.append(
            "| {model} | {runs} | {mean_test_accuracy:.6f} | {std_test_accuracy:.6f} | "
            "{best_test_accuracy:.6f} | {best_experiment} |".format(**row)
        )
    md_lines.extend(
        [
            "",
            "| 模型 | run | seed | test acc | checkpoint |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['run']} | {row['seed']} | "
            f"{float(row['test_accuracy']):.6f} | {row['checkpoint']} |"
        )
    (output_dir / "final_test_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def best_rows_by_model(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select the highest test-accuracy run for each model family."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    return [max(model_rows, key=lambda row: float(row["test_accuracy"])) for model_rows in grouped.values()]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiments = config["experiments"]
    ensure_dir(args.outputs_dir)
    ensure_dir(args.eval_dir)
    ensure_dir(args.gradcam_dir)

    summary_rows: list[dict[str, object]] = []
    for run_index, experiment in enumerate(experiments, start=1):
        name = experiment["name"]
        train_output_dir = args.outputs_dir / name
        merged_args = dict(experiment.get("args", {}))
        # Command-line overrides apply uniformly across every configured final trial.
        if args.epochs is not None:
            merged_args["epochs"] = args.epochs
        if args.num_workers is not None:
            merged_args["num_workers"] = args.num_workers
        merged_args["device"] = args.device
        epochs = int(merged_args.get("epochs", 150))

        if not args.skip_train and not train_is_complete(train_output_dir, epochs):
            command = [
                sys.executable,
                "-m",
                "src.train",
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(train_output_dir),
                "--experiment-name",
                name,
            ]
            append_args(command, merged_args)
            print(f"training_start name={name}", flush=True)
            run_logged(command, train_output_dir / "train_command.log")
            print(f"training_done name={name}", flush=True)
        else:
            print(f"training_skip name={name}", flush=True)

        train_config = read_train_config(train_output_dir)
        checkpoint = train_output_dir / "last_model.pth"
        eval_output_dir = args.eval_dir / name
        if not args.skip_eval:
            # Evaluate the last full-train checkpoint because there is no validation split to pick "best".
            eval_command = [
                sys.executable,
                "-m",
                "src.evaluate",
                "--data-dir",
                str(args.data_dir),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(eval_output_dir),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            if args.num_workers is not None:
                eval_command.extend(["--num-workers", str(args.num_workers)])
            print(f"eval_start name={name}", flush=True)
            run_logged(eval_command, eval_output_dir / "evaluate_command.log")
            print(f"eval_done name={name} test_accuracy={read_test_accuracy(eval_output_dir):.6f}", flush=True)

        summary_rows.append(
            {
                "model": train_config["model"],
                "run": ((run_index - 1) % 3) + 1,
                "experiment": name,
                "seed": train_config["seed"],
                "optimizer": train_config["optimizer"],
                "lr": train_config["lr"],
                "scheduler": train_config["scheduler"],
                "test_accuracy": read_test_accuracy(eval_output_dir),
                "checkpoint": str(checkpoint),
                "eval_dir": str(eval_output_dir),
            }
        )

    write_summary(summary_rows, args.eval_dir)

    if not args.skip_gradcam:
        for row in best_rows_by_model(summary_rows):
            gradcam_output_dir = args.gradcam_dir / str(row["experiment"])
            if gradcam_output_dir.exists():
                shutil.rmtree(gradcam_output_dir)
            # Generate explanations only for the best seed of each model to keep output volume manageable.
            command = [
                sys.executable,
                "-m",
                "src.gradcam",
                "--data-dir",
                str(args.data_dir),
                "--checkpoint",
                str(row["checkpoint"]),
                "--output-dir",
                str(gradcam_output_dir),
                "--split",
                "test",
                "--target-mode",
                "both",
                "--max-wrong",
                str(args.max_wrong),
                "--save-raw-heatmap",
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            print(f"gradcam_start model={row['model']} name={row['experiment']}", flush=True)
            run_logged(command, gradcam_output_dir / "gradcam_command.log")
            print(f"gradcam_done model={row['model']} output_dir={gradcam_output_dir}", flush=True)

    print(f"summary={args.eval_dir / 'final_test_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
