"""Run a batch of configured training experiments and summarize their results."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.summarize_experiments import summarize, write_summary
from src.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured STL-10 experiments sequentially.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("STL10"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for every experiment.")
    parser.add_argument("--device", type=str, default=None, help="Override device for every experiment.")
    parser.add_argument("--random-seed", action="store_true", help="Override every experiment to use a fresh random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiments = config["experiments"]
    ensure_dir(args.outputs_dir)

    for experiment in experiments:
        name = experiment["name"]
        output_dir = args.outputs_dir / name
        # Launch training as a subprocess so every experiment uses the same CLI path.
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            name,
        ]
        merged_args = dict(experiment.get("args", {}))
        if args.epochs is not None:
            merged_args["epochs"] = args.epochs
        if args.device is not None:
            merged_args["device"] = args.device
        if args.random_seed:
            merged_args["random_seed"] = True

        # Convert JSON config keys like "batch_size" into CLI flags like "--batch-size".
        for key, value in merged_args.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])

        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    rows = summarize(args.outputs_dir)
    write_summary(rows, args.outputs_dir)
    if rows:
        # Keep a convenient copy of the overall best checkpoint at the outputs root.
        best = max(rows, key=lambda row: float(row["best_valid_acc"]))
        best_checkpoint = Path(str(best["checkpoint"]))
        destination = args.outputs_dir / "best_model.pth"
        shutil.copy2(best_checkpoint, destination)
        save_json(best, args.outputs_dir / "best_experiment.json")
        print(f"best_experiment={best['experiment']} best_valid_acc={best['best_valid_acc']:.6f}")
        print(f"best_checkpoint={destination}")


if __name__ == "__main__":
    main()
