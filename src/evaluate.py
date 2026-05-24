"""Evaluate a saved checkpoint and export test metrics, predictions, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import build_test_loader
from src.models import create_model_from_checkpoint_kwargs
from src.utils import class_names_from_mapping, ensure_dir, get_device, save_json, torch_load, worker_count


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Render a labeled confusion matrix as a PNG for quick error inspection."""
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Switch text color at half the max cell count so labels remain readable.
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Small ResNet on STL-10 test data.")
    parser.add_argument("--data-dir", type=Path, default=Path("STL10"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=worker_count())
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)

    # The checkpoint carries both class order and model constructor kwargs.
    checkpoint = torch_load(args.checkpoint, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    class_names = class_names_from_mapping(class_to_idx)
    model_kwargs = checkpoint.get("model_kwargs", {"model_name": "small_resnet18", "num_classes": len(class_names)})
    model = create_model_from_checkpoint_kwargs(model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader, dataset = build_test_loader(
        data_dir=args.data_dir,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    prediction_rows: list[dict[str, object]] = []
    sample_paths = [sample[0] for sample in dataset.samples]
    seen = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            batch_conf, batch_pred = probabilities.max(dim=1)

            targets_list = targets.tolist()
            preds_list = batch_pred.cpu().tolist()
            conf_list = batch_conf.cpu().tolist()
            # DataLoader preserves dataset order because shuffle=False, so slice paths by offset.
            paths = sample_paths[seen : seen + len(targets_list)]
            seen += len(targets_list)

            y_true.extend(targets_list)
            y_pred.extend(preds_list)
            confidences.extend(conf_list)

            for path, true_idx, pred_idx, confidence in zip(paths, targets_list, preds_list, conf_list):
                prediction_rows.append(
                    {
                        "path": path,
                        "true": class_names[true_idx],
                        "pred": class_names[pred_idx],
                        "confidence": f"{confidence:.6f}",
                        "correct": int(true_idx == pred_idx),
                    }
                )

    accuracy = accuracy_score(y_true, y_pred)
    # Export both human-readable text and structured JSON/CSV for reports or notebooks.
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_json({"test_accuracy": accuracy, "checkpoint": str(args.checkpoint)}, output_dir / "metrics.json")
    plot_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png")
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "true", "pred", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"test_accuracy={accuracy:.6f}")
    print(report_text)


if __name__ == "__main__":
    main()
