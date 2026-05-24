"""Generate Grad-CAM overlays for trained STL-10 classifiers."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.dataset import build_eval_transform, stratified_train_valid_indices
from src.models import create_model_from_checkpoint_kwargs
from src.utils import class_names_from_mapping, ensure_dir, get_device, load_json, torch_load, worker_count


class GradCAM:
    """Minimal Grad-CAM helper that records activations and gradients from one layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # Hooks keep the model untouched while exposing the tensors needed for Grad-CAM.
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        """Store target-layer feature maps from the forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        """Store gradients flowing back through the target layer."""
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        """Remove hooks explicitly so repeated runs do not accumulate callbacks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Return a normalized heatmap for the requested class and the model prediction."""
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        output = self.model(image)
        pred_idx = int(output.argmax(dim=1).item())
        target_idx = pred_idx if class_idx is None else class_idx
        score = output[:, target_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        # Global-average pooled gradients are the channel weights in the Grad-CAM paper.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam_map = F.interpolate(cam_map, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam_map = cam_map[0, 0]
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        return cam_map.cpu().numpy(), pred_idx


def safe_name(value: str) -> str:
    """Make class names safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def overlay_heatmap(image_path: str | Path, heatmap: np.ndarray, output_path: Path, alpha: float = 0.38) -> None:
    """Blend a normalized heatmap over the original image and save the RGB overlay."""
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image).astype(np.float32) / 255.0
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size, resample=Image.BILINEAR)
    heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0
    heatmap_rgb = matplotlib.colormaps["jet"](heatmap_array)[..., :3]
    overlay = np.clip((1.0 - alpha) * image_array + alpha * heatmap_rgb, 0.0, 1.0)
    Image.fromarray(np.uint8(overlay * 255)).save(output_path)


def save_heatmap(heatmap: np.ndarray, output_path: Path) -> None:
    """Save a normalized heatmap as a grayscale PNG."""
    Image.fromarray(np.uint8(255 * heatmap)).save(output_path)


def target_layer_for_model(
    model: torch.nn.Module,
    model_name: str,
    layer_name: str = "auto",
) -> tuple[str, torch.nn.Module]:
    """Resolve the layer used for Grad-CAM, with model-specific defaults."""
    if layer_name != "auto":
        return layer_name, model.get_submodule(layer_name)

    if model_name == "small_resnet18":
        return "layer4.1.conv2", model.layer4[-1].conv2
    if model_name == "shufflenetv2_x1_0":
        return "conv5.0", model.conv5[0]
    if model_name == "efficientnet_b0":
        return "features.8.0", model.features[-1][0]
    raise ValueError(f"No default Grad-CAM target layer for model={model_name!r}.")


def resolve_split_indices(
    dataset: datasets.ImageFolder,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    split: str,
) -> list[int]:
    """Find the sample indices that correspond to train, validation, or test."""
    if split == "test":
        return list(range(len(dataset)))

    config = checkpoint.get("config", {})
    output_dir = Path(str(config.get("output_dir", checkpoint_path.parent))) if isinstance(config, dict) else checkpoint_path.parent
    split_path = output_dir / "split_indices.json"
    if split_path.exists():
        # Prefer the exact split recorded during training, if it is available.
        split_info = load_json(split_path)
        key = f"{split}_indices"
        if key in split_info:
            return [int(index) for index in split_info[key]]

    if not isinstance(config, dict):
        config = {}
    targets = [sample[1] for sample in dataset.samples]
    seed = int(config.get("seed", 42))
    valid_ratio = float(config.get("valid_ratio", 0.2))
    # Reconstruct the split from checkpoint config when the metadata file is missing.
    train_indices, valid_indices = stratified_train_valid_indices(targets, valid_ratio=valid_ratio, seed=seed)
    return valid_indices if split == "valid" else train_indices


def build_visualization_dataset(
    data_dir: Path,
    class_to_idx: dict[str, int],
    input_size: int,
    split: str,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
) -> tuple[datasets.ImageFolder, list[int]]:
    """Load the requested split using deterministic transforms for attribution."""
    root = data_dir / ("test" if split == "test" else "train")
    if not root.is_dir():
        raise FileNotFoundError(f"{split} directory not found: {root}")

    dataset = datasets.ImageFolder(root, transform=build_eval_transform(input_size=input_size))
    if dataset.class_to_idx != class_to_idx:
        raise RuntimeError(f"{split} class mapping differs from checkpoint mapping.")
    return dataset, resolve_split_indices(dataset, checkpoint, checkpoint_path, split)


def predict_dataset(
    model: torch.nn.Module,
    dataset: datasets.ImageFolder,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, object]]:
    """Run predictions for candidate samples and keep enough metadata for selection."""
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    class_names = dataset.classes
    rows: list[dict[str, object]] = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            conf, preds = probabilities.max(dim=1)
            for offset, (true_idx, pred_idx, confidence) in enumerate(
                zip(targets.tolist(), preds.cpu().tolist(), conf.cpu().tolist())
            ):
                # ``Subset`` positions must be translated back to original ImageFolder indices.
                split_position = seen + offset
                sample_index = int(indices[split_position])
                rows.append(
                    {
                        "split_position": split_position,
                        "index": sample_index,
                        "path": dataset.samples[sample_index][0],
                        "true_idx": true_idx,
                        "pred_idx": pred_idx,
                        "true": class_names[true_idx],
                        "pred": class_names[pred_idx],
                        "confidence": confidence,
                        "correct": true_idx == pred_idx,
                    }
                )
            seen += len(targets)
    return rows


def select_samples(rows: list[dict[str, object]], class_names: list[str], max_wrong: int) -> list[dict[str, object]]:
    """Pick one representative image per class, plus the most confident mistakes."""
    selected: list[dict[str, object]] = []
    used_indices: set[int] = set()

    for class_name in class_names:
        # Prefer a correct example for each class, but still show the class if all are wrong.
        correct = next((row for row in rows if row["true"] == class_name and row["correct"]), None)
        fallback = next((row for row in rows if row["true"] == class_name), None)
        chosen = correct or fallback
        if chosen is not None:
            selected.append(chosen)
            used_indices.add(int(chosen["index"]))

    wrong_rows = [row for row in rows if not row["correct"] and int(row["index"]) not in used_indices]
    # High-confidence mistakes are often the most useful explanations to inspect.
    wrong_rows = sorted(wrong_rows, key=lambda row: float(row["confidence"]), reverse=True)
    selected.extend(wrong_rows[:max_wrong])
    return selected


def target_requests(true_idx: int, pred_idx: int, target_mode: str) -> list[tuple[str, int]]:
    """Translate target-mode CLI input into the class indices Grad-CAM should explain."""
    if target_mode == "pred":
        return [("pred", pred_idx)]
    if target_mode == "true":
        return [("true", true_idx)]
    requests = [("pred", pred_idx)]
    if true_idx != pred_idx:
        requests.append(("true", true_idx))
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for trained STL-10 classifiers.")
    parser.add_argument("--data-dir", type=Path, default=Path("STL10"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--split", choices=["valid", "train", "test"], default="valid")
    parser.add_argument("--target-layer", type=str, default="auto")
    parser.add_argument("--target-mode", choices=["pred", "true", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=worker_count())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-wrong", type=int, default=10)
    parser.add_argument("--save-raw-heatmap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)

    checkpoint = torch_load(args.checkpoint, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    class_names = class_names_from_mapping(class_to_idx)
    model_kwargs = checkpoint.get("model_kwargs", {"model_name": "small_resnet18", "num_classes": len(class_names)})
    model_name = str(model_kwargs.get("model_name", "small_resnet18"))
    model = create_model_from_checkpoint_kwargs(model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset, indices = build_visualization_dataset(
        data_dir=args.data_dir,
        class_to_idx=class_to_idx,
        input_size=args.input_size,
        split=args.split,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
    )

    prediction_rows = predict_dataset(model, dataset, indices, device, args.batch_size, args.num_workers)
    selected_rows = select_samples(prediction_rows, class_names, args.max_wrong)

    # Create hooks once and reuse them for every selected image.
    target_layer_name, target_layer = target_layer_for_model(model, model_name, args.target_layer)
    gradcam = GradCAM(model, target_layer)
    index_rows: list[dict[str, object]] = []
    try:
        for row in selected_rows:
            sample_index = int(row["index"])
            tensor, true_idx = dataset[sample_index]
            pred_idx = int(row["pred_idx"])
            tensor = tensor.unsqueeze(0).to(device)
            status = "correct" if true_idx == pred_idx else "wrong"

            for target_type, target_idx in target_requests(true_idx, pred_idx, args.target_mode):
                heatmap, recomputed_pred_idx = gradcam(tensor, class_idx=target_idx)
                target_name = class_names[target_idx]
                # Filename contains enough context to inspect images without opening the CSV index.
                filename = (
                    f"{sample_index:04d}_{status}_true-{safe_name(class_names[true_idx])}"
                    f"_pred-{safe_name(class_names[recomputed_pred_idx])}_{target_type}-{safe_name(target_name)}_gradcam.png"
                )
                output_path = output_dir / filename
                overlay_heatmap(row["path"], heatmap, output_path)

                heatmap_path = ""
                if args.save_raw_heatmap:
                    raw_path = output_path.with_name(output_path.stem + "_raw.png")
                    save_heatmap(heatmap, raw_path)
                    heatmap_path = str(raw_path)

                index_rows.append(
                    {
                        "split": args.split,
                        "split_position": row["split_position"],
                        "index": sample_index,
                        "path": row["path"],
                        "true": class_names[true_idx],
                        "pred": class_names[recomputed_pred_idx],
                        "confidence": f"{float(row['confidence']):.6f}",
                        "correct": int(true_idx == recomputed_pred_idx),
                        "target_type": target_type,
                        "target": target_name,
                        "target_layer": target_layer_name,
                        "output": str(output_path),
                        "raw_heatmap": heatmap_path,
                    }
                )
    finally:
        gradcam.close()

    with (output_dir / "gradcam_index.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "split_position",
            "index",
            "path",
            "true",
            "pred",
            "confidence",
            "correct",
            "target_type",
            "target",
            "target_layer",
            "output",
            "raw_heatmap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(
        f"saved_gradcam_images={len(index_rows)} split={args.split} "
        f"model={model_name} target_layer={target_layer_name} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
