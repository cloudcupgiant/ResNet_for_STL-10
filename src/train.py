"""Training entrypoint and reusable epoch utilities for STL-10 experiments."""

from __future__ import annotations

import argparse
import csv
import secrets
import time
from pathlib import Path

import torch
from torch import nn

from src.dataset import build_dataloaders
from src.models import SUPPORTED_MODELS, create_model
from src.plot_curves import plot_curves
from src.utils import (
    count_parameters,
    current_lr,
    ensure_dir,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
    torch_load,
    worker_count,
)


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """Accumulate L1 regularization over trainable parameters only."""
    penalty = None
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.abs().sum()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        raise RuntimeError("No trainable parameters found for L1 penalty.")
    return penalty


def synchronize_if_cuda(device: torch.device) -> None:
    """Make CUDA timings comparable to CPU timings when measuring elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    l1_lambda: float = 0.0,
) -> tuple[float, float, float, float]:
    """Run either a training or evaluation epoch, depending on whether optimizer is set."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    inference_seconds = 0.0

    synchronize_if_cuda(device)
    epoch_start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if not is_train:
                # Time only the forward pass during evaluation; data loading is excluded.
                synchronize_if_cuda(device)
                inference_start = time.perf_counter()
            outputs = model(inputs)
            if not is_train:
                synchronize_if_cuda(device)
                inference_seconds += time.perf_counter() - inference_start
            loss = criterion(outputs, targets)

            if is_train and l1_lambda > 0.0:
                loss = loss + l1_lambda * l1_penalty(model)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        # Accuracy is measured against the batch labels for a simple progress signal.
        total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

    synchronize_if_cuda(device)
    epoch_seconds = time.perf_counter() - epoch_start
    return total_loss / total_samples, total_correct / total_samples, epoch_seconds, inference_seconds


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    """Create the optimizer requested by command-line arguments."""
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def wrap_with_warmup(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Prepend a linear warmup phase to schedulers that support epoch stepping."""
    if args.warmup_epochs == 0:
        return scheduler
    if args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be smaller than --epochs")
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, scheduler],
        milestones=[args.warmup_epochs],
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    """Build the selected learning-rate schedule and validate incompatible options."""
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.eta_min < 0.0:
        raise ValueError("--eta-min must be non-negative")
    if args.scheduler == "none":
        if args.warmup_epochs > 0:
            raise ValueError("--warmup-epochs requires a scheduler")
        return None
    if args.scheduler == "cosine":
        t_max = args.epochs - args.warmup_epochs if args.warmup_epochs > 0 else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.eta_min)
        return wrap_with_warmup(args, optimizer, scheduler)
    if args.scheduler in {"warm_restart", "cosine_restart"}:
        if args.restart_t0 <= 0:
            raise ValueError("--restart-t0 must be positive")
        if args.restart_t_mult < 1:
            raise ValueError("--restart-t-mult must be at least 1")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_t0,
            T_mult=args.restart_t_mult,
            eta_min=args.eta_min,
        )
        return wrap_with_warmup(args, optimizer, scheduler)
    if args.scheduler == "linear":
        if args.warmup_epochs >= args.epochs:
            raise ValueError("--warmup-epochs must be smaller than --epochs")

        def lr_lambda(step: int) -> float:
            # LambdaLR receives the scheduler step count, which maps to epochs in this loop.
            if args.warmup_epochs > 0 and step <= args.warmup_epochs:
                warmup_progress = step / float(args.warmup_epochs)
                return args.warmup_start_factor + (1.0 - args.warmup_start_factor) * warmup_progress
            decay_steps = max(1, args.epochs - args.warmup_epochs)
            decay_progress = (step - args.warmup_epochs) / float(decay_steps)
            return max(0.0, 1.0 - decay_progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        return wrap_with_warmup(args, optimizer, scheduler)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classifier on STL-10 ImageFolder data.")
    parser.add_argument("--data-dir", type=Path, default=Path("STL10"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exp0_baseline"))
    parser.add_argument("--experiment-name", type=str, default="exp0_baseline")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="small_resnet18")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--augmentation", choices=["weak", "strong"], default="weak")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "step", "linear", "warm_restart", "cosine_restart"],
        default="cosine",
    )
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--restart-t0", type=int, default=30)
    parser.add_argument("--restart-t-mult", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--l1-lambda", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--se-activation", choices=["sigmoid", "relu", "tanh"], default="sigmoid")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", action="store_true", help="Generate a fresh random seed for this run.")
    parser.add_argument("--num-workers", type=int, default=worker_count())
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint and continue logging.")
    return parser.parse_args()


def resolve_seed(args: argparse.Namespace) -> int:
    """Resolve a fixed or freshly generated seed before all randomness is initialized."""
    if args.random_seed:
        return secrets.randbelow(2**31 - 1)
    return int(args.seed)


def main() -> None:
    args = parse_args()
    args.seed = resolve_seed(args)
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)

    train_loader, valid_loader, class_to_idx, split_info = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        augmentation=args.augmentation,
        input_size=args.input_size,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=output_dir,
    )

    model_kwargs = {
        "model_name": args.model,
        "num_classes": len(class_to_idx),
        "dropout": args.dropout,
        "activation": args.activation,
        "se_activation": args.se_activation,
        "base_channels": args.base_channels,
    }
    model = create_model(args.model, **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    # Persist the exact runtime configuration alongside checkpoints for reproducibility.
    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(output_dir)
    config["resume"] = str(args.resume) if args.resume else None
    config["device"] = str(device)
    config["num_parameters"] = count_parameters(model)
    config["train_size"] = len(split_info["train_indices"])
    config["valid_size"] = len(split_info["valid_indices"])
    config["monitor_split"] = "valid" if valid_loader is not None else "train"
    save_json(config, output_dir / "config.json")

    log_path = output_dir / "train_log.csv"
    best_valid_acc = -1.0
    best_valid_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    if args.resume is not None:
        # Resume model state first, then optional optimizer/scheduler states when present.
        checkpoint = torch_load(args.resume, map_location=device)
        checkpoint_classes = checkpoint.get("class_to_idx")
        if checkpoint_classes is not None and checkpoint_classes != class_to_idx:
            raise RuntimeError(f"Checkpoint class mapping differs from current data: {checkpoint_classes} != {class_to_idx}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_valid_acc = float(checkpoint.get("best_valid_acc", -1.0))
        best_epoch = int(checkpoint["epoch"])
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as existing_log:
                rows = list(csv.DictReader(existing_log))
            if rows:
                best_row = max(rows, key=lambda row: float(row["valid_acc"]))
                best_epoch = int(best_row["epoch"])
                best_valid_acc = float(best_row["valid_acc"])
                best_valid_loss = float(best_row["valid_loss"])
        print(
            f"[{args.experiment_name}] resumed from {args.resume} "
            f"at epoch {start_epoch} best_valid_acc={best_valid_acc:.4f}",
            flush=True,
            )

    append_log = args.resume is not None and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_acc",
            "valid_loss",
            "valid_acc",
            "lr",
            "train_seconds",
            "valid_seconds",
            "valid_infer_seconds",
            "valid_infer_ms_per_sample",
            "epoch_seconds",
        ]
        if append_log:
            # Refuse to append to logs written by older versions with different columns.
            with log_path.open("r", encoding="utf-8") as existing_log:
                existing_fieldnames = csv.DictReader(existing_log).fieldnames
            if existing_fieldnames != fieldnames:
                raise RuntimeError(
                    f"Cannot resume {log_path} because its CSV header does not match current timing-aware format."
                )
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        if not append_log:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            lr_value = current_lr(optimizer)
            train_loss, train_acc, train_seconds, _ = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                l1_lambda=args.l1_lambda,
            )
            if valid_loader is None:
                # Full-train runs intentionally monitor the training split when no validation split exists.
                valid_loss = train_loss
                valid_acc = train_acc
                valid_seconds = 0.0
                valid_infer_seconds = 0.0
                valid_infer_ms_per_sample = 0.0
            else:
                valid_loss, valid_acc, valid_seconds, valid_infer_seconds = run_one_epoch(
                    model,
                    valid_loader,
                    criterion,
                    device,
                    optimizer=None,
                )
                valid_infer_ms_per_sample = valid_infer_seconds * 1000.0 / len(valid_loader.dataset)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "valid_loss": f"{valid_loss:.6f}",
                    "valid_acc": f"{valid_acc:.6f}",
                    "lr": f"{lr_value:.8f}",
                    "train_seconds": f"{train_seconds:.6f}",
                    "valid_seconds": f"{valid_seconds:.6f}",
                    "valid_infer_seconds": f"{valid_infer_seconds:.6f}",
                    "valid_infer_ms_per_sample": f"{valid_infer_ms_per_sample:.6f}",
                    "epoch_seconds": f"{train_seconds + valid_seconds:.6f}",
                }
            )
            handle.flush()

            if valid_acc > best_valid_acc:
                # The "best" checkpoint is selected by validation accuracy, with loss kept for reporting.
                best_valid_acc = valid_acc
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_checkpoint(
                    output_dir / "best_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_valid_acc,
                    config,
                    class_to_idx,
                    model_kwargs,
                )

            save_checkpoint(
                output_dir / "last_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_valid_acc,
                config,
                class_to_idx,
                model_kwargs,
            )

            if scheduler is not None:
                scheduler.step()

            print(
                f"[{args.experiment_name}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} "
                f"train_time={train_seconds:.1f}s valid_infer={valid_infer_ms_per_sample:.3f}ms/img "
                f"lr={lr_value:.6f} best_valid_acc={best_valid_acc:.4f}@{best_epoch}",
                flush=True,
            )

    plot_curves(log_path, output_dir, args.experiment_name)
    save_json(
        {"best_epoch": best_epoch, "best_valid_acc": best_valid_acc, "best_valid_loss": best_valid_loss},
        output_dir / "best_metrics.json",
    )


if __name__ == "__main__":
    main()
