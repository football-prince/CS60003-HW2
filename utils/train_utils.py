from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import save_json
from utils.metrics import AverageMeter, accuracy_top1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_classifier_module(model: nn.Module) -> nn.Module | None:
    for name in ("fc", "head", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            return module
    return None


def split_backbone_classifier_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = get_classifier_module(model)
    if classifier is None:
        return list(model.parameters()), []

    classifier_ids = {id(param) for param in classifier.parameters() if param.requires_grad}
    classifier_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in classifier_ids:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, classifier_params


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    backbone_lr: float | None = None,
    classifier_lr: float | None = None,
    lr: float | None = None,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if lr is not None:
        param_groups: Iterable[Any] = model.parameters()
    else:
        if backbone_lr is None or classifier_lr is None:
            raise ValueError("backbone_lr and classifier_lr are required when lr is not set.")
        backbone_params, classifier_params = split_backbone_classifier_params(model)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": classifier_lr})
        if not param_groups:
            raise ValueError("No trainable parameters were found.")

    base_lr = lr if lr is not None else classifier_lr

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_string(optimizer: torch.optim.Optimizer) -> str:
    return ";".join(f"{group['lr']:.6g}" for group in optimizer.param_groups)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        acc = accuracy_top1(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            acc=f"{acc_meter.avg:.2f}",
            lr=get_lr_string(optimizer),
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        acc = accuracy_top1(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.2f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    return evaluate(model, loader, criterion, device, desc="Test")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
        },
        path,
    )


def save_train_log(log_rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not log_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)


def plot_curves(log_rows: list[dict[str, Any]], path: str | Path) -> None:
    if not log_rows:
        return
    df = pd.DataFrame(log_rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    log_rows: list[dict[str, Any]] = []
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_metrics = evaluate(model, val_loader, criterion, device, desc=f"Val {epoch}/{epochs}")

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "learning_rate": get_lr_string(optimizer),
        }
        log_rows.append(row)
        save_train_log(log_rows, output_dir / "train_log.csv")
        plot_curves(log_rows, output_dir / "accuracy_loss_curve.png")

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}%"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                epoch,
                best_val_acc,
                config,
            )

    save_checkpoint(
        output_dir / "last_model.pth",
        model,
        optimizer,
        epochs,
        best_val_acc,
        config,
    )

    try:
        checkpoint = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = test(model, test_loader, criterion, device)

    result = {
        "model_name": config.get("model"),
        "attention": config.get("attention"),
        "pretrained": config.get("pretrained"),
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "hyperparameters": config,
    }
    save_json(result, output_dir / "test_result.json")
    return result
