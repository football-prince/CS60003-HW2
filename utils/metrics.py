from __future__ import annotations

import torch


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a percentage for one batch."""
    if targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return 100.0 * correct / targets.size(0)


class AverageMeter:
    """Track a running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

