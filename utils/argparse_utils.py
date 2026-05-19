from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
from typing import Any


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ArgumentTypeError(f"Expected a boolean value, got {value}")


def add_common_train_args(parser: ArgumentParser) -> None:
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--color_jitter", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)


def namespace_to_config(args: Any) -> dict[str, Any]:
    return vars(args).copy()
