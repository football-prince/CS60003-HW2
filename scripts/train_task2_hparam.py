from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import build_model
from utils.argparse_utils import add_common_train_args, namespace_to_config
from utils.dataset import NUM_CLASSES, build_dataloaders
from utils.logger import format_float_for_dir, print_config, resolve_output_dir
from utils.train_utils import build_optimizer, run_training, select_device, set_seed


def default_task2_dir(args) -> Path:
    blr = format_float_for_dir(args.backbone_lr)
    clr = format_float_for_dir(args.classifier_lr)
    name = f"task2_{args.model}_epoch{args.epochs}_blr{blr}_clr{clr}_bs{args.batch_size}"
    return PROJECT_ROOT / "outputs" / name


def main() -> None:
    parser = ArgumentParser(description="Task 2: Hyperparameter analysis.")
    add_common_train_args(parser)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.model not in {"resnet18", "resnet34"}:
        raise ValueError("Task 2 supports --model resnet18 or resnet34.")

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = resolve_output_dir(args.output_dir or default_task2_dir(args), args.overwrite)

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True, attention=None)
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        color_jitter=args.color_jitter,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    optimizer = build_optimizer(
        model,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
        momentum=args.momentum,
    )

    config = namespace_to_config(args)
    config.update(
        {
            "task": "task2_hparam",
            "pretrained": True,
            "attention": "none",
            "device": str(device),
            "output_dir": str(output_dir),
            "num_classes": NUM_CLASSES,
            "pretrained_load_info": getattr(model, "pretrained_load_info", {}),
        }
    )
    print("Current experiment config:")
    print_config(config)

    result = run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        output_dir,
        config,
        device,
        args.epochs,
        optimizer,
    )
    print("Final test result:")
    print_config(result)


if __name__ == "__main__":
    main()

