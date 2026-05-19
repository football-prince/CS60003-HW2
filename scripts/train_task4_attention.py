from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import build_model
from utils.argparse_utils import add_common_train_args, namespace_to_config, str2bool
from utils.dataset import NUM_CLASSES, build_dataloaders
from utils.logger import print_config, resolve_output_dir
from utils.train_utils import build_optimizer, run_training, select_device, set_seed


def default_task4_dir(args) -> Path:
    if args.model in {"vit_tiny", "swin_t"}:
        return PROJECT_ROOT / "outputs" / f"task4_{args.model}"
    return PROJECT_ROOT / "outputs" / f"task4_{args.model}_{args.attention}"


def main() -> None:
    parser = ArgumentParser(description="Task 4: Attention modules and Transformers.")
    add_common_train_args(parser)
    parser.add_argument("--attention", type=str, default="se", choices=["none", "se", "cbam"])
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    valid_models = {"resnet18", "resnet34", "vit_tiny", "swin_t"}
    if args.model not in valid_models:
        raise ValueError(f"Task 4 supports --model in {sorted(valid_models)}.")
    if args.model in {"vit_tiny", "swin_t"}:
        args.attention = "none"
    elif args.attention == "none":
        print("Task 4 ResNet is running without attention because --attention none was set.")

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = resolve_output_dir(args.output_dir or default_task4_dir(args), args.overwrite)

    model = build_model(
        args.model,
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        attention=args.attention,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        color_jitter=args.color_jitter,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    if args.model in {"vit_tiny", "swin_t"}:
        optimizer = build_optimizer(
            model,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            lr=args.lr,
            momentum=args.momentum,
        )
    else:
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
            "task": "task4_attention",
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

