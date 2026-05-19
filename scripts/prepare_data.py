from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchvision import datasets

from utils.dataset import NUM_CLASSES, dataset_summary


def split_available(data_dir: Path, split: str) -> bool:
    try:
        datasets.Flowers102(root=str(data_dir), split=split, download=False)
        return True
    except RuntimeError:
        return False


def main() -> None:
    parser = ArgumentParser(description="Prepare and verify the Flowers102 dataset.")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = ("train", "val", "test")
    available = {split: split_available(data_dir, split) for split in splits}
    if not all(available.values()):
        missing = [split for split, ok in available.items() if not ok]
        print(f"Missing split(s): {missing}. Downloading Flowers102 to {data_dir} ...")
        for split in splits:
            datasets.Flowers102(root=str(data_dir), split=split, download=True)
    else:
        print(f"Flowers102 already exists in {data_dir}.")

    summary = dataset_summary(data_dir, download=False)
    print("Dataset summary:")
    for split in splits:
        print(
            f"  {split}: {summary[f'{split}_samples']} samples, "
            f"{summary[f'{split}_classes']} classes"
        )
    print(f"  total classes: {summary.get('num_classes', NUM_CLASSES)}")


if __name__ == "__main__":
    main()

