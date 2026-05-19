from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "summary_results.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"


def read_rows() -> list[dict[str, str]]:
    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def label_hparam(row: dict[str, str]) -> str:
    return (
        f"e{row['epochs']}\n"
        f"bs{row['batch_size']}\n"
        f"clr={float(row['classifier_lr']):.0e}"
    )


def plot_hparams(rows: list[dict[str, str]], model: str) -> None:
    subset = [
        row
        for row in rows
        if row["task_name"] == "task2_hparam" and row["model"] == model
    ]
    subset = sorted(
        subset,
        key=lambda row: (
            int(row["epochs"]),
            int(row["batch_size"]),
            float(row["classifier_lr"]),
        ),
    )
    labels = [label_hparam(row) for row in subset]
    test = [f(row, "test_acc") for row in subset]
    val = [f(row, "best_val_acc") for row in subset]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    x = range(len(subset))
    ax.plot(x, val, marker="o", label="Best val acc")
    ax.plot(x, test, marker="s", label="Test acc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Task 2 Hyperparameter Sweep ({model})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"task2_{model}_hparam.png")
    plt.close(fig)


def plot_task4(rows: list[dict[str, str]]) -> None:
    order = [
        "task4_resnet18_se",
        "task4_resnet18_cbam",
        "task4_resnet34_se",
        "task4_resnet34_cbam",
        "task4_vit_tiny",
        "task4_swin_t",
    ]
    lookup = {row["output_folder"]: row for row in rows}
    subset = [lookup[name] for name in order if name in lookup]
    labels = []
    for row in subset:
        if row["model"].startswith("resnet"):
            labels.append(f"{row['model']}\n{row['attention_type']}")
        else:
            labels.append(row["model"].replace("_", "-"))
    test = [f(row, "test_acc") for row in subset]
    val = [f(row, "best_val_acc") for row in subset]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    x = range(len(subset))
    width = 0.38
    ax.bar([i - width / 2 for i in x], val, width=width, label="Best val acc")
    ax.bar([i + width / 2 for i in x], test, width=width, label="Test acc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(75, 96)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task 4 Attention and Transformer Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "task4_attention_transformer_comparison.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    plot_hparams(rows, "resnet18")
    plot_hparams(rows, "resnet34")
    plot_task4(rows)
    print(f"Wrote report figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

