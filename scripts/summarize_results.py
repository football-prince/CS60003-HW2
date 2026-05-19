from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"


COLUMNS = [
    "task_name",
    "model",
    "attention_type",
    "initialization",
    "pretrained",
    "epochs",
    "batch_size",
    "backbone_lr",
    "classifier_lr",
    "lr",
    "optimizer",
    "weight_decay",
    "best_val_acc",
    "test_acc",
    "test_loss",
    "final_train_acc",
    "final_val_acc",
    "final_train_loss",
    "final_val_loss",
    "best_epoch",
    "device",
    "seed",
    "status",
    "output_folder",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_train_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any, digits: int = 2) -> str:
    value = to_float(value)
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_lr(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "--"
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def tex_escape(text: Any) -> str:
    text = "" if text is None else str(text)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def infer_task(folder: str, config: dict[str, Any]) -> str:
    task = config.get("task", "")
    if task:
        return task
    match = re.match(r"(task\d+)", folder)
    return match.group(1) if match else "unknown"


def read_experiment(folder: Path) -> dict[str, Any]:
    config = load_json(folder / "config.json") or {}
    result = load_json(folder / "test_result.json") or {}
    log_rows = load_train_log(folder / "train_log.csv")

    status_parts = []
    if not (folder / "config.json").exists():
        status_parts.append("missing_config")
    if not (folder / "test_result.json").exists():
        status_parts.append("missing_test_result")
    if not (folder / "train_log.csv").exists():
        status_parts.append("missing_train_log")

    final_row = log_rows[-1] if log_rows else {}
    best_epoch = ""
    if log_rows:
        best = max(log_rows, key=lambda row: to_float(row.get("val_acc")) or -1)
        best_epoch = best.get("epoch", "")

    model = config.get("model") or result.get("model_name") or ""
    attention = config.get("attention") or result.get("attention") or "none"
    pretrained = config.get("pretrained", result.get("pretrained", ""))
    if pretrained is True:
        initialization = "pretrained"
    elif pretrained is False:
        initialization = "scratch"
    else:
        initialization = ""

    row = {
        "task_name": infer_task(folder.name, config),
        "model": model,
        "attention_type": attention or "none",
        "initialization": initialization,
        "pretrained": pretrained,
        "epochs": config.get("epochs", ""),
        "batch_size": config.get("batch_size", ""),
        "backbone_lr": config.get("backbone_lr", ""),
        "classifier_lr": config.get("classifier_lr", ""),
        "lr": config.get("lr", ""),
        "optimizer": config.get("optimizer", ""),
        "weight_decay": config.get("weight_decay", ""),
        "best_val_acc": result.get("best_val_acc", ""),
        "test_acc": result.get("test_acc", ""),
        "test_loss": result.get("test_loss", ""),
        "final_train_acc": final_row.get("train_acc", ""),
        "final_val_acc": final_row.get("val_acc", ""),
        "final_train_loss": final_row.get("train_loss", ""),
        "final_val_loss": final_row.get("val_loss", ""),
        "best_epoch": best_epoch,
        "device": config.get("device", ""),
        "seed": config.get("seed", ""),
        "status": "ok" if not status_parts else ";".join(status_parts),
        "output_folder": folder.name,
    }
    return row


def sort_key(row: dict[str, Any]) -> tuple:
    folder = row["output_folder"]
    task_rank = {"task1_baseline": 1, "task2_hparam": 2, "task3_ablation": 3, "task4_attention": 4}
    model_rank = {"resnet18": 1, "resnet34": 2, "vit_tiny": 3, "swin_t": 4}
    return (
        task_rank.get(row["task_name"], 99),
        model_rank.get(row["model"], 99),
        row["attention_type"],
        int(row["epochs"] or 0),
        int(row["batch_size"] or 0),
        str(row["classifier_lr"]),
        folder,
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def latex_table(
    path: Path,
    headers: list[str],
    body_rows: list[list[Any]],
    align: str,
    resize: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for body in body_rows:
        lines.append(" & ".join(tex_escape(v) for v in body) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_by_folder(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    lookup = {row["output_folder"]: row for row in rows}
    return [lookup[name] for name in names if name in lookup]


def make_tables(rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]

    table1_names = ["task1_resnet18", "task1_resnet18_20260517_120412", "task1_resnet34"]
    table1 = rows_by_folder(ok_rows, table1_names)
    latex_table(
        TABLES_DIR / "table1_baseline.tex",
        ["Folder", "Model", "Pretrained", "Epochs", "BS", "Backbone LR", "Classifier LR", "Best Val", "Test"],
        [
            [
                r["output_folder"],
                r["model"],
                r["pretrained"],
                r["epochs"],
                r["batch_size"],
                fmt_lr(r["backbone_lr"]),
                fmt_lr(r["classifier_lr"]),
                fmt_num(r["best_val_acc"]),
                fmt_num(r["test_acc"]),
            ]
            for r in table1
        ],
        "llrrrrrrr",
        resize=True,
    )

    for model in ("resnet18", "resnet34"):
        table = [
            r
            for r in ok_rows
            if r["task_name"] == "task2_hparam" and r["model"] == model
        ]
        table = sorted(
            table,
            key=lambda r: (
                int(r["epochs"]),
                int(r["batch_size"]),
                to_float(r["classifier_lr"]) or 0,
                r["output_folder"],
            ),
        )
        latex_table(
            TABLES_DIR / f"table2_hparam_{model}.tex",
            ["Epochs", "BS", "Backbone LR", "Classifier LR", "Best Val", "Test"],
            [
                [
                    r["epochs"],
                    r["batch_size"],
                    fmt_lr(r["backbone_lr"]),
                    fmt_lr(r["classifier_lr"]),
                    fmt_num(r["best_val_acc"]),
                    fmt_num(r["test_acc"]),
                ]
                for r in table
            ],
            "rrrrrr",
        )

    table4 = rows_by_folder(
        ok_rows,
        [
            "task3_resnet18_pretrained",
            "task3_resnet18_scratch",
            "task3_resnet34_pretrained",
            "task3_resnet34_scratch",
        ],
    )
    latex_table(
        TABLES_DIR / "table4_pretraining.tex",
        ["Model", "Initialization", "Best Val", "Test"],
        [
            [r["model"], r["initialization"], fmt_num(r["best_val_acc"]), fmt_num(r["test_acc"])]
            for r in table4
        ],
        "llrr",
    )

    table5 = rows_by_folder(
        ok_rows,
        [
            "task4_resnet18_se",
            "task4_resnet18_cbam",
            "task4_resnet34_se",
            "task4_resnet34_cbam",
            "task4_vit_tiny",
            "task4_swin_t",
        ],
    )
    latex_table(
        TABLES_DIR / "table5_attention_transformers.tex",
        ["Model", "Attention / Architecture", "Pretrained", "Best Val", "Test"],
        [
            [
                r["model"],
                r["attention_type"] if r["model"].startswith("resnet") else r["model"],
                r["pretrained"],
                fmt_num(r["best_val_acc"]),
                fmt_num(r["test_acc"]),
            ]
            for r in table5
        ],
        "lllrr",
    )

    latex_table(
        TABLES_DIR / "table_appendix_all_results.tex",
        ["Task", "Model", "Attn", "Init", "Epochs", "BS", "BB LR", "Cls LR", "LR", "Best Val", "Test", "Folder"],
        [
            [
                r["task_name"],
                r["model"],
                r["attention_type"],
                r["initialization"],
                r["epochs"],
                r["batch_size"],
                fmt_lr(r["backbone_lr"]),
                fmt_lr(r["classifier_lr"]),
                fmt_lr(r["lr"]),
                fmt_num(r["best_val_acc"]),
                fmt_num(r["test_acc"]),
                r["output_folder"],
            ]
            for r in ok_rows
        ],
        "llllrrrrrrrl",
        resize=True,
    )


def copy_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "task1_resnet18_curve.png": OUTPUTS_DIR / "task1_resnet18" / "accuracy_loss_curve.png",
        "task3_resnet18_pretrained_curve.png": OUTPUTS_DIR / "task3_resnet18_pretrained" / "accuracy_loss_curve.png",
        "task3_resnet18_scratch_curve.png": OUTPUTS_DIR / "task3_resnet18_scratch" / "accuracy_loss_curve.png",
        "task4_resnet18_se_curve.png": OUTPUTS_DIR / "task4_resnet18_se" / "accuracy_loss_curve.png",
    }
    for name, src in figure_map.items():
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / name)


def main() -> None:
    folders = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("task")]
    rows = [read_experiment(folder) for folder in folders]
    rows = sorted(rows, key=sort_key)
    write_csv(rows, OUTPUTS_DIR / "summary_results.csv")
    make_tables(rows)
    copy_figures()

    ok = sum(1 for row in rows if row["status"] == "ok")
    missing = len(rows) - ok
    print(f"Wrote {OUTPUTS_DIR / 'summary_results.csv'}")
    print(f"Experiments summarized: {len(rows)}; ok={ok}; missing={missing}")
    print(f"Wrote LaTeX tables to {TABLES_DIR}")
    print(f"Copied representative figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

