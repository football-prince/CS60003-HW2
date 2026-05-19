from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_dir(output_dir: str | Path, overwrite: bool = False) -> Path:
    """Create an output directory without silently overwriting old results."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.with_name(f"{output_dir.name}_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def print_config(config: dict[str, Any]) -> None:
    print(json.dumps(config, ensure_ascii=False, indent=2))


def format_float_for_dir(value: float) -> str:
    """Format learning rates compactly, e.g. 1e-4 and 5e-5."""
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        text = f"{value:.0e}"
        text = text.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
        return text
    return f"{value:g}"

