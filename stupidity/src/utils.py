from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"


def ensure_data_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, GRAPH_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def next_id(items: list[dict[str, Any]], prefix: str) -> str:
    max_seen = 0
    for item in items:
        item_id = str(item.get("id", ""))
        if item_id.startswith(prefix) and item_id[len(prefix) :].isdigit():
            max_seen = max(max_seen, int(item_id[len(prefix) :]))
    return f"{prefix}{max_seen + 1}"
