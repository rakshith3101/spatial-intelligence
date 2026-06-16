from __future__ import annotations

import re
from pathlib import Path

from .utils import PROCESSED_DIR, RAW_DIR, ensure_data_dirs, write_json


def load_text(path: str | Path | None = None) -> str:
    ensure_data_dirs()
    input_path = Path(path) if path else RAW_DIR / "input.txt"
    if not input_path.exists():
        input_path.write_text("", encoding="utf-8")
    return input_path.read_text(encoding="utf-8")


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def create_nodes(sentences: list[str] | None = None, source: str = "input.txt") -> list[dict]:
    if sentences is None:
        sentences = split_sentences(load_text())
    return [
        {
            "id": f"w{index}",
            "text": sentence,
            "embedding": [],
            "source": source,
            "created_from": None,
            "type": "original",
        }
        for index, sentence in enumerate(sentences, start=1)
    ]


def save_nodes(nodes: list[dict], path: str | Path | None = None) -> None:
    write_json(Path(path) if path else PROCESSED_DIR / "nodes.json", nodes)


def load_multiple_texts(file_names: list[str] | None = None) -> dict[str, str]:
    """
    Load text from multiple files in the raw data directory.

    Args:
        file_names: List of filenames to load. If None, loads all .txt files.

    Returns:
        Dictionary mapping filename to text content
    """
    ensure_data_dirs()

    if file_names is None:
        file_names = [f.name for f in RAW_DIR.glob("*.txt")]

    texts = {}
    for fname in file_names:
        path = RAW_DIR / fname
        if path.exists():
            texts[fname] = path.read_text(encoding="utf-8")
        else:
            print(f"Warning: {fname} not found in {RAW_DIR}")

    return texts


def create_nodes_from_sources(
    sources: dict[str, str] | None = None,
) -> list[dict]:
    """
    Create nodes from multiple text sources.

    Args:
        sources: Dictionary mapping source name to text content.
                If None, loads from all .txt files in raw directory.

    Returns:
        List of node dictionaries with unique IDs
    """
    if sources is None:
        sources = load_multiple_texts()

    nodes = []
    node_counter = 1

    for source_name, text in sources.items():
        sentences = split_sentences(text)
        for sentence in sentences:
            nodes.append(
                {
                    "id": f"w{node_counter}",
                    "text": sentence,
                    "embedding": [],
                    "source": source_name,
                    "created_from": None,
                    "type": "original",
                }
            )
            node_counter += 1

    return nodes
