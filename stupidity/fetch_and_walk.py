"""Fetch content from URLs and run semantic walks."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from canopy_detection.src.embed import generate_embeddings
from canopy_detection.src.fetcher import extract_sentences, fetch_and_extract
from canopy_detection.src.graph_builder import build_graph, connect_similar_nodes, extract_edges, save_graph
from canopy_detection.src.ingest import create_nodes
from canopy_detection.src.loop import run_loop
from canopy_detection.src.mutator import generate_hypothesis, score_hypothesis
from canopy_detection.src.utils import PROCESSED_DIR, ensure_data_dirs, next_id, read_json, write_json
from canopy_detection.src.visualizer import generate_report
from canopy_detection.src.walker import random_walk


def fetch_url_content(url: str, source_name: str | None = None) -> list[dict] | None:
    """Fetch and process content from a URL."""
    print(f"Fetching {url}...")
    content = fetch_and_extract(url)

    if not content:
        print(f"Failed to fetch {url}")
        return None

    sentences = extract_sentences(content)
    if not sentences:
        print(f"No sentences extracted from {url}")
        return None

    source = source_name or url.split("/")[2]
    nodes = create_nodes(sentences, source=source)

    print(f"Extracted {len(nodes)} sentences from {url}")
    return nodes


def merge_nodes(existing_nodes: list[dict], new_nodes: list[dict]) -> list[dict]:
    """Merge new nodes, updating IDs to avoid conflicts."""
    if not existing_nodes:
        return new_nodes

    max_id = max([int(n["id"][1:]) for n in existing_nodes if n["id"][0] == "w"], default=0)

    for i, node in enumerate(new_nodes, start=max_id + 1):
        node["id"] = f"w{i}"

    return existing_nodes + new_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch web content and run semantic mutations.")
    parser.add_argument("--urls", nargs="+", help="URLs to fetch content from")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations.")
    parser.add_argument("--walk-steps", type=int, default=4, help="Steps per walk.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Similarity threshold.")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations.")

    args = parser.parse_args()

    if not args.urls:
        print("Please provide URLs using --urls")
        return

    ensure_data_dirs()

    # Load existing nodes or create empty list
    existing_nodes = read_json(PROCESSED_DIR / "nodes.json", [])
    all_nodes = existing_nodes.copy()

    # Fetch and merge URLs
    for url in args.urls:
        url_nodes = fetch_url_content(url)
        if url_nodes:
            all_nodes = merge_nodes(all_nodes, url_nodes)

    if not all_nodes:
        print("No nodes to process")
        return

    # Generate embeddings for new nodes
    print("Generating embeddings...")
    generate_embeddings(all_nodes)
    write_json(PROCESSED_DIR / "nodes.json", all_nodes)

    # Build graph
    print("Building graph...")
    graph = build_graph(all_nodes, threshold=args.threshold)

    # Run mutations
    print("Running semantic mutations...")
    generated = run_loop(graph, iterations=args.iterations, walk_steps=args.walk_steps)

    # Print results
    print(f"\n=== Results ===")
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")
    print(f"Generated hypotheses: {len(generated)}\n")

    for i, record in enumerate(sorted(generated, key=lambda x: x.get("score", 0), reverse=True)[:5], 1):
        print(f"{i}. [{record.get('score', 0):.4f}] {record.get('generated_text', '')}")
        print(f"   Path: {' → '.join(record.get('path', []))}\n")

    # Visualize if requested
    if args.visualize:
        print("Generating visualizations...")
        generate_report(graph, generated)


if __name__ == "__main__":
    main()
