"""Explore interesting facts from various domains."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from canopy_detection.src.embed import generate_embeddings
from canopy_detection.src.graph_builder import build_graph, save_graph
from canopy_detection.src.ingest import create_nodes_from_sources, load_multiple_texts
from canopy_detection.src.loop import run_loop
from canopy_detection.src.utils import PROCESSED_DIR, ensure_data_dirs, write_json
from canopy_detection.src.visualizer import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore interesting facts and connections from domain sources."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["india_history.txt", "aerospace.txt"],
        help="Source files to load from data/raw/",
    )
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterations.")
    parser.add_argument("--walk-steps", type=int, default=5, help="Steps per walk.")
    parser.add_argument("--threshold", type=float, default=0.42, help="Similarity threshold.")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualizations.")

    args = parser.parse_args()

    ensure_data_dirs()

    # Load text from specified sources
    print(f"Loading sources: {', '.join(args.sources)}")
    sources = load_multiple_texts(args.sources)

    if not sources:
        print("No source files found!")
        return

    # Create nodes from all sources
    print("Creating nodes from all sources...")
    nodes = create_nodes_from_sources(sources)
    print(f"Created {len(nodes)} nodes from {len(sources)} sources")

    # Show sources summary
    print("\nSources loaded:")
    for source_name in sources:
        source_nodes = [n for n in nodes if n["source"] == source_name]
        print(f"  - {source_name}: {len(source_nodes)} nodes")

    # Generate embeddings
    print("\nGenerating semantic embeddings...")
    generate_embeddings(nodes)

    # Save nodes
    write_json(PROCESSED_DIR / "nodes.json", nodes)

    # Build graph
    print(f"Building semantic graph (threshold: {args.threshold})...")
    graph = build_graph(nodes, threshold=args.threshold)
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Run semantic mutations
    print(f"\nRunning {args.iterations} iterations of semantic mutations...")
    generated = run_loop(graph, iterations=args.iterations, walk_steps=args.walk_steps)

    # Print top interesting facts
    print("\n" + "=" * 80)
    print("TOP DISCOVERED CONNECTIONS & INTERESTING FACTS")
    print("=" * 80)

    sorted_generated = sorted(generated, key=lambda x: x.get("score", 0), reverse=True)

    for i, record in enumerate(sorted_generated[:10], 1):
        score = record.get("score", 0)
        text = record.get("generated_text", "")
        path = record.get("path", [])

        # Get source names for each node in path
        path_sources = []
        for node_id in path:
            for node in nodes:
                if node["id"] == node_id:
                    path_sources.append(f"{node_id}({node['source'].replace('.txt', '')})")
                    break

        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {text}")
        print(f"   Path: {' → '.join(path_sources)}")

    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")
    print(f"Generated hypotheses: {len(generated)}")
    print(f"Avg hypothesis score: {sum(r.get('score', 0) for r in generated) / len(generated):.4f}" if generated else "N/A")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        generate_report(graph, generated)
        print("✓ Visualizations saved to visualizations/")


if __name__ == "__main__":
    main()
