from __future__ import annotations

import argparse

from src.embed import generate_embeddings
from src.graph_builder import build_graph, save_graph
from src.ingest import create_nodes, load_text, save_nodes, split_sentences
from src.loop import run_loop
from src.utils import ensure_data_dirs
from src.visualizer import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the semantic mutation engine.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of recursive mutations to generate.")
    parser.add_argument("--walk-steps", type=int, default=4, help="Number of nodes to visit in each random walk.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Semantic similarity threshold for graph edges.")
    args = parser.parse_args()

    ensure_data_dirs()
    sentences = split_sentences(load_text())
    nodes = create_nodes(sentences)
    generate_embeddings(nodes)
    save_nodes(nodes)

    graph = build_graph(nodes, threshold=args.threshold)
    save_graph(graph)

    generated = run_loop(graph, iterations=args.iterations, walk_steps=args.walk_steps)
    print(f"Built {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Generated {len(generated)} total ideas.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_report(graph, generated)


if __name__ == "__main__":
    main()
