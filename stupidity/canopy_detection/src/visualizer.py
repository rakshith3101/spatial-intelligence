"""Visualization utilities for semantic graph exploration."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

from .utils import DATA_DIR


def visualize_graph(
    graph: nx.Graph,
    title: str = "Semantic Graph",
    output_file: str | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """
    Visualize the semantic graph with generated nodes highlighted.

    Args:
        graph: NetworkX graph to visualize
        title: Title for the plot
        output_file: Path to save the figure
        figsize: Figure size (width, height)
    """
    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout for better visualization
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # Separate node types
    original_nodes = [
        node for node in graph.nodes() if graph.nodes[node].get("source") != "generated"
    ]
    generated_nodes = [
        node for node in graph.nodes() if graph.nodes[node].get("source") == "generated"
    ]

    # Draw original nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=original_nodes,
        node_color="#3498db",
        node_size=800,
        label="Original Nodes",
        ax=ax,
    )

    # Draw generated nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=generated_nodes,
        node_color="#e74c3c",
        node_size=1000,
        node_shape="s",
        label="Generated Nodes",
        ax=ax,
    )

    # Draw edges with transparency based on weight
    edges = graph.edges()
    weights = [graph[u][v].get("weight", 0.5) for u, v in edges]
    max_weight = max(weights) if weights else 1.0

    for (u, v), weight in zip(edges, weights):
        alpha = max(0.1, min(weight / max_weight, 1.0))
        nx.draw_networkx_edges(
            graph, pos, [(u, v)], alpha=alpha, width=2 * alpha, ax=ax
        )

    # Draw labels
    labels = {
        node: node
        for node in graph.nodes()
        if graph.nodes[node].get("type") != "generated"
    }
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(scatterpoints=1, loc="upper left")
    ax.axis("off")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Graph saved to {output_file}")
    else:
        plt.show()

    plt.close()


def visualize_walk_chain(
    graph: nx.Graph,
    path: list[str],
    generated_text: str,
    output_file: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Visualize a single walk path and its generated hypothesis.

    Args:
        graph: NetworkX graph
        path: List of node IDs in the walk
        generated_text: Generated hypothesis from the walk
        output_file: Path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # Color all nodes gray except path nodes
    all_nodes = list(graph.nodes())
    gray_nodes = [n for n in all_nodes if n not in path]
    path_nodes = [n for n in path if n in graph.nodes()]

    nx.draw_networkx_nodes(
        graph, pos, nodelist=gray_nodes, node_color="#bdc3c7", node_size=500, ax=ax
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=path_nodes, node_color="#f39c12", node_size=1200, ax=ax
    )

    # Draw all edges lightly
    nx.draw_networkx_edges(graph, pos, alpha=0.1, ax=ax)

    # Highlight path edges
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1) if path[i + 1] in graph.nodes()]
    nx.draw_networkx_edges(
        graph, pos, edgelist=path_edges, edge_color="#e74c3c", width=3, ax=ax
    )

    # Draw labels for path nodes
    path_labels = {node: node for node in path_nodes}
    nx.draw_networkx_labels(graph, pos, path_labels, font_size=10, font_weight="bold", ax=ax)

    # Add generated hypothesis as text
    ax.text(
        0.5,
        -0.05,
        f"Generated: {generated_text}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", edgecolor="#95a5a6"),
        wrap=True,
    )

    ax.set_title(f"Semantic Walk: {' → '.join(path_nodes)}", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Walk visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()


def generate_report(
    graph: nx.Graph,
    generated_records: list[dict],
    output_dir: Path | None = None,
) -> None:
    """
    Generate a comprehensive visualization report.

    Args:
        graph: NetworkX graph
        generated_records: List of generated hypotheses
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = DATA_DIR.parent / "visualizations"

    output_dir.mkdir(exist_ok=True)

    # Main graph visualization
    visualize_graph(
        graph,
        title=f"Semantic Hypothesis Graph\n({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)",
        output_file=str(output_dir / "graph.png"),
    )

    # Top hypothesis walks
    if generated_records:
        sorted_records = sorted(
            generated_records, key=lambda x: x.get("score", 0), reverse=True
        )

        for i, record in enumerate(sorted_records[:5], 1):
            path = record.get("path", [])
            text = record.get("generated_text", "")
            if path and text:
                visualize_walk_chain(
                    graph,
                    path,
                    text,
                    output_file=str(output_dir / f"walk_{i:02d}.png"),
                )

    print(f"\nVisualization report saved to {output_dir}")
