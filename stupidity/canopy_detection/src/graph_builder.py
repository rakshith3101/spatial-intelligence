from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from .embed import cosine_similarity
from .utils import GRAPH_DIR, PROCESSED_DIR, write_json


def build_graph(nodes: list[dict], threshold: float = 0.45, top_k: int = 3) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], **node)
    connect_similar_nodes(graph, nodes, threshold=threshold, top_k=top_k)
    return graph


def connect_similar_nodes(
    graph: nx.Graph,
    nodes: list[dict],
    threshold: float = 0.45,
    top_k: int = 3,
) -> list[dict]:
    scored_edges: list[dict] = []
    for index, source in enumerate(nodes):
        for target in nodes[index + 1 :]:
            if not source.get("embedding") or not target.get("embedding"):
                continue
            score = cosine_similarity(source["embedding"], target["embedding"])
            if score >= threshold:
                scored_edges.append(
                    {
                        "source": source["id"],
                        "target": target["id"],
                        "weight": round(score, 4),
                        "relation": "semantic_similarity",
                    }
                )

    by_source: dict[str, list[dict]] = {}
    for edge in scored_edges:
        by_source.setdefault(edge["source"], []).append(edge)
        by_source.setdefault(edge["target"], []).append(edge)

    selected: dict[tuple[str, str], dict] = {}
    for edges in by_source.values():
        for edge in sorted(edges, key=lambda item: item["weight"], reverse=True)[:top_k]:
            key = tuple(sorted((edge["source"], edge["target"])))
            selected[key] = edge

    for edge in selected.values():
        graph.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            relation=edge["relation"],
        )
    return list(selected.values())


def extract_edges(graph: nx.Graph) -> list[dict]:
    return [
        {
            "source": source,
            "target": target,
            "weight": data.get("weight", 1.0),
            "relation": data.get("relation", "semantic_similarity"),
        }
        for source, target, data in graph.edges(data=True)
    ]


def save_graph(graph: nx.Graph, path: str | Path | None = None) -> None:
    output_path = Path(path) if path else GRAPH_DIR / "graph.gpickle"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(graph, file)
    write_json(PROCESSED_DIR / "edges.json", extract_edges(graph))


def load_graph(path: str | Path | None = None) -> nx.Graph:
    graph_path = Path(path) if path else GRAPH_DIR / "graph.gpickle"
    with graph_path.open("rb") as file:
        return pickle.load(file)
