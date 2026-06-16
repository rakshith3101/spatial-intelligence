from __future__ import annotations

import random

import networkx as nx

from .embed import generate_embeddings
from .graph_builder import connect_similar_nodes, extract_edges, save_graph
from .mutator import generate_hypothesis, score_hypothesis
from .utils import PROCESSED_DIR, next_id, read_json, write_json
from .walker import random_walk


def _path_texts(graph: nx.Graph, path: list[str]) -> list[str]:
    return [graph.nodes[node_id]["text"] for node_id in path if node_id in graph.nodes]


def _is_trivial_hypothesis(generated_text: str) -> bool:
    """Filter out trivial/obvious hypotheses."""
    trivial_patterns = [
        "associated with",  # Too weak
        "may be associated",  # Weak signal
    ]
    
    # Reject if it only uses weak association language
    if "associated with" in generated_text and "connect" not in generated_text:
        return True
    
    return False


def _add_generated_node(graph: nx.Graph, path: list[str], generated_text: str) -> dict:
    nodes = [dict(data) for _, data in graph.nodes(data=True)]
    generated_node = {
        "id": next_id(nodes, "g"),
        "text": generated_text,
        "embedding": [],
        "source": "generated",
        "created_from": path,
        "type": "generated",
    }
    generate_embeddings([generated_node])
    graph.add_node(generated_node["id"], **generated_node)

    all_nodes = [dict(data) for _, data in graph.nodes(data=True)]
    connect_similar_nodes(graph, all_nodes, threshold=0.35, top_k=4)
    return generated_node


def run_loop(graph: nx.Graph, iterations: int = 5, walk_steps: int = 4) -> list[dict]:
    generated_records = read_json(PROCESSED_DIR / "generated.json", [])

    for _ in range(iterations):
        if graph.number_of_nodes() == 0:
            break
        start_node = random.choice(list(graph.nodes))
        path = random_walk(graph, start_node=start_node, steps=walk_steps)
        texts = _path_texts(graph, path)
        generated_text = generate_hypothesis(texts)
        
        # Skip trivial hypotheses
        if _is_trivial_hypothesis(generated_text):
            continue
        
        generated_node = _add_generated_node(graph, path, generated_text)
        score = score_hypothesis(texts, generated_text)

        generated_records.append(
            {
                "id": generated_node["id"],
                "path": path,
                "generated_text": generated_text,
                "score": score,
            }
        )

        write_json(PROCESSED_DIR / "nodes.json", [dict(data) for _, data in graph.nodes(data=True)])
        write_json(PROCESSED_DIR / "edges.json", extract_edges(graph))
        write_json(PROCESSED_DIR / "generated.json", generated_records)
        save_graph(graph)

    return generated_records
