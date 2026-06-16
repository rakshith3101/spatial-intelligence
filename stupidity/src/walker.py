from __future__ import annotations

import random

import networkx as nx


def random_walk(graph: nx.Graph, start_node: str | None = None, steps: int = 4) -> list[str]:
    if graph.number_of_nodes() == 0:
        return []

    current = start_node or random.choice(list(graph.nodes))
    path = [current]

    for _ in range(max(steps - 1, 0)):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        weights = [graph[current][neighbor].get("weight", 1.0) for neighbor in neighbors]
        current = random.choices(neighbors, weights=weights, k=1)[0]
        path.append(current)

    return path
