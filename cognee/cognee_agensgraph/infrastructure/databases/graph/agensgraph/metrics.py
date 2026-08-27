"""
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Graph metrics, in the shape cognee's other adapters return them."""

import logging
from typing import Any, Dict, Iterable, List, Tuple

from psycopg import sql

logger = logging.getLogger(__name__)

# Always present. The last four are computed only when asked for, and read -1 otherwise.
METRIC_KEYS = (
    "num_nodes",
    "num_edges",
    "mean_degree",
    "edge_density",
    "num_connected_components",
    "sizes_of_connected_components",
    "num_selfloops",
    "diameter",
    "avg_shortest_path_length",
    "avg_clustering",
)

# Diameter and average shortest path visit every pair of nodes. Above this many nodes
# they are estimated instead, and the result says so.
EXACT_MAX_NODES = 5000
SAMPLED_SOURCES = 200


def connected_components(node_ids: Iterable[Any], edges: Iterable[Tuple[Any, Any]]) -> List[int]:
    """Sizes of the connected components, largest first, ignoring edge direction."""
    parent: Dict[Any, Any] = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a not in parent or b not in parent:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    sizes: Dict[Any, int] = {}
    for n in parent:
        root = find(n)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


async def graph_metrics(adapter, include_optional: bool = False) -> Dict[str, Any]:
    """Counts and structure of the whole graph.

    Counts come from the label tables in SQL. Components are found in Python from the
    edge list read once. The optional metrics use networkx, built the way cognee's own
    networkx adapter builds them, so the numbers agree.
    """
    graph = adapter.graph_name
    vertices = sql.SQL("{}.{}").format(sql.Identifier(graph), sql.Identifier("ag_vertex"))
    edges_table = sql.SQL("{}.{}").format(sql.Identifier(graph), sql.Identifier("ag_edge"))

    ((num_nodes, num_edges, selfloops),) = await adapter._read_raw(
        sql.SQL(
            "SELECT (SELECT count(*) FROM {v}), "
            "(SELECT count(*) FROM {e}), "
            '(SELECT count(*) FROM {e} WHERE start = "end")'
        ).format(v=vertices, e=edges_table)
    )
    num_nodes, num_edges = int(num_nodes), int(num_edges)

    node_ids = [gid for (gid,) in await adapter._read_raw(sql.SQL("SELECT id FROM {v}").format(v=vertices))]
    pairs = await adapter._read_raw(sql.SQL('SELECT start, "end" FROM {e}').format(e=edges_table))
    sizes = connected_components(node_ids, pairs)

    metrics: Dict[str, Any] = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "mean_degree": (2 * num_edges) / num_nodes if num_nodes else 0,
        "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
        "num_connected_components": len(sizes),
        "sizes_of_connected_components": sizes,
        "num_selfloops": -1,
        "diameter": -1,
        "avg_shortest_path_length": -1,
        "avg_clustering": -1,
    }
    if include_optional:
        metrics["num_selfloops"] = int(selfloops)
        metrics.update(_optional_metrics(node_ids, pairs))
    return metrics


def _optional_metrics(node_ids: List[Any], pairs: List[Tuple[Any, Any]]) -> Dict[str, Any]:
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(pairs)
    undirected = nx.DiGraph(graph.to_undirected())

    out: Dict[str, Any] = {}
    try:
        out["avg_clustering"] = nx.average_clustering(undirected)
    except Exception as e:  # networkx raises on shapes it cannot measure
        logger.warning("could not compute the clustering coefficient: %s", e)
        out["avg_clustering"] = None

    if len(node_ids) <= EXACT_MAX_NODES:
        try:
            out["diameter"] = nx.diameter(undirected)
        except Exception as e:
            logger.warning("could not compute the diameter: %s", e)
            out["diameter"] = None
        try:
            out["avg_shortest_path_length"] = nx.average_shortest_path_length(undirected)
        except Exception as e:
            logger.warning("could not compute the average shortest path length: %s", e)
            out["avg_shortest_path_length"] = None
        return out

    # Too many nodes for every pair. Estimate from a sample and say so.
    out["approximate"] = True
    try:
        out["diameter"] = nx.approximation.diameter(undirected)
    except Exception as e:
        logger.warning("could not estimate the diameter: %s", e)
        out["diameter"] = None
    try:
        import random

        sources = random.Random(0).sample(node_ids, min(SAMPLED_SOURCES, len(node_ids)))
        total, count = 0, 0
        for source in sources:
            for target, length in nx.single_source_shortest_path_length(undirected, source).items():
                if target != source:
                    total += length
                    count += 1
        out["avg_shortest_path_length"] = total / count if count else None
    except Exception as e:
        logger.warning("could not estimate the average shortest path length: %s", e)
        out["avg_shortest_path_length"] = None
    return out
