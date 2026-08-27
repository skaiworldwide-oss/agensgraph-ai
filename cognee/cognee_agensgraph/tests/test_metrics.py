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

"""Graph metrics against cognee's own ground truth, and the whole-graph read."""

import json
from pathlib import Path

import pytest
import pytest_asyncio
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
)
from cognee_agensgraph.infrastructure.databases.graph.agensgraph.metrics import (
    METRIC_KEYS,
    connected_components,
)

pytestmark = pytest.mark.asyncio

GROUND_TRUTH = json.loads(
    (Path(__file__).parent / "tasks" / "descriptive_metrics" / "ground_truth_metrics.json").read_text()
)


class Node(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture
async def adapter(conn_url):
    a = AgensgraphAdapter(conn_url)
    await a.initialize()
    await a.delete_graph()
    try:
        yield a
    finally:
        await a.delete_graph()
        await a.finalize()


async def _connected(adapter):
    """cognee's connected fixture: a document, a chunk, two entities, a type; six edges
    including one self-loop."""
    doc, chunk, e1, e2, kind = (Node(name=n) for n in ("doc", "chunk", "e1", "e2", "kind"))
    await adapter.add_nodes([doc, chunk, e1, e2, kind])
    await adapter.add_edges(
        [
            (chunk.id, doc.id, "is_part_of", {}),
            (chunk.id, e1.id, "contains", {}),
            (chunk.id, e2.id, "contains", {}),
            (chunk.id, chunk.id, "contains", {}),
            (e1.id, kind.id, "is_a", {}),
            (e2.id, kind.id, "is_a", {}),
        ]
    )


async def _disconnected(adapter):
    """cognee's disconnected fixture: the connected graph without its self-loop, plus a
    second component of four nodes and three edges."""
    doc, chunk, e1, e2, kind = (Node(name=n) for n in ("doc", "chunk", "e1", "e2", "kind"))
    d2, c2, f1, k2 = (Node(name=n) for n in ("doc2", "chunk2", "f1", "kind2"))
    await adapter.add_nodes([doc, chunk, e1, e2, kind, d2, c2, f1, k2])
    await adapter.add_edges(
        [
            (chunk.id, doc.id, "is_part_of", {}),
            (chunk.id, e1.id, "contains", {}),
            (chunk.id, e2.id, "contains", {}),
            (e1.id, kind.id, "is_a", {}),
            (e2.id, kind.id, "is_a", {}),
            (c2.id, d2.id, "is_part_of", {}),
            (c2.id, f1.id, "contains", {}),
            (f1.id, k2.id, "is_a", {}),
        ]
    )


async def test_metrics_match_the_ground_truth_with_optional_metrics(adapter):
    await _connected(adapter)
    metrics = await adapter.get_graph_metrics(include_optional=True)
    assert set(metrics) == set(METRIC_KEYS) == set(GROUND_TRUTH["connected"])
    for key, value in GROUND_TRUTH["connected"].items():
        assert metrics[key] == pytest.approx(value), key


async def test_metrics_match_the_ground_truth_without_optional_metrics(adapter):
    await _disconnected(adapter)
    metrics = await adapter.get_graph_metrics(include_optional=False)
    assert set(metrics) == set(GROUND_TRUTH["disconnected"])
    for key, value in GROUND_TRUTH["disconnected"].items():
        assert metrics[key] == pytest.approx(value), key


async def test_metrics_of_an_empty_graph(adapter):
    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == 0 and metrics["num_edges"] == 0
    assert metrics["num_connected_components"] == 0
    assert metrics["mean_degree"] == 0 and metrics["edge_density"] == 0


def test_connected_components_ignore_direction_and_unknown_endpoints():
    assert connected_components([1, 2, 3, 4], [(1, 2), (3, 2)]) == [3, 1]
    assert connected_components([1], [(1, 9)]) == [1]
    assert connected_components([], []) == []


async def test_the_whole_graph_read_projects_into_cognee(adapter):
    await _disconnected(adapter)
    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 9 and len(edges) == 8
    assert all(isinstance(n[0], str) and n[1]["id"] == n[0] for n in nodes)
    assert all(e[3]["relationship_name"] == e[2] for e in edges)
    graph = CogneeGraph()
    await graph.project_graph_from_db(
        adapter,
        node_properties_to_project=["id", "name", "type"],
        edge_properties_to_project=["relationship_name"],
    )
    assert len(graph.nodes) == 9 and len(graph.edges) == 8
    assert {e.attributes["relationship_type"] for e in graph.edges} == {"is_part_of", "contains", "is_a"}
