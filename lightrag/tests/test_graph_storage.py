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

import pytest
import pytest_asyncio

from lightrag_agensgraph.kg.agensgraph_impl import AgensgraphStorage

from conftest import requires_agens

pytestmark = [requires_agens, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def graph():
    store = AgensgraphStorage(
        namespace="lr_test_graph", workspace="", global_config={}, embedding_func=None
    )
    await store.initialize()
    await store.drop()
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def _seed(graph):
    await graph.upsert_nodes_batch(
        [
            ("Alice", {"entity_id": "Alice", "source_id": "c1"}),
            ("Bob", {"entity_id": "Bob", "source_id": "c2"}),
            ("Paris", {"entity_id": "Paris", "source_id": "c1"}),
            ("Rome", {"entity_id": "Rome", "source_id": "c3"}),
        ]
    )
    await graph.upsert_edges_batch(
        [
            ("Alice", "Bob", {"rel": "knows", "source_id": "c1"}),
            ("Alice", "Paris", {"rel": "visited", "source_id": "c1"}),
            ("Bob", "Rome", {"rel": "visited", "source_id": "c3"}),
        ]
    )


async def test_upsert_and_get(graph):
    await graph.upsert_node("Alice", {"entity_id": "Alice", "kind": "person"})
    node = await graph.get_node("Alice")
    assert node["entity_id"] == "Alice" and node["kind"] == "person"
    assert await graph.get_node("Ghost") is None


async def test_has_node_and_edge_present_and_absent(graph):
    await _seed(graph)
    assert await graph.has_node("Alice") is True
    assert await graph.has_node("Ghost") is False  # must not raise
    assert await graph.has_edge("Alice", "Bob") is True
    assert await graph.has_edge("Alice", "Ghost") is False  # must not raise
    assert await graph.has_edge("Bob", "Alice") is True  # undirected


async def test_batches(graph):
    await _seed(graph)
    assert await graph.has_nodes_batch(["Alice", "Bob", "Ghost"]) == {"Alice", "Bob"}
    nodes = await graph.get_nodes_batch(["Alice", "Rome"])
    assert set(nodes) == {"Alice", "Rome"}
    degrees = await graph.node_degrees_batch(["Alice", "Rome"])
    assert degrees["Alice"] == 2 and degrees["Rome"] == 1


async def test_all_nodes_edges_and_labels(graph):
    await _seed(graph)
    assert {n["entity_id"] for n in await graph.get_all_nodes()} == {
        "Alice", "Bob", "Paris", "Rome",
    }
    edges = await graph.get_all_edges()
    assert len(edges) == 3  # deduped (undirected), not doubled
    assert await graph.get_all_labels() == ["Alice", "Bob", "Paris", "Rome"]


async def test_popular_and_search_labels(graph):
    await _seed(graph)
    assert (await graph.get_popular_labels(limit=1))[0] == "Alice"  # highest degree
    assert await graph.search_labels("ar") == ["Paris"]
    assert await graph.search_labels("ALICE") == ["Alice"]  # case-insensitive


async def test_remove_nodes_and_edges(graph):
    await _seed(graph)
    await graph.remove_edges([("Alice", "Paris")])
    assert await graph.has_edge("Alice", "Paris") is False
    await graph.remove_nodes(["Paris", "Rome"])
    assert await graph.get_all_labels() == ["Alice", "Bob"]


async def test_knowledge_graph_all_and_labeled(graph):
    await _seed(graph)
    kg = await graph.get_knowledge_graph("*", max_nodes=10)
    assert {n.id for n in kg.nodes} == {"Alice", "Bob", "Paris", "Rome"}
    assert len(kg.edges) == 3
    assert kg.is_truncated is False

    kg2 = await graph.get_knowledge_graph("Alice", max_depth=1)
    assert {"Bob", "Paris"}.issubset({n.id for n in kg2.nodes})


async def test_knowledge_graph_truncation(graph):
    await graph.upsert_nodes_batch(
        [(f"n{i}", {"entity_id": f"n{i}"}) for i in range(20)]
    )
    kg = await graph.get_knowledge_graph("*", max_nodes=5)
    assert len(kg.nodes) == 5
    assert kg.is_truncated is True


async def test_chunk_id_lookups(graph):
    await _seed(graph)
    nodes = await graph.get_nodes_by_chunk_ids(["c1"])
    assert {n["entity_id"] for n in nodes} == {"Alice", "Paris"}
    edges = await graph.get_edges_by_chunk_ids(["c3"])
    assert any(e["source"] in ("Bob", "Rome") for e in edges)
