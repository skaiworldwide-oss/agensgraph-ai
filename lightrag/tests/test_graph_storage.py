# Copyright (c) 2025, SKAI Worldwide Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import pytest_asyncio
from conftest import explain

from lightrag_agensgraph.kg import agensgraph_impl as impl
from lightrag_agensgraph.kg.agensgraph_impl import AgensgraphStorage

pytestmark = pytest.mark.asyncio


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


async def test_edges_are_undirected_and_stored_once(graph):
    await _seed(graph)
    assert await graph.has_edge("Bob", "Alice") is True
    assert (await graph.get_edge("Bob", "Alice"))["rel"] == "knows"
    batch = await graph.get_edges_batch([{"src": "Bob", "tgt": "Alice"}, {"src": "Alice", "tgt": "Paris"}])
    assert set(batch) == {("Bob", "Alice"), ("Alice", "Paris")}  # keyed as asked, whichever way round
    await graph.upsert_edge("Bob", "Alice", {"rel": "knows well"})  # the other way round: same edge
    assert await graph.node_degree("Alice") == 2 and await graph.node_degree("Bob") == 2
    assert (await graph.get_edge("Alice", "Bob"))["rel"] == "knows well"
    assert (await graph.edge_degrees_batch([("Alice", "Bob")]))[("Alice", "Bob")] == 4


async def test_node_edges_tell_absent_from_isolated(graph):
    await _seed(graph)
    await graph.upsert_node("Hermit", {"entity_id": "Hermit"})
    assert await graph.get_node_edges("Ghost") is None
    assert await graph.get_node_edges("Hermit") == []
    assert sorted(await graph.get_node_edges("Alice")) == [("Alice", "Bob"), ("Alice", "Paris")]
    assert await graph.get_node_edges("Bob") == [("Bob", "Alice"), ("Bob", "Rome")] or sorted(
        await graph.get_node_edges("Bob")
    ) == [("Bob", "Alice"), ("Bob", "Rome")]
    batch = await graph.get_nodes_edges_batch(["Rome", "Hermit", "Ghost"])
    assert batch == {"Rome": [("Rome", "Bob")], "Hermit": [], "Ghost": []}


async def test_popular_labels_rank_everything_and_search_ranks_matches(graph):
    await _seed(graph)
    await graph.upsert_node("Zed", {"entity_id": "Zed"})  # isolated: last, but present
    assert await graph.get_popular_labels(limit=10) == ["Alice", "Bob", "Paris", "Rome", "Zed"]
    assert await graph.get_popular_labels(limit=2) == ["Alice", "Bob"]
    await graph.upsert_node("Rom", {"entity_id": "Rom"})
    assert await graph.search_labels("rom") == ["Rom", "Rome"]  # exact match first, then the prefix
    assert await graph.search_labels("%") == [] and await graph.search_labels("  ") == []


async def test_knowledge_graph_names_its_nodes_and_edges(graph):
    await _seed(graph)
    kg = await graph.get_knowledge_graph("Alice", max_depth=1)
    assert {n.id for n in kg.nodes} == {"Alice", "Bob", "Paris"}
    assert all(n.labels == [n.id] for n in kg.nodes)
    assert {(e.source, e.target) for e in kg.edges} == {("Alice", "Bob"), ("Alice", "Paris")}
    assert all(e.type == "DIRECTED" and "rel" in e.properties for e in kg.edges)
    two = await graph.get_knowledge_graph("Alice", max_depth=2)
    assert {n.id for n in two.nodes} == {"Alice", "Bob", "Paris", "Rome"} and len(two.edges) == 3
    capped = await graph.get_knowledge_graph("Alice", max_depth=2, max_nodes=2)
    assert len(capped.nodes) == 2 and capped.is_truncated is True


@pytest_asyncio.fixture
async def big_graph(graph):
    # Enough rows for the planner to prefer an index over a scan, with a few hubs. On a
    # small table a scan is the right plan and says nothing about these statements.
    names = [f"n{i}" for i in range(12000)]
    await graph.upsert_nodes_batch([(n, {"entity_id": n, "source_id": f"c{i % 50}"}) for i, n in enumerate(names)])
    edges = [(names[i % 40], names[(i * 7 + 3) % 12000], {"weight": 1.0}) for i in range(24000)]
    await graph.upsert_edges_batch(edges)
    graph.expected_degree = {}
    for a, b, _ in edges:
        pair = tuple(sorted((a, b)))
        if pair not in graph.expected_degree.setdefault("pairs", set()):
            graph.expected_degree["pairs"].add(pair)
            for name in pair:
                graph.expected_degree[name] = graph.expected_degree.get(name, 0) + 1
    async with graph._engine.connection() as conn:
        await conn.execute(f'ANALYZE "{graph.graph_name}".base')
        await conn.execute(f'ANALYZE "{graph.graph_name}"."DIRECTED"')
    return graph


async def _plan(graph, template, params, by_index):
    async with graph._engine.connection() as conn:
        return await explain(conn, graph._sql(template).as_string(conn), params, by_index=by_index)


async def test_edge_reads_probe_the_pair_index_whatever_the_degree(big_graph):
    g = big_graph
    hub = "n0"
    for template, params, by_index in [
        (impl.HAS_EDGE, {"a": hub, "b": "n3"}, False),
        (impl.GET_EDGE, {"a": hub, "b": "n3"}, False),
        (impl.GET_EDGES, {"names": impl.Jsonb([hub, "n3", "n1", "n10"]), "src": [hub, "n1"], "tgt": ["n3", "n10"]}, True),
        (impl.DEGREES, {"names": impl.Jsonb([hub, "n1", "n2"])}, True),
        (impl.NODE_EDGES, {"names": impl.Jsonb([hub])}, True),
    ]:
        plan = await _plan(g, template, params, by_index)
        # The empty parent tables (ag_vertex, ag_edge) appear in a Cypher plan and are
        # never executed; the label tables themselves must be reached through an index.
        assert "Seq Scan on base" not in plan and 'Seq Scan on "DIRECTED"' not in plan, plan
        assert "Join Filter" not in plan and "Rows Removed by Filter" not in plan, plan
    assert await g.has_edge("n3", hub) is True
    expected = g.expected_degree[hub]  # the generator repeats targets; a pair is one edge
    assert (await g.node_degrees_batch([hub]))[hub] == expected
    assert len(await g.get_node_edges(hub)) == expected
