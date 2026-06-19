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

from cognee.infrastructure.engine import DataPoint

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
)

from conftest import requires_agens

pytestmark = [requires_agens, pytest.mark.asyncio]


class Ent(DataPoint):
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


async def _seed(adapter):
    alice, bob, paris = Ent(name="Alice"), Ent(name="Bob"), Ent(name="Paris")
    await adapter.add_nodes([alice, bob, paris])
    await adapter.add_edges(
        [
            (str(alice.id), str(bob.id), "knows", {"w": 1}),
            (str(alice.id), str(paris.id), "visited", {"w": 2}),
        ]
    )
    return alice, bob, paris


async def test_add_node_and_get(adapter):
    alice = Ent(name="Alice")
    await adapter.add_node(alice)  # delegates to add_nodes
    assert await adapter.has_node(str(alice.id)) is True
    assert await adapter.has_node("missing") is False  # must not raise
    node = await adapter.get_node(str(alice.id))
    assert node["name"] == "Alice"


async def test_edges_and_has_edge(adapter):
    alice, bob, paris = await _seed(adapter)
    assert await adapter.has_edge(str(alice.id), str(bob.id), "knows") is True
    assert await adapter.has_edge(str(alice.id), str(bob.id), "visited") is False
    assert len(await adapter.get_edges(str(alice.id))) == 2
    assert len(await adapter.get_neighbors(str(alice.id))) == 2
    assert len(await adapter.get_connections(alice.id)) == 2


async def test_predecessors_successors(adapter):
    alice, bob, paris = await _seed(adapter)
    assert len(await adapter.get_successors(str(alice.id))) == 2
    assert len(await adapter.get_predecessors(str(bob.id))) == 1


async def test_graph_data_and_metrics(adapter):
    await _seed(adapter)
    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 3 and len(edges) == 2
    metrics = await adapter.get_graph_metrics(include_optional=False)
    assert metrics["num_nodes"] == 3 and metrics["num_edges"] == 2


async def test_get_nodes_and_delete(adapter):
    alice, bob, paris = await _seed(adapter)
    got = await adapter.get_nodes([str(alice.id), str(bob.id)])
    assert len(got) == 2
    await adapter.delete_node(str(paris.id))
    assert await adapter.has_node(str(paris.id)) is False


async def test_nodeset_subgraph(adapter):
    alice, bob, paris = await _seed(adapter)
    nodes, edges = await adapter.get_nodeset_subgraph(Ent, ["Alice"])
    names = {n[1]["name"] for n in nodes}
    assert "Alice" in names and "Bob" in names  # Alice + its neighbors


async def test_name_lookup_uses_index(adapter):
    await adapter.add_nodes([Ent(name=f"e{i}") for i in range(30)])
    async with adapter._engine.aconnection(graph_path=adapter.graph_name) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                'EXPLAIN MATCH (n:"__Node__" {name: \'"e5"\'}) RETURN n'
            )
            plan = "\n".join(r[0] for r in await cur.fetchall())
        await conn.rollback()
    assert "base_name_idx" in plan and "Seq Scan" not in plan


async def test_ingest_and_lookup_use_id_index(adapter):
    # The MERGE-by-id ingest and id lookups must use base_id_idx, not seq-scan.
    await adapter.add_nodes([Ent(name=f"n{i}") for i in range(30)])
    async with adapter._engine.aconnection(graph_path=adapter.graph_name) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                'EXPLAIN MATCH (n:"__Node__" {id: \'x\'}) RETURN n'
            )
            plan = "\n".join(r[0] for r in await cur.fetchall())
        await conn.rollback()
    assert "base_id_idx" in plan and "Seq Scan" not in plan
