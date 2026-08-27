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

from psycopg import sql
from psycopg.types.json import Jsonb

from conftest import explain, requires_agens, scanned_tables

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


async def _analyzed(adapter, labels):
    async with adapter._engine.connection() as conn:
        for label in labels:
            await conn.execute(
                sql.SQL("ANALYZE {}.{}").format(
                    sql.Identifier(adapter.graph_name), sql.Identifier(label)
                )
            )


async def _plan(adapter, statement, params=None, *, by_index=False):
    async with adapter._engine.connection() as conn:
        return await explain(conn, statement, params, by_index=by_index)


async def test_id_lookups_use_an_index(adapter):
    # Plans are read with the planner's default settings, on enough rows that a
    # sequential scan would be a real choice, and after ANALYZE so the choice is made
    # on real statistics.
    await adapter.add_nodes([Ent(name=f"n{i}") for i in range(2000)])
    await _analyzed(adapter, ["ent", "__node__"])
    for statement in (
        'MATCH (n:"ent" {id: %(id)s}) RETURN n',
        'MATCH (n:"__node__" {id: %(id)s}) RETURN n',
        'MERGE (n:"ent" {id: %(id)s}) SET n += %(props)s',
    ):
        plan = await _plan(adapter, statement, {"id": Jsonb("x"), "props": Jsonb({"a": 1})})
        assert scanned_tables(plan) == [], plan


async def test_name_lookup_uses_an_index(adapter):
    await adapter.add_nodes([Ent(name=f"e{i}") for i in range(2000)])
    await _analyzed(adapter, ["ent", "__node__"])
    plan = await _plan(adapter, 'MATCH (n:"ent" {name: %(name)s}) RETURN n', {"name": Jsonb("e5")})
    assert scanned_tables(plan) == [], plan
    plan = await _plan(
        adapter,
        'MATCH (n:"ent") WHERE n.name = %(a)s OR n.name = %(b)s RETURN n',
        {"a": Jsonb("e5"), "b": Jsonb("e6")},
    )
    assert scanned_tables(plan) == [], plan


async def test_edge_reads_from_a_bound_list_use_an_index(adapter):
    ents = [Ent(name=f"e{i}") for i in range(2000)]
    await adapter.add_nodes(ents)
    await adapter.add_edges(
        [(str(ents[i].id), str(ents[i + 1].id), "knows", {}) for i in range(1999)]
    )
    await _analyzed(adapter, ["ent", "__node__", "knows"])
    ids = [str(e.id) for e in ents[:50]]
    # the statement get_nodeset_subgraph runs for the edges among a node set
    plan = await _plan(
        adapter,
        'UNWIND %(ids)s AS wanted MATCH (a:"__node__" {id: wanted})-[r]->(b:"__node__") '
        "RETURN label(r), properties(r)",
        {"ids": Jsonb(ids)},
        by_index=True,
    )
    assert scanned_tables(plan) == [], plan
    # the statement has_edges runs
    plan = await _plan(
        adapter,
        'UNWIND %(edges)s AS e MATCH (a:"__node__" {id: e.from_node})-[r]->(b:"__node__" {id: e.to_node}) '
        "WHERE label(r) = e.relationship_name RETURN e.from_node",
        {"edges": Jsonb([{"from_node": ids[0], "to_node": ids[1], "relationship_name": "knows"}])},
        by_index=True,
    )
    assert scanned_tables(plan) == [], plan
    # the merge add_nodes runs
    plan = await _plan(
        adapter,
        'UNWIND %(rows)s AS row MERGE (n:"ent" {id: row.id}) SET n += row.props',
        {"rows": Jsonb([{"id": i, "props": {"name": "x"}} for i in ids])},
        by_index=True,
    )
    assert scanned_tables(plan) == [], plan
