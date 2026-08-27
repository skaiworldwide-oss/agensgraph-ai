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

"""The shared engine and the query() boundary."""

import asyncio

import psycopg
import pytest
import pytest_asyncio
from agensgraph.errors import ReadOnlyGraphWrite
from cognee.infrastructure.engine import DataPoint

from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import (
    AgensgraphAdapter,
)
from cognee_agensgraph.infrastructure.databases.vector.agensgraph.adapter import (
    AgensgraphVectorAdapter,
)

pytestmark = pytest.mark.asyncio


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


async def test_a_second_initialize_sends_nothing(adapter, statements):
    statements.reset()
    await adapter.initialize()
    assert len(statements) == 0


async def test_one_statement_costs_one_round_trip(adapter, statements):
    ent = Ent(name="Alice")
    await adapter.add_nodes([ent])
    statements.reset()
    await adapter.get_node(str(ent.id))
    # No SET graph_path, no BEGIN, no COMMIT: the graph is bound to the connection
    # and the connection is in autocommit mode.
    assert len(statements) == 1


async def test_graph_and_vector_adapters_share_one_pool(adapter, conn_url, embedding_engine):
    vector = AgensgraphVectorAdapter(conn_url, embedding_engine=embedding_engine)
    engine = await vector._ensure_engine()
    assert engine is adapter._engine
    assert await engine.pool() is await adapter._engine.pool()


def test_the_adapter_works_from_a_second_event_loop(conn_url):
    # cognee caches one adapter for the life of the process; a script or a test runner
    # can start more than one event loop in that time.
    a = AgensgraphAdapter(conn_url)

    async def use():
        await a.initialize()
        return await a.query('MATCH (n:"__node__") RETURN count(n) AS c')

    for _ in range(3):
        rows = asyncio.run(asyncio.wait_for(use(), 30))
        assert rows[0]["c"] >= 0

    asyncio.run(a.finalize())


async def test_query_refuses_a_write(adapter):
    with pytest.raises(ReadOnlyGraphWrite):
        await adapter.query("CREATE (n:\"__node__\" {id: 'x'})")
    assert await adapter.has_node("x") is False


async def test_query_refuses_a_second_statement(adapter):
    with pytest.raises(ValueError):
        await adapter.query("MATCH (n) RETURN count(n); CREATE (m:\"__node__\" {id: 'y'})")


async def test_query_writes_when_asked_to(conn_url):
    a = AgensgraphAdapter(conn_url, query_read_only=False)
    await a.initialize()
    try:
        await a.query("CREATE (n:\"__node__\" {id: 'written-through-query'})")
        assert await a.has_node("written-through-query") is True
    finally:
        await a.delete_graph()
        await a.finalize()


async def test_query_returns_plain_values(adapter):
    alice = Ent(name="Alice")
    await adapter.add_nodes([alice])
    rows = await adapter.query('MATCH (n:"__node__") RETURN n, n.name AS name, id(n) AS gid')
    assert rows[0]["n"]["name"] == "Alice"
    assert rows[0]["name"] == "Alice"
    assert isinstance(rows[0]["gid"], str)


async def test_a_bad_statement_raises_the_driver_error(adapter):
    with pytest.raises(psycopg.errors.SyntaxError):
        await adapter.query("MATCH (n RETRUN n")


async def test_borrowing_a_connection_does_not_open_the_pool_again(adapter):
    # Opening an already open pool is not free: the driver checks the server with a
    # connection of its own each time (about 3 ms). The engine opens its pool once.
    pool = await adapter._engine.pool()
    calls = []
    original = pool.open

    async def counted(*args, **kwargs):
        calls.append(1)
        return await original(*args, **kwargs)

    pool.open = counted
    try:
        for _ in range(20):
            await adapter.has_node("nobody")
        assert calls == []
    finally:
        pool.open = original


def test_edge_shorthand_is_spelled_out_outside_literals():
    from cognee_agensgraph.infrastructure.databases.graph.agensgraph.adapter import spell_out_edges

    assert spell_out_edges("MATCH (a)--(b) RETURN b") == "MATCH (a)-[]-(b) RETURN b"
    assert spell_out_edges("MATCH (a) --> (b) RETURN b") == "MATCH (a)-[]->(b) RETURN b"
    assert spell_out_edges("MATCH (a)<--(b:Entity) RETURN b") == "MATCH (a)<-[]-(b:Entity) RETURN b"
    # already spelled out, or a comment that is not between two nodes, is left alone
    assert spell_out_edges("MATCH (a)-[r]->(b) RETURN r") == "MATCH (a)-[r]->(b) RETURN r"
    assert spell_out_edges("MATCH (a) RETURN a -- trailing comment") == "MATCH (a) RETURN a -- trailing comment"
    assert spell_out_edges("MATCH (a {name: ')--('}) RETURN a") == "MATCH (a {name: ')--('}) RETURN a"


async def test_query_accepts_the_neo4j_edge_shorthand(adapter):
    alice, bob = Ent(name="Alice"), Ent(name="Bob")
    await adapter.add_nodes([alice, bob])
    await adapter.add_edges([(alice.id, bob.id, "knows", {})])
    rows = await adapter.query("MATCH (a:Ent {name: 'Alice'})--(b) RETURN b.name AS name")
    assert [r["name"] for r in rows] == ["Bob"]
    rows = await adapter.query("MATCH (a:Ent)-->(b:Ent) RETURN a.name AS a, b.name AS b")
    assert rows == [{"a": "Alice", "b": "Bob"}]
