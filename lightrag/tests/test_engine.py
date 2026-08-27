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

"""The shared engine: one pool per graph, one round trip per statement."""

import asyncio

import pytest_asyncio

from lightrag_agensgraph.kg._base import graph_name_for
from lightrag_agensgraph.kg.agensgraph_impl import AgensgraphStorage
from lightrag_agensgraph.kg.agensgraph_kv_impl import AgensgraphKVStorage


def _graph(workspace=""):
    # LightRAG's own graph namespace, so the other stores compute the same graph name.
    return AgensgraphStorage(
        namespace="chunk_entity_relation", workspace=workspace, global_config={}, embedding_func=None
    )


@pytest_asyncio.fixture
async def graph():
    store = _graph()
    await store.initialize()
    await store.drop()
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def test_a_second_initialize_sends_nothing(graph, statements):
    statements.reset()
    await graph.initialize()
    assert len(statements) == 0


async def test_one_statement_costs_one_round_trip(graph, statements):
    await graph.upsert_node("Alice", {"entity_id": "Alice"})
    statements.reset()
    await graph.get_node("Alice")
    # No SET graph_path, no BEGIN, no COMMIT: the graph is bound to the connection and
    # the connection is in autocommit mode. The pool may set another connection up in
    # the background meanwhile, so only the connection that ran the read is judged.
    reads = [r for r in statements.records if r.statement.strip().upper().startswith("MATCH")]
    assert len(reads) == 1, statements.statements
    on_that_connection = [r.statement for r in statements.records if r.connection == reads[0].connection]
    assert on_that_connection == [reads[0].statement]


async def test_stores_of_one_workspace_share_one_pool(graph):
    kv = AgensgraphKVStorage(namespace="full_docs", workspace="", global_config={}, embedding_func=None)
    await kv.initialize()
    try:
        assert kv._engine is graph._engine
        assert await kv._engine.pool() is await graph._engine.pool()
    finally:
        await kv.finalize()


async def test_two_workspaces_get_two_graphs_and_two_pools():
    a, b = _graph("tenant_a"), _graph("tenant_b")
    await a.initialize()
    await b.initialize()
    try:
        assert a.graph_name != b.graph_name
        assert a._engine is not b._engine
        await a.upsert_node("Shared", {"entity_id": "Shared"})
        assert await b.has_node("Shared") is False
    finally:
        for s in (a, b):
            await s.drop()
            await s.finalize()


def test_the_store_works_from_a_second_event_loop():
    # A script or a test runner can start more than one event loop in one process;
    # the pool of a finished loop is not reused.
    store = _graph()

    async def use():
        await store.initialize()
        try:
            await store.upsert_node("Loop", {"entity_id": "Loop"})
            return await store.has_node("Loop")
        finally:
            await store.drop()
            await store.finalize()

    for _ in range(3):
        assert asyncio.run(asyncio.wait_for(use(), 30)) is True


async def test_concurrent_writers_on_one_key_make_one_node(graph):
    # LightRAG merges entities from up to two dozen coroutines at once; two of them can
    # meet on the same name. The uniqueness constraint plus the retry turn that into one
    # node and no error.
    async def writer(i):
        await graph.upsert_node("Race", {"entity_id": "Race", "writer": i})
        await graph.upsert_edge("Race", "Other", {"weight": float(i)})

    await graph.upsert_node("Other", {"entity_id": "Other"})
    results = await asyncio.gather(*(writer(i) for i in range(24)), return_exceptions=True)
    assert [r for r in results if isinstance(r, Exception)] == []
    rows = await graph._fetch("MATCH (n:base {entity_id: %(id)s}) RETURN count(n) AS c", {"id": "Race"})
    assert rows[0]["c"] == 1


async def test_names_that_look_like_statements_are_parameters(graph):
    hostile = "x'); MATCH (n) DETACH DELETE n; --"
    await graph.upsert_node("Alice", {"entity_id": "Alice"})
    await graph.upsert_node(hostile, {"entity_id": hostile, "description": 'say "hi"; --'})
    node = await graph.get_node(hostile)
    assert node["entity_id"] == hostile and node["description"] == 'say "hi"; --'
    assert await graph.has_node("Alice") is True
    assert await graph.search_labels("DETACH") == [hostile]


def test_a_workspace_becomes_a_valid_graph_name():
    assert graph_name_for("chunk_entity_relation", "") == "chunk_entity_relation"
    assert graph_name_for("g", 'Tenant-A"; drop schema public;') == "tenant_a_drop_schema_public_g"
    assert graph_name_for("g", "acme") == "acme_g"
    assert graph_name_for("g", "9lives").startswith("w_9lives")
    assert graph_name_for("g", "pg_catalog").startswith("w_pg_catalog")
    assert len(graph_name_for("g", "x" * 200)) <= 63
