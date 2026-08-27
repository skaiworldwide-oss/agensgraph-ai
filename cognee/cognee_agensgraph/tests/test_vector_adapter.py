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

import asyncio
from uuid import UUID

import pytest
import pytest_asyncio
from cognee.infrastructure.databases.vector.exceptions.exceptions import (
    CollectionNotFoundError,
)
from cognee.infrastructure.engine import DataPoint

from cognee_agensgraph.infrastructure.databases.vector.agensgraph.adapter import (
    AgensgraphVectorAdapter,
)

from conftest import explain, requires_agens, scanned_tables

pytestmark = [requires_agens, pytest.mark.asyncio]

COLLECTION = "test_vec_coll"


class Item(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


@pytest_asyncio.fixture
async def vector(conn_url, embedding_engine):
    adapter = AgensgraphVectorAdapter(conn_url, embedding_engine=embedding_engine)
    await adapter.prune()  # start from a clean collection set
    try:
        yield adapter
    finally:
        await adapter.prune()


@pytest_asyncio.fixture
async def wide(conn_url, wide_embedding_engine):
    adapter = AgensgraphVectorAdapter(conn_url, embedding_engine=wide_embedding_engine)
    await adapter.prune()
    try:
        yield adapter
    finally:
        await adapter.prune()


async def test_create_and_search(vector):
    apple, banana, cherry = Item(text="apple"), Item(text="banana"), Item(text="cherry")
    await vector.create_data_points(COLLECTION, [apple, banana, cherry])
    assert await vector.has_collection(COLLECTION) is True

    results = await vector.search(COLLECTION, query_text="apple", limit=3)
    assert results[0].payload["text"] == "apple"  # identical vector ranks first
    assert results[0].score == pytest.approx(0.0, abs=1e-6)  # score is the cosine distance
    assert results[0].score <= results[1].score <= results[2].score
    assert isinstance(results[0].id, UUID)


async def test_a_second_write_updates_the_row(vector):
    apple = Item(text="apple")
    await vector.create_data_points(COLLECTION, [apple])
    changed = Item(id=apple.id, text="apricot")
    await vector.create_data_points(COLLECTION, [changed])
    got = await vector.retrieve(COLLECTION, [str(apple.id)])
    assert len(got) == 1 and got[0].payload["text"] == "apricot"
    (nearest,) = await vector.search(COLLECTION, query_text="apricot", limit=1)
    assert nearest.score == pytest.approx(0.0, abs=1e-6)


async def test_retrieve(vector):
    apple, banana = Item(text="apple"), Item(text="banana")
    await vector.create_data_points(COLLECTION, [apple, banana])
    got = await vector.retrieve(COLLECTION, [str(apple.id), str(banana.id)])
    assert {str(r.id) for r in got} == {str(apple.id), str(banana.id)}
    assert all(r.score == 0 for r in got)


async def test_batch_search(vector):
    await vector.create_data_points(COLLECTION, [Item(text="apple"), Item(text="cherry")])
    results = await vector.batch_search(COLLECTION, ["apple", "cherry"], limit=1)
    assert sorted(r[0].payload["text"] for r in results) == ["apple", "cherry"]


async def test_limit_zero_returns_every_row(vector):
    await vector.create_data_points(COLLECTION, [Item(text=f"t{i}") for i in range(60)])
    assert len(await vector.search(COLLECTION, query_text="t1", limit=0)) == 60
    assert len(await vector.search(COLLECTION, query_text="t1", limit=None)) == 60


async def test_a_limit_above_the_index_default_is_honoured(vector):
    # HNSW hands back at most hnsw.ef_search rows (40 by default); asking for 100 of 200
    # must give 100, not 40.
    await vector.create_data_points(COLLECTION, [Item(text=f"t{i}") for i in range(200)])
    assert len(await vector.search(COLLECTION, query_text="t1", limit=100)) == 100


async def test_delete_and_prune(vector):
    apple, banana = Item(text="apple"), Item(text="banana")
    await vector.create_data_points(COLLECTION, [apple, banana])
    await vector.delete_data_points(COLLECTION, [str(apple.id)])
    assert await vector.retrieve(COLLECTION, [str(apple.id)]) == []
    await vector.prune()
    assert await vector.has_collection(COLLECTION) is False


async def test_a_missing_collection_is_reported(vector):
    with pytest.raises(CollectionNotFoundError):
        await vector.search("no_such_collection", query_text="apple", limit=3)
    with pytest.raises(CollectionNotFoundError):
        await vector.retrieve("no_such_collection", ["x"])
    await vector.delete_data_points("no_such_collection", ["x"])  # nothing to delete: no error
    assert await vector.has_collection("no_such_collection") is False


async def test_a_collection_known_to_exist_costs_no_lookup(vector, statements):
    await vector.create_data_points(COLLECTION, [Item(text="apple")])
    statements.reset()
    assert await vector.has_collection(COLLECTION) is True
    assert len(statements) == 0
    statements.reset()
    await vector.search(COLLECTION, query_text="apple", limit=3)
    # the search itself and its transaction preamble, nothing to check the table first
    assert not any("pg_class" in s for s in statements.statements)


async def test_top_k_search_uses_the_hnsw_index_on_real_width_vectors(wide):
    await wide.create_data_points(COLLECTION, [Item(text=f"text {i}") for i in range(300)])
    q = (await wide.embed_data(["text 7"]))[0]
    from agensgraph import Vector

    async with (await wide._ensure_engine()).connection() as conn:
        await conn.execute(f'ANALYZE "{COLLECTION}"')
        statement = f'SELECT id FROM "{COLLECTION}" ORDER BY vector <=> %(q)s LIMIT 15'
        # the form the adapter runs a top-k search in
        forced = await explain(conn, statement, {"q": Vector(q.tolist())}, by_index=True)
        assert scanned_tables(forced) == [], forced
        assert "hnsw" in forced.lower()
        # a whole-collection read keeps the sequential scan on purpose
        plain = await explain(conn, f'SELECT id FROM "{COLLECTION}" ORDER BY vector <=> %(q)s', {"q": Vector(q.tolist())})
        assert scanned_tables(plain), plain


async def test_eight_writers_on_shared_ids_all_succeed(conn_url, embedding_engine):
    items = [Item(text=f"shared {i}") for i in range(25)]
    adapters = [AgensgraphVectorAdapter(conn_url, embedding_engine=embedding_engine) for _ in range(8)]
    await adapters[0].prune()
    try:
        await adapters[0].create_collection(COLLECTION)
        results = await asyncio.gather(
            *[a.create_data_points(COLLECTION, items) for a in adapters], return_exceptions=True
        )
        assert [r for r in results if isinstance(r, BaseException)] == []
        got = await adapters[0].retrieve(COLLECTION, [str(i.id) for i in items])
        assert len(got) == 25
    finally:
        await adapters[0].prune()


async def test_bulk_ingest_builds_the_index_once_at_the_end(vector):
    await vector.create_data_points(COLLECTION, [Item(text="seed")])

    async def index_exists():
        async with (await vector._ensure_engine()).connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) FROM pg_indexes WHERE indexname = %s", (f"{COLLECTION}_hnsw",)
            )
            return (await cur.fetchone())[0] == 1

    assert await index_exists()
    async with vector.bulk_ingest():
        await vector.create_data_points(COLLECTION, [Item(text=f"t{i}") for i in range(50)])
        assert not await index_exists()  # dropped for the load
        await vector.create_data_points("second_collection", [Item(text="x")])
        # reads inside the block still answer
        assert len(await vector.search(COLLECTION, query_text="t1", limit=3)) == 3
    assert await index_exists()
    async with (await vector._ensure_engine()).connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM pg_indexes WHERE indexname = 'second_collection_hnsw'")
        assert (await cur.fetchone())[0] == 1
    assert len(await vector.search(COLLECTION, query_text="t1", limit=3)) == 3


async def test_hnsw_parameters_are_applied(conn_url, embedding_engine):
    adapter = AgensgraphVectorAdapter(
        conn_url, embedding_engine=embedding_engine, hnsw_m=8, hnsw_ef_construction=32
    )
    await adapter.prune()
    try:
        await adapter.create_collection(COLLECTION)
        async with (await adapter._ensure_engine()).connection() as conn:
            cur = await conn.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (f"{COLLECTION}_hnsw",))
            (definition,) = await cur.fetchone()
        assert "m='8'" in definition and "ef_construction='32'" in definition
    finally:
        await adapter.prune()
