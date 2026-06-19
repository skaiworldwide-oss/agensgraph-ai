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

from uuid import UUID

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint

from cognee_agensgraph.infrastructure.databases.vector.agensgraph.adapter import (
    AgensgraphVectorAdapter,
)

from conftest import requires_agens

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


async def test_create_and_search(vector):
    apple, banana, cherry = Item(text="apple"), Item(text="banana"), Item(text="cherry")
    await vector.create_data_points(COLLECTION, [apple, banana, cherry])
    assert await vector.has_collection(COLLECTION) is True

    results = await vector.search(COLLECTION, query_text="apple", limit=3)
    assert results[0].payload["text"] == "apple"  # identical vector ranks first
    assert isinstance(results[0].score, float)
    assert isinstance(results[0].id, UUID)


async def test_retrieve(vector):
    apple, banana = Item(text="apple"), Item(text="banana")
    await vector.create_data_points(COLLECTION, [apple, banana])
    got = await vector.retrieve(COLLECTION, [str(apple.id), str(banana.id)])
    assert {str(r.id) for r in got} == {str(apple.id), str(banana.id)}
    assert all(r.score == 0 for r in got)


async def test_batch_search(vector):
    await vector.create_data_points(
        COLLECTION, [Item(text="apple"), Item(text="cherry")]
    )
    results = await vector.batch_search(COLLECTION, ["apple", "cherry"], limit=1)
    assert sorted(r[0].payload["text"] for r in results) == ["apple", "cherry"]


async def test_delete_and_prune(vector):
    apple, banana = Item(text="apple"), Item(text="banana")
    await vector.create_data_points(COLLECTION, [apple, banana])
    await vector.delete_data_points(COLLECTION, [str(apple.id)])
    assert await vector.retrieve(COLLECTION, [str(apple.id)]) == []
    await vector.prune()
    assert await vector.has_collection(COLLECTION) is False


async def test_search_uses_hnsw_index(vector):
    await vector.create_data_points(
        COLLECTION, [Item(text=f"text {i}") for i in range(30)]
    )
    async with vector._engine.aconnection(graph_path=None) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                f'EXPLAIN SELECT id FROM "{COLLECTION}" '
                "ORDER BY vector <=> '[0,0,0,0,0,0,0,1]'::vector LIMIT 5"
            )
            plan = "\n".join(r[0] for r in await cur.fetchall())
        await conn.rollback()
    assert "hnsw" in plan.lower() and "Seq Scan" not in plan
