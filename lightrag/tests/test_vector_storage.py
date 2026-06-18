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

from lightrag_agensgraph.kg.agensgraph_vector_impl import AgensgraphVectorStorage

from conftest import requires_agens

pytestmark = [requires_agens, pytest.mark.asyncio]


def _vec(namespace, embedding_func):
    return AgensgraphVectorStorage(
        namespace=namespace,
        workspace="",
        global_config={"vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.0}},
        embedding_func=embedding_func,
    )


@pytest_asyncio.fixture
async def entities(embedding_func):
    store = _vec("entities", embedding_func)
    await store.initialize()
    await store.drop()
    try:
        yield store
    finally:
        await store.drop()
        await store.finalize()


async def test_upsert_query_ranking(entities):
    await entities.upsert(
        {
            "ent-1": {"entity_name": "apple", "content": "apple", "source_id": "c1<SEP>c2"},
            "ent-2": {"entity_name": "zebra", "content": "zebra", "source_id": "c3"},
        }
    )
    res = await entities.query("apple", top_k=2)
    assert res[0]["entity_name"] == "apple"  # identical vector ranks first


async def test_get_by_id_strips_vector_and_splits_chunks(entities):
    await entities.upsert(
        {"ent-1": {"entity_name": "apple", "content": "a", "source_id": "c1<SEP>c2"}}
    )
    rec = await entities.get_by_id("ent-1")
    assert "content_vector" not in rec
    assert rec["chunk_ids"] == ["c1", "c2"]
    vecs = await entities.get_vectors_by_ids(["ent-1"])
    assert len(vecs["ent-1"]) == 8


async def test_delete_and_delete_entity(entities):
    await entities.upsert(
        {
            "ent-1": {"entity_name": "apple", "content": "a", "source_id": "c1"},
            "ent-2": {"entity_name": "pear", "content": "p", "source_id": "c2"},
        }
    )
    await entities.delete(["ent-2"])
    assert await entities.get_by_id("ent-2") is None
    await entities.delete_entity("apple")
    assert await entities.get_by_id("ent-1") is None


async def test_chunks_namespace(embedding_func):
    chunks = _vec("chunks", embedding_func)
    await chunks.initialize()
    await chunks.drop()
    try:
        await chunks.upsert(
            {
                "chunk-1": {
                    "content": "apple pie recipe",
                    "tokens": 3,
                    "chunk_order_index": 0,
                    "full_doc_id": "d1",
                    "file_path": "f1",
                }
            }
        )
        res = await chunks.query("apple pie", top_k=1)
        assert res and res[0]["id"] == "chunk-1" and res[0]["content"] == "apple pie recipe"
    finally:
        await chunks.drop()
        await chunks.finalize()


async def test_query_uses_hnsw_index(entities, embedding_func):
    # Seed a few rows, then prove the HNSW index is usable for the ANN order-by
    # (small tables otherwise prefer a seq scan, so we disable seq scans in-txn).
    await entities.upsert(
        {
            f"ent-{i}": {"entity_name": f"e{i}", "content": f"text {i}", "source_id": "c"}
            for i in range(20)
        }
    )
    async with entities._engine.aconnection(graph_path=None) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET LOCAL enable_seqscan = off")
            await cur.execute(
                "EXPLAIN SELECT id FROM LIGHTRAG_VDB_ENTITY "
                "ORDER BY content_vector <=> '[0,0,0,0,0,0,0,1]'::vector LIMIT 5"
            )
            plan = "\n".join(r[0] for r in await cur.fetchall())
        await conn.rollback()
    assert "hnsw" in plan.lower() and "Seq Scan" not in plan
