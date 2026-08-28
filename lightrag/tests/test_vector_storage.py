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

"""The vector store: batched writes, binary vectors, and a search that uses its index."""

import numpy as np
import pytest
import pytest_asyncio
from conftest import EMBED_DIM, embed_one, explain
from lightrag.utils import EmbeddingFunc

from lightrag_agensgraph.kg.agensgraph_vector_impl import AgensgraphVectorStorage

pytestmark = pytest.mark.asyncio


def _vec(namespace, embedding_func, **cls_kwargs):
    return AgensgraphVectorStorage(
        namespace=namespace,
        workspace="",
        global_config={
            # The test embedder puts unrelated texts at a distance of about 1, LightRAG's
            # default cutoff, so the stores accept everything unless a test says otherwise.
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": -1.0, **cls_kwargs},
            "embedding_batch_num": 10,
        },
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


def _entity(i, name=None):
    name = name or f"e{i}"
    return {"entity_name": name, "content": f"{name}\ndescription {i}", "source_id": f"c{i}", "file_path": "f"}


async def test_upsert_query_ranking(entities):
    await entities.upsert(
        {
            "ent-1": {"entity_name": "apple", "content": "apple", "source_id": "c1<SEP>c2"},
            "ent-2": {"entity_name": "zebra", "content": "zebra", "source_id": "c3"},
        }
    )
    res = await entities.query("apple", top_k=2)
    assert res[0]["entity_name"] == "apple"  # identical vector ranks first
    assert res[0]["distance"] < res[1]["distance"] and res[0]["id"] == "ent-1"


async def test_records_are_written_when_the_document_is_done(entities, statements):
    await entities.upsert({f"ent-{i}": _entity(i) for i in range(25)})
    assert len(statements) == 0  # nothing sent yet
    await entities.index_done_callback()
    writes = [s for s in statements.statements if s.lstrip().startswith("INSERT")]
    assert len(writes) == 1  # one statement for the batch
    assert (await entities.get_by_id("ent-7"))["entity_name"] == "e7"
    assert len(await entities.get_by_ids([f"ent-{i}" for i in range(25)])) == 25


async def test_embeddings_are_made_in_batches(embedding_func):
    calls = []

    async def counting(texts, **kwargs):
        calls.append(len(texts))
        return np.array([embed_one(t) for t in texts], dtype=float)

    store = _vec("entities", EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=counting))
    await store.initialize()
    await store.drop()
    try:
        for i in range(25):
            await store.upsert({f"ent-{i}": _entity(i)})  # one record per call, as LightRAG does
        assert calls == []
        await store.index_done_callback()
        assert calls == [10, 10, 5]
    finally:
        await store.drop()
        await store.finalize()


async def test_a_pending_record_is_read_and_a_delete_cancels_it(entities):
    await entities.upsert({"ent-1": _entity(1, "apple"), "ent-2": _entity(2, "pear")})
    assert (await entities.get_by_id("ent-1"))["entity_name"] == "apple"  # read through the buffer
    vectors = await entities.get_vectors_by_ids(["ent-1", "ent-2"])
    assert set(vectors) == {"ent-1", "ent-2"} and len(vectors["ent-1"]) == EMBED_DIM
    await entities.delete(["ent-2"])
    await entities.index_done_callback()
    assert await entities.get_by_id("ent-2") is None and await entities.get_by_id("ent-1") is not None
    # A delete before a new write of the same id leaves the new record.
    await entities.delete(["ent-1"])
    await entities.upsert({"ent-1": _entity(1, "apple again")})
    await entities.index_done_callback()
    assert (await entities.get_by_id("ent-1"))["entity_name"] == "apple again"
    await entities.upsert({"ent-9": _entity(9)})
    await entities.drop_pending_index_ops()
    await entities.index_done_callback()
    assert await entities.get_by_id("ent-9") is None


async def test_get_by_id_strips_vector_and_splits_chunks(entities):
    await entities.upsert({"ent-1": {"entity_name": "apple", "content": "a", "source_id": "c1<SEP>c2"}})
    await entities.index_done_callback()
    rec = await entities.get_by_id("ent-1")
    assert "content_vector" not in rec
    assert rec["chunk_ids"] == ["c1", "c2"] and rec["source_id"] == "c1<SEP>c2"
    assert isinstance(rec["created_at"], int)
    vecs = await entities.get_vectors_by_ids(["ent-1"])
    assert len(vecs["ent-1"]) == EMBED_DIM
    assert np.allclose(vecs["ent-1"], embed_one("a"), atol=1e-6)


async def test_delete_and_delete_entity(entities):
    await entities.upsert(
        {
            "ent-1": {"entity_name": "apple", "content": "a", "source_id": "c1"},
            "ent-2": {"entity_name": "pear", "content": "p", "source_id": "c2"},
        }
    )
    await entities.index_done_callback()
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
        assert res[0]["tokens"] == 3 and res[0]["full_doc_id"] == "d1"
    finally:
        await chunks.drop()
        await chunks.finalize()


async def test_relations_are_found_by_either_endpoint(embedding_func):
    rel = _vec("relationships", embedding_func)
    await rel.initialize()
    await rel.drop()
    try:
        await rel.upsert(
            {f"r{i}": {"src_id": f"e{i}", "tgt_id": f"e{i + 1}", "content": "c", "source_id": "s"} for i in range(5)}
        )
        await rel.index_done_callback()
        hits = await rel.query("c", top_k=5)
        assert {h["src_id"] for h in hits} == {f"e{i}" for i in range(5)}
        await rel.delete_entity_relation("e2")  # r1 (tgt) and r2 (src)
        assert {r["id"] for r in await rel.get_by_ids([f"r{i}" for i in range(5)])} == {"r0", "r3", "r4"}
        async with rel._engine.connection() as conn:
            plan = await explain(
                conn,
                "DELETE FROM LIGHTRAG_VDB_RELATION WHERE workspace = '' AND (src_id = 'e1' OR tgt_id = 'e1')",
                by_index=True,
            )
        assert "Index" in plan and "Seq Scan" not in plan
    finally:
        await rel.drop()
        await rel.finalize()


@pytest_asyncio.fixture
async def many(entities):
    await entities.upsert({f"ent-{i}": _entity(i) for i in range(200)})
    await entities.index_done_callback()
    async with entities._engine.connection() as conn:
        await conn.execute("ANALYZE LIGHTRAG_VDB_ENTITY")
    return entities


async def test_a_search_runs_on_the_hnsw_index(many):
    # The planner left to itself scans the table (it cannot see the out-of-line vectors);
    # the store's own preamble makes it walk the index. The plan is read the way the store
    # runs the statement.
    probe = embed_one("e7\ndescription 7")
    statement = (
        "SELECT id FROM LIGHTRAG_VDB_ENTITY WHERE workspace = %(ws)s AND content_vector <=> %(v)b < %(t)s "
        "ORDER BY content_vector <=> %(v)b LIMIT 40"
    )
    from agensgraph import Vector

    params = {"ws": "", "v": Vector(probe), "t": 1.0}
    async with many._engine.connection() as conn:
        default = await explain(conn, statement, params)
        async with conn.transaction(force_rollback=True):
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
            await conn.execute("SET LOCAL enable_sort = off")
            await conn.vector_search_options({"hnsw.iterative_scan": "relaxed_order"})
            cur = await conn.execute("EXPLAIN (ANALYZE, COSTS OFF) " + statement, params)
            tuned = "\n".join(r[0] for r in await cur.fetchall())
    assert "hnsw" not in default.lower()  # what the planner does on its own
    assert "hnsw" in tuned.lower() and "Seq Scan" not in tuned and "Bitmap Heap Scan" not in tuned
    hits = await many.query("e7\ndescription 7", top_k=5)
    assert hits[0]["id"] == "ent-7"


async def test_a_limit_above_the_default_candidate_count_is_honoured(many):
    hits = await many.query("anything", top_k=120)
    assert len(hits) == 120  # ef_search is raised with the limit; the index would stop at 40


async def test_bulk_ingest_rebuilds_the_index(entities):
    async with entities.bulk_ingest():
        await entities.upsert({f"ent-{i}": _entity(i) for i in range(30)})
    rows = await entities._fetch_tuples(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'lightrag_vdb_entity' AND indexname = 'lightrag_vdb_entity_hnsw'"
    )
    assert rows and "hnsw" in rows[0][0]
    assert len(await entities.get_by_ids([f"ent-{i}" for i in range(30)])) == 30


async def test_a_different_embedding_width_is_refused(entities):
    async def eight(texts, **kwargs):
        return np.zeros((len(texts), 8))

    store = _vec("entities", EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=eight))
    with pytest.raises(ValueError, match="vector\\(1536\\)"):
        await store.initialize()
    await store.finalize()
