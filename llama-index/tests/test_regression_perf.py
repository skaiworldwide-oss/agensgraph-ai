'''
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
'''

import os

import pytest

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

agens_available = bool(agens_db and agens_user and agens_password)

pytestmark = pytest.mark.skipif(
    not agens_available,
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)


def _conf():
    return {
        "dbname": agens_db,
        "user": agens_user,
        "password": agens_password,
        "host": agens_host,
        "port": agens_port,
    }


@pytest.fixture()
def vec_store() -> AgensPropertyGraphStore:
    """A property graph store with the HNSW vector index enabled (dim=4)."""
    store = AgensPropertyGraphStore(
        "test_regression", conf=_conf(), vector_dimension=4, create=True
    )
    store.structured_query("MATCH (n) DETACH DELETE n")
    return store


def test_vector_query_tracks_query_embedding(vec_store: AgensPropertyGraphStore):
    """Regression guard for the old ``vector_query``.

    The previous implementation hard-coded ``::vector(3)`` and ordered by a
    hard-coded literal vector, so the ranking ignored the query embedding (and
    erred at any dimension other than 3). Here, at dim=4, the nearest neighbour
    must change when the query embedding changes.
    """
    far = EntityNode(label="POINT", name="far", properties={"embedding": [0.0, 0.0, 0.0, 1.0]})
    near = EntityNode(label="POINT", name="near", properties={"embedding": [1.0, 0.0, 0.0, 0.0]})
    vec_store.upsert_nodes([far, near])

    # Query close to "near"
    res1, _ = vec_store.vector_query(
        VectorStoreQuery(query_embedding=[0.95, 0.0, 0.0, 0.05], similarity_top_k=2)
    )
    assert res1[0].name == "near"

    # Query close to "far" -> ranking must flip (proves it tracks the embedding)
    res2, _ = vec_store.vector_query(
        VectorStoreQuery(query_embedding=[0.05, 0.0, 0.0, 0.95], similarity_top_k=2)
    )
    assert res2[0].name == "far"


def test_vector_query_uses_hnsw_index(vec_store: AgensPropertyGraphStore):
    """The indexed vector query path must be able to use the HNSW index.

    With a small table the planner prefers a sequential scan regardless of any
    index, so we disable seq scans for one transaction: if the HNSW index
    expression matches the query's embedding cast, the planner then uses an
    ``Index Scan`` on the ``entity`` index. (A mismatch -- e.g. the old
    ``->>'embedding'`` vs a Cypher cast -- could not, which is what we guard.)
    """
    from psycopg import sql

    nodes = [
        EntityNode(
            label="POINT",
            name=f"p{i}",
            properties={"embedding": [float(i), 0.0, 0.0, 0.0]},
        )
        for i in range(50)
    ]
    vec_store.upsert_nodes(nodes)

    built = vec_store._build_vector_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=3)
    )
    assert built is not None
    query, params = built

    with vec_store.connection.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute((sql.SQL("EXPLAIN ") + query).as_string(cur), params)
        plan_text = " ".join(" ".join(str(c) for c in row) for row in cur.fetchall())
    vec_store.connection.rollback()

    assert "entity" in plan_text and "Index Scan" in plan_text


def test_enhanced_schema_samples_examples():
    """``enhanced_schema=True`` should surface bounded example values."""
    plain = AgensPropertyGraphStore("test_schema_plain", conf=_conf(), create=True)
    plain.structured_query("MATCH (n) DETACH DELETE n")
    plain.upsert_nodes(
        [EntityNode(label="PERSON", name=f"n{i}") for i in range(3)]
    )
    plain.refresh_schema()
    # Default schema carries no example values.
    for props in plain.structured_schema["node_props"].values():
        assert all("values" not in p for p in props)
    assert "(e.g." not in plain.get_schema_str()

    enhanced = AgensPropertyGraphStore(
        "test_schema_plain", conf=_conf(), enhanced_schema=True, create=True
    )
    enhanced.refresh_schema()
    assert "(e.g." in enhanced.get_schema_str()


def test_structured_query_sanitizes_oversized_lists():
    """``sanitize_query_output`` drops oversized list properties (embedding-like)."""
    big_list = list(range(200))  # >= LIST_LIMIT (128)

    sanitized = AgensPropertyGraphStore("test_sanitize", conf=_conf(), create=True)
    sanitized.structured_query("MATCH (n) DETACH DELETE n")
    sanitized.upsert_nodes(
        [EntityNode(label="PERSON", name="A", properties={"big": big_list})]
    )
    rows = sanitized.structured_query(
        'MATCH (n:"__Node__") RETURN properties(n) AS props'
    )
    assert "big" not in rows[0]["props"]

    raw = AgensPropertyGraphStore(
        "test_sanitize", conf=_conf(), sanitize_query_output=False, create=True
    )
    rows = raw.structured_query(
        'MATCH (n:"__Node__") RETURN properties(n) AS props'
    )
    assert "big" in rows[0]["props"]


@pytest.mark.asyncio
async def test_async_parity_matches_sync(vec_store: AgensPropertyGraphStore):
    """The true-async hot paths must produce the same results as their sync siblings."""
    await vec_store.aupsert_nodes(
        [
            EntityNode(label="PERSON", name="async_a", properties={"embedding": [1.0, 0.0, 0.0, 0.0]}),
            EntityNode(label="PERSON", name="async_b", properties={"embedding": [0.0, 1.0, 0.0, 0.0]}),
        ]
    )
    got = await vec_store.aget()
    assert {n.name for n in got} == {"async_a", "async_b"}

    res, _ = await vec_store.avector_query(
        VectorStoreQuery(query_embedding=[0.9, 0.1, 0.0, 0.0], similarity_top_k=2)
    )
    assert res[0].name == "async_a"


def test_engine_pooling_roundtrip():
    """A store backed by an AgensEngine pool performs a full upsert/get round-trip."""
    engine = AgensEngine.from_conf(_conf(), min_size=1, max_size=4)
    try:
        store = AgensPropertyGraphStore(
            "test_pool", conf=_conf(), vector_dimension=4, engine=engine
        )
        store.structured_query("MATCH (n) DETACH DELETE n")
        store.upsert_nodes([EntityNode(label="PERSON", name="pooled")])
        assert [n.name for n in store.get()] == ["pooled"]
    finally:
        engine.close()


# --------------------------------------------------------------------------- #
# Performance-audit regression guards (index usage, batching)
# --------------------------------------------------------------------------- #


def _url() -> str:
    return (
        f"postgresql://{agens_user}:{agens_password}"
        f"@{agens_host}:{agens_port}/{agens_db}"
    )


def _plan_noseqscan(conn, query, params):
    """EXPLAIN plan text with sequential scans disabled, so the assertion
    reflects whether an index is *usable* regardless of table size."""
    from psycopg import sql

    with conn.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(sql.SQL("EXPLAIN ") + query, params)
        plan = " ".join(r[0] for r in cur.fetchall())
    conn.rollback()
    return plan


def test_pg_get_ids_uses_index(vec_store: AgensPropertyGraphStore):
    """get(ids=...) must be index-backed (OR-of-equalities, not `id <@ list`)."""
    vec_store.upsert_nodes([EntityNode(name=f"e{i}", label="PERSON") for i in range(20)])
    query, params = vec_store._build_get(ids=["e1", "e2", "e3"])
    plan = _plan_noseqscan(vec_store.connection, query, params)
    assert "Seq Scan" not in plan
    assert "Index Scan" in plan or "Bitmap" in plan


def test_vector_get_nodes_ids_uses_index():
    """get_nodes(node_ids=...) must be index-backed (OR-of-equalities)."""
    vs = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_perf_vec", node_label="Chunk"
    )
    vs.clear()
    query, params = vs._build_get_nodes(node_ids=["a", "b", "c"], filters=None)
    plan = _plan_noseqscan(vs._connection, query, params)
    assert "Seq Scan" not in plan
    assert "Index Scan" in plan or "Bitmap" in plan


def test_vector_ref_doc_id_index_present():
    """The vector store indexes ref_doc_id so delete(ref_doc_id) is not a seq scan."""
    vs = AgensgraphVectorStore(
        url=_url(), embedding_dimension=4, graph_name="test_perf_vec2", node_label="Chunk"
    )
    rows = vs.database_query(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'test_perf_vec2' AND tablename = 'Chunk'"
    )
    names = {r["indexname"] for r in rows}
    assert "Chunk_ref_doc_id_idx" in names
    assert "Chunk_id_idx" in names


def test_relation_batching_one_query_per_label(vec_store: AgensPropertyGraphStore):
    """upsert_relations batches per label (was one query per relation)."""
    rels = (
        [Relation(source_id=f"e{i}", target_id=f"e{i + 1}", label="KNOWS") for i in range(5)]
        + [Relation(source_id=f"e{i}", target_id=f"e{i + 1}", label="LIKES") for i in range(5)]
    )
    ops = vec_store._build_upsert_relations_ops(rels)
    assert len(ops) == 2  # one batched UNWIND per distinct label, not 10


def test_create_property_index_enables_filtered_index_scan(vec_store: AgensPropertyGraphStore):
    """After create_property_index, a metadata-filtered vector_query can use it."""
    vec_store.upsert_nodes(
        [
            EntityNode(
                name=f"p{i}",
                label="PERSON",
                properties={
                    "embedding": [float(i), 0.0, 0.0, 0.0],
                    "country": "FR" if i % 2 else "US",
                },
            )
            for i in range(20)
        ]
    )
    vec_store.create_property_index("country")
    built = vec_store._build_vector_query(
        VectorStoreQuery(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            similarity_top_k=3,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="country", value="FR", operator=FilterOperator.EQ)]
            ),
        )
    )
    assert built is not None
    query, params = built
    plan = _plan_noseqscan(vec_store.connection, query, params)
    assert "country_idx" in plan and "Index Scan" in plan
