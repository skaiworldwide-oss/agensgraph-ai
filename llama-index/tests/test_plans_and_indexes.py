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

"""What each statement reaches, and what it does not read.

Named test_regression_perf before, which suggested timings: it held none, and no
thresholds either. What it does hold is worth more -- the plan a statement gets.
Almost everything this migration got wrong was an index that existed and was
never used, and a timing on a small fixture cannot see that while the plan says
it outright.

Where a number is asserted it is a shape rather than a duration: that planning
does not grow with the size of an id list, that a read makes one statement per
label rather than one per row.
"""

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


def _drop_graph(name: str) -> None:
    """Start from nothing.

    A fixture that keeps its graph and only deletes the elements keeps the schema
    too, so a test about what construction declares reads what an earlier run
    declared instead.
    """
    import agensgraph

    conn = agensgraph.Connection.connect(autocommit=True, **_conf())
    conn.execute(f"DROP GRAPH IF EXISTS {name} CASCADE")
    conn.close()


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


def test_create_property_index_indexes_the_label_that_holds_the_rows(
    vec_store: AgensPropertyGraphStore,
):
    """The index has to be on the label the elements are written on.

    An index belongs to one label's storage and does not reach the labels that
    inherit from it, so one built on the base label alone served nothing: the
    base holds no elements.
    """
    from psycopg import sql

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

    # Asked for on its own, the filter is served by the index rather than by
    # reading the label. (Ordered by distance as well, the planner may instead
    # take the vector index and filter after it, which is its decision to make.)
    filter_only = sql.SQL(
        'SELECT t.name FROM (MATCH (n:{label}) WHERE n.country = \'"FR"\'::jsonb '
        "RETURN n.id AS name)t"
    ).format(label=sql.Identifier("PERSON"))
    with vec_store.connection.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute((sql.SQL("EXPLAIN ") + filter_only).as_string(cur))
        plan = " ".join(" ".join(str(c) for c in row) for row in cur.fetchall())
    vec_store.connection.rollback()
    assert "PERSON_country_idx" in plan and "Seq Scan" not in plan


def test_every_label_holding_embeddings_is_vector_indexed(
    vec_store: AgensPropertyGraphStore,
):
    """Not only the base label, which holds none of them.

    Measured on two thousand elements: the child branch of the plan was a
    sequential scan at 5.27 ms with only the base indexed, and an index scan at
    0.38 ms once the label itself was.
    """
    vec_store.upsert_nodes(
        [
            EntityNode(
                name=f"q{i}",
                label="ANIMAL",
                properties={"embedding": [float(i), 1.0, 0.0, 0.0]},
            )
            for i in range(20)
        ]
    )
    names = {
        index.name
        for index in vec_store.connection.indexes(
            "ANIMAL", graph=vec_store.graph_name
        )
    }
    assert "ANIMAL_entity" in names

    built = vec_store._build_vector_query(
        VectorStoreQuery(query_embedding=[1.0, 1.0, 0.0, 0.0], similarity_top_k=3)
    )
    assert built is not None
    plan = _plan_noseqscan(vec_store.connection, *built)
    assert "ANIMAL_entity" in plan and "Seq Scan" not in plan


def test_query_embedding_is_bound_as_a_vector(vec_store: AgensPropertyGraphStore):
    """The embedding travels as itself, not as its decimal spelling.

    A list sent as jsonb is written out as text for the server to parse back: on
    1536 numbers that is 31 KB on the wire against 6 KB, and vector_query measured
    1.39x slower. The statement must lose the cast to match -- a vector needs
    none -- so both are checked together.
    """
    from agensgraph.vector import Vector

    built = vec_store._build_vector_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=3)
    )
    assert built is not None
    query, params = built
    assert isinstance(params["query_embedding"], Vector)
    assert "%(query_embedding)s::vector" not in query.as_string(vec_store.connection)


def test_a_bound_vector_still_reaches_the_hnsw_index(
    vec_store: AgensPropertyGraphStore,
):
    """Binding it differently must not cost the index."""
    from psycopg import sql

    vec_store.upsert_nodes(
        [
            EntityNode(
                label="POINT",
                name=f"v{i}",
                properties={"embedding": [float(i), 0.0, 0.0, 0.0]},
            )
            for i in range(50)
        ]
    )
    query, params = vec_store._build_vector_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=3)
    )
    with vec_store.connection.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute((sql.SQL("EXPLAIN ") + query).as_string(cur), params)
        plan = " ".join(" ".join(str(c) for c in row) for row in cur.fetchall())
    vec_store.connection.rollback()
    assert "entity" in plan and "Index Scan" in plan


def test_bulk_ingest_puts_the_vector_indexes_back(
    vec_store: AgensPropertyGraphStore,
):
    """Including for a label first written inside the block, and after a failure.

    Keeping the indexes current costs more than the writing does: 3,000 elements
    of 384 numbers took 11.02 s written normally and 6.86 s inside the block.
    """

    def vector_indexes():
        return {
            index.name
            for label in ("__Node__", "BIRD")
            for index in vec_store.connection.indexes(
                label, graph=vec_store.graph_name
            )
            if index.name.endswith("entity")
        }

    with vec_store.bulk_ingest():
        assert vector_indexes() == set()
        vec_store.upsert_nodes(
            [
                EntityNode(
                    name=f"b{i}",
                    label="BIRD",
                    properties={"embedding": [float(i), 0.0, 1.0, 0.0]},
                )
                for i in range(5)
            ]
        )
        # Still absent: building it here is the cost this block exists to avoid.
        assert vector_indexes() == set()
    assert "BIRD_entity" in vector_indexes()

    class Boom(Exception):
        pass

    try:
        with vec_store.bulk_ingest():
            raise Boom
    except Boom:
        pass
    assert "BIRD_entity" in vector_indexes()


def test_the_embedding_gets_a_column_and_the_index_matches_it():
    """Read out of the property map an embedding is text in a TOASTed bag, parsed
    once per element a filter kept -- which is where a filtered search spent its
    time: 538.9 ms against 3.0 ms over 20,000 entities with a filter keeping one
    in ten.

    The index has to be spelled for where the value lives. Against a column the
    query says ``n.embedding`` and the cast falls away, so an index built over the
    cast cannot serve it: 568 ms against 1.8 ms, with nothing to say it went
    unused.
    """
    # A graph of its own, made here: sharing one with the other tests meant the
    # column an earlier run added was still there, and this passed with promotion
    # turned off entirely.
    _drop_graph("test_promotion")
    vec_store = AgensPropertyGraphStore(
        "test_promotion", conf=_conf(), vector_dimension=4, create=True
    )
    declared = {
        prop.name
        for prop in vec_store.connection.declared_properties(
            "__Node__", graph=vec_store.graph_name
        )
    }
    vec_store.connection.commit()
    assert "embedding" in declared

    vec_store.upsert_nodes(
        [
            EntityNode(
                name=f"c{i}",
                label="CAT",
                properties={"embedding": [float(i), 0.0, 0.0, 1.0]},
            )
            for i in range(20)
        ]
    )
    definitions = {
        index.name: index.definition
        for index in vec_store.connection.indexes("CAT", graph="test_promotion")
    }
    vec_store.connection.commit()
    assert "::vector(" not in definitions["CAT_entity"]

    built = vec_store._build_vector_query(
        VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 1.0], similarity_top_k=3)
    )
    assert built is not None
    plan = _plan_noseqscan(vec_store.connection, *built)
    assert "CAT_entity" in plan and "Seq Scan" not in plan


def test_promotion_can_be_declined():
    """A server that cannot give a property a column keeps it in the map, and so
    does a caller who says so -- and the index is then spelled with the cast."""
    _drop_graph("test_no_promotion")
    store = AgensPropertyGraphStore(
        "test_no_promotion",
        conf=_conf(),
        vector_dimension=4,
        create=True,
        promote_embedding=False,
    )
    declared = {
        prop.name
        for prop in store.connection.declared_properties(
            "__Node__", graph="test_no_promotion"
        )
    }
    store.connection.commit()
    assert "embedding" not in declared

    store.upsert_nodes(
        [
            EntityNode(
                name="d1", label="DOG", properties={"embedding": [1.0, 0.0, 0.0, 0.0]}
            )
        ]
    )
    definitions = {
        index.name: index.definition
        for index in store.connection.indexes("DOG", graph="test_no_promotion")
    }
    store.connection.commit()
    assert "::vector(4)" in definitions["DOG_entity"]


def test_reading_many_ids_costs_the_same_to_plan_as_reading_one(
    vec_store: AgensPropertyGraphStore,
):
    """An OR of equalities reaches the index and executes at the same speed, but
    the planner works through every term: on 5,000 ids that was 151.7 ms of
    planning against 0.2 ms, roughly half the statement.

    The form matters -- subscripting the list inside the predicate is not
    something the index can serve -- so the plan is checked as well as the cost.
    """
    from psycopg import sql

    vec_store.upsert_nodes(
        [EntityNode(name=f"m{i}", label="MOUSE") for i in range(200)]
    )

    def planning_ms(count: int) -> float:
        query, params = vec_store._build_get(
            None, [f"m{i}" for i in range(count)]
        )
        with vec_store.connection.cursor() as cur:
            cur.execute((sql.SQL("EXPLAIN (ANALYZE) ") + query).as_string(cur), params)
            rows = [r[0] for r in cur.fetchall()]
        vec_store.connection.rollback()
        return min(
            float(r.split(":")[1].split("ms")[0]) for r in rows if "Planning Time" in r
        )

    one = planning_ms(1)
    many = planning_ms(200)
    # Generous: the point is that it does not grow with the list, and the OR form
    # was 16x here for the same step.
    assert many < one * 4 + 1.0, f"planning grew from {one:.2f} ms to {many:.2f} ms"

    query, params = vec_store._build_get(None, [f"m{i}" for i in range(200)])
    plan = _plan_noseqscan(vec_store.connection, query, params)
    assert "MOUSE_unique_id" in plan


def test_reading_many_ids_returns_them(vec_store: AgensPropertyGraphStore):
    """The rewrite must still answer the same question."""
    vec_store.upsert_nodes(
        [EntityNode(name=f"r{i}", label="RAT") for i in range(50)]
    )
    got = vec_store.get(ids=[f"RAT_r{i}" for i in range(0)])
    assert got == []
    ids = [n.id for n in vec_store.get(properties={"name": "r7"})]
    assert len(ids) == 1
    assert {n.name for n in vec_store.get(ids=ids)} == {"r7"}


def test_every_element_label_carries_its_own_uniqueness_on_id(
    vec_store: AgensPropertyGraphStore,
):
    """A constraint on the parent does not reach a child -- measured, the same id
    written to two child labels gave two elements -- so each label needs one, or
    two writers merging the same id each create rather than find."""
    vec_store.upsert_nodes(
        [EntityNode(name=f"u{i}", label="UNIQUELABEL") for i in range(3)]
    )
    unique = {
        constraint.label
        for constraint in vec_store.connection.constraints(graph=vec_store.graph_name)
        if constraint.unique
    }
    vec_store.connection.commit()
    assert "UNIQUELABEL" in unique, sorted(unique)
