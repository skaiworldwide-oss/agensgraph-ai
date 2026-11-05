"""The vector store against a server: writes, reads, filters, searches, the index, lifecycle."""

from __future__ import annotations

import numpy as np
import psycopg
import pytest
from agensgraph import Jsonb
from agensgraph.errors import StaleVectorStorage
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from psycopg import sql
from psycopg.conninfo import make_conninfo

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph._core.program import Program, Statement
from langchain_agensgraph.errors import SchemaMismatchError, UnknownLabel
from langchain_agensgraph.graphs import AgensGraph
from langchain_agensgraph.retrievers import (
    AgensGraphContextRetriever,
    AgensVectorRetriever,
    render_graph_context,
)
from langchain_agensgraph.retrievers.graph_context import walk
from langchain_agensgraph.roles import render_grants
from langchain_agensgraph.vectorstores import (
    AgensgraphVector,
    DistanceStrategy,
    KeywordConfig,
    SearchType,
)
from langchain_agensgraph.vectorstores.config import IndexConfig
from tests import datagen
from tests.filter_oracle import matches
from tests.goldens.golden import FixedEmbeddings
from tests.integration_tests.fixtures import filtering_test_cases as cases

pytestmark = pytest.mark.server

DIMS = 8
FILTERS = [
    {"s10": 3},
    {"lang": {"$in": ["ko", "ur"]}},
    {"lang": {"$nin": ["ko"]}},
    {"year": {"$gte": 2020}},
    {"price": {"$lt": 10}},
    {"price": {"$between": [10, 20]}},
    {"flag": True},
    {"note": {"$exists": True}},
    {"note": {"$exists": False}},
    {"lang": {"$ne": "en"}},
    {"note": {"$like": "n1%"}},
    {"note": {"$ilike": "N2_"}},
    {"$and": [{"s10": 3}, {"year": {"$lt": 2010}}]},
    {"$or": [{"s1": 7}, {"s1": 8}]},
    {"$not": {"flag": True}},
    {"year": {"$gt": "2000"}},
]


@pytest.fixture
def chunks() -> list[datagen.Chunk]:
    return datagen.chunks(200, DIMS, seed=11)


@pytest.fixture
def embedding(chunks: list[datagen.Chunk]) -> FixedEmbeddings:
    return FixedEmbeddings({c.text: c.embedding for c in chunks}, DIMS)


def store_for(
    engine: AgensEngine, graph_name: str, embedding: object, **kwargs: object
) -> AgensgraphVector:
    return AgensgraphVector(
        embedding,  # type: ignore[arg-type]
        engine=engine,
        graph_name=graph_name,
        create_graph=True,
        **kwargs,  # type: ignore[arg-type]
    )


def load(store: AgensgraphVector, chunks: list[datagen.Chunk]) -> list[str]:
    return store.add_texts(
        [c.text for c in chunks], [dict(c.metadata) for c in chunks], ids=[c.id for c in chunks]
    )


def brute_force(
    chunks: list[datagen.Chunk], query: np.ndarray, k: int, where: dict | None = None
) -> list[str]:
    kept = [c for c in chunks if matches(where, c.metadata)]
    distances = [1.0 - float(np.dot(c.embedding, query)) for c in kept]
    order = np.argsort(distances, kind="stable")
    return [kept[i].id for i in order[:k]]


def test_texts_are_stored_and_found_nearest_first(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    assert load(store, chunks) == [c.id for c in chunks]
    target = chunks[17]
    found = store.similarity_search_with_score(target.text, k=3)
    assert found[0][0] == Document(id=target.id, page_content=target.text, metadata=target.metadata)
    assert found[0][1] == pytest.approx(0.0, abs=1e-6)
    assert [d.id for d, _ in found] == brute_force(chunks, target.embedding, 3)
    assert store.dims == DIMS
    assert store.index_status()["indexed"]


def test_every_filter_matches_its_python_meaning(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding, search_mode="exact")
    load(store, chunks)
    query = chunks[5].embedding
    for where in FILTERS:
        got = [d.id for d in store.similarity_search_by_vector(query.tolist(), k=7, filter=where)]
        assert got == brute_force(chunks, query, 7, where), where


def test_the_langchain_postgres_filter_cases_hold(engine: AgensEngine, graph_name: str) -> None:
    store = store_for(
        engine, graph_name, DeterministicFakeEmbedding(size=DIMS), search_mode="exact"
    )
    store.add_documents(cases.DOCUMENTS, ids=[str(d.metadata["id"]) for d in cases.DOCUMENTS])
    wrong = []
    for where, expected in (
        cases.TYPE_1_FILTERING_TEST_CASES
        + cases.TYPE_2_FILTERING_TEST_CASES
        + cases.TYPE_3_FILTERING_TEST_CASES
        + cases.TYPE_4_FILTERING_TEST_CASES
        + cases.TYPE_5_FILTERING_TEST_CASES
    ):
        found = sorted(d.metadata["id"] for d in store.similarity_search("meow", k=5, filter=where))
        if found != expected:
            wrong.append((where, expected, found))
    assert wrong == []


def test_a_filtered_approximate_search_still_returns_k(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding, filter_keys=["s10"])
    load(store, chunks)
    assert store.index_status()["indexed"]
    found = store.similarity_search_by_vector(chunks[0].embedding.tolist(), k=10, filter={"s10": 3})
    assert len(found) == 10
    assert all(d.metadata["s10"] == 3 for d in found)


def test_adding_an_id_again_replaces_the_chunk(
    engine: AgensEngine, graph_name: str, embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, DeterministicFakeEmbedding(size=DIMS))
    store.add_texts(["foo", "bar"], [{"a": 1, "gone": True}, {"a": 2}], ids=["1", "2"])
    store.add_texts(["new foo"], [{"a": 3}], ids=["1"])
    store.add_texts(["new foo"], [{"a": 3}], ids=["1"])
    assert store.get_by_ids(["2", "missing", "1"]) == [
        Document(id="2", page_content="bar", metadata={"a": 2}),
        Document(id="1", page_content="new foo", metadata={"a": 3}),
    ]
    assert store.similarity_search("new foo", k=5)[0].id == "1"
    assert len(store.similarity_search("anything", k=5)) == 2


def test_metadata_drops_none_and_refuses_the_keys_the_store_uses(
    engine: AgensEngine, graph_name: str
) -> None:
    store = store_for(engine, graph_name, DeterministicFakeEmbedding(size=DIMS))
    store.add_texts(["a"], [{"kept": 1, "dropped": None}], ids=["a"])
    assert store.get_by_ids(["a"])[0].metadata == {"kept": 1}
    for reserved in ("__id__", "text"):
        with pytest.raises(ValueError, match="metadata may not hold"):
            store.add_texts(["b"], [{reserved: "x"}])
    with pytest.raises(ValueError, match="2 texts but 1 ids"):
        store.add_texts(["c", "d"], ids=["c"])


def test_deleting_by_id_by_filter_and_everything(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:40])
    store.delete([chunks[0].id, "not stored"])
    assert store.get_by_ids([chunks[0].id]) == []
    store.delete(filter={"s10": 1})
    left = store.similarity_search(chunks[2].text, k=100)
    assert len(left) == 35 and all(d.metadata["s10"] != 1 for d in left)
    for everything in ({}, {"$and": [{}]}):
        # A filter testing no field matches every chunk, so it is refused as delete() is.
        with pytest.raises(ValueError, match="delete_all"):
            store.delete(filter=everything)
        with pytest.raises(ValueError, match="delete_all"):
            store.delete([chunks[1].id], filter=everything)
    assert len(store.similarity_search(chunks[2].text, k=100)) == 35
    with pytest.raises(ValueError, match="delete_all"):
        store.delete()
    store.delete(delete_all=True)
    assert store.similarity_search(chunks[2].text, k=5) == []


def test_a_store_already_made_says_its_own_type_to_every_first_call(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    load(store_for(engine, graph_name, embedding, vector_type="halfvec"), chunks[:5])
    # A store opened without a type, first called with embeddings, takes halfvec from the table.
    later = store_for(engine, graph_name, embedding)
    rest = chunks[5:10]
    later.add_embeddings(
        [c.text for c in rest], [c.embedding.tolist() for c in rest], ids=[c.id for c in rest]
    )
    assert later.vector_type == "halfvec"
    assert later.similarity_search(chunks[7].text, k=1)[0].id == chunks[7].id


def test_a_k_above_one_index_scan_is_read_on_iteratively(
    engine: AgensEngine, graph_name: str
) -> None:
    many = datagen.chunks(1300, 16, seed=11)
    embedding = FixedEmbeddings({c.text: c.embedding for c in many}, 16)
    store = store_for(engine, graph_name, embedding, search_type="hybrid")
    load(store, many)
    assert store.index_status()["indexed"]
    # pgvector takes hnsw.ef_search up to 1000; past it the scan reads on until k rows are found.
    assert len(store.similarity_search(many[0].text, k=1200)) == 1200
    assert len(store.hybrid_search_with_details(many[0].text, k=300)) == 300
    capped = store_for(engine, graph_name, embedding, search_type="hybrid", iterative_scan="off")
    with pytest.raises(ValueError, match="iterative scan"):
        capped.similarity_search(many[0].text, k=1200)


def test_mmr_trades_nearness_for_variety(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding, search_mode="exact")
    load(store, chunks)
    query = chunks[9].embedding.tolist()
    nearest = [d.id for d in store.similarity_search_by_vector(query, k=4)]
    assert [
        d.id for d in store.max_marginal_relevance_search_by_vector(query, k=4, lambda_mult=1.0)
    ] == nearest
    varied = store.max_marginal_relevance_search_by_vector(query, k=4, fetch_k=30, lambda_mult=0.1)
    assert len(varied) == 4 and varied[0].id == nearest[0]


def test_stored_embeddings_come_back_as_given(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:10])
    (document, distance, vector), *_ = store.search_with_embeddings(chunks[3].text, k=1)
    assert document.id == chunks[3].id
    assert np.allclose(vector, chunks[3].embedding)


def test_relevance_scores_run_from_nearest_to_furthest(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    for strategy in (DistanceStrategy.COSINE, DistanceStrategy.EUCLIDEAN_DISTANCE):
        store = store_for(
            engine, graph_name, embedding, distance_strategy=strategy, label=f"L{strategy.name}"
        )
        load(store, chunks[:20])
        scored = store.similarity_search_with_relevance_scores(chunks[4].text, k=5)
        assert scored[0][1] == pytest.approx(1.0, abs=1e-5)
        assert all(0.0 <= s <= 1.0 for _, s in scored)
        assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)


def test_a_hybrid_search_finds_the_rare_word(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding, search_type=SearchType.HYBRID)
    load(store, chunks)
    rare = next(c for c in chunks if datagen.RARE_WORD in c.text)
    hits = store.hybrid_search_with_details(datagen.RARE_WORD, k=5)
    assert hits[0].document.id == rare.id
    assert hits[0].keyword_rank == 1 and hits[0].keyword_score is not None
    scored = store.similarity_search_with_relevance_scores(datagen.RARE_WORD, k=3)
    assert scored[0][0].id == rare.id and 0.0 < scored[0][1] <= 1.0
    mixed = store_for(
        engine,
        graph_name,
        embedding,
        search_type="hybrid",
        fusion="max",
        label="Max",
        keyword=KeywordConfig(regconfig="simple"),
    )
    load(mixed, chunks)
    assert mixed.similarity_search(datagen.RARE_WORD, k=1)[0].id == rare.id


def test_relevance_score_fn_says_what_a_distance_means(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    def near(distance: float) -> float:
        return 1.0 / (1.0 + distance)

    taxicab = {"distance_strategy": "TAXICAB", "index": None}
    plain = store_for(engine, graph_name, embedding, relevance_score_fn=near, **taxicab)
    load(plain, chunks[:20])
    (doc, relevance), *_ = plain.similarity_search_with_relevance_scores(chunks[3].text, k=3)
    assert doc.id == chunks[3].id and relevance == 1.0
    # A max-fused search needs a relevance for its vector half, which TAXICAB has only when given.
    fused = {"search_type": "hybrid", "fusion": "max", **taxicab}
    said = store_for(engine, graph_name, embedding, label="Said", relevance_score_fn=near, **fused)
    load(said, chunks[:20])
    assert said.similarity_search(chunks[3].text, k=1)[0].id == chunks[3].id
    unsaid = store_for(engine, graph_name, embedding, label="Unsaid", **fused)
    load(unsaid, chunks[:20])
    with pytest.raises(ValueError, match="relevance_score_fn"):
        unsaid.similarity_search(chunks[3].text, k=1)


def test_the_index_waits_for_a_callers_transaction_to_end(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    with engine.transaction():
        load(store, chunks[:20])
        assert store.index_status() == {
            "indexed": False,
            "pending": True,
            "refused": False,
            "state": "absent",
        }
    assert store.build_index()
    assert store.index_status()["indexed"]


def test_an_index_a_rolled_back_transaction_built_is_built_again(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    store.setup()
    with pytest.raises(RuntimeError), engine.transaction():
        load(store, chunks[:20])
        assert store.build_index()
        raise RuntimeError
    # The rollback took the rows and the index; the next write builds the index again.
    load(store, chunks[:20])
    assert store.index_status()["indexed"]


@pytest.mark.timeout(60)
def test_setup_inside_a_callers_transaction_runs_in_it(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    graph = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    graph.ensure_labels(["Chunk"])
    store = store_for(engine, graph_name, embedding)
    with pytest.raises(RuntimeError):
        with engine.transaction():
            # Setup puts indexes and a foreign key on Chunk, which this transaction has written
            # to: on a connection of its own it would wait for the transaction to end.
            graph.query('CREATE (:"Chunk" {note: %s})', ("written first",))
            load(store, chunks[:5])
            with pytest.raises(ValueError, match="cannot run inside a transaction"):
                store.build_index(concurrently=True)
            raise RuntimeError
    # The rollback took the store's tables with it, so the next call makes them again.
    load(store, chunks[:5])
    assert [d.id for d in store.similarity_search(chunks[0].text, k=1)] == [chunks[0].id]


@pytest.mark.parametrize("concurrently", [False, True])
def test_an_index_too_big_for_shared_memory_is_built_serially(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
    monkeypatch: pytest.MonkeyPatch,
    concurrently: bool,
) -> None:
    store = store_for(engine, graph_name, embedding, index_build="manual")
    load(store, chunks)
    table = store.layout.vector_table_regclass
    # Parallel workers share a segment of all of maintenance_work_mem from /dev/shm, which is
    # 256 MB on the test image. A table this small gets no workers unless it asks for them.
    monkeypatch.setattr(store, "_build_memory", lambda rows: 1024)
    build_with = store._build_with
    serial_builds: list[bool] = []

    def recorded(memory: str, *, concurrently: bool, serial: bool) -> Program[None]:
        serial_builds.append(serial)
        return build_with(memory, concurrently=concurrently, serial=serial)

    monkeypatch.setattr(store, "_build_with", recorded)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"ALTER TABLE {table} SET (parallel_workers = 2)")
    assert store.build_index(concurrently=concurrently)
    # The parallel build was refused its shared memory, and the serial one built the index.
    assert serial_builds == [False, True]
    assert store.index_status()["indexed"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        invalid = conn.execute(
            "SELECT count(*) FROM pg_index WHERE indrelid = %s::regclass AND NOT indisvalid",
            (table,),
        ).fetchone()
    assert invalid == (0,)


def test_a_failed_transaction_takes_the_chunks_with_it(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    store.setup()
    with pytest.raises(RuntimeError):
        with engine.transaction():
            load(store, chunks[:5])
            raise RuntimeError
    assert store.get_by_ids([c.id for c in chunks[:5]]) == []


def test_an_existing_store_is_found_again_with_its_dimension(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    load(store_for(engine, graph_name, embedding), chunks[:5])
    again = AgensgraphVector.from_existing_index(embedding, engine=engine, graph_name=graph_name)
    assert again.dims == DIMS
    assert again.get_by_ids([chunks[0].id])[0].page_content == chunks[0].text
    with pytest.raises(SchemaMismatchError):
        AgensgraphVector.from_existing_index(
            embedding, engine=engine, graph_name=graph_name, label="Absent"
        )


def test_a_missing_graph_is_not_made_unless_asked(engine: AgensEngine, graph_name: str) -> None:
    store = AgensgraphVector(
        DeterministicFakeEmbedding(size=DIMS), engine=engine, graph_name=graph_name
    )
    with pytest.raises(UnknownLabel, match="create_graph=True"):
        store.add_texts(["x"])


def test_vectors_left_by_a_dropped_graph_are_refused(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    embedding: FixedEmbeddings,
    chunks: list[datagen.Chunk],
) -> None:
    load(store_for(engine, graph_name, embedding), chunks[:5])
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP GRAPH {} CASCADE").format(sql.Identifier(graph_name)))
    fresh = store_for(engine, graph_name, embedding)
    with pytest.raises(StaleVectorStorage):
        fresh.add_texts(["after"], ids=["after"])


def test_a_retrieval_query_reads_each_hit_through_cypher(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:30])
    retriever = AgensVectorRetriever(
        vectorstore=store,
        k=2,
        retrieval_query=(
            "RETURN node.text AS text, score, {lang: node.lang, shout: 'yes'} AS metadata, "
            "node.__id__ AS doc_id"
        ),
    )
    docs = retriever.invoke(chunks[6].text)
    assert docs[0].id == chunks[6].id
    assert docs[0].metadata == {"lang": chunks[6].metadata["lang"], "shout": "yes", "score": 0.0}


def test_a_retrieval_query_sees_the_hits_best_first_whatever_the_join(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:30])
    ranked = [d.id for d in store.similarity_search(chunks[20].text, k=5)]
    retriever = AgensVectorRetriever(
        vectorstore=store,
        k=5,
        retrieval_query="RETURN node.text AS text, score, {} AS metadata, node.__id__ AS doc_id",
    )

    aggregating = AgensVectorRetriever(
        vectorstore=store,
        k=5,
        retrieval_query=(
            "OPTIONAL MATCH (node)-[]->(x) WITH node, score, count(x) AS near "
            "RETURN node.text AS text, score, {near: near} AS metadata, node.__id__ AS doc_id"
        ),
    )

    def hashing() -> Program[object]:
        # A hash join or a hash aggregate hands rows on in an order of its own, not the hits'.
        return (
            yield Statement(
                "SELECT set_config('enable_nestloop', 'off', true), "
                "set_config('enable_mergejoin', 'off', true), "
                "set_config('enable_sort', 'off', true)"
            )
        )

    with engine.transaction():
        engine.run(hashing, graph=graph_name)
        assert [d.id for d in retriever.invoke(chunks[20].text)] == ranked
        assert [d.id for d in aggregating.invoke(chunks[20].text)] == ranked


def test_the_retriever_takes_k_from_the_call_then_the_search_kwargs(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:30])
    retriever = AgensVectorRetriever(vectorstore=store, k=1, search_kwargs={"k": 3})
    assert len(retriever.invoke(chunks[0].text)) == 3
    assert len(retriever.invoke(chunks[0].text, k=5)) == 5
    assert all("score" in d.metadata for d in retriever.invoke(chunks[0].text))


def test_graph_context_brings_the_neighbourhood(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:10])
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("SET graph_path = {}").format(sql.Identifier(graph_name)))
        conn.execute("CREATE VLABEL Entity")
        conn.execute("CREATE ELABEL MENTIONS")
        conn.execute("CREATE ELABEL KNOWS")
        conn.execute(
            "MATCH (c:\"Chunk\" {__id__: %s}) CREATE (c)-[:MENTIONS]->(:Entity {name: 'Ada'})"
            "-[:KNOWS]->(:Entity {name: 'Bob'})",
            (Jsonb(chunks[0].id),),
        )
    retriever = AgensGraphContextRetriever(
        vectorstore=store, k=1, hops=2, document_formatter=render_graph_context
    )
    (doc,) = retriever.invoke(chunks[0].text)
    names = sorted(n["properties"]["name"] for n in doc.metadata["_context_nodes_"])
    assert names == ["Ada", "Bob"]
    assert sorted(r["type"] for r in doc.metadata["_context_rels_"]) == ["knows", "mentions"]
    assert "entity:Ada -[knows]-> entity:Bob" in doc.page_content
    one_hop = AgensGraphContextRetriever(vectorstore=store, k=1, hops=1)
    assert len(one_hop.invoke(chunks[0].text)[0].metadata["_context_nodes_"]) == 1
    typed = AgensGraphContextRetriever(vectorstore=store, k=1, hops=2, relationship_type="mentions")
    assert len(typed.invoke(chunks[0].text)[0].metadata["_context_nodes_"]) == 1
    # A retrieval query the store holds is for AgensVectorRetriever; the seeds here go past it.
    queried = store_for(
        engine, graph_name, embedding, retrieval_query="RETURN node.text AS text, score, {} AS m"
    )
    (seed,) = AgensGraphContextRetriever(vectorstore=queried, k=1, hops=2).invoke(chunks[0].text)
    assert seed.metadata["_seed_id_"] != "None"
    assert sorted(n["properties"]["name"] for n in seed.metadata["_context_nodes_"]) == [
        "Ada",
        "Bob",
    ]


def test_graph_context_is_capped_and_a_seed_alone_has_none(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:3])
    numbers = ", ".join(str(i) for i in range(30))
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("SET graph_path = {}").format(sql.Identifier(graph_name)))
        conn.execute("CREATE VLABEL Entity")
        conn.execute("CREATE ELABEL MENTIONS")
        conn.execute(
            f'MATCH (c:"Chunk" {{__id__: %s}}) UNWIND [{numbers}] AS i '
            "CREATE (c)-[:MENTIONS]->(:Entity {i: i})",
            (Jsonb(chunks[0].id),),
        )
    few_nodes = AgensGraphContextRetriever(vectorstore=store, k=1, max_context_nodes=5)
    (hub,) = few_nodes.invoke(chunks[0].text)
    assert len(hub.metadata["_context_nodes_"]) == 5
    assert len(hub.metadata["_context_rels_"]) <= 20
    few_edges = AgensGraphContextRetriever(vectorstore=store, k=1, max_context_rels=4)
    (hub,) = few_edges.invoke(chunks[0].text)
    assert len(hub.metadata["_context_rels_"]) == len(hub.metadata["_context_nodes_"]) == 4
    (alone,) = few_nodes.invoke(chunks[1].text)
    assert alone.metadata["_context_nodes_"] == [] and alone.metadata["_context_rels_"] == []


def test_graph_context_reads_only_what_its_caps_keep_and_keeps_what_the_whole_walk_would(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
) -> None:
    store = store_for(engine, graph_name, embedding)
    load(store, chunks[:3])
    numbers = ", ".join(str(i) for i in range(12))
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("SET graph_path = {}").format(sql.Identifier(graph_name)))
        conn.execute("CREATE VLABEL Entity")
        conn.execute("CREATE ELABEL MENTIONS")
        conn.execute("CREATE ELABEL KNOWS")
        conn.execute(
            f'MATCH (c:"Chunk" {{__id__: %s}}) UNWIND [{numbers}] AS i '
            "CREATE (c)-[:MENTIONS]->(:Entity {i: i})",
            (Jsonb(chunks[0].id),),
        )
        conn.execute("MATCH (e:Entity) UNWIND [0, 1, 2] AS j CREATE (e)-[:KNOWS]->(:Entity {j: j})")
    retriever = AgensGraphContextRetriever(
        vectorstore=store, k=1, hops=2, max_context_nodes=7, max_context_rels=9
    )
    (doc,) = retriever.invoke(chunks[0].text)
    hits = store._search_hits(chunks[0].text, 1, None, retrieval_query=None)
    seeds = [gid for _, _, gid in hits]
    capped = engine.run(lambda: retriever._walk(hits), graph=graph_name)
    whole = engine.run(
        lambda: walk(
            graph_name, seeds, hops=2, cap=9, nodes=10**6, rels=10**6, relationship_type=None
        ),
        graph=graph_name,
    )
    assert len(capped) < len(whole)
    ((expected, _),) = retriever._with_context(hits, whole)
    assert doc.metadata["_context_nodes_"] == expected.metadata["_context_nodes_"]
    assert doc.metadata["_context_rels_"] == expected.metadata["_context_rels_"]
    assert len(doc.metadata["_context_nodes_"]) == 7
    assert len(doc.metadata["_context_rels_"]) == 9


@pytest.mark.parametrize(
    ("index", "built_with"),
    [
        (IndexConfig(method="ivfflat", lists=4), "lists='4'"),
        (IndexConfig(m=8, ef_construction=32), "m='8', ef_construction='32'"),
    ],
)
def test_each_index_is_built_with_its_settings_and_searched(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
    index: IndexConfig,
    built_with: str,
) -> None:
    store = store_for(engine, graph_name, embedding, index=index)
    load(store, chunks)
    assert store.index_status()["indexed"]
    with psycopg.connect(dsn) as conn:
        definitions = [
            row[0]
            for row in conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = %s",
                (f"{graph_name}_store",),
            )
        ]
    assert any(f"USING {index.method}" in d and built_with in d for d in definitions)
    found = store.similarity_search_by_vector(list(chunks[7].embedding), k=3)
    assert found[0].id == chunks[7].id


def test_a_least_privileged_role_writes_and_searches_but_leaves_the_index_to_the_owner(
    dsn: str,
    engine: AgensEngine,
    graph_name: str,
    chunks: list[datagen.Chunk],
    embedding: FixedEmbeddings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner_store = store_for(engine, graph_name, embedding, index_build="manual")
    owner_store.setup()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(render_grants(graph_name))
    with AgensEngine(make_conninfo(dsn, user="app_rw", password="agens")) as app:
        store = AgensgraphVector(embedding, engine=app, graph_name=graph_name)
        load(store, chunks[:20])
        assert store.similarity_search(chunks[1].text, k=1)[0].id == chunks[1].id
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            store.build_index()
        store.delete([chunks[1].id])
    assert owner_store.build_index()


async def test_the_awaiting_interface_does_the_same(
    engine: AgensEngine, graph_name: str, chunks: list[datagen.Chunk], embedding: FixedEmbeddings
) -> None:
    store = store_for(engine, graph_name, embedding, search_type="hybrid")
    ids = await store.aadd_texts(
        [c.text for c in chunks[:30]],
        [dict(c.metadata) for c in chunks[:30]],
        ids=[c.id for c in chunks[:30]],
    )
    assert ids == [c.id for c in chunks[:30]]
    assert (await store.asimilarity_search(chunks[3].text, k=1))[0].id == chunks[3].id
    assert [d.id for d in await store.aget_by_ids([chunks[2].id])] == [chunks[2].id]
    await store.adelete([chunks[3].id])
    assert await store.aget_by_ids([chunks[3].id]) == []
    retriever = AgensGraphContextRetriever(vectorstore=store, k=2)
    assert len(await retriever.ainvoke(chunks[4].text)) == 2
    await engine.aclose()


async def test_an_awaiting_store_infers_its_dimension(engine: AgensEngine, graph_name: str) -> None:
    store = store_for(engine, graph_name, DeterministicFakeEmbedding(size=5))
    await store.aadd_texts(["one", "two"])
    assert store.dims == 5
    await engine.aclose()
