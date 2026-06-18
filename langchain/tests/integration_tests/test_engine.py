"""Integration tests for AgensEngine connection pooling."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector
from tests.integration_tests.fake_embeddings import FakeEmbeddings


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


@pytest.fixture
def engine():
    eng = AgensEngine.from_conf(_conf(), min_size=2, max_size=8)
    yield eng
    eng.close()


def test_graph_query_via_engine(engine: AgensEngine):
    g = AgensGraph("enginetest", _conf(), engine=engine, create=True)
    out = g.query("MATCH (n) RETURN count(n) AS c")
    assert out and out[0]["c"] >= 0
    g.close()


def test_concurrent_searches_through_one_engine(engine: AgensEngine):
    store = AgensgraphVector.from_texts(
        texts=["alpha", "beta", "gamma", "delta", "epsilon"],
        embedding=FakeEmbeddings(),
        graph_name="enginetest",
        engine=engine,
        node_label="EngineChunk",
        pre_delete_collection=True,
    )

    def search(i: int):
        return store.similarity_search("alpha", k=2)

    # 8 concurrent workers sharing one pool; must not raise
    # "connection already in use".
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(search, range(32)))
    assert all(len(r) == 2 for r in results)
    store.close()


def test_engine_shared_between_graph_and_vector(engine: AgensEngine):
    g = AgensGraph("enginetest", _conf(), engine=engine, create=True)
    store = AgensgraphVector.from_texts(
        texts=["one", "two"],
        embedding=FakeEmbeddings(),
        graph=g,
        node_label="SharedChunk",
        pre_delete_collection=True,
    )
    # the vector store inherits the graph's engine
    assert store._engine is engine
    assert store.similarity_search("one", k=1)
    store.close()
    g.close()


def test_no_engine_path_unchanged():
    # Sanity: without an engine, a plain graph still works (no pool).
    g = AgensGraph("enginetest", _conf(), create=True)
    assert g._engine is None
    assert g.query("MATCH (n) RETURN count(n) AS c")[0]["c"] >= 0
    g.close()


async def test_async_query_via_engine(engine: AgensEngine):
    g = AgensGraph("enginetest", _conf(), engine=engine, create=True)
    out = await g.aquery("MATCH (n) RETURN count(n) AS c")
    assert out and out[0]["c"] >= 0
    g.close()
    await engine.aclose()
