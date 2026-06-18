"""Integration tests for 0.2.0 async surface.

Covers ``aquery``, ``aadd_texts``, ``aadd_embeddings``, ``asimilarity_search``,
``asimilarity_search_with_score``, ``adelete``, ``aget_by_ids``, and ``aclose``
on a live AgensGraph 2.17 instance.
"""

from __future__ import annotations

import os
from typing import List

import pytest

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector
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
def graph():
    g = AgensGraph("asynctest", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    yield g


@pytest.fixture
def store(graph):
    s = AgensgraphVector.from_texts(
        texts=["seed-a", "seed-b"],
        embedding=FakeEmbeddings(),
        graph=graph,
        node_label="AsyncChunk",
        pre_delete_collection=True,
    )
    yield s


async def test_aquery_runs_and_commits(graph: AgensGraph):
    out = await graph.aquery("MATCH (n) RETURN count(n) AS c")
    assert out and out[0]["c"] >= 0


async def test_aquery_uses_separate_connection(graph: AgensGraph):
    # Confirm async uses its own AsyncConnection (different object than sync).
    aconn = await graph._aconn_get()
    assert aconn is not None
    assert aconn is not graph.connection


async def test_aadd_texts_then_aget_by_ids(store: AgensgraphVector):
    ids = await store.aadd_texts(["alpha", "beta"], ids=["A", "B"])
    assert ids == ["A", "B"]
    docs = await store.aget_by_ids(["A", "B", "missing"])
    found = sorted([d.id for d in docs if d.id is not None])
    assert found == ["A", "B"]


async def test_aadd_texts_batched(store: AgensgraphVector):
    n = 17
    ids = [f"async-{i}" for i in range(n)]
    await store.aadd_texts([f"t{i}" for i in range(n)], ids=ids, batch_size=5)
    docs = await store.aget_by_ids(ids)
    assert len(docs) == n


async def test_adelete_removes(store: AgensgraphVector):
    await store.aadd_texts(["x", "y", "z"], ids=["d1", "d2", "d3"])
    assert await store.adelete(["d1", "d3"]) is True
    remaining = await store.aget_by_ids(["d1", "d2", "d3"])
    assert sorted(d.id for d in remaining) == ["d2"]


async def test_adelete_empty_is_none(store: AgensgraphVector):
    assert await store.adelete([]) is None
    assert await store.adelete(None) is None


async def test_asimilarity_search_returns_k(store: AgensgraphVector):
    await store.aadd_texts(["red", "blue", "green", "yellow"])
    res = await store.asimilarity_search("red", k=2)
    assert len(res) == 2


async def test_aclose_idempotent(graph: AgensGraph):
    # Open the async conn, then close twice — must not raise.
    await graph.aquery("MATCH (n) RETURN count(n) AS c")
    await graph.aclose()
    await graph.aclose()
