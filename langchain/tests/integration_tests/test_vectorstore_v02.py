"""Integration tests for 0.2.0 additions: delete, get_by_ids, batch_size,
effective_search_ratio.

Requires a live AgensGraph database with the pgvector and meta extensions
installed; configuration is read from the same env vars as the existing
integration tests.
"""

from __future__ import annotations

import os
from typing import List

import pytest
from langchain_core.documents import Document

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


def _drop_label(store: AgensgraphVector) -> None:
    """Best-effort cleanup so tests are repeatable."""
    try:
        store.query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


@pytest.fixture
def store():
    AgensGraph("v02test", _conf(), create=True)  # ensure graph exists
    s = AgensgraphVector.from_texts(
        texts=["alpha", "beta", "gamma", "delta"],
        embedding=FakeEmbeddings(),
        graph_name="v02test",
        url=os.environ.get("AGENSGRAPH_URL"),
        node_label="VecChunk",
        pre_delete_collection=True,
    )
    yield s
    _drop_label(s)


def test_delete_removes_nodes(store: AgensgraphVector) -> None:
    # Seed two known ids
    store.add_texts(["one", "two", "three"], ids=["id-1", "id-2", "id-3"])
    deleted = store.delete(["id-1", "id-3"])
    assert deleted is True
    remaining = store.get_by_ids(["id-1", "id-2", "id-3"])
    remaining_ids = sorted([d.id for d in remaining if d.id is not None])
    assert remaining_ids == ["id-2"]


def test_delete_empty_returns_none(store: AgensgraphVector) -> None:
    assert store.delete([]) is None
    assert store.delete(None) is None


def test_get_by_ids_returns_docs(store: AgensgraphVector) -> None:
    store.add_texts(["hello"], ids=["only"])
    docs: List[Document] = store.get_by_ids(["only", "missing"])
    # Missing is silently absent; "only" is present.
    assert len(docs) == 1
    assert docs[0].page_content == "hello"
    assert docs[0].id == "only"


def test_get_by_ids_excludes_embedding_from_metadata(store: AgensgraphVector) -> None:
    store.add_texts(["x"], ids=["x1"])
    docs = store.get_by_ids(["x1"])
    assert docs
    assert store.embedding_node_property not in docs[0].metadata


def test_add_texts_batched(store: AgensgraphVector) -> None:
    # batch_size smaller than payload should still ingest every row.
    n = 25
    texts = [f"t{i}" for i in range(n)]
    ids = [f"b-{i}" for i in range(n)]
    out = store.add_texts(texts, ids=ids, batch_size=4)
    assert out == ids
    docs = store.get_by_ids(ids)
    assert len(docs) == n


def test_effective_search_ratio_does_not_crash_and_returns_k(
    store: AgensgraphVector,
) -> None:
    # Just exercises the parameter path. With ratio > 1 the index fetches
    # more candidates; trimming Python-side must still respect k.
    res = store.similarity_search("alpha", k=2, effective_search_ratio=3.0)
    assert len(res) == 2
