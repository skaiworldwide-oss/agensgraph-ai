"""Regression tests: the integration creates the property indexes that keep
its hot paths index-backed (MERGE/delete/get_by_ids, per-session lookups,
checkpoint reads). These guard against silently shipping seq-scan paths.
"""

from __future__ import annotations

import os

from langchain_agensgraph import (
    AgensChatMessageHistory,
    AgensGraph,
    AgensSaver,
    AgensgraphVector,
)
from tests.integration_tests.fake_embeddings import FakeEmbeddings


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


def _indexdefs(graph: AgensGraph, label: str):
    rows = graph.query(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = %(g)s AND tablename = %(t)s",
        {"g": graph.graph_name, "t": label},
    )
    return [r["indexdef"] for r in rows]


def test_vectorstore_indexes_id_on_merge_key():
    store = AgensgraphVector.from_texts(
        ["a", "b", "c"],
        embedding=FakeEmbeddings(),
        url=os.environ.get("AGENSGRAPH_URL"),
        graph_name="idxtest",
        node_label="IdxChunk",
        index_name="idxchunk_vec",
        pre_delete_collection=True,
    )
    graph = AgensGraph("idxtest", _conf())
    defs = _indexdefs(graph, "IdxChunk")
    # The unique import index must be on the actual MERGE key (__id__), not id.
    assert any("'__id__'" in d and "UNIQUE" in d for d in defs), defs
    # And the vector (HNSW) index must exist.
    assert any("hnsw" in d.lower() for d in defs), defs
    graph.close()
    store.close()


def test_chat_history_indexes_session_id():
    g = AgensGraph("idxtest", _conf(), create=True)
    AgensChatMessageHistory("sess", graph=g)
    defs = _indexdefs(g, "Session")
    assert any("'id'" in d for d in defs), defs
    g.close()


def test_checkpoint_indexes_thread_keys():
    g = AgensGraph("idxtest", _conf(), create=True)
    AgensSaver(graph=g)
    for label in ("Checkpoint", "CheckpointBlob", "CheckpointWrite"):
        defs = _indexdefs(g, label)
        assert any("'thread_id'" in d for d in defs), (label, defs)
    g.close()


def test_add_graph_documents_indexes_id():
    from langchain_agensgraph import GraphDocument, Node, Relationship
    from langchain_core.documents import Document

    g = AgensGraph("idxtest", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    doc = GraphDocument(
        nodes=[Node(id="a", type="Widget"), Node(id="b", type="Widget")],
        relationships=[
            Relationship(
                source=Node(id="a", type="Widget"),
                target=Node(id="b", type="Widget"),
                type="LINKS",
            )
        ],
        source=Document(page_content="x"),
    )
    g.add_graph_documents([doc])
    defs = _indexdefs(g, "Widget")
    assert any("'id'" in d for d in defs), defs
    g.close()
