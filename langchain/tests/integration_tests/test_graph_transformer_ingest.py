"""Integration test: LLMGraphTransformer output ingests into AgensGraph."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_agensgraph import AgensGraph, LLMGraphTransformer


class FakeStructuredLLM:
    def __init__(self, payload: dict):
        self._payload = payload

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return RunnableLambda(lambda _messages: dict(self._payload))


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


def test_transformer_output_ingests():
    payload = {
        "nodes": [
            {"id": "Alice", "type": "Person", "properties": []},
            {"id": "Acme", "type": "Company", "properties": []},
        ],
        "relationships": [
            {
                "source_id": "Alice",
                "source_type": "Person",
                "target_id": "Acme",
                "target_type": "Company",
                "type": "WORKS_AT",
                "properties": [],
            }
        ],
    }
    t = LLMGraphTransformer(FakeStructuredLLM(payload))
    gdocs = t.convert_to_graph_documents([Document(page_content="Alice works at Acme.")])

    g = AgensGraph("transformer_test", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    g.add_graph_documents(gdocs)

    counts = g.query("MATCH (n) RETURN count(n) AS c")
    assert counts[0]["c"] == 2
    rels = g.query('MATCH ()-[r]->() RETURN type(r) AS t')
    assert any(r["t"].upper() == "WORKS_AT" for r in rels)
    g.query("MATCH (n) DETACH DELETE n")
    g.close()
