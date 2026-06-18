"""Unit tests for LLMGraphTransformer using a fake structured-output model."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_agensgraph.graph_transformers import LLMGraphTransformer


class FakeStructuredLLM:
    """Minimal stand-in: returns a canned structured graph regardless of input."""

    def __init__(self, payload: dict):
        self._payload = payload

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return RunnableLambda(lambda _messages: dict(self._payload))


CANNED = {
    "nodes": [
        {"id": "Alice", "type": "Person", "properties": []},
        {"id": "Acme", "type": "Company", "properties": []},
        {"id": "Berlin", "type": "City", "properties": []},
    ],
    "relationships": [
        {
            "source_id": "Alice",
            "source_type": "Person",
            "target_id": "Acme",
            "target_type": "Company",
            "type": "WORKS_AT",
            "properties": [],
        },
        {
            "source_id": "Acme",
            "source_type": "Company",
            "target_id": "Berlin",
            "target_type": "City",
            "type": "LOCATED_IN",
            "properties": [],
        },
    ],
}


def test_basic_extraction():
    t = LLMGraphTransformer(FakeStructuredLLM(CANNED))
    docs = t.convert_to_graph_documents([Document(page_content="Alice works at Acme.")])
    gd = docs[0]
    assert {n.id for n in gd.nodes} == {"Alice", "Acme", "Berlin"}
    rels = {(r.source.id, r.type, r.target.id) for r in gd.relationships}
    assert ("Alice", "WORKS_AT", "Acme") in rels
    assert ("Acme", "LOCATED_IN", "Berlin") in rels
    assert gd.source.page_content == "Alice works at Acme."


def test_allowed_nodes_filtering():
    # Only keep Person/Company; City node and the rel referencing it must drop.
    t = LLMGraphTransformer(
        FakeStructuredLLM(CANNED), allowed_nodes=["Person", "Company"]
    )
    gd = t.convert_to_graph_documents([Document(page_content="x")])[0]
    assert {n.id for n in gd.nodes} == {"Alice", "Acme"}
    rels = {(r.source.id, r.type, r.target.id) for r in gd.relationships}
    assert rels == {("Alice", "WORKS_AT", "Acme")}


def test_allowed_relationships_filtering():
    t = LLMGraphTransformer(
        FakeStructuredLLM(CANNED), allowed_relationships=["WORKS_AT"]
    )
    gd = t.convert_to_graph_documents([Document(page_content="x")])[0]
    rels = {r.type for r in gd.relationships}
    assert rels == {"WORKS_AT"}


def test_node_properties_extracted_when_enabled():
    payload = {
        "nodes": [
            {
                "id": "Alice",
                "type": "Person",
                "properties": [{"key": "role", "value": "engineer"}],
            }
        ],
        "relationships": [],
    }
    t = LLMGraphTransformer(FakeStructuredLLM(payload), node_properties=True)
    gd = t.convert_to_graph_documents([Document(page_content="x")])[0]
    assert gd.nodes[0].properties == {"role": "engineer"}


def test_requires_structured_output():
    import pytest

    class NoStructured:
        pass

    with pytest.raises(ValueError):
        LLMGraphTransformer(NoStructured())


async def test_async_conversion():
    t = LLMGraphTransformer(FakeStructuredLLM(CANNED))
    docs = await t.aconvert_to_graph_documents(
        [Document(page_content="a"), Document(page_content="b")]
    )
    assert len(docs) == 2
    assert all(d.nodes for d in docs)
