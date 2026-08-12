"""Unit tests for LLMGraphTransformer using a fake structured-output model."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
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


class TestOneEntityHoweverTheModelSpellsItsType:
    """A type is admitted whatever its case, and an endpoint is looked up by that.

    A model naming the same entity ``Person`` among the nodes and ``PERSON`` in a
    relationship is naming one entity. Looked up by what it wrote, the node it described
    is missed and a second one of the same type is built carrying none of the properties
    -- and that one is written last, so it wins.
    """

    PAYLOAD = {
        "nodes": [
            {
                "id": "Alice",
                "type": "Person",
                "properties": [{"key": "role", "value": "engineer"}],
            }
        ],
        "relationships": [
            {
                "source_id": "Alice",
                "source_type": "PERSON",
                "target_id": "Acme",
                "target_type": "company",
                "type": "works_at",
                "properties": [],
            }
        ],
    }

    def _transform(self):
        transformer = LLMGraphTransformer(
            FakeStructuredLLM(self.PAYLOAD),
            allowed_nodes=["Person", "Company"],
            allowed_relationships=["WORKS_AT"],
            node_properties=True,
        )
        return transformer.process_response(Document(page_content="Alice at Acme."))

    def test_the_relationships_endpoint_is_the_node_that_was_described(self):
        doc = self._transform()
        assert [(n.id, n.type) for n in doc.nodes] == [("Alice", "Person")]
        assert doc.relationships[0].source.properties == {"role": "engineer"}

    def test_the_endpoint_keeps_the_case_the_caller_asked_for(self):
        doc = self._transform()
        rel = doc.relationships[0]
        assert (rel.source.type, rel.target.type, rel.type) == (
            "Person",
            "Company",
            "WORKS_AT",
        )


class TestCancellationIsNotAFailedDocument:
    """``CancelledError`` is a ``BaseException``, so it arrives like any other result.

    Classified as a failure it would be logged and skipped; classified as a result it
    lands in the returned list of graph documents and fails later in whatever is handed
    it. A caller that cancelled is owed the cancellation.
    """

    class CancellingLLM(FakeStructuredLLM):
        """One call is cancelled, as a client cleaning up after a timeout does."""

        def __init__(self, payload: dict):
            super().__init__(payload)
            self.calls = 0

        def with_structured_output(self, schema: Any, **kwargs: Any):
            async def answer(_messages):
                self.calls += 1
                if self.calls == 2:
                    raise asyncio.CancelledError()
                return dict(self._payload)

            return RunnableLambda(func=lambda m: dict(self._payload), afunc=answer)

    @pytest.mark.asyncio
    async def test_an_inner_cancellation_is_raised_on(self):
        transformer = LLMGraphTransformer(self.CancellingLLM(CANNED))
        docs = [Document(page_content=f"doc {i}") for i in range(3)]
        with pytest.raises(asyncio.CancelledError):
            await transformer.aconvert_to_graph_documents(docs)

    @pytest.mark.asyncio
    async def test_an_ordinary_failure_is_still_skipped(self):
        class FailingLLM(FakeStructuredLLM):
            def with_structured_output(self, schema: Any, **kwargs: Any):
                async def answer(_messages):
                    raise RuntimeError("the answer hit the output-token limit")

                return RunnableLambda(
                    func=lambda m: dict(self._payload), afunc=answer
                )

        transformer = LLMGraphTransformer(FailingLLM(CANNED))
        out = await transformer.aconvert_to_graph_documents(
            [Document(page_content="doc")]
        )
        assert out == []
