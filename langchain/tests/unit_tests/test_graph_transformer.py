"""Unit tests for LLMGraphTransformer using a fake structured-output model."""

from __future__ import annotations

import asyncio
import json
import threading
import time
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
    t = LLMGraphTransformer(FakeStructuredLLM(CANNED), allowed_nodes=["Person", "Company"])
    gd = t.convert_to_graph_documents([Document(page_content="x")])[0]
    assert {n.id for n in gd.nodes} == {"Alice", "Acme"}
    rels = {(r.source.id, r.type, r.target.id) for r in gd.relationships}
    assert rels == {("Alice", "WORKS_AT", "Acme")}


def test_allowed_relationships_filtering():
    t = LLMGraphTransformer(FakeStructuredLLM(CANNED), allowed_relationships=["WORKS_AT"])
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

                return RunnableLambda(func=lambda m: dict(self._payload), afunc=answer)

        transformer = LLMGraphTransformer(FailingLLM(CANNED), on_error="skip")
        out = await transformer.aconvert_to_graph_documents([Document(page_content="doc")])
        assert out == []
        with pytest.raises(RuntimeError, match="output-token limit"):
            await LLMGraphTransformer(FailingLLM(CANNED)).aconvert_to_graph_documents(
                [Document(page_content="doc")]
            )


def test_an_answer_that_is_no_graph_is_skipped_like_a_failed_call():
    class NoToolCallLLM(FakeStructuredLLM):
        """Structured output gives None when the model's reply makes no tool call."""

        def with_structured_output(self, schema: Any, **kwargs: Any):
            return RunnableLambda(lambda m: None if "second" in str(m) else dict(self._payload))

    docs = [Document(page_content=f"{n} doc") for n in ("first", "second", "third")]
    kept = LLMGraphTransformer(NoToolCallLLM(CANNED), on_error="skip")
    assert len(kept.convert_to_graph_documents(docs)) == 2
    with pytest.raises(TypeError, match="not a graph"):
        LLMGraphTransformer(NoToolCallLLM(CANNED)).convert_to_graph_documents(docs)


class RecordingLLM(FakeStructuredLLM):
    """Answers with a payload and keeps the messages it was sent, and how many ran at once."""

    def __init__(self, payload: dict, pause: float = 0.0):
        super().__init__(payload)
        self.seen: list[Any] = []
        self.running = 0
        self.peak = 0
        self.pause = pause
        self.lock = threading.Lock()

    def with_structured_output(self, schema: Any, **kwargs: Any):
        def answer(messages: Any) -> dict:
            with self.lock:
                self.seen.append(messages)
                self.running += 1
                self.peak = max(self.peak, self.running)
            time.sleep(self.pause)
            with self.lock:
                self.running -= 1
            return dict(self._payload)

        return RunnableLambda(answer)


TRIPLED = {
    "nodes": [
        {"id": "Ada", "type": "person", "properties": [{"key": "Age", "value": 36}]},
        {"id": "Acme", "type": "Company", "properties": [{"key": "public", "value": True}]},
    ],
    "relationships": [
        {
            "source_id": "Ada",
            "source_type": "Person",
            "target_id": "Acme",
            "target_type": "Company",
            "type": "works_at",
            "properties": [{"key": "since", "value": 2019}],
        },
        {
            "source_id": "Acme",
            "source_type": "Company",
            "target_id": "Ada",
            "target_type": "Person",
            "type": "WORKS_AT",
            "properties": [],
        },
    ],
}


def test_a_triple_keeps_only_its_own_direction_in_the_callers_spelling():
    transformer = LLMGraphTransformer(
        RecordingLLM(TRIPLED),
        allowed_nodes=["Person", "Company"],
        allowed_relationships=[("Person", "WORKS_AT", "Company")],
    )
    (document,) = transformer.convert_to_graph_documents([Document(page_content="x")])
    assert [(r.source.type, r.type, r.target.type) for r in document.relationships] == [
        ("Person", "WORKS_AT", "Company")
    ]
    assert "(Person, WORKS_AT, Company)" in transformer._system


@pytest.mark.parametrize(
    ("relationships", "message"),
    [
        ([("Person", "WORKS_AT", "City")], "not among the allowed nodes"),
        (["KNOWS", ("Person", "WORKS_AT", "Company")], "not both"),
    ],
)
def test_allowed_relationships_are_checked(relationships, message):
    with pytest.raises(ValueError, match=message):
        LLMGraphTransformer(
            RecordingLLM(TRIPLED),
            allowed_nodes=["Person", "Company"],
            allowed_relationships=relationships,
        )


def test_properties_are_kept_by_name_with_their_types():
    transformer = LLMGraphTransformer(
        RecordingLLM(TRIPLED), node_properties=["age"], relationship_properties=True
    )
    (document,) = transformer.convert_to_graph_documents([Document(page_content="x")])
    assert [n.properties for n in document.nodes] == [{"age": 36}, {}]
    assert document.relationships[0].properties == {"since": 2019}
    assert "only these node properties, as key and value: age" in transformer._system
    plain = LLMGraphTransformer(RecordingLLM(TRIPLED))
    (document,) = plain.convert_to_graph_documents([Document(page_content="x")])
    assert all(not n.properties for n in document.nodes)
    assert all(not r.properties for r in document.relationships)


def test_additional_instructions_reach_the_model_as_written():
    llm = RecordingLLM(CANNED)
    LLMGraphTransformer(llm, additional_instructions="Name {people} by surname.").process_response(
        Document(page_content="x")
    )
    system, human = llm.seen[0]
    assert system.content.endswith("Name {people} by surname.") and human.content == "x"


@pytest.mark.parametrize("ignore_tool_usage", [True, False])
def test_a_model_without_structured_output_answers_in_json(ignore_tool_usage):
    from langchain_core.language_models import FakeListChatModel

    reply = "```json\n" + json.dumps(CANNED) + "\n```"
    transformer = LLMGraphTransformer(
        FakeListChatModel(responses=[reply]), ignore_tool_usage=ignore_tool_usage
    )
    assert "JSON" in transformer._system
    document = transformer.process_response(Document(page_content="Alice works at Acme."))
    assert {n.id for n in document.nodes} == {"Alice", "Acme", "Berlin"}


def test_documents_convert_at_most_max_concurrency_at_a_time_in_their_order():
    llm = RecordingLLM(CANNED, pause=0.02)
    transformer = LLMGraphTransformer(llm, max_concurrency=2)
    documents = [Document(page_content=f"doc {i}") for i in range(6)]
    out = transformer.convert_to_graph_documents(documents)
    assert [d.source.page_content for d in out] == [d.page_content for d in documents]
    assert llm.peak == 2


def test_a_conversion_runs_under_the_callers_config():
    from langchain_core.tracers.context import collect_runs

    with collect_runs() as runs:
        LLMGraphTransformer(RecordingLLM(CANNED)).convert_to_graph_documents(
            [Document(page_content="x")], config={"tags": ["ingest"]}
        )
    (run,) = runs.traced_runs
    assert run.name == "LLMGraphTransformer" and "ingest" in run.tags
