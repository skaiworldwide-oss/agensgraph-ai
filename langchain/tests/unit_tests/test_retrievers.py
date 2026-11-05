"""The retrievers' pure parts: the graph walk's statement, arguments, and rendering."""

from collections.abc import Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_agensgraph.retrievers import AgensGraphContextRetriever, AgensVectorRetriever
from langchain_agensgraph.retrievers.graph_context import neighbourhood, render_graph_context
from langchain_agensgraph.vectorstores import AgensgraphVector


@pytest.fixture
def store() -> Iterator[AgensgraphVector]:
    made = AgensgraphVector(
        DeterministicFakeEmbedding(size=4), engine="host=nowhere dbname=none", graph_name="g"
    )
    yield made
    made.close()


def test_the_walk_reads_every_edge_label_or_one() -> None:
    assert '"g".ag_edge e' in neighbourhood("g", None)
    assert '"g"."knows" e' in neighbourhood("g", "knows")
    assert "LIMIT %s) n WHERE w.depth < %s" in neighbourhood("g", None)


def test_the_call_wins_over_the_search_kwargs_which_win_over_the_field(
    store: AgensgraphVector,
) -> None:
    retriever = AgensVectorRetriever(
        vectorstore=store, k=1, search_kwargs={"k": 3, "filter": {"a": 1}}
    )
    assert retriever._arguments({}) == (3, {"filter": {"a": 1}})
    assert retriever._arguments({"k": 5, "filter": None}) == (5, {"filter": None})
    assert AgensVectorRetriever(vectorstore=store, k=2)._arguments({}) == (2, {})


def test_the_graph_context_retriever_reads_the_graph_itself(store: AgensgraphVector) -> None:
    retriever = AgensGraphContextRetriever(vectorstore=store, retrieval_query="RETURN 1")
    with pytest.raises(ValueError, match="reads the graph itself"):
        retriever._seeds({})
    with pytest.raises(ValueError):
        AgensGraphContextRetriever(vectorstore=store, hops=4)


class TestRenderGraphContext:
    def test_a_document_without_context_is_returned_untouched(self) -> None:
        doc = Document(page_content="seed", metadata={"score": 0.5})
        assert render_graph_context(doc) is doc

    def test_context_is_rendered_under_the_seed_text(self) -> None:
        doc = Document(
            id="d1",
            page_content="seed",
            metadata={
                "title": "The Seed",
                "_seed_id_": "9.9",
                "_seed_label_": "doc",
                "_context_nodes_": [
                    {"id": "3.1", "label": "person", "properties": {"name": "Ada"}},
                    {"id": "3.2", "label": "person", "properties": {}},
                ],
                "_context_rels_": [
                    {"type": "knows", "start": "3.1", "end": "3.2", "properties": {}},
                    {"type": "wrote", "start": "3.1", "end": "9.9", "properties": {}},
                ],
            },
        )
        rendered = render_graph_context(doc)
        assert rendered.page_content.startswith("seed\n\nGraph context:\n")
        assert "- person:Ada" in rendered.page_content
        assert "- person:3.2" in rendered.page_content
        assert "- person:Ada -[knows]-> person:3.2" in rendered.page_content
        # The edge into the seed names the seed by its own label, not by a graph id, and the
        # seed is not listed among its own neighbours.
        assert "- person:Ada -[wrote]-> doc:The Seed" in rendered.page_content
        assert "- doc:The Seed\n" not in rendered.page_content
        assert rendered.id == "d1"
        assert rendered.metadata == doc.metadata

    def test_a_seed_without_a_label_is_still_named(self) -> None:
        doc = Document(
            page_content="seed",
            metadata={
                "name": "Anon",
                "_seed_id_": "9.9",
                "_context_nodes_": [
                    {"id": "3.1", "label": "person", "properties": {"name": "Ada"}}
                ],
                "_context_rels_": [
                    {"type": "wrote", "start": "3.1", "end": "9.9", "properties": {}}
                ],
            },
        )
        assert "- person:Ada -[wrote]-> Anon" in render_graph_context(doc).page_content
