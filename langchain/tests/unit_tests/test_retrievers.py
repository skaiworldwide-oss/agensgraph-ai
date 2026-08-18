"""The pure pieces of the retrievers: query building and rendering."""

import pytest
from langchain_core.documents import Document

from langchain_agensgraph.retrievers.graph_context import (
    build_expansion_query,
    render_graph_context,
)


class TestBuildExpansionQuery:
    def test_hops_are_inlined_as_a_literal_bound(self) -> None:
        query = build_expansion_query(hops=2, hybrid=False)
        assert "*1..2" in query
        assert "{hops}" not in query

    def test_the_vector_variant_expands_from_the_seed_vertex(self) -> None:
        query = build_expansion_query(hops=1, hybrid=False)
        assert "OPTIONAL MATCH (node)-[rels*1..1]-(peer)" in query
        assert "__id__ = node.__id__" not in query

    def test_the_hybrid_variant_finds_its_seed_again_first(self) -> None:
        query = build_expansion_query(hops=1, hybrid=True)
        assert "OPTIONAL MATCH (seed:{label}) WHERE seed.__id__ = node.__id__" in query
        assert "OPTIONAL MATCH (seed)-[rels*1..1]-(peer)" in query

    def test_a_relationship_type_narrows_the_pattern(self) -> None:
        query = build_expansion_query(hops=1, hybrid=False, relationship_type="knows")
        assert '-[rels:"knows"*1..1]-' in query

    def test_the_caps_are_parameters_not_literals(self) -> None:
        query = build_expansion_query(hops=1, hybrid=False)
        assert "%(max_context_nodes)s" in query
        assert "%(max_context_rels)s" in query

    def test_the_seed_identity_travels_with_the_context(self) -> None:
        # An edge can point at the seed itself, and the seed is never among
        # its own neighbours -- the id is what lets a renderer name it.
        assert "id(node) AS seed_id" in build_expansion_query(hops=1, hybrid=False)
        assert "id(seed) AS seed_id" in build_expansion_query(hops=1, hybrid=True)

    @pytest.mark.parametrize("hops", [0, 4, -1])
    def test_out_of_range_hops_are_refused(self, hops: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 3"):
            build_expansion_query(hops=hops, hybrid=False)

    @pytest.mark.parametrize(
        "rel_type",
        ['a"b', "a-b", "1abc", "", 'x"]->() MATCH (m) DETACH DELETE m //'],
    )
    def test_a_type_that_is_not_an_identifier_is_refused(self, rel_type: str) -> None:
        # The type is inlined into the pattern, so anything but a plain
        # identifier is refused outright rather than quoted heroically.
        with pytest.raises(ValueError, match="plain identifier"):
            build_expansion_query(hops=1, hybrid=False, relationship_type=rel_type)


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
        # The edge into the seed names the seed, not its graph id.
        assert "- person:Ada -[wrote]-> The Seed" in rendered.page_content
        assert rendered.id == "d1"
        assert rendered.metadata == doc.metadata
