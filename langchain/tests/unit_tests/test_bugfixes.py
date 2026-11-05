"""Regressions that need no server: what reaches a prompt, errors, and names refused early."""

from __future__ import annotations

import pytest
from agensgraph import SparseVector, Vector, Vertex
from agensgraph._protocol.graphid import GraphId
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_agensgraph.errors import AgensGraphError, AgensQueryError
from langchain_agensgraph.graphs.agensgraph import AgensGraph, _sanitize_value
from langchain_agensgraph.vectorstores import AgensgraphVector

OFFLINE = "host=nowhere dbname=none"


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}),
        # An oversized list takes its key with it, rather than leaving a null behind.
        ({"a": list(range(200))}, {}),
        ({"a": 1, "b": "x", "c": None}, {"a": 1, "b": "x", "c": None}),
        ({"a": {"b": list(range(200))}}, {"a": {}}),
    ],
)
def test_sanitize_drops_only_oversized_lists(value: dict, expected: dict) -> None:
    assert _sanitize_value(value) == expected


def test_sanitize_describes_a_graph_element_rather_than_walking_past_it() -> None:
    vertex = Vertex(GraphId(3, 1), "Chunk", {"text": "hi", "embedding": list(range(200))})
    out = _sanitize_value(vertex)
    assert out["label"] == "Chunk"
    assert out["id"] == "3.1"
    assert out["properties"]["text"] == "hi"
    assert "embedding" not in out["properties"]


class TestSanitizeSeesAVector:
    """A vector is not a ``list``, though it compares equal to one, so a guard written against
    ``list`` alone would let 1,536 numbers into a prompt."""

    def test_a_wide_vector_is_dropped(self) -> None:
        assert _sanitize_value({"embedding": Vector([1.0] * 1536)}) == {}

    def test_a_narrow_one_is_kept_as_its_numbers(self) -> None:
        assert _sanitize_value({"embedding": Vector([1.0, 2.0])}) == {"embedding": [1.0, 2.0]}

    def test_a_sparse_vector_is_measured_by_what_it_holds(self) -> None:
        held = SparseVector({1: 1.0, 7: 2.0}, 4096)
        assert _sanitize_value({"embedding": held}) == {"embedding": {1: 1.0, 7: 2.0}}
        crowded = SparseVector({i: 1.0 for i in range(200)}, 4096)
        assert _sanitize_value({"embedding": crowded}) == {}

    def test_a_vector_is_not_a_list(self) -> None:
        held = Vector([1.0] * 1536)
        assert not isinstance(held, list)
        assert held == [1.0] * 1536


def test_a_query_error_carries_the_servers_code() -> None:
    error = AgensQueryError("syntax error at or near x", sqlstate="42601")
    assert isinstance(error, AgensGraphError)
    assert error.sqlstate == "42601" and str(error) == "syntax error at or near x"


@pytest.mark.parametrize("name", ["Ses\x00sion", "", "x" * 64])
def test_a_name_that_would_not_arrive_intact_is_refused_before_anything_is_sent(
    name: str,
) -> None:
    with pytest.raises(ValueError):
        AgensGraph(OFFLINE, graph_name=name)
    with pytest.raises(ValueError):
        AgensgraphVector(
            DeterministicFakeEmbedding(size=4), engine=OFFLINE, graph_name="g", label=name
        )
