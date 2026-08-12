import json

from agensgraph import Result
from agensgraph._protocol.graphid import GraphId
from agensgraph.types import Edge, Vertex
from mcp_agensgraph_common.results import (
    OMITTED,
    column_names,
    count_tokens,
    fit_rows,
    rows_of,
    value_sanitize,
)


def _result(keys, *records):
    return Result(records=list(records), keys=list(keys), counts=None)


def _vertex(label, labid, locid, props):
    return Vertex(GraphId(labid, locid), label, props)


def _edge(label, labid, locid, start, end, props):
    return Edge(GraphId(labid, locid), label, GraphId(*start), GraphId(*end), props)


def test_rows_of_vertex():
    out = rows_of(_result(["n"], (_vertex("Person", 3, 1, {"id": 1, "name": "alice"}),)))
    assert out == [
        {
            "n": {
                "id": "3.1",
                "label": "Person",
                "properties": {"id": 1, "name": "alice"},
            }
        }
    ]


def test_rows_of_edge_names_both_endpoints_on_its_own():
    """An edge read without its vertices still says what is at each end."""
    out = rows_of(_result(["r"], (_edge("KNOWS", 5, 1, (3, 1), (3, 2), {"since": 2020}),)))
    assert out == [
        {
            "r": {
                "id": "5.1",
                "label": "KNOWS",
                "start": "3.1",
                "end": "3.2",
                "properties": {"since": 2020},
            }
        }
    ]


def test_rows_of_walks_containers():
    out = rows_of(_result(["everyone"], ([_vertex("Person", 3, 1, {"name": "alice"})],)))
    assert out[0]["everyone"][0]["properties"] == {"name": "alice"}


def test_rows_of_scalars_passthrough():
    out = rows_of(_result(["count", "name", "flag"], (42, "plain", True)))
    assert out == [{"count": 42, "name": "plain", "flag": True}]


def test_a_repeated_column_name_keeps_both_values():
    """`RETURN n.a AS x, n.b AS x` names two columns, and both were asked for."""
    out = rows_of(_result(["x", "x"], (1, "alice")))
    assert out == [{"x": 1, "x (column 2)": "alice"}]


def test_a_column_named_after_a_keyword_is_the_name_it_was_given():
    """Building rows as namedtuples refused this one outright."""
    assert rows_of(_result(["class"], (1,))) == [{"class": 1}]


def test_a_leading_underscore_is_not_renamed():
    """A namedtuple silently made this `f_x`, so the caller's own alias was unreachable."""
    assert rows_of(_result(["_x"], (1,))) == [{"_x": 1}]


def test_column_names_leaves_a_name_used_once_alone():
    assert column_names(["a", "b", "c"]) == ["a", "b", "c"]


def test_column_names_distinguishes_three_of_a_kind():
    assert column_names(["x", "x", "x"]) == ["x", "x (column 2)", "x (column 3)"]


def test_value_sanitize_marks_an_oversized_list_rather_than_dropping_it():
    data = {"name": "x", "embedding": list(range(500)), "tags": [1, 2, 3]}
    out = value_sanitize(data, list_limit=128)
    assert out["embedding"] == OMITTED.format(count=500)
    assert out["tags"] == [1, 2, 3]
    assert out["name"] == "x"


def test_value_sanitize_marks_a_bare_oversized_list():
    assert value_sanitize(list(range(200)), list_limit=128) == OMITTED.format(count=200)


def test_value_sanitize_reaches_into_nested_maps():
    out = value_sanitize({"n": {"v": list(range(300))}}, list_limit=128)
    assert out["n"]["v"] == OMITTED.format(count=300)


def test_fit_rows_keeps_whole_rows():
    rows = [{"text": "word " * 100} for _ in range(20)]
    kept, dropped = fit_rows(rows, token_limit=250)
    assert 0 < len(kept) < len(rows)
    assert dropped == len(rows) - len(kept)
    # Whole rows, so what is kept still serializes and parses.
    assert json.loads(json.dumps(kept)) == kept


def test_fit_rows_keeps_everything_that_fits():
    rows = [{"i": i} for i in range(5)]
    assert fit_rows(rows, token_limit=10_000) == (rows, 0)


def test_count_tokens_falls_back_for_an_unknown_model():
    assert count_tokens("hello world", model="not-a-real-model") > 0
