import json
from collections import namedtuple

from agensgraph._protocol.graphid import GraphId
from agensgraph.types import Edge, Vertex
from mcp_agensgraph_common.results import (
    OMITTED,
    count_tokens,
    fit_rows,
    record_to_dict,
    value_sanitize,
)


def _rec(**fields):
    R = namedtuple("R", list(fields))
    return R(**fields)


def _vertex(label, labid, locid, props):
    return Vertex(GraphId(labid, locid), label, props)


def _edge(label, labid, locid, start, end, props):
    return Edge(GraphId(labid, locid), label, GraphId(*start), GraphId(*end), props)


def test_record_to_dict_vertex():
    rec = _rec(n=_vertex("Person", 3, 1, {"id": 1, "name": "alice"}))
    out = record_to_dict(rec)
    assert out["n"] == {
        "id": "3.1",
        "label": "Person",
        "properties": {"id": 1, "name": "alice"},
    }


def test_record_to_dict_edge_names_both_endpoints_on_its_own():
    """An edge read without its vertices still says what is at each end."""
    rec = _rec(r=_edge("KNOWS", 5, 1, (3, 1), (3, 2), {"since": 2020}))
    out = record_to_dict(rec)
    assert out["r"] == {
        "id": "5.1",
        "label": "KNOWS",
        "start": "3.1",
        "end": "3.2",
        "properties": {"since": 2020},
    }


def test_record_to_dict_walks_containers():
    rec = _rec(everyone=[_vertex("Person", 3, 1, {"name": "alice"})])
    assert record_to_dict(rec)["everyone"][0]["properties"] == {"name": "alice"}


def test_record_to_dict_scalars_passthrough():
    rec = _rec(count=42, name="plain", flag=True)
    assert record_to_dict(rec) == {"count": 42, "name": "plain", "flag": True}


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
