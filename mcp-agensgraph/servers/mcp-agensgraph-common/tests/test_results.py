from collections import namedtuple

from mcp_agensgraph_common.results import (
    record_to_dict,
    truncate_to_tokens,
    value_sanitize,
)


def _rec(**fields):
    R = namedtuple("R", list(fields))
    return R(**fields)


def test_record_to_dict_vertex():
    rec = _rec(n='Person[3.1]{"id": 1, "name": "alice"}')
    out = record_to_dict(rec)
    assert out["n"] == {"id": 1, "name": "alice"}


def test_record_to_dict_edge_resolves_endpoints():
    rec = _rec(
        a='Person[3.1]{"name": "alice"}',
        b='Person[3.2]{"name": "bob"}',
        r='KNOWS[5.1][3.1, 3.2]{"since": 2020}',
    )
    out = record_to_dict(rec)
    start, label, end = out["r"]
    assert start == {"name": "alice"}
    assert label == "KNOWS"
    assert end == {"name": "bob"}


def test_record_to_dict_scalars_passthrough():
    rec = _rec(count=42, name="plain", flag=True)
    assert record_to_dict(rec) == {"count": 42, "name": "plain", "flag": True}


def test_record_to_dict_malformed_props_does_not_crash():
    rec = _rec(n="Person[3.1]{not valid json}")
    out = record_to_dict(rec)
    assert out["n"] == "{not valid json}"


def test_value_sanitize_drops_oversized_lists():
    data = {"name": "x", "embedding": list(range(500)), "tags": [1, 2, 3]}
    out = value_sanitize(data, list_limit=128)
    assert "embedding" not in out
    assert out["tags"] == [1, 2, 3]
    assert out["name"] == "x"


def test_truncate_to_tokens():
    text = "word " * 100
    truncated = truncate_to_tokens(text, token_limit=10)
    assert len(truncated) < len(text)
    # unknown model falls back to a generic encoding rather than crashing
    assert truncate_to_tokens("hello world", 5, model="not-a-real-model")
