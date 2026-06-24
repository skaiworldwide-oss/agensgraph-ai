"""Unit tests for connection helpers that don't need a database."""

from psycopg.types.json import Jsonb

from mcp_agensgraph_common.connection import build_dsn, jsonb_params


def test_build_dsn_encodes_credentials():
    dsn = build_dsn("postgresql://host:55432", "u ser", "p@ss", "mydb")
    assert dsn == "postgresql://u%20ser:p%40ss@host:55432/mydb"
    # missing host/port fall back to defaults
    assert build_dsn("postgresql://", "u", "", "db") == "postgresql://u:@localhost:5432/db"


def test_jsonb_params_wraps_lists_and_dicts():
    out = jsonb_params({"records": [{"a": 1}], "obj": {"k": "v"}, "name": "JFK", "n": 3, "flag": True})
    # list/dict values are wrapped so psycopg can bind them for Cypher JSONB params
    assert isinstance(out["records"], Jsonb)
    assert isinstance(out["obj"], Jsonb)
    # scalars are left untouched
    assert out["name"] == "JFK"
    assert out["n"] == 3 and out["flag"] is True


def test_jsonb_params_passthrough_empty():
    assert jsonb_params(None) is None
    assert jsonb_params({}) == {}


def test_jsonb_params_does_not_double_wrap():
    already = Jsonb([1, 2, 3])
    out = jsonb_params({"x": already})
    assert out["x"] is already  # Jsonb isn't a list/dict, so it's not re-wrapped
