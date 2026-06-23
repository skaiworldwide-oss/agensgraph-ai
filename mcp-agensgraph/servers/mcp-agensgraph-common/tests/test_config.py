import argparse

import pytest

from mcp_agensgraph_common.config import (
    connection_config,
    format_namespace,
    parse_boolean_safely,
    read_controls,
    transport_config,
)

# All possible CLI attributes; individual tests set what they need.
_ATTRS = dict(
    db_url=None, username=None, password=None, database=None, graphname=None,
    namespace=None, transport=None, server_host=None, server_port=None,
    server_path=None, allow_origins=None, allowed_hosts=None,
    read_timeout=None, token_limit=None, read_only=False,
)


def ns(**overrides):
    return argparse.Namespace(**{**_ATTRS, **overrides})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("AGENSGRAPH_"):
            monkeypatch.delenv(key, raising=False)


def test_parse_boolean_safely():
    assert parse_boolean_safely("true") is True
    assert parse_boolean_safely("FALSE") is False
    assert parse_boolean_safely(True) is True
    with pytest.raises(ValueError):
        parse_boolean_safely("yes")


def test_format_namespace():
    assert format_namespace("") == ""
    assert format_namespace("db") == "db-"
    assert format_namespace("db-") == "db-"


def test_connection_defaults_and_graphname_override():
    cfg = connection_config(ns(), default_graphname="memory")
    assert cfg["graphname"] == "memory"
    assert cfg["database"] == "agens"
    assert cfg["db_url"].startswith("postgresql://")


def test_connection_cli_beats_env(monkeypatch):
    monkeypatch.setenv("AGENSGRAPH_USERNAME", "from_env")
    assert connection_config(ns())["username"] == "from_env"
    assert connection_config(ns(username="from_cli"))["username"] == "from_cli"


def test_connection_env_aliases(monkeypatch):
    monkeypatch.setenv("AGENSGRAPH_DB", "viadbalias")
    monkeypatch.setenv("AGENSGRAPH_GRAPH_NAME", "viagraphalias")
    cfg = connection_config(ns())
    assert cfg["database"] == "viadbalias"
    assert cfg["graphname"] == "viagraphalias"


def test_transport_stdio_nulls_http_fields():
    cfg = transport_config(ns(transport="stdio"))
    assert cfg["transport"] == "stdio"
    assert cfg["host"] is None and cfg["port"] is None and cfg["path"] is None


def test_transport_http_defaults():
    cfg = transport_config(ns(transport="http"))
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 8000
    assert cfg["path"] == "/mcp/"
    assert cfg["allowed_hosts"] == ["localhost", "127.0.0.1"]


def test_transport_csv_parsing():
    cfg = transport_config(ns(transport="http", allow_origins="a.com, b.com ,"))
    assert cfg["allow_origins"] == ["a.com", "b.com"]


def test_read_controls_defaults_and_cli():
    assert read_controls(ns()) == {"read_timeout": 30, "token_limit": None, "read_only": False}
    cfg = read_controls(ns(read_timeout=10, token_limit=500, read_only=True))
    assert cfg == {"read_timeout": 10, "token_limit": 500, "read_only": True}


def test_read_controls_env(monkeypatch):
    monkeypatch.setenv("AGENSGRAPH_READ_ONLY", "true")
    monkeypatch.setenv("AGENSGRAPH_READ_TIMEOUT", "45")
    cfg = read_controls(ns())
    assert cfg["read_only"] is True
    assert cfg["read_timeout"] == 45
