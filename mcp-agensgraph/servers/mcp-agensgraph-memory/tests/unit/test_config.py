"""Memory-server config composition.

Detailed parsing is unit-tested in ``mcp-agensgraph-common``; here we only check
that ``process_config`` stitches the connection + transport sections together and
defaults the graph to ``memory``.
"""

import argparse
import os

import pytest

from mcp_agensgraph_memory.utils import process_config

_ATTRS = dict(
    db_url=None, username=None, password=None, database=None, graphname=None,
    namespace=None, transport=None, server_host=None, server_port=None,
    server_path=None, allow_origins=None, allowed_hosts=None,
)


def ns(**overrides):
    return argparse.Namespace(**{**_ATTRS, **overrides})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("AGENSGRAPH_"):
            monkeypatch.delenv(key, raising=False)


def test_process_config_sections_and_default_graph():
    cfg = process_config(ns())
    for key in (
        "db_url", "username", "password", "database", "graphname",
        "namespace", "transport", "host", "port", "path",
        "allow_origins", "allowed_hosts",
    ):
        assert key in cfg, f"missing {key}"
    assert cfg["graphname"] == "memory"  # memory default
    assert cfg["transport"] == "stdio"
    # memory has no read controls
    assert "read_only" not in cfg


def test_graphname_env_override(monkeypatch):
    monkeypatch.setenv("AGENSGRAPH_GRAPHNAME", "mygraph")
    assert process_config(ns())["graphname"] == "mygraph"
