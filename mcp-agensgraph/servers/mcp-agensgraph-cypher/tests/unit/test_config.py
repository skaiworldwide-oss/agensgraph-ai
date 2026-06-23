"""Cypher-server config composition.

The detailed parsing/quoting/result-parsing behaviour is unit-tested in
``mcp-agensgraph-common``; here we only check that this server's ``process_config``
stitches the connection + transport + read-control sections together correctly.
"""

import argparse

import pytest

from mcp_agensgraph_cypher.utils import process_config

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
    import os

    for key in list(os.environ):
        if key.startswith("AGENSGRAPH_"):
            monkeypatch.delenv(key, raising=False)


def test_process_config_has_all_sections():
    cfg = process_config(ns())
    # connection + transport + read controls, all present with defaults
    for key in (
        "db_url", "username", "password", "database", "graphname",
        "namespace", "transport", "host", "port", "path",
        "allow_origins", "allowed_hosts",
        "read_timeout", "token_limit", "read_only",
    ):
        assert key in cfg, f"missing {key}"
    assert cfg["graphname"] == "agens"  # cypher default
    assert cfg["transport"] == "stdio"
    assert cfg["read_only"] is False
    assert cfg["read_timeout"] == 30


def test_process_config_read_only_via_cli():
    assert process_config(ns(read_only=True))["read_only"] is True
