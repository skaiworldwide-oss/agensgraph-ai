"""Data-modeling utils: the DDL-aware identifier quoter + config composition.

Transport/config parsing is unit-tested in ``mcp-agensgraph-common``; here we keep
coverage for the locally-retained ``_quote_identifiers`` (which handles
VLABEL/ELABEL/ON/ASSERT and camelCase) and the thin ``process_config``.
"""

import argparse

import pytest

from mcp_agensgraph_data_modeling.utils import _quote_identifiers, process_config


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MATCH (p:Person) RETURN p", 'MATCH (p: "Person") RETURN p'),
        ("RETURN p.FirstName", 'RETURN p."FirstName"'),
        ("RETURN p.firstName", 'RETURN p."firstName"'),  # camelCase
        (
            "CREATE VLABEL IF NOT EXISTS Person",
            'CREATE VLABEL IF NOT EXISTS "Person"',
        ),
        (
            "CREATE ELABEL IF NOT EXISTS Friend",
            'CREATE ELABEL IF NOT EXISTS "Friend"',
        ),
        (
            "CREATE CONSTRAINT c ON Person ASSERT personId IS UNIQUE",
            'CREATE CONSTRAINT c ON "Person" ASSERT "personId" IS UNIQUE',
        ),
    ],
)
def test_quote_identifiers_ddl_cases(raw, expected):
    assert _quote_identifiers(raw) == expected


def test_process_config_is_transport_only():
    args = argparse.Namespace(
        namespace=None, transport=None, server_host=None, server_port=None,
        server_path=None, allow_origins=None, allowed_hosts=None,
    )
    cfg = process_config(args)
    assert set(cfg) == {
        "namespace", "transport", "host", "port", "path",
        "allow_origins", "allowed_hosts",
    }
    assert cfg["transport"] == "stdio"
