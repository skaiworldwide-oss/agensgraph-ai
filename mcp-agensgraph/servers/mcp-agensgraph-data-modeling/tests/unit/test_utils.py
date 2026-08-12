"""Config composition, and the quoting the generators depend on.

Quoting is the driver's, applied where a name is placed into a statement. What is asserted here
is the property the generators rely on: whatever a name holds, it comes out as one identifier,
so a name can never become a second statement or a second clause.
"""

import argparse

import pytest
from agensgraph.cypher import quote_identifier

from mcp_agensgraph_data_modeling.utils import process_config

HOSTILE = [
    "Zap2; CREATE TEMP TABLE exfil(l text); COPY exfil FROM PROGRAM 'id'; --",
    "Victim {id: record.id}) DETACH DELETE n WITH record MERGE (n: Pwned",
    'a"b',
    "a) ; drop table t; --",
    "My Label",
    "2Cool",
]


@pytest.mark.parametrize("name", HOSTILE)
def test_a_hostile_name_becomes_one_quoted_identifier(name: str) -> None:
    quoted = quote_identifier(name)
    assert quoted.startswith('"') and quoted.endswith('"')
    # Every quote inside is doubled, so none of them closes the identifier early.
    assert quoted[1:-1].replace('""', "") .count('"') == 0


def test_a_null_byte_is_refused_rather_than_quoted() -> None:
    """The server's lexer stops at one, so the statement would end somewhere unexpected."""
    with pytest.raises(ValueError, match="null byte"):
        quote_identifier("a\x00b")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("person", "person"),
        ("Person", '"Person"'),
        ("firstName", '"firstName"'),
        ("sourceId", '"sourceId"'),
        ("create", '"create"'),
    ],
)
def test_case_is_preserved_because_an_unquoted_name_is_lowered(
    name: str, expected: str
) -> None:
    assert quote_identifier(name) == expected


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
