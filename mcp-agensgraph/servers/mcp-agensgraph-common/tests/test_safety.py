import pytest

from mcp_agensgraph_common.safety import (
    is_write_query,
    quote_identifiers,
    quote_label,
    strip_comments_and_strings,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MATCH (p:Person) RETURN p", 'MATCH (p:"Person") RETURN p'),
        ("RETURN p.FirstName", 'RETURN p."FirstName"'),
        (
            "CREATE (n:MyLabel {MyProp: 'value'})",
            'CREATE (n:"MyLabel" {"MyProp": \'value\'})',
        ),
        # lowercase identifiers are left alone
        ("MATCH (n:person) RETURN n.name", "MATCH (n:person) RETURN n.name"),
        # already-quoted identifiers are not double-quoted
        ('MATCH (p:"Person") RETURN p', 'MATCH (p:"Person") RETURN p'),
    ],
)
def test_quote_identifiers(raw, expected):
    assert quote_identifiers(raw) == expected


@pytest.mark.parametrize("label", ["KNOWS", "works_at", "Likes2", "_private"])
def test_quote_label_valid(label):
    assert quote_label(label) == f'"{label}"'


@pytest.mark.parametrize(
    "bad",
    ['KNOWS"]->(x)-[:"BACKDOOR', "has space", "with-dash", "", "1leading", "a;b"],
)
def test_quote_label_rejects_injection(bad):
    with pytest.raises(ValueError):
        quote_label(bad)


def test_is_write_query_detects_writes():
    assert is_write_query("MATCH (n) CREATE (m) RETURN m")
    assert is_write_query("merge (n:X)")
    assert is_write_query("MATCH (n) DETACH DELETE n")


def test_is_write_query_allows_reads():
    assert not is_write_query("MATCH (n) RETURN n")
    assert not is_write_query("MATCH (n) RETURN n.created_at")  # 'CREATE' substring


def test_is_write_query_not_fooled_by_comments_or_strings():
    # 'CREATE' only appears in a comment / string literal -> still a read
    assert not is_write_query("MATCH (n) RETURN n  // CREATE later")
    assert not is_write_query("MATCH (n) WHERE n.note = 'please CREATE' RETURN n")
    # a real write hidden after a comment is still caught
    assert is_write_query("// comment\nCREATE (n:X)")


def test_strip_comments_and_strings():
    out = strip_comments_and_strings("MATCH (n) // CREATE\nRETURN 'DELETE'")
    assert "CREATE" not in out
    assert "DELETE" not in out
    assert "MATCH" in out and "RETURN" in out
