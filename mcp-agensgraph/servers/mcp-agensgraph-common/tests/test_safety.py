import pytest

from mcp_agensgraph_common.safety import quote_identifiers


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


@pytest.mark.parametrize(
    "raw",
    [
        "MATCH (n) WHERE n.tag = 'a:Bcd' RETURN n",
        "MATCH (n) WHERE n.file = 'report.PDF' RETURN n",
        "MATCH (n) RETURN n -- a comment about :Labels",
    ],
)
def test_a_value_being_searched_for_is_not_edited(raw):
    """Rewriting inside a literal turned one row into none: 'a:Bcd' became 'a:"Bcd"'."""
    assert quote_identifiers(raw) == raw
