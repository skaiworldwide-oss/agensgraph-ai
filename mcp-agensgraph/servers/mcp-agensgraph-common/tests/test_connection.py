"""Unit tests for connection helpers that don't need a database."""

import pytest
from psycopg.types.json import Jsonb

from mcp_agensgraph_common.connection import build_dsn, jsonb_params, paged_statement


def test_build_dsn_carries_awkward_credentials_intact():
    """What matters is what libpq reads back, not how it was spelled on the way."""
    from psycopg.conninfo import conninfo_to_dict

    read = conninfo_to_dict(build_dsn("postgresql://host:55432", "u ser", "p@ss", "mydb"))
    assert read["user"] == "u ser"
    assert read["password"] == "p@ss"
    assert read["dbname"] == "mydb"
    assert read["host"] == "host"
    assert read["port"] == "55432"


def test_build_dsn_keeps_what_the_url_asked_for():
    """A dropped `sslmode=require` is a connection made in the clear."""
    from psycopg.conninfo import conninfo_to_dict

    read = conninfo_to_dict(
        build_dsn("postgresql://host:55432/theirs?sslmode=require", "u", "p", "")
    )
    assert read["sslmode"] == "require"
    assert read["dbname"] == "theirs", "the URL's own database survives when none is given"


@pytest.mark.parametrize(
    "database",
    ["db?host=/tmp", "db?sslmode=disable", "db?options=-c%20search_path%3Devil"],
)
def test_a_database_name_cannot_carry_connection_parameters(database):
    """It was pasted in front of a query string, so a name moved the connection elsewhere."""
    from psycopg.conninfo import conninfo_to_dict

    read = conninfo_to_dict(build_dsn("postgresql://host:55432", "u", "p", database))
    assert read["dbname"] == database
    assert read["host"] == "host"
    assert "options" not in read
    assert read.get("sslmode") is None


def test_an_empty_credential_is_left_for_libpq_to_resolve():
    """Sending an empty one made .pgpass, PGPASSWORD and peer authentication unreachable."""
    from psycopg.conninfo import conninfo_to_dict

    read = conninfo_to_dict(build_dsn("postgresql://host:55432", "", "", "db"))
    assert "user" not in read
    assert "password" not in read


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


def test_a_page_of_one_query_part_is_appended_to_it():
    paged = paged_statement('MATCH (n:"Person") RETURN n', limit=5, offset=10)
    assert paged == 'MATCH (n:"Person") RETURN n\nSKIP 10 LIMIT 6'


def test_a_mid_query_limit_is_left_where_the_caller_put_it():
    """The read-clause continuation set has no arm for LIMIT after WITH, so this cannot wrap."""
    query = "MATCH (p:Person)\nWITH p LIMIT 3\nRETURN p.name AS person"
    assert paged_statement(query, limit=5, offset=0).endswith("\nSKIP 0 LIMIT 6")


def test_a_union_is_paged_from_outside():
    """Appended, SKIP and LIMIT bind to the last arm: a five-row page of a two-arm union had
    the server produce 50,006 rows, and Python then kept five."""
    query = "MATCH (a:X) RETURN a.n AS x\nUNION ALL\nMATCH (b:Y) RETURN b.n AS x"
    paged = paged_statement(query, limit=5, offset=0)
    assert paged.startswith("SELECT * FROM (")
    assert paged.endswith(") AS _page LIMIT 6 OFFSET 0")


def test_the_callers_own_paging_is_paged_from_outside():
    paged = paged_statement("MATCH (n) RETURN n LIMIT 10", limit=2, offset=0)
    assert paged.startswith("SELECT * FROM (")


def test_a_query_ending_in_a_comment_still_closes():
    """`--` is a comment in AgensGraph, so the closing bracket goes on a line of its own."""
    paged = paged_statement("MATCH (n) RETURN n LIMIT 10 -- capped", limit=2, offset=0)
    assert paged.endswith("\n) AS _page LIMIT 3 OFFSET 0")


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) RETURN n.a AS x NEXT RETURN x LIMIT 10",
        "MATCH (a:X) RETURN a.n AS x UNION MATCH (b:Y) RETURN b.n AS x NEXT RETURN x",
    ],
)
def test_a_page_that_can_be_neither_appended_nor_wrapped_is_refused(query):
    with pytest.raises(ValueError, match="top of a statement"):
        paged_statement(query, limit=5, offset=0)


def test_a_clause_name_inside_a_literal_is_not_a_clause():
    """`next` as a property and NEXT as a clause are not the same word."""
    query = "MATCH (n) WHERE n.t = 'NEXT' RETURN n.next AS x LIMIT 4"
    assert paged_statement(query, limit=1, offset=0).startswith("SELECT * FROM (")
