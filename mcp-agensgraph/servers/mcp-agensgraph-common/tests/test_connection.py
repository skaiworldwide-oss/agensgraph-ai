"""Unit tests for connection helpers that don't need a database."""

import pytest
from psycopg.types.json import Jsonb

from mcp_agensgraph_common.connection import build_dsn, jsonb_params


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
