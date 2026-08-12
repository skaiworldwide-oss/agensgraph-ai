"""What survives the trip from the server to a map a model can read."""

from __future__ import annotations

from mcp_agensgraph_common.connection import run_query

from .conftest import ALLOW_SERVER_PROGRAMS


async def read(pool, query):
    return await run_query(
        pool, query, read_only=True, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )


async def test_an_edge_read_on_its_own_carries_its_properties_and_both_ends(seeded):
    """Matching the text the server printed reported an empty map at each end and no `since`."""
    rows = await read(seeded, 'MATCH ()-[r:"KNOWS"]->() RETURN r ORDER BY r.since')
    first = rows[0]["r"]
    assert first["properties"] == {"since": 2020}
    assert first["label"] == "KNOWS"
    assert first["start"] != first["end"]
    assert first["id"] not in (first["start"], first["end"])


async def test_a_vertex_carries_its_label_its_id_and_its_properties(seeded):
    rows = await read(seeded, 'MATCH (n:"Person" {name: \'Alice\'}) RETURN n')
    assert rows[0]["n"]["label"] == "Person"
    assert rows[0]["n"]["properties"] == {"name": "Alice", "age": 30}
    assert "." in rows[0]["n"]["id"]


async def test_a_vertex_inside_a_list_is_still_a_vertex(seeded):
    rows = await read(seeded, 'MATCH (n:"Person") RETURN collect(n) AS everyone')
    everyone = rows[0]["everyone"]
    assert len(everyone) == 3
    assert {person["properties"]["name"] for person in everyone} == {"Alice", "Bob", "Carol"}


async def test_a_string_that_looks_like_a_vertex_stays_a_string(seeded):
    """Reading the printed form reinterpreted any matching text as a vertex."""
    rows = await read(seeded, """RETURN 'Person[3.1]{"secret": "not a vertex"}' AS s""")
    assert rows[0]["s"] == 'Person[3.1]{"secret": "not a vertex"}'


async def test_two_columns_of_one_name_both_arrive(seeded):
    """Built as a namedtuple this raised `ValueError: duplicate field name`."""
    rows = await read(
        seeded, 'MATCH (n:"Person" {name: \'Alice\'}) RETURN n.name AS x, n.age AS x'
    )
    assert rows == [{"x": "Alice", "x (column 2)": 30}]


async def test_a_column_named_after_a_python_keyword_arrives(seeded):
    """Built as a namedtuple this raised `field names cannot be a keyword`."""
    assert await read(seeded, "RETURN 1 AS class") == [{"class": 1}]


async def test_a_column_whose_name_starts_with_an_underscore_keeps_it(seeded):
    """Built as a namedtuple this was silently renamed `f_x`."""
    assert await read(seeded, 'RETURN 1 AS "_x"') == [{"_x": 1}]


async def test_a_null_is_a_null_rather_than_a_missing_key(seeded):
    rows = await read(seeded, 'MATCH (n:"Person" {name: \'Alice\'}) RETURN n.nickname AS nick')
    assert rows == [{"nick": None}]


async def test_a_string_parameter_matches_what_the_same_literal_matches(seeded):
    """The driver sends a `str` as text; unspecified, the server parsed it as JSON.

    `{"name": "Alice"}` was `22P02 invalid input syntax for type json`, and a numeric-looking
    string matched nothing at all.
    """
    bound = await run_query(
        seeded,
        'MATCH (n:"Person" {name: %(name)s}) RETURN n.age AS age',
        {"name": "Alice"},
        read_only=True,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    literal = await read(seeded, 'MATCH (n:"Person" {name: \'Alice\'}) RETURN n.age AS age')
    assert bound == literal == [{"age": 30}]
