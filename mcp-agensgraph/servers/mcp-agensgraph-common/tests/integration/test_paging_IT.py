"""A page, and how many rows the server produced to make it.

The row count the caller sees says nothing about the cost: a page of five is a page of five
whether the server produced five rows or fifty thousand. So each test here runs the paged
statement itself and counts what came back, which for a statement whose outermost clause is the
page is exactly what the server produced.
"""

from __future__ import annotations

import pytest

from mcp_agensgraph_common.connection import paged_statement, read_page, run_query

from .conftest import ALLOW_SERVER_PROGRAMS

UNION = (
    'MATCH (n:"Person") RETURN n.name AS name\n'
    "UNION ALL\n"
    'MATCH (n:"Person") RETURN n.name AS name'
)


async def produced(pool, query, *, limit, offset=0):
    """How many rows the server sent for this page."""
    rows = await run_query(
        pool,
        paged_statement(query, limit=limit, offset=offset),
        read_only=True,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    return len(rows)


async def test_a_page_asks_for_one_row_more_than_it_returns(seeded):
    assert await produced(seeded, 'MATCH (n:"Person") RETURN n.name AS name', limit=2) == 3
    rows, has_more = await read_page(
        seeded,
        'MATCH (n:"Person") RETURN n.name AS name',
        limit=2,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    assert len(rows) == 2 and has_more is True


async def test_a_page_of_a_union_is_a_page_of_the_whole_union(seeded):
    """Appended, SKIP and LIMIT page the last arm alone -- so the server produces a whole arm.

    Measured on 50,000 people: 50,006 rows in 140 ms for a five-row page, and 6 in 2 ms now.
    """
    assert await produced(seeded, UNION, limit=2) == 3
    rows, has_more = await read_page(
        seeded, UNION, limit=2, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )
    assert len(rows) == 2 and has_more is True

    everything, _ = await read_page(
        seeded, UNION, limit=100, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )
    assert len(everything) == 6, "both arms are still read when the page is big enough"


async def test_the_callers_own_limit_still_bounds_the_result(seeded):
    query = 'MATCH (n:"Person") RETURN n.name AS name ORDER BY name LIMIT 2'
    rows, has_more = await read_page(
        seeded, query, limit=10, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )
    assert [row["name"] for row in rows] == ["Alice", "Bob"]
    assert has_more is False


async def test_a_query_ending_in_a_comment_is_paged_rather_than_broken(seeded):
    """`--` is the comment AgensGraph has, and the closing bracket must not land inside one."""
    query = 'MATCH (n:"Person") RETURN n.name AS name ORDER BY name LIMIT 3 -- capped'
    rows, _ = await read_page(
        seeded, query, limit=2, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )
    assert [row["name"] for row in rows] == ["Alice", "Bob"]


async def test_a_limit_in_the_middle_of_a_query_still_runs(seeded):
    """The read-clause continuation set has no arm for LIMIT after WITH, so this cannot wrap."""
    query = (
        'MATCH (p:"Person")\n'
        "WITH p ORDER BY p.name LIMIT 2\n"
        'MATCH (p)-[:"KNOWS"]->(other)\n'
        "RETURN p.name AS person, other.name AS other ORDER BY person"
    )
    rows, has_more = await read_page(
        seeded, query, limit=10, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )
    assert [(row["person"], row["other"]) for row in rows] == [("Alice", "Bob"), ("Bob", "Carol")]
    assert has_more is False


async def test_an_offset_is_the_offset_of_the_whole_query(seeded):
    first, _ = await read_page(
        seeded,
        'MATCH (n:"Person") RETURN n.name AS name ORDER BY name',
        limit=1,
        offset=0,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    second, _ = await read_page(
        seeded,
        'MATCH (n:"Person") RETURN n.name AS name ORDER BY name',
        limit=1,
        offset=1,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    assert first == [{"name": "Alice"}]
    assert second == [{"name": "Bob"}]


async def test_a_page_that_can_be_neither_appended_nor_wrapped_says_so(seeded):
    with pytest.raises(ValueError, match="top of a statement"):
        await read_page(
            seeded,
            'MATCH (n:"Person") RETURN n.name AS name NEXT RETURN name LIMIT 3',
            limit=2,
            allow_server_programs=ALLOW_SERVER_PROGRAMS,
        )
