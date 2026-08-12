"""How many connections the pool has, asked of the database rather than of the pool."""

from __future__ import annotations

import asyncio

import pytest

from mcp_agensgraph_common.config import DEFAULT_POOL_MAX_SIZE, DEFAULT_POOL_MIN_SIZE
from mcp_agensgraph_common.connection import create_pool, run_query

from .conftest import ALLOW_SERVER_PROGRAMS

BACKENDS = (
    "select count(*) as n from pg_stat_activity where application_name = %(name)s"
)
SLOW = "RETURN pg_sleep(1) AS slept"


async def live_backends(dsn, graphname, name):
    """How many connections the server holds for this application, counted by the server."""
    probe = create_pool(dsn, graphname, min_size=1, max_size=1, read_timeout=30)
    await probe.open()
    try:
        rows = await run_query(
            probe, BACKENDS, {"name": name}, read_only=True,
            allow_server_programs=ALLOW_SERVER_PROGRAMS,
        )
        return rows[0]["n"]
    finally:
        await probe.close()


def test_both_ends_of_the_default_size_are_given_values():
    """A pool told only its lower bound is that wide at the top as well."""
    assert DEFAULT_POOL_MAX_SIZE > DEFAULT_POOL_MIN_SIZE >= 1


async def test_the_upper_bound_is_a_bound(dsn, graphname):
    """Borrowing went round the pool on a timeout, adding connections it did not count.

    Measured then: `max_size` of 2 serving 6 live backends while `get_stats` reported 2.
    """
    name = "mcp_common_it_cap"
    capped = create_pool(
        dsn, graphname, min_size=1, max_size=2, read_timeout=30, timeout=0.2,
        kwargs={"application_name": name},
    )
    await capped.open()
    try:
        async def one():
            try:
                await run_query(capped, SLOW, read_only=True,
                                allow_server_programs=ALLOW_SERVER_PROGRAMS)
            except Exception:
                pass  # a pool with nothing to give says so; that is the point of a bound

        for _ in range(3):
            await asyncio.gather(*[one() for _ in range(6)])
        live = await live_backends(dsn, graphname, name)
        assert live <= 2, "max_size is the number of connections, not a suggestion"
        assert capped.get_stats()["pool_size"] == live
    finally:
        await capped.close()


async def test_a_caller_the_pool_cannot_serve_is_told_so(dsn, graphname):
    """Rather than being given a connection the pool does not know it has."""
    from psycopg_pool import PoolTimeout

    capped = create_pool(
        dsn, graphname, min_size=1, max_size=1, read_timeout=30, timeout=0.2,
        kwargs={"application_name": "mcp_common_it_timeout"},
    )
    await capped.open()
    try:
        held = asyncio.create_task(
            run_query(capped, SLOW, read_only=True,
                      allow_server_programs=ALLOW_SERVER_PROGRAMS)
        )
        await asyncio.sleep(0.1)
        with pytest.raises(PoolTimeout):
            await run_query(capped, "RETURN 1 AS one", read_only=True,
                            allow_server_programs=ALLOW_SERVER_PROGRAMS)
        await held
    finally:
        await capped.close()


async def test_a_pool_bound_to_a_graph_that_is_not_there_says_so_at_startup(dsn):
    """Rather than every read returning nothing and reporting no error."""
    missing = create_pool(
        dsn, "mcp_common_it_no_such_graph", min_size=1, max_size=1, read_timeout=30, timeout=2.0
    )
    await missing.open()
    try:
        with pytest.raises(Exception):
            await missing.wait()
    finally:
        await missing.close()
