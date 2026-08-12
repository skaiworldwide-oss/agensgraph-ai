"""What a read-only server does before it serves anything.

A read-only server that creates a graph has written to the database it was told not to write
to, and the write is one nobody sees: a graph name with a typo in it makes an empty graph, and
every read then returns nothing and reports no error.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from mcp_agensgraph_common.connection import run_query

from conftest import ALLOW_SERVER_PROGRAMS, free_port

MISSING_GRAPH = "mcp_cypher_it_never_made"


async def graph_exists(pool, name: str) -> bool:
    rows = await run_query(
        pool,
        "select count(*) as n from pg_catalog.ag_graph where graphname = %(name)s",
        {"name": name},
        read_only=True,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )
    return rows[0]["n"] > 0


async def spawn(db_url: str, graphname: str, *extra: str):
    return await asyncio.create_subprocess_exec(
        "uv", "run", "mcp-agensgraph-cypher",
        "--transport", "http",
        "--server-host", "127.0.0.1",
        "--server-port", str(free_port()),
        "--db-url", db_url,
        "--database", os.getenv("AGENSGRAPH_DB", ""),
        "--username", os.getenv("AGENSGRAPH_USERNAME", ""),
        "--graphname", graphname,
        *extra,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
        start_new_session=True,
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_a_read_only_server_makes_no_graph(setup, db_url):
    """It stops instead, which is where a name with a typo in it can still be reported."""
    assert not await graph_exists(setup, MISSING_GRAPH)
    process = await spawn(db_url, MISSING_GRAPH, "--read-only")
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    except asyncio.TimeoutError:  # pragma: no cover - it should not start at all
        process.kill()
        raise AssertionError("a read-only server started against a graph that is not there")
    assert process.returncode != 0
    assert MISSING_GRAPH in (stderr.decode() + stdout.decode())
    assert not await graph_exists(setup, MISSING_GRAPH), (
        "a read-only server created the graph it was pointed at"
    )
