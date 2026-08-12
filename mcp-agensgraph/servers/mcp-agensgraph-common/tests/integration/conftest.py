"""What the shared core does against a real server.

Everything here needs one, because everything here is about the boundary between this package
and the database: which graph a connection is reading, what a transaction is allowed to do,
what the server produced for a page, and how many connections exist. None of that can be
answered without asking.

The tests skip rather than fail where no server is named, so the unit suite still runs
anywhere.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from mcp_agensgraph_common.connection import build_dsn, create_pool, ensure_graph, run_query

# The role a developer runs these as is usually the bootstrap superuser, which can run a command
# on the database server's host through `COPY ... TO PROGRAM` -- so the driver refuses to open a
# read-only transaction for it unless that is accepted out loud. What is under test here is the
# transaction, which refuses a write from any role; one test asserts the refusal itself.
ALLOW_SERVER_PROGRAMS = True


def settings() -> dict[str, str]:
    """Where to connect, and as whom.

    No password is required to reach a server -- trust and peer authentication have none, and a
    local one usually does.
    """
    return {
        "url": os.getenv("AGENSGRAPH_URL", "postgresql://localhost:5432"),
        "database": os.getenv("AGENSGRAPH_DB", os.getenv("AGENSGRAPH_DATABASE", "")),
        "user": os.getenv("AGENSGRAPH_USERNAME", ""),
        "password": os.getenv("AGENSGRAPH_PASSWORD", ""),
        "host": os.getenv("AGENSGRAPH_HOST", ""),
        "port": os.getenv("AGENSGRAPH_PORT", ""),
    }


@pytest.fixture(scope="session")
def dsn() -> str:
    where = settings()
    if not where["database"] or not where["user"]:
        pytest.skip("set AGENSGRAPH_DB and AGENSGRAPH_USERNAME to run the tests that need a server")
    url = where["url"]
    if where["host"]:
        url = f"postgresql://{where['host']}:{where['port'] or '5432'}"
    return build_dsn(url, where["user"], where["password"], where["database"])


@pytest.fixture(scope="session")
def graphname() -> str:
    return os.getenv("AGENSGRAPH_COMMON_GRAPH", "mcp_common_it")


@pytest_asyncio.fixture
async def pool(dsn, graphname):
    """A pool reading the test graph, opened and closed inside one test's event loop.

    Per test, because a pool returns a connection through a worker task on the loop that opened
    it and each test here runs in a loop of its own -- one that outlives its loop is handed
    nothing back.
    """
    await ensure_graph(dsn, graphname)
    made = create_pool(dsn, graphname, min_size=1, max_size=4, read_timeout=30)
    await made.open()
    await made.wait()
    try:
        yield made
    finally:
        await made.close()


@pytest_asyncio.fixture
async def seeded(pool):
    """Three people and the two relationships between them, and nothing else."""
    await run_query(pool, 'MATCH (n) DETACH DELETE n')
    await run_query(
        pool,
        """
        CREATE (a:"Person" {name: 'Alice', age: 30}),
               (b:"Person" {name: 'Bob', age: 25}),
               (c:"Person" {name: 'Carol', age: 41}),
               (a)-[:"KNOWS" {since: 2020}]->(b),
               (b)-[:"KNOWS" {since: 2021}]->(c)
        """,
    )
    return pool
