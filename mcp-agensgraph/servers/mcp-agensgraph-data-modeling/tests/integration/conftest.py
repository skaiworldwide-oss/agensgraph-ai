import asyncio
import os
import socket
import subprocess
import time

import pytest
import pytest_asyncio
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool  # type: ignore

from mcp_agensgraph_data_modeling.server import create_mcp_server

# SQL function to check if a property has unique constraint
SQL_PROPERTY_CONSTRAINT_FUNCTION = """
CREATE OR REPLACE FUNCTION property_has_unique_constraint(key_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    found BOOLEAN;
BEGIN
    -- Either mechanism makes a property unique, and which one was used is not the question
    -- being asked. A constraint is an exclusion constraint; a unique property index is a plain
    -- unique index over the property expression.
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint r
        JOIN pg_catalog.ag_label l ON r.conrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        WHERE g.graphname = current_setting('graph_path')
        AND r.contype IN ('c', 'x')
        AND pg_catalog.ag_get_graphconstraintdef(r.oid) ILIKE '%(' || key_name || ') IS UNIQUE%'
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_index x
        JOIN pg_catalog.pg_class i ON i.oid = x.indexrelid
        JOIN pg_catalog.ag_label l ON x.indrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        WHERE g.graphname = current_setting('graph_path')
        AND x.indisunique
        AND i.relkind = 'i'
        -- PostgreSQL's own printer, which returns a definition for any index. AgensGraph's
        -- raises on one that is not a property index, and a toast index reaches this.
        AND pg_catalog.pg_get_indexdef(i.oid) ILIKE '%''' || key_name || '''%'
    ) INTO found;

    RETURN found;
END;
$$ LANGUAGE plpgsql;
"""


def get_pool_connection(pool: AsyncConnectionPool):
    """Context manager for getting a connection from the pool."""
    return pool.connection()


def free_port() -> int:
    """A port the kernel chose, rather than one written down here.

    The SSE fixture named 8002, which is the port the cypher and memory suites' SSE
    fixtures name too, so no two of the three could run at once. Binding 0 and reading back
    what was assigned means each fixture takes a port nothing else holds.
    """
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        return int(taken.getsockname()[1])


class Spawned:
    """A server process and where it is listening.

    The tests need the port, not just the process, now that no one writes it down.
    """

    def __init__(self, process, port: int) -> None:
        self.process = process
        self.port = port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp/"

    @property
    def returncode(self):
        return self.process.returncode


async def port_is_open(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def wait_for_server(process, port: int, timeout: float = 60.0) -> None:
    """Wait until the server accepts connections, or say why it will not.

    These fixtures slept three seconds and then posted. A server ready sooner was waited on
    anyway, and one that needed longer was reported as a connection refused to a port nobody
    was listening on -- which is what `test_trusted_host_security` failed with, intermittently,
    because two of these servers start one after the other in the same suite.
    """
    deadline = time.monotonic() + timeout
    while not await port_is_open(port):
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise RuntimeError(
                f"server exited before taking port {port}. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"server did not take port {port} within {timeout}s")
        await asyncio.sleep(0.05)


# ===== Transport Testing Fixtures =====


@pytest_asyncio.fixture(scope="function")
async def mcp_server():
    """Create MCP server instance for transport testing."""
    mcp = create_mcp_server()
    return mcp


@pytest_asyncio.fixture
async def sse_server():
    """Start the MCP server in SSE mode."""
    port = free_port()
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-data-modeling",
        "--transport",
        "sse",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        str(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    await wait_for_server(process, port)

    yield Spawned(process, port)

    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


# ===== Database Testing Fixtures =====


@pytest.fixture(scope="module")
def graphname():
    """Graph name for database testing."""
    return os.getenv("AGENSGRAPH_GRAPH_NAME", "test_data_modeling")


@pytest_asyncio.fixture(scope="function", autouse=False)
async def db_setup(graphname):
    """Setup AgensGraph connection pool for database tests."""
    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    # A password is not required to reach a server: trust and peer authentication have none, and
    # a local one usually does. Requiring one here is what kept these tests from ever running --
    # they are the only ones that execute the generated Cypher, which is the thing worth checking.
    if not db_name or not db_user:
        pytest.skip(
            "Database integration tests skipped: AGENSGRAPH_DB, AGENSGRAPH_USERNAME, "
            "and AGENSGRAPH_PASSWORD environment variables must be set."
        )

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    agensgraph_pool = AsyncConnectionPool(db_url, open=False)

    await agensgraph_pool.open()

    # Ensure graph exists and create helper function
    async with get_pool_connection(agensgraph_pool) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            await cursor.execute(f"CREATE GRAPH IF NOT EXISTS {graphname}")
            await cursor.execute(SQL_PROPERTY_CONSTRAINT_FUNCTION)
            await conn.commit()

    yield agensgraph_pool

    await agensgraph_pool.close()


@pytest_asyncio.fixture(scope="function")
async def clean_graph(db_setup, graphname):
    """Clean the graph before each database test."""
    async with get_pool_connection(db_setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            await cursor.execute(f"SET graph_path = {graphname}")

            # Delete all nodes and relationships
            await cursor.execute("MATCH (n) DETACH DELETE n")

            # Drop all constraints
            await cursor.execute("""
                SELECT r.conname, l.labname
                FROM pg_catalog.pg_constraint r
                JOIN pg_catalog.ag_label l ON r.conrelid = l.relid
                JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
                WHERE g.graphname = current_setting('graph_path')
                AND r.contype IN ('c', 'x')
            """)

            constraints = await cursor.fetchall()
            for constraint in constraints:
                constraint_name, label_name = constraint[0], constraint[1]
                try:
                    await cursor.execute(
                        f'DROP CONSTRAINT {constraint_name} ON "{label_name}"'
                    )
                except Exception:
                    pass

            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def db_connection(db_setup, clean_graph, graphname):
    """Provide a database connection for tests."""
    async with get_pool_connection(db_setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            await cursor.execute(f"SET graph_path = {graphname}")
            await conn.commit()

            yield conn, cursor
