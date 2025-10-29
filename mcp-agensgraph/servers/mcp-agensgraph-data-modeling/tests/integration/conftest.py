import asyncio
import os
import subprocess

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
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint r
        JOIN pg_catalog.ag_label l ON r.conrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        WHERE g.graphname = current_setting('graph_path')
        AND r.contype IN ('c', 'x')
        AND pg_catalog.ag_get_graphconstraintdef(r.oid) ILIKE '%(' || key_name || ') IS UNIQUE%'
    ) INTO found;

    RETURN found;
END;
$$ LANGUAGE plpgsql;
"""


def get_pool_connection(pool: AsyncConnectionPool):
    """Context manager for getting a connection from the pool."""
    return pool.connection()


# ===== Transport Testing Fixtures =====


@pytest_asyncio.fixture(scope="function")
async def mcp_server():
    """Create MCP server instance for transport testing."""
    mcp = create_mcp_server()
    return mcp


@pytest_asyncio.fixture
async def sse_server():
    """Start the MCP server in SSE mode."""
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-data-modeling",
        "--transport",
        "sse",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8002",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    await asyncio.sleep(3)

    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield process

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
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    if not db_name or not db_user or not db_password:
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
