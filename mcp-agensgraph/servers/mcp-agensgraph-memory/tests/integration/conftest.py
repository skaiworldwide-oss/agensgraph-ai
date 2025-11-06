import asyncio
import os
import subprocess

import pytest
import pytest_asyncio
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool

from mcp_agensgraph_memory.server import create_mcp_server
from mcp_agensgraph_memory.agensgraph_memory import AgensGraphMemory


def get_pool_connection(pool: AsyncConnectionPool):
    """Context manager for getting a connection from the pool."""
    return pool.connection()


# ===== Transport Testing Fixtures =====

@pytest_asyncio.fixture(scope="function")
async def mcp_server():
    """Create MCP server instance for transport testing."""
    # For transport tests, we don't need a real database connection
    # Create a mock connection pool
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    pool = AsyncConnectionPool(db_url, open=False)

    try:
        await pool.open()
        memory = AgensGraphMemory(pool, graphname)
        mcp = create_mcp_server(memory)
        yield mcp
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def sse_server():
    """Start the MCP server in SSE mode."""
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")
    db_url = f"postgresql://{db_host}:{db_port}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-memory",
        "--transport",
        "sse",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8002",
        "--db-url",
        db_url,
        "--username",
        db_user,
        "--password",
        db_password,
        "--database",
        db_name,
        "--graphname",
        graphname,
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


@pytest_asyncio.fixture
async def http_server():
    """Start the MCP server in HTTP mode."""
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")
    db_url = f"postgresql://{db_host}:{db_port}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-memory",
        "--transport",
        "http",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8001",
        "--db-url",
        db_url,
        "--username",
        db_user,
        "--password",
        db_password,
        "--database",
        db_name,
        "--graphname",
        graphname,
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


@pytest_asyncio.fixture
async def http_server_restricted_cors():
    """Start the MCP server in HTTP mode with restricted CORS origins."""
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")
    db_url = f"postgresql://{db_host}:{db_port}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-memory",
        "--transport",
        "http",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8003",
        "--allow-origins",
        "http://localhost:3000,https://trusted-site.com",
        "--db-url",
        db_url,
        "--username",
        db_user,
        "--password",
        db_password,
        "--database",
        db_name,
        "--graphname",
        graphname,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    await asyncio.sleep(3)

    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Restricted CORS server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield process

    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest_asyncio.fixture
async def http_server_custom_hosts():
    """Start the MCP server in HTTP mode with custom allowed hosts."""
    db_name = os.getenv("AGENSGRAPH_DB", "test_memory")
    db_user = os.getenv("AGENSGRAPH_USERNAME", "agens")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "agens")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")
    graphname = os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")
    db_url = f"postgresql://{db_host}:{db_port}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-memory",
        "--transport",
        "http",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8004",
        "--allowed-hosts",
        "example.com,test.local",
        "--db-url",
        db_url,
        "--username",
        db_user,
        "--password",
        db_password,
        "--database",
        db_name,
        "--graphname",
        graphname,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    await asyncio.sleep(3)

    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Custom hosts server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
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
    return os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory")


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

    # Ensure graph exists
    async with get_pool_connection(agensgraph_pool) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            await cursor.execute(f"CREATE GRAPH IF NOT EXISTS {graphname}")
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

            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def memory(db_setup, clean_graph, graphname):
    """Provide an AgensGraphMemory instance for tests."""
    memory = AgensGraphMemory(db_setup, graphname)
    await memory.create_fulltext_index()
    yield memory
