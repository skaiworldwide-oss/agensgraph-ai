import os
from typing import Any
import pytest
import pytest_asyncio

from mcp_agensgraph_cypher.server import create_mcp_server, get_pool_connection
from mcp_agensgraph_cypher.utils import _quote_identifiers
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool, PoolTimeout  # type: ignore

@pytest.fixture(scope="module")
def graphname():
    return os.getenv("AGENSGRAPH_GRAPH_NAME", "test")

@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup(graphname):
    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    if not db_name or not db_user or not db_password:
        raise ValueError("Environment variables AGENSGRAPH_DB, AGENSGRAPH_USERNAME, and AGENSGRAPH_PASSWORD must be set.")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    agensgraph_driver = AsyncConnectionPool(db_url, open=False)

    await agensgraph_driver.open()

    # Ensure graph exists
    async with get_pool_connection(agensgraph_driver) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            await cursor.execute(f"CREATE GRAPH IF NOT EXISTS {graphname}")
            await conn.commit()

    yield agensgraph_driver

    await agensgraph_driver.close()


@pytest_asyncio.fixture(scope="function")
async def mcp_server(setup, graphname):
    mcp = create_mcp_server(setup, graphname=graphname)

    return mcp


@pytest_asyncio.fixture(scope="function")
async def mcp_server_short_timeout(setup, graphname):
    """MCP server with very short timeout for testing timeout behavior."""
    mcp = create_mcp_server(setup, graphname=graphname, read_timeout=0.01)
    return mcp


@pytest_asyncio.fixture(scope="function")
async def init_data(setup, clear_data: Any, graphname):
    async with get_pool_connection(setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            # Create test data
            await cursor.execute(f"CREATE GRAPH IF NOT EXISTS {graphname}")
            await cursor.execute(f"SET graph_path = {graphname}")
            query = """
                CREATE (a:Person {name: 'Alice', age: 30}),
                       (b:Person {name: 'Bob', age: 25}),
                       (c:Person {name: 'Charlie', age: 35}),
                       (a)-[:FRIEND]->(b),
                       (b)-[:FRIEND]->(c)
            """
            # Quote identifiers to preserve case sensitivity
            query = _quote_identifiers(query)
            await cursor.execute(query)
            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def clear_data(setup):
    async with get_pool_connection(setup) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            # Clear existing data
            await cursor.execute("""
                SET graph_path = 'test'
            """)
            await cursor.execute("""
                MATCH (n) DETACH DELETE n
            """)
            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def http_server(setup, graphname):
    """HTTP server fixture on port 8001 with default settings."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Start server process in HTTP mode using the installed binary
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
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

    # Wait for server to start
    await asyncio.sleep(3)

    # Check if process is still running
    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield process

    # Cleanup
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest_asyncio.fixture(scope="function")
async def http_server_read_only(setup, graphname):
    """HTTP server fixture on port 8005 with read-only mode enabled."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Start server process in HTTP mode with read-only
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
        "--transport",
        "http",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        "8005",
        "--read-only",
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
        env=os.environ.copy(),
        # Remove stdout and stderr pipes to see output directly
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )

    # Wait for server to start
    await asyncio.sleep(3)

    # Check if process is still running
    if process.returncode is not None:
        raise RuntimeError(f"Read-only server failed to start with return code: {process.returncode}")

    yield process

    # Cleanup
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest_asyncio.fixture(scope="function")
async def http_server_restricted_cors(setup, graphname):
    """HTTP server fixture on port 8003 with restricted CORS settings."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Start server process in HTTP mode with restricted CORS
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
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

    # Wait for server to start
    await asyncio.sleep(3)

    # Check if process is still running
    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Restricted CORS server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield process

    # Cleanup
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest_asyncio.fixture(scope="function")
async def http_server_custom_hosts(setup, graphname):
    """HTTP server fixture on port 8004 with custom allowed hosts."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Start server process in HTTP mode with custom allowed hosts
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
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

    # Wait for server to start
    await asyncio.sleep(3)

    # Check if process is still running
    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Custom hosts server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield process

    # Cleanup
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest_asyncio.fixture(scope="function")
async def sse_server(setup, graphname):
    """Start the MCP server in SSE mode."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
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
