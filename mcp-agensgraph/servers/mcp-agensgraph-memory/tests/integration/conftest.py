import asyncio
import os
import signal
import subprocess
import time

import pytest
import pytest_asyncio

from mcp_agensgraph_memory.agensgraph_memory import AgensGraphMemory
from mcp_agensgraph_memory.bootstrap import bootstrap, ensure_graph, make_pool
from mcp_agensgraph_memory.server import create_mcp_server


def settings():
    """Where to connect, and as whom.

    No password is required to reach a server -- trust and peer authentication have none, and a
    local one usually does. Requiring one is what kept these tests from running anywhere.
    """
    return {
        "database": os.getenv("AGENSGRAPH_DB", "test_memory"),
        "user": os.getenv("AGENSGRAPH_USERNAME", "agens"),
        "password": os.getenv("AGENSGRAPH_PASSWORD", ""),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": os.getenv("AGENSGRAPH_PORT", "5432"),
        "graphname": os.getenv("AGENSGRAPH_GRAPH_NAME", "test_memory"),
    }


def dsn(where=None):
    where = where or settings()
    return (
        f"host={where['host']} port={where['port']} dbname={where['database']} "
        f"user={where['user']}"
        + (f" password={where['password']}" if where["password"] else "")
    )


def server_arguments(where, extra=()):
    """The arguments every transport fixture starts the server with."""
    return [
        "uv",
        "run",
        "mcp-agensgraph-memory",
        *extra,
        "--db-url",
        f"postgresql://{where['host']}:{where['port']}",
        "--username",
        where["user"],
        "--password",
        where["password"],
        "--database",
        where["database"],
        "--graphname",
        where["graphname"],
    ]


async def port_is_open(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def wait_for_port_free(port: int, timeout: float = 30.0) -> None:
    """Wait until nothing on localhost holds the port the next server has to bind."""
    deadline = time.monotonic() + timeout
    while await port_is_open(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"port {port} still in use after {timeout}s")
        await asyncio.sleep(0.05)


async def wait_for_server(process, port: int, what: str, timeout: float = 60.0) -> None:
    """Wait until the server accepts connections, or say why it will not.

    Polling beats sleeping a fixed interval twice over: one that is ready sooner is not
    waited on, and one that needs longer -- this server makes labels and indexes before it
    serves -- is not called broken. If it exits instead, report what it printed.
    """
    deadline = time.monotonic() + timeout
    while not await port_is_open(port):
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise RuntimeError(
                f"{what} exited before taking port {port}. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{what} did not take port {port} within {timeout}s")
        await asyncio.sleep(0.05)


async def start_server(where, extra, what, port=None, stdin=None):
    if port is not None:
        await wait_for_port_free(port)
    process = await asyncio.create_subprocess_exec(
        *server_arguments(where, extra),
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
        # Its own session, so that stopping it signals the process group: the launcher runs
        # the server as its own child, and signalling the launcher alone leaves the server
        # running. Without this the group is the test runner's own.
        start_new_session=True,
    )
    if port is not None:
        await wait_for_server(process, port, what)
    return process


async def stop_server(process, port=None):
    """Stop a spawned server and wait for its port to come free.

    The launcher runs the server as its own child, so the signal goes to the process group:
    signalling the launcher alone leaves the server holding the port that the next test binds.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.returncode is not None:
            break
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
            break
        except asyncio.TimeoutError:
            continue
    if port is not None:
        await wait_for_port_free(port)


# ===== Transport Testing Fixtures =====


@pytest_asyncio.fixture(scope="function")
async def mcp_server():
    """Create MCP server instance for transport testing."""
    where = settings()
    await ensure_graph(dsn(where), where["graphname"])
    pool = make_pool(dsn(where), where["graphname"])
    try:
        await pool.open()
        await bootstrap(pool, where["graphname"])
        memory = AgensGraphMemory(pool, where["graphname"])
        yield create_mcp_server(memory)
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def sse_server():
    """Start the MCP server in SSE mode."""
    where = settings()
    process = await start_server(
        where,
        ["--transport", "sse", "--server-host", "127.0.0.1", "--server-port", "8002"],
        "SSE server",
        port=8002,
    )
    yield process
    await stop_server(process, 8002)


@pytest_asyncio.fixture
async def http_server():
    """Start the MCP server in HTTP mode."""
    where = settings()
    process = await start_server(
        where,
        ["--transport", "http", "--server-host", "127.0.0.1", "--server-port", "8001"],
        "HTTP server",
        port=8001,
    )
    yield process
    await stop_server(process, 8001)


@pytest_asyncio.fixture
async def http_server_restricted_cors():
    """Start the MCP server in HTTP mode with restricted CORS origins."""
    where = settings()
    process = await start_server(
        where,
        [
            "--transport",
            "http",
            "--server-host",
            "127.0.0.1",
            "--server-port",
            "8003",
            "--allow-origins",
            "http://localhost:3000,https://trusted-site.com",
        ],
        "Restricted CORS server",
        port=8003,
    )
    yield process
    await stop_server(process, 8003)


@pytest_asyncio.fixture
async def http_server_custom_hosts():
    """Start the MCP server in HTTP mode with custom allowed hosts."""
    where = settings()
    process = await start_server(
        where,
        [
            "--transport",
            "http",
            "--server-host",
            "127.0.0.1",
            "--server-port",
            "8004",
            "--allowed-hosts",
            "example.com,test.local",
        ],
        "Custom hosts server",
        port=8004,
    )
    yield process
    await stop_server(process, 8004)


# ===== Database Testing Fixtures =====


@pytest.fixture(scope="module")
def graphname():
    """Graph name for database testing."""
    return settings()["graphname"]


@pytest_asyncio.fixture(scope="function", autouse=False)
async def db_setup(graphname):
    """A pool whose connections are already reading the test graph."""
    where = settings()
    if not where["database"] or not where["user"]:
        raise ValueError(
            "Set AGENSGRAPH_DB and AGENSGRAPH_USERNAME to run the tests that need a server."
        )
    await ensure_graph(dsn(where), graphname)
    pool = make_pool(dsn(where), graphname)
    await pool.open()
    yield pool
    await pool.close()


@pytest_asyncio.fixture(scope="function")
async def clean_graph(db_setup, graphname):
    """Empty the graph before each database test."""
    async with db_setup.connection() as conn:
        await conn.execute_query("MATCH (n) DETACH DELETE n")
        await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def memory(db_setup, clean_graph, graphname):
    """Provide an AgensGraphMemory instance for tests."""
    await bootstrap(db_setup, graphname)
    yield AgensGraphMemory(db_setup, graphname)
