import asyncio
import os
import signal
import socket
import time
from typing import Any
import pytest
import pytest_asyncio

from mcp_agensgraph_cypher.server import (
    create_mcp_server,
    create_pool,
    ensure_graph,
)
from mcp_agensgraph_common.safety import quote_identifiers as _quote_identifiers

# The role these tests run as is whoever the developer is connected as, which locally is the
# bootstrap superuser -- one that can run a command on the server's host, and so one the read
# tools refuse to serve unless it is accepted out loud. What is under test here is the
# transaction, which refuses a write from any role.
ALLOW_SERVER_PROGRAMS = True


def free_port() -> int:
    """A port the kernel chose, rather than one written down here.

    Every fixture below used to name a port -- 8001 for the plain HTTP server, and the
    memory server's fixtures name the same one. So the two suites cannot run at once, and
    when they did, the cypher test that asserts on a tools list was answered by the memory
    server and passed against the wrong list. Binding 0 and reading back what was assigned
    means each fixture takes a port nothing else holds, and the suites are independent.
    """
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        return int(taken.getsockname()[1])


class Spawned:
    """A server process and where it is listening.

    The tests need the port, not just the process, now that no one writes it down.
    """

    def __init__(self, process: Any, port: int) -> None:
        self.process = process
        self.port = port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp/"

    @property
    def returncode(self):
        return self.process.returncode


async def _wait_for_port_free(port: int, timeout: float = 30.0) -> None:
    """Wait until nothing on localhost holds ``port``."""
    deadline = time.monotonic() + timeout
    while await _port_is_open(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"port {port} still in use after {timeout}s")
        await asyncio.sleep(0.05)


async def _wait_for_server(process: Any, port: int, timeout: float = 30.0) -> None:
    """Wait until a spawned server accepts connections on ``port``.

    Polling beats sleeping a fixed interval twice over: a server that is ready sooner is
    not waited on, and one that needs longer is not called broken. If it exits instead,
    report its output rather than time out.
    """
    deadline = time.monotonic() + timeout
    while not await _port_is_open(port):
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise RuntimeError(
                f"server exited before taking port {port}. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"server did not take port {port} within {timeout}s")
        await asyncio.sleep(0.05)


async def _port_is_open(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def _stop_server(process: Any, port: int) -> None:
    """Stop a spawned server and wait for its port to come free.

    The launcher runs the server as its own child, so the signal goes to the process
    group: signalling the launcher alone can leave the server holding the port, which the
    next test then tries to bind.
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
    await _wait_for_port_free(port)

@pytest.fixture(scope="module")
def graphname():
    return os.getenv("AGENSGRAPH_GRAPH_NAME", "test")

@pytest.fixture(scope="module")
def db_url():
    """Where the server is, out of the environment."""
    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    if not db_name or not db_user:
        # No password is required to reach a server -- trust and peer authentication have
        # none, and a local one usually does. Requiring one here is what kept these tests from
        # running anywhere.
        raise ValueError(
            "Set AGENSGRAPH_DB and AGENSGRAPH_USERNAME to run the tests that need a server."
        )
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup(graphname, db_url):
    """A pool reading the test graph, opened and closed inside one test's event loop.

    Per test rather than per module, because a pool returns a connection through a worker task
    on the loop that opened it, and each test here runs in a loop of its own. A pool outliving
    its loop is handed nothing back: measured, the second test in a module waited the pool's
    full 30 s and was given no connection at all.
    """
    # Before the pool: its connections each select the graph as they are made, so a graph that
    # is not there yet fails all of them.
    await ensure_graph(db_url, graphname)
    pool = create_pool(db_url, graphname, min_size=1, max_size=4, read_timeout=30)

    await pool.open()
    await pool.wait()

    yield pool

    await pool.close()


@pytest_asyncio.fixture(scope="function")
async def mcp_server(setup, graphname):
    return create_mcp_server(
        setup, graphname=graphname, allow_server_programs=ALLOW_SERVER_PROGRAMS
    )


@pytest_asyncio.fixture(scope="function")
async def mcp_server_short_timeout(setup, graphname, db_url):
    """MCP server whose reads have almost no time at all.

    A pool of its own, because how long a read may take is the connections' own
    ``statement_timeout`` rather than something the server sets per call.
    """
    pool = create_pool(db_url, graphname, min_size=1, max_size=2, read_timeout=0.01)
    await pool.open()
    await pool.wait()
    try:
        yield create_mcp_server(
            pool, graphname=graphname, allow_server_programs=ALLOW_SERVER_PROGRAMS
        )
    finally:
        await pool.close()


@pytest_asyncio.fixture(scope="function")
async def mcp_server_tiny_sample(setup, graphname):
    """MCP server whose schema sample is smaller than one label's node count."""
    return create_mcp_server(
        setup,
        graphname=graphname,
        schema_sample=2,
        allow_server_programs=ALLOW_SERVER_PROGRAMS,
    )


@pytest_asyncio.fixture(scope="function")
async def two_label_data(setup, clear_data: Any, graphname):
    """Two labels, the first with more nodes than the tiny sample, plus a relationship
    type that reaches both labels."""
    async with setup.connection() as conn:
        async with conn.cursor() as cursor:
            query = """
                CREATE (a:Person {name: 'Alice', age: 30}),
                       (b:Person {name: 'Bob', age: 25}),
                       (c:Person {name: 'Charlie', age: 35}),
                       (d:Person {name: 'Dana', age: 40}),
                       (x:Company {name: 'Acme', founded: 1999}),
                       (y:Company {name: 'Globex', founded: 2005}),
                       (a)-[:KNOWS]->(b),
                       (a)-[:KNOWS]->(x)
            """
            await cursor.execute(_quote_identifiers(query))
            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def init_data(setup, clear_data: Any, graphname):
    async with setup.connection() as conn:
        async with conn.cursor() as cursor:
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
async def clear_data(setup, graphname):
    async with setup.connection() as conn:
        async with conn.cursor() as cursor:
            # Clear existing data in the graph the pool is reading
            await cursor.execute("MATCH (n) DETACH DELETE n")
            await conn.commit()


@pytest_asyncio.fixture(scope="function")
async def http_server(setup, graphname):
    """HTTP server fixture on a free port with default settings."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    port = free_port()

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
        str(port),
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
        start_new_session=True,
    )

    await _wait_for_server(process, port)

    yield Spawned(process, port)

    # Cleanup
    await _stop_server(process, port)


@pytest_asyncio.fixture(scope="function")
async def http_server_read_only(setup, graphname):
    """HTTP server fixture on a free port with read-only mode enabled."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    port = free_port()

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
        str(port),
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
        start_new_session=True,
    )

    await _wait_for_server(process, port)

    # Check if process is still running
    if process.returncode is not None:
        raise RuntimeError(f"Read-only server failed to start with return code: {process.returncode}")

    yield Spawned(process, port)

    # Cleanup
    await _stop_server(process, port)


@pytest_asyncio.fixture(scope="function")
async def http_server_restricted_cors(setup, graphname):
    """HTTP server fixture on a free port with restricted CORS settings."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    port = free_port()

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
        str(port),
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
        start_new_session=True,
    )

    await _wait_for_server(process, port)

    yield Spawned(process, port)

    # Cleanup
    await _stop_server(process, port)


@pytest_asyncio.fixture(scope="function")
async def http_server_custom_hosts(setup, graphname):
    """HTTP server fixture on a free port with custom allowed hosts."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    port = free_port()

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
        str(port),
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
        start_new_session=True,
    )

    await _wait_for_server(process, port)

    yield Spawned(process, port)

    # Cleanup
    await _stop_server(process, port)


@pytest_asyncio.fixture(scope="function")
async def sse_server(setup, graphname):
    """Start the MCP server in SSE mode."""
    import asyncio
    import subprocess

    db_name = os.getenv("AGENSGRAPH_DB")
    db_user = os.getenv("AGENSGRAPH_USERNAME")
    db_password = os.getenv("AGENSGRAPH_PASSWORD", "")
    db_host = os.getenv("AGENSGRAPH_HOST", "localhost")
    db_port = os.getenv("AGENSGRAPH_PORT", "5432")

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    port = free_port()

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "mcp-agensgraph-cypher",
        "--transport",
        "sse",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        str(port),
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
        start_new_session=True,
    )

    await _wait_for_server(process, port)

    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        raise RuntimeError(
            f"Server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    yield Spawned(process, port)

    await _stop_server(process, port)
