"""Connect a FastMCP ``Client`` to each AgensGraph MCP server.

The demos drive the servers exactly as an MCP agent would — through the client/tool
protocol — but in-process (no LLM, no subprocess) for speed and reproducibility. The
transport helpers at the bottom are used by the transports demo to prove the same
tools work over stdio and Streamable HTTP against the real server process.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from typing import Any, Optional

with warnings.catch_warnings():  # quiet fastmcp's transitive authlib deprecation notice
    warnings.simplefilter("ignore")
    from fastmcp import Client

from mcp_agensgraph_common.connection import create_pool, ensure_graph

from . import config


# ---- result helpers -------------------------------------------------------------
def text(result) -> str:
    """The first text block of a CallToolResult."""
    return result.content[0].text if getattr(result, "content", None) else ""


def data(result) -> Any:
    """Parse the first text block of a CallToolResult as JSON."""
    raw = text(result)
    return json.loads(raw) if raw else None


# ---- in-memory clients ----------------------------------------------------------
@contextlib.asynccontextmanager
async def cypher_client(database: str, graphname: str, *, read_only: bool = False, **kwargs):
    """In-memory client for the cypher server against ``database``/``graphname``.

    Opens a pool, builds the server, and yields a connected client. Pass server knobs as
    kwargs (``page_size``, ``schema_sample``, ``read_timeout``, ``token_limit``,
    ``namespace``).

    A read-only client creates nothing: not the database, not the graph. This is the same
    rule the server's own ``main()`` follows, and it is what lets a demo point at a graph
    somebody else owns and mean the words "never writes". Creating the graph so that the
    read tools have something to read is a write, and on a shared database it is a write to
    a database the demo does not own.

    The server asks the connection whether it understands the GQL clauses, because that
    decides what the tools tell a model they accept; the demos ask the same question rather
    than hard-coding either answer.
    """
    from mcp_agensgraph_cypher.server import create_mcp_server, server_has_gql_clauses

    if not read_only:
        config.ensure_db(database)
    dsn = config.dsn(database)
    if not read_only:
        # Before the pool: its connections each select the graph as they are made, so one that
        # is not there yet fails all of them.
        await ensure_graph(dsn, graphname)
    pool = create_pool(dsn, graphname, read_timeout=kwargs.get("read_timeout", 30))
    await pool.open()
    try:
        gql_clauses = await server_has_gql_clauses(pool)
        mcp = create_mcp_server(
            pool,
            graphname,
            read_only=read_only,
            gql_clauses=gql_clauses,
            # The demos run as whoever the developer is connected as, which locally is the
            # bootstrap superuser -- a role that can run a command on the server's host, and
            # so one the read tools refuse to serve unless it is accepted out loud.
            allow_server_programs=True,
            **kwargs,
        )
        async with Client(mcp) as client:
            yield client
    finally:
        await pool.close()


@contextlib.asynccontextmanager
async def memory_client(database: str, graphname: str, **kwargs):
    """In-memory client for the memory server against ``database``/``graphname``.

    Started the way the memory server starts itself: the graph on a connection of its own,
    then a pool that selects that graph once per connection rather than once per call, then
    ``bootstrap`` to make the labels and the three indexes the tools write through.
    """
    from mcp_agensgraph_memory.agensgraph_memory import AgensGraphMemory
    from mcp_agensgraph_memory.bootstrap import bootstrap, ensure_graph, make_pool
    from mcp_agensgraph_memory.server import create_mcp_server

    config.ensure_db(database)
    dsn = config.dsn(database)
    await ensure_graph(dsn, graphname)
    pool = make_pool(dsn, graphname)
    await pool.open()
    try:
        await bootstrap(pool, graphname)
        memory = AgensGraphMemory(pool, graphname)
        mcp = create_mcp_server(memory, **kwargs)
        async with Client(mcp) as client:
            yield client
    finally:
        await pool.close()


@contextlib.asynccontextmanager
async def data_modeling_client(**kwargs):
    """In-memory client for the (DB-less) data-modeling server."""
    from mcp_agensgraph_data_modeling.server import create_mcp_server

    mcp = create_mcp_server(**kwargs)
    async with Client(mcp) as client:
        yield client


# ---- transport clients (used by the transports demo) ----------------------------
def stdio_client(command: str, args: list[str], env: Optional[dict] = None) -> Client:
    """A client that spawns ``command`` as a stdio MCP server subprocess."""
    from fastmcp.client.transports import StdioTransport

    full_env = {**os.environ, **(env or {})}
    return Client(StdioTransport(command=command, args=args, env=full_env))


def http_client(url: str) -> Client:
    """A client for a Streamable-HTTP MCP server already listening at ``url``."""
    from fastmcp.client.transports import StreamableHttpTransport

    return Client(StreamableHttpTransport(url))
