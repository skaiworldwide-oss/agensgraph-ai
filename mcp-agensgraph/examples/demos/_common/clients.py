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

from mcp_agensgraph_common.connection import (
    create_pool,
    ensure_graph,
    get_pool_connection,
)

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

    Opens a pool, ensures the graph + schema-helper functions exist, builds the
    server, and yields a connected client. Pass server knobs as kwargs
    (``page_size``, ``schema_sample``, ``read_timeout``, ``token_limit``, ``namespace``).
    """
    from mcp_agensgraph_cypher.server import _ensure_helper_functions, create_mcp_server

    config.ensure_db(database)
    pool = create_pool(config.dsn(database))
    await pool.open()
    try:
        await ensure_graph(pool, graphname)  # CREATE GRAPH IF NOT EXISTS (no-op if present)
        await _ensure_helper_functions(pool, graphname)
        mcp = create_mcp_server(pool, graphname, read_only=read_only, **kwargs)
        async with Client(mcp) as client:
            yield client
    finally:
        await pool.close()


@contextlib.asynccontextmanager
async def memory_client(database: str, graphname: str, **kwargs):
    """In-memory client for the memory server against ``database``/``graphname``."""
    from mcp_agensgraph_memory.agensgraph_memory import AgensGraphMemory
    from mcp_agensgraph_memory.server import create_mcp_server, jsonb_to_string

    config.ensure_db(database)
    pool = create_pool(config.dsn(database))
    await pool.open()
    try:
        await ensure_graph(pool, graphname)
        async with get_pool_connection(pool) as conn:
            async with conn.cursor() as cur:
                await cur.execute(jsonb_to_string)
        memory = AgensGraphMemory(pool, graphname)
        await memory.create_fulltext_index()
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
