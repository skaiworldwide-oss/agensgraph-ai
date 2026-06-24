"""AgensGraph connection + query execution for the DB-backed MCP servers.

Centralizes the connection-pool lifecycle, graph bootstrap, and a single query
executor that applies (per transaction, so it is pool-safe):

- ``SET TRANSACTION READ ONLY`` when ``read_only`` — AgensGraph rejects any Cypher
  write at the database level (verified). This is the real read-only guarantee.
- ``SET LOCAL statement_timeout`` for read queries.
- ``SET LOCAL graph_path`` to the (identifier-quoted) graph name.

Identifiers (graph name) are composed with ``psycopg.sql`` rather than f-strings.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional
from urllib.parse import quote, urlparse

import psycopg
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .results import record_to_dict

logger = logging.getLogger("mcp_agensgraph_common")


def build_dsn(db_url: str, username: str, password: str, database: str) -> str:
    """Build a PostgreSQL DSN from a base URL plus explicit credentials/database."""
    parsed = urlparse(db_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = quote(username or "", safe="")
    pw = quote(password or "", safe="")
    return f"postgresql://{user}:{pw}@{host}:{port}/{database}"


def create_pool(dsn: str, **kwargs: Any) -> AsyncConnectionPool:
    """Create a (not-yet-opened) async connection pool."""
    return AsyncConnectionPool(dsn, open=False, **kwargs)


@asynccontextmanager
async def get_pool_connection(pool: AsyncConnectionPool, timeout: Optional[float] = None):
    """Borrow a connection from the pool, returning it on exit.

    Includes a workaround for a psycopg_pool edge case where the pool can time out
    while reporting capacity; ``putconn`` resets the connection (rolling back any
    open transaction), so callers manage their own transaction explicitly.
    """
    try:
        connection = await pool.getconn(timeout=timeout)
    except PoolTimeout:
        await pool._add_connection(None)  # pragma: no cover - pool workaround
        connection = await pool.getconn(timeout=timeout)
    try:
        # `async with connection` commits on clean exit / rolls back on error, so
        # callers that don't manage their own transaction still get committed work.
        async with connection:
            yield connection
    finally:
        await pool.putconn(connection)


async def ensure_graph(pool: AsyncConnectionPool, graphname: str) -> None:
    """``CREATE GRAPH IF NOT EXISTS`` with an identifier-quoted graph name."""
    async with get_pool_connection(pool) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(sql.Identifier(graphname))
            )
        await conn.commit()
    logger.info("Ensured graph %r exists", graphname)


async def run_query(
    pool: AsyncConnectionPool,
    graphname: str,
    query: str,
    params: Optional[dict[str, Any]] = None,
    *,
    read_only: bool = False,
    timeout: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Execute a Cypher query against ``graphname`` and return parsed rows.

    When ``read_only`` is set, the statement runs in a READ ONLY transaction so the
    database itself rejects writes (defense in depth, independent of any client-side
    keyword check).
    """
    set_path = sql.SQL("SET LOCAL graph_path = {}").format(sql.Identifier(graphname))
    async with get_pool_connection(pool) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cur:
            try:
                if read_only:
                    # Must precede any snapshot-taking statement in the transaction.
                    await cur.execute("SET TRANSACTION READ ONLY")
                if timeout is not None:
                    # SET does not accept bind parameters; inline the validated int.
                    await cur.execute(
                        sql.SQL("SET LOCAL statement_timeout = {}").format(
                            sql.Literal(int(timeout * 1000))
                        )
                    )
                await cur.execute(set_path)
                if params:
                    await cur.execute(query, params)
                else:
                    await cur.execute(query)
                await conn.commit()
            except psycopg.Error:
                await conn.rollback()
                raise

            try:
                rows = await cur.fetchall()
            except psycopg.ProgrammingError:
                # Statement returned no result set (e.g. SET, write with no RETURN).
                return []

    return [record_to_dict(r) for r in rows]


async def run_paginated_query(
    pool: AsyncConnectionPool,
    graphname: str,
    query: str,
    params: Optional[dict[str, Any]] = None,
    *,
    read_only: bool = False,
    timeout: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """Run a Cypher query and return one page of rows plus a ``has_more`` flag.

    The query is wrapped as an AgensGraph SQL subquery so ``LIMIT``/``OFFSET`` are
    applied **by the database** — it can short-circuit instead of materializing the
    whole result set (the point of paginating an arbitrary read). One extra row is
    fetched to detect whether more results exist beyond this page. Vertex/edge
    values survive the wrap, so the normal parsing still applies.

    Returns ``(rows, has_more)`` where ``rows`` has at most ``limit`` items.
    """
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    inner = query.rstrip().rstrip(";").rstrip()
    # limit/offset are validated ints, so inlining them is injection-safe (and SQL
    # LIMIT/OFFSET would accept binds, but the inner query owns the param namespace).
    wrapped = f"SELECT * FROM (\n{inner}\n) AS _page LIMIT {limit + 1} OFFSET {offset}"
    rows = await run_query(
        pool, graphname, wrapped, params, read_only=read_only, timeout=timeout
    )
    has_more = len(rows) > limit
    return rows[:limit], has_more
