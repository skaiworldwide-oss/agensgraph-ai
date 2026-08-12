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
import re
from contextlib import asynccontextmanager
from typing import Any, Optional

import agensgraph
import psycopg
from agensgraph.cypher import without_literals
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import namedtuple_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from .results import record_to_dict

logger = logging.getLogger("mcp_agensgraph_common")

# Paging clauses the caller ended their own query with. Cypher takes one set of them per
# query part, so ours cannot follow theirs (see ``run_paginated_query``). ``OFFSET`` is
# ``SKIP`` under another name, so a query ending in one takes the same shape as a query
# ending in the other: appending to `... OFFSET 5` produces `... OFFSET 5 SKIP 0 LIMIT 101`,
# which the server rejects with "Cypher query must end with RETURN, FINISH or update clause".
_TRAILING_PAGING = re.compile(r"\b(?:SKIP|OFFSET|LIMIT)\s+\S+\s*$", re.IGNORECASE)

# Clauses the grammar keeps for the top of a statement. `SELECT * FROM (<cypher>) AS _page`
# routes the query through the read-clause continuation set instead, which holds
# MATCH/WITH/LET/LOAD/UNWIND/FOR/CALL{} and none of these -- so wrapping one is a syntax
# error at the word, not a slower plan. Measured: `... FILTER n.t IS NOT NULL RETURN n`,
# `... NEXT RETURN t` and `... CALL jsonb_each(...) YIELD key RETURN key` each come back
# `ERROR: syntax error at or near "FILTER" / "NEXT" / "CALL"`.
_TOP_LEVEL_ONLY = re.compile(r"(?<![A-Za-z0-9_.\"])(FILTER|NEXT|YIELD)(?![A-Za-z0-9_])", re.IGNORECASE)


def unwrappable_clause(query: str) -> Optional[str]:
    """The clause in this query that cannot be read as a subquery, if there is one.

    Located against the statement with its strings and comments blanked, so a property named
    ``next`` and the word ``FILTER`` inside a quoted value are not read as clauses.
    """
    found = _TOP_LEVEL_ONLY.search(without_literals(query))
    return found.group(1).upper() if found else None


def jsonb_params(params: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Wrap list/dict param values as JSONB.

    Cypher parameters are JSONB-shaped (a query like ``UNWIND $records AS r`` or
    ``WHERE n.name IN $names`` expects a JSONB array/object), but psycopg cannot adapt
    a bare Python ``list``/``dict``. Params reaching an MCP tool arrive as plain JSON
    values, so wrap any list/dict in ``Jsonb`` here; scalars and already-wrapped
    values pass through unchanged.
    """
    if not params:
        return params
    return {
        k: (Jsonb(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()
    }


def build_dsn(db_url: str, username: str, password: str, database: str) -> str:
    """Compose a connection string from a base URL plus whichever parts were given.

    Everything the URL carries is kept and everything given separately is layered on top, by
    psycopg, which quotes each value for the format it is writing. Reading the URL for a host
    and a port and rebuilding the rest by hand lost and invented settings:

    * ``?sslmode=require`` in the URL was dropped, so a connection asked to be encrypted was
      made in the clear;
    * a user and password in the URL were dropped and replaced with empty ones;
    * a database name is not a place for a query string, but it was pasted in front of one --
      ``db?host=/tmp`` moved the connection to another host, and ``db?sslmode=disable`` turned
      encryption off;
    * a URL with no scheme parsed as nothing at all and became ``localhost:5432``, which is a
      different server than the one asked for.

    An empty user or password is left out rather than sent as empty. libpq then resolves it the
    way it resolves everything else -- ``PGUSER``, ``PGPASSWORD``, ``.pgpass``, ``PGSERVICE``,
    peer authentication -- none of which was reachable while an empty string was always sent.
    """
    given = {"dbname": database or None, "user": username or None, "password": password or None}
    return make_conninfo(db_url, **{k: v for k, v in given.items() if v is not None})


def create_pool(dsn: str, **kwargs: Any) -> AsyncConnectionPool:
    """Create a (not-yet-opened) async connection pool of graph connections.

    The connection class is the driver's, which is what makes a vertex arrive as a vertex.
    Read over a plain psycopg connection, a graph value is the text the server printed and has
    to be matched with a regular expression; the endpoints of an edge are two identities that
    expression cannot resolve, so ``MATCH ()-[r]->() RETURN r`` reported an empty map at each
    end. The driver decodes the wire form, so both ends are named whether or not the query
    also returned the vertices.

    It also refuses a server it cannot read at connect time, from the version in the startup
    packet, rather than at whichever later statement first wants a catalog that is not there.
    """
    return AsyncConnectionPool(
        dsn, open=False, connection_class=agensgraph.AsyncConnection, **kwargs
    )


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


SERVER_PROGRAM_QUERY = """
select rolsuper or pg_has_role(current_user, 'pg_execute_server_program', 'member')
from pg_roles where rolname = current_user
"""


async def check_role_cannot_run_programs(
    pool: AsyncConnectionPool, *, allow_server_programs: bool = False
) -> None:
    """Refuse to serve as a role that can run a command on the server's host.

    ``COPY ... TO PROGRAM`` does exactly that, and a read-only transaction does not stop it: it
    takes rows out of the database rather than putting any in, so there is no write for the
    server to refuse. Reading the statement does not stop it either -- a second statement after
    a semicolon and a leading comment both get one past, and both were demonstrated. What stops
    it is not holding the privilege.

    So the role is asked about once, here, at startup, and one that holds it is refused rather
    than left to find out. A server that advertises a read-only tool while connected as such a
    role is making a claim it cannot keep. Asked as ``member`` rather than ``usage``, because a
    membership granted ``WITH INHERIT FALSE`` carries nothing until ``SET ROLE`` names it --
    and ``SET ROLE`` moves no rows, so a read-only transaction permits it.
    """
    if allow_server_programs:
        logger.warning(
            "Serving as a role that may run a command on the server's host, because "
            "--allow-server-programs was given. A read-only tool is not a boundary for it."
        )
        return
    async with get_pool_connection(pool) as conn:
        async with conn.cursor() as cur:
            await cur.execute(SERVER_PROGRAM_QUERY)
            row = await cur.fetchone()
        await conn.rollback()
    if row and row[0]:
        raise RuntimeError(
            "This role can run a command on the server's host, through COPY ... TO PROGRAM, "
            "which a read-only transaction does not stop -- so the read tools would not be a "
            "boundary. Connect as a role that is neither a superuser nor a member of "
            "pg_execute_server_program, or pass --allow-server-programs to accept it."
        )


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
                bound = jsonb_params(params)
                if bound:
                    await cur.execute(query, bound)
                else:
                    await cur.execute(query)
                if read_only:
                    # Nothing to commit -- the transaction could not write. What it *could* do
                    # is `SET`, and a committed setting belongs to the session rather than the
                    # transaction, so on a pooled connection it would be inherited by whoever
                    # borrows it next. Measured on a pool of one: `SET ROLE`, `SET search_path`
                    # and `SET work_mem` all reached the following call.
                    await conn.rollback()
                else:
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

    Paging is applied **by the database**, so it can short-circuit instead of
    materializing the whole result set (the point of paginating an arbitrary read). One
    extra row is fetched to detect whether more results exist beyond this page.

    Cypher's ``SKIP``/``LIMIT`` carries the page. A query the caller already ended with
    paging clauses -- ``SKIP``, its GQL spelling ``OFFSET``, or ``LIMIT`` -- takes another set
    only from outside, wrapped as ``SELECT * FROM (<cypher>) AS _page``.

    Neither shape fits a query that both ends in paging and holds a clause the grammar keeps
    for the top of a statement, so that is refused here in terms the caller can act on rather
    than sent to produce a syntax error naming a word they did not write.

    Vertex/edge values survive both forms, so the normal parsing still applies.

    Returns ``(rows, has_more)`` where ``rows`` has at most ``limit`` items.
    """
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    inner = query.rstrip().rstrip(";").rstrip()
    # Located against the statement with its strings and comments blanked, so that a property
    # named `next` and the word LIMIT inside a quoted value are not read as clauses.
    blanked = without_literals(inner)
    # limit/offset are validated ints, so inlining them is injection-safe (and both
    # forms accept binds, but the caller's query owns the param namespace).
    if not _TRAILING_PAGING.search(blanked):
        paged = f"{inner}\nSKIP {offset} LIMIT {limit + 1}"
    elif (clause := unwrappable_clause(inner)) is not None:
        raise ValueError(
            f"this query ends with its own paging clause and also uses {clause}, which the "
            f"grammar accepts only at the top of a statement. A page can be taken by appending "
            f"SKIP and LIMIT, which cannot follow the paging already there, or by reading the "
            f"query as a subquery, which {clause} cannot be part of. Take the page in the query "
            f"itself -- write the SKIP and LIMIT you want -- and ask for it with limit and "
            f"offset left alone."
        )
    else:
        paged = f"SELECT * FROM (\n{inner}\n) AS _page LIMIT {limit + 1} OFFSET {offset}"
    rows = await run_query(
        pool, graphname, paged, params, read_only=read_only, timeout=timeout
    )
    has_more = len(rows) > limit
    return rows[:limit], has_more
