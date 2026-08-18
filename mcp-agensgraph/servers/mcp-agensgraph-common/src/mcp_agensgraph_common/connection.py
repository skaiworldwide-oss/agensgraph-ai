"""AgensGraph connection and query execution for the DB-backed MCP servers.

Three things are settled once, when the pool is made, rather than per tool call:

* **which graph to read.** The driver selects it on every connection it hands out, and fills
  the label table for it. Selecting it per statement is a statement per call, and the driver
  does not read a graph path set behind its back.
* **how long a statement may take.** It travels in the connection's own startup options, so
  the limit is already in force when the connection arrives.
* **how many connections there may be.** Both bounds are given values, because a pool that
  names neither is four connections wide however much work arrives.

What a call decides for itself is whether its transaction may write. A read runs inside the
driver's ``read_only_transaction``, so the refusal is the server's -- ``25006`` for a Cypher
write, an ``INSERT``, a ``TRUNCATE`` or a ``DROP`` alike -- rather than a reading of the text.
It ends by rolling back: a transaction that could not write has nothing to commit, and the one
thing it can do is ``SET``, which committed would belong to whoever borrows the connection next.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import agensgraph
from agensgraph import AsyncConnection, AsyncConnectionPool
from agensgraph.cypher import quote_identifier, without_literals
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from .config import DEFAULT_POOL_MAX_SIZE, DEFAULT_POOL_MIN_SIZE
from .results import rows_of

logger = logging.getLogger("mcp_agensgraph_common")

# Paging clauses the caller ended their own query with. Cypher takes one set of them per query
# part, so ours cannot follow theirs: appending to `... OFFSET 5` produces
# `... OFFSET 5 SKIP 0 LIMIT 101`, which the server rejects with "Cypher query must end with
# RETURN, FINISH or update clause". `OFFSET` is `SKIP` under another name.
_TRAILING_PAGING = re.compile(r"\b(?:SKIP|OFFSET|LIMIT)\s+\S+\s*$", re.IGNORECASE)

# Clauses the grammar keeps for the top of a statement. `SELECT * FROM (<cypher>) AS _page`
# routes the query through the read-clause continuation set instead, which holds
# MATCH/WITH/LET/LOAD/UNWIND/FOR/CALL{} and none of these -- so wrapping one is a syntax
# error at the word, not a slower plan. Measured: `... FILTER n.t IS NOT NULL RETURN n`,
# `... NEXT RETURN t` and `... CALL jsonb_each(...) YIELD key RETURN key` each come back
# `ERROR: syntax error at or near "FILTER" / "NEXT" / "CALL"`.
_TOP_LEVEL_ONLY = re.compile(r"(?<![A-Za-z0-9_.\"])(FILTER|NEXT|YIELD)(?![A-Za-z0-9_])", re.IGNORECASE)

# A query made of several parts. Which one a trailing SKIP and LIMIT belong to is the reason
# this is worth finding: they bind to the last part alone.
_UNION = re.compile(r"(?<![A-Za-z0-9_.\"])UNION(?![A-Za-z0-9_])", re.IGNORECASE)


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
    ``WHERE n.name IN $names`` expects a JSONB array/object), but psycopg adapts a bare Python
    ``list`` as a PostgreSQL array. Params reaching an MCP tool arrive as plain JSON values, so
    wrap any list/dict in ``Jsonb`` here; scalars and already-wrapped values pass through
    unchanged.
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


def create_pool(
    dsn: str,
    graphname: str,
    *,
    min_size: int = DEFAULT_POOL_MIN_SIZE,
    max_size: int = DEFAULT_POOL_MAX_SIZE,
    read_timeout: Optional[float] = None,
    **kwargs: Any,
) -> AsyncConnectionPool:
    """A (not-yet-opened) pool of connections already reading ``graphname``.

    The driver selects the graph on every connection it makes, which is where the saving is:
    a statement that selects a graph is a round trip, and every tool call sends one statement.
    It is also the only way to tell the driver which graph its label table describes, since it
    does not read a graph path set behind its back.

    ``read_timeout`` becomes the connection's own ``statement_timeout``, in its startup options,
    so it is in force before the first statement and costs nothing per call. A limit set inside
    the transaction instead is one more round trip on every read; a per-caller deadline is two,
    because it opens the transaction to carry the setting and the read-only block then nests
    inside it. Measured over a proxy counting client-to-server flushes: 5.00 round trips per
    read call this way, 6.00 with the limit set per statement, and 9.00 with a deadline.

    Every statement on the connection is bounded by it, a write included. Lifting it for a write
    is a round trip of its own, and what it buys is a statement somebody else wrote holding a
    pooled connection for as long as it likes.
    """
    options = kwargs.pop("kwargs", None) or {}
    if read_timeout is not None:
        milliseconds = int(read_timeout * 1000)
        # Three limits, because a statement is only one of the ways a caller holds a connection.
        # A statement that waits on a lock it will never get is bounded by lock_timeout; a
        # transaction whose caller went away between statements is bounded by the third, which
        # statement_timeout does not see because nothing is running.
        settings = " ".join(
            (
                f"-c statement_timeout={milliseconds}",
                f"-c lock_timeout={milliseconds}",
                f"-c idle_in_transaction_session_timeout={milliseconds}",
            )
        )
        options = {**options, "options": f"{options.get('options', '')} {settings}".strip()}
    return agensgraph.AsyncConnectionPool(
        dsn,
        graph=graphname,
        min_size=min_size,
        max_size=max_size,
        kwargs=options or None,
        **kwargs,
    )


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
    role is making a claim it cannot keep.

    The question is the driver's ``can_run_server_programs``, which asks membership as
    ``member`` rather than ``usage`` -- a membership granted ``WITH INHERIT FALSE`` carries
    nothing until ``SET ROLE`` names it, and ``SET ROLE`` moves no rows, so a read-only
    transaction permits it.
    """
    if allow_server_programs:
        logger.warning(
            "Serving as a role that may run a command on the server's host, because "
            "--allow-server-programs was given. A read-only tool is not a boundary for it."
        )
        return
    async with pool.connection() as conn:
        held = await conn.can_run_server_programs()
        await conn.rollback()
    if held:
        raise RuntimeError(
            "This role can run a command on the server's host, through COPY ... TO PROGRAM, "
            "which a read-only transaction does not stop -- so the read tools would not be a "
            "boundary. Connect as a role that is neither a superuser nor a member of "
            "pg_execute_server_program, or pass --allow-server-programs to accept it."
        )


async def ensure_graph(dsn: str, graphname: str) -> None:
    """Make the graph, on a connection of its own, before the pool that reads it.

    Before the pool rather than through it: a pool told which graph to read selects it on every
    connection it makes, and a graph that is not there yet fails all of them.
    """
    conn = await AsyncConnection.connect(dsn, autocommit=True)
    try:
        await conn.execute(f"CREATE GRAPH IF NOT EXISTS {quote_identifier(graphname)}")
    finally:
        await conn.close()
    logger.info("Graph %r is there", graphname)


async def run_query(
    pool: AsyncConnectionPool,
    query: str,
    params: Optional[dict[str, Any]] = None,
    *,
    read_only: bool = False,
    allow_server_programs: bool = False,
) -> list[dict[str, Any]]:
    """Run one statement on a connection from the pool and return its rows as maps.

    With ``read_only`` the statement runs inside a transaction the server will not let write,
    which is the boundary itself rather than a message about one: a Cypher write, an ``INSERT``,
    a ``TRUNCATE`` and a ``DROP`` are each refused with ``25006`` and leave nothing behind. The
    block ends by rolling back, so a ``SET`` the statement performed does not reach the next
    caller to borrow the connection.

    The graph is the pool's, so nothing here selects one.
    """
    bound = jsonb_params(params) or None
    async with pool.connection() as conn:
        if read_only:
            async with conn.read_only_transaction(
                allow_server_programs=allow_server_programs
            ):
                result = await conn.execute_query(query, bound)
        else:
            async with conn.transaction():
                result = await conn.execute_query(query, bound)
    return rows_of(result)


def paged_statement(query: str, *, limit: int, offset: int) -> str:
    """The caller's query, asking the server for one page of it and one row more.

    ``SKIP`` and ``LIMIT`` appended to the query is the form the grammar takes anywhere, and it
    is what a single-part query gets. It binds to the **last query part**, though, so two
    queries cannot have it:

    * a ``UNION``, where the last part is one arm. Measured: a five-row page of a two-arm union
      had the server produce 50,006 rows and Python then kept five. Read as a subquery, the
      server produces six.
    * a query that already ends in paging of its own, where a second ``SKIP`` cannot follow the
      first -- the server answers "Cypher query must end with RETURN, FINISH or update clause".

    Both are taken from outside instead, as ``SELECT * FROM (<cypher>) AS _page``, and the
    closing bracket goes on a line of its own so that a query the caller ended with a ``--``
    comment does not comment it out. Three clauses the grammar keeps for the top of a statement
    cannot be read as a subquery at all, so a query holding one of those *and* needing to be
    wrapped is refused here, in terms the caller can act on, rather than sent to produce a wrong
    count or a syntax error naming a word they did place correctly.

    ``limit`` and ``offset`` are integers this function bounds, so writing them into the
    statement binds nothing a caller chose.
    """
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    inner = query.rstrip().rstrip(";").rstrip()
    # Located against the statement with its strings and comments blanked, so that a property
    # named `next` and the word LIMIT inside a quoted value are not read as clauses.
    blanked = without_literals(inner)
    own_paging = _TRAILING_PAGING.search(blanked) is not None
    several_parts = _UNION.search(blanked) is not None
    if not (own_paging or several_parts):
        return f"{inner}\nSKIP {offset} LIMIT {limit + 1}"
    if (clause := unwrappable_clause(inner)) is not None:
        reason = (
            "is a UNION, so a page of it has to be taken from outside"
            if several_parts
            else "ends with its own paging clause, which a second SKIP cannot follow"
        )
        raise ValueError(
            f"this query {reason}, and it also uses {clause}, which the grammar accepts only at "
            f"the top of a statement -- so it cannot be read as the subquery that taking the "
            f"page from outside needs. Take the page in the query itself, by writing the SKIP "
            f"and LIMIT you want, and ask for it with limit and offset left alone."
        )
    return f"SELECT * FROM (\n{inner}\n) AS _page LIMIT {limit + 1} OFFSET {offset}"


async def read_page(
    pool: AsyncConnectionPool,
    query: str,
    params: Optional[dict[str, Any]] = None,
    *,
    limit: int = 100,
    offset: int = 0,
    allow_server_programs: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """One page of a read, and whether there is more behind it.

    The page is taken **by the server**, so it can stop rather than produce the whole result and
    have the rows thrown away here. One row more than the page is asked for, which is how a full
    page is told from a page with more behind it.

    Returns ``(rows, has_more)`` where ``rows`` holds at most ``limit`` items.
    """
    limit = max(1, int(limit))
    rows = await run_query(
        pool,
        paged_statement(query, limit=limit, offset=offset),
        params,
        read_only=True,
        allow_server_programs=allow_server_programs,
    )
    return rows[:limit], len(rows) > limit
