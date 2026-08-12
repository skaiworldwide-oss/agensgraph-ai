"""AgensGraph Cypher MCP server.

Schema introspection, read and write Cypher, plan inspection, index advice and health, built
on the shared ``mcp_agensgraph_common`` core (connection pool, identifier-quoted graph
bootstrap, read-only-transaction enforcement, result shaping, transport) and on the driver for
everything the driver already knows: what a statement is allowed to do, what is in a graph,
what a write changed, and how to read a large result without materialising it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional

import psycopg
from agensgraph.cypher import check_single_statement, writable_counters
from agensgraph.introspect import index_properties
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from mcp.types import ToolAnnotations
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import Field

from mcp_agensgraph_common.config import format_namespace
from mcp_agensgraph_common.connection import (
    build_dsn,
    create_pool,
    ensure_graph,
    get_pool_connection,
    jsonb_params,
    run_paginated_query,
    run_query,
    unwrappable_clause,
)
from mcp_agensgraph_common.results import (
    as_builtins,
    count_tokens,
    fit_rows,
    record_to_dict,
    value_sanitize,
)
from mcp_agensgraph_common.safety import quote_identifiers
from mcp_agensgraph_common.transport import run_server

from mcp_agensgraph_cypher.perf import (
    analyze_plan,
    existing_indexes_query,
    explain_statement,
    extensions_query,
    format_findings,
    health_queries,
    indexed_properties,
    label_stats_query,
    missing_extension_note,
    relname_to_label,
    top_cypher_queries_query,
)

logger = logging.getLogger("mcp_agensgraph_cypher")


# Default cap on nodes sampled *per label* when introspecting the schema (bounds cost on
# large graphs); overridable via AGENSGRAPH_SCHEMA_SAMPLE.
DEFAULT_SCHEMA_SAMPLE = 1000

# Read-result pagination: default page size and hard ceiling on rows per call, so an
# unbounded query can't flood the agent's context or the server's memory. Overridable
# via AGENSGRAPH_PAGE_SIZE / AGENSGRAPH_MAX_PAGE_SIZE.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

# How many rows a streamed walk fetches per round trip.
STREAM_CHUNK = 1000

# A uniqueness constraint names its property inside its rendered definition, and is kept as an
# exclusion constraint rather than an index -- which is why the property-index view does not
# show it and ``constraints()`` is asked separately.
_ASSERTED_UNIQUE = re.compile(r"\(([^()]+)\)\s+IS\s+UNIQUE", re.IGNORECASE)


def _refuse_unless_one_statement(query: str) -> None:
    """Refuse text that is more than one statement, in terms the caller can act on.

    The driver's message names the part of the caller's own text that is the reason, which is
    exactly what a model needs to correct itself, so it is passed through rather than replaced.
    """
    try:
        check_single_statement(query)
    except ValueError as exc:
        raise ToolError(str(exc)) from None


def _failure(what: str, exc: Exception) -> ToolError:
    """A failure a model can act on, without the parts that carry row data.

    The four things that go wrong here read identically from the outside -- a syntax error, a
    statement that ran out of time, a pool with no connection to give, and a server that is not
    answering -- and a model told only that the query "failed" has nothing to correct. So the
    SQLSTATE and the server's primary message go back.

    ``DETAIL`` and ``CONTEXT`` do not. PostgreSQL puts row data in them: a uniqueness failure's
    detail reads ``Key (email)=(alice@example.com) already exists``. The driver leaves them out
    of a message by default, which is the behaviour wanted here, and they stay reachable on the
    exception for a log the operator owns.
    """
    logger.error("%s: %s", what, exc, exc_info=True)
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate:
        return ToolError(f"{what}: [{sqlstate}] {exc}")
    if isinstance(exc, PoolTimeout):
        return ToolError(
            f"{what}: no database connection was free within the pool's timeout. The server is "
            f"busy rather than wrong; the same query is worth trying again."
        )
    if isinstance(exc, psycopg.OperationalError):
        return ToolError(f"{what}: the database connection failed -- {exc}")
    return ToolError(f"{what}: {exc}")


def _relationship_scan_query(graphname: str) -> str:
    """SQL reporting the graph's (start label, type, end label) triples, by reading the edges.

    The planner's own triple catalog answers this without reading anything, and is what
    ``describe()`` uses. It is filled by a gather, and a transaction that wrote to the graph
    marks it out of date -- so this is the way to an answer when nobody has gathered since the
    last write.

    A ``graphid`` carries the label id of the row it identifies, so an edge already names both
    of its endpoint labels: ``start``, ``end`` and ``id`` off the edge tables give the triples
    without reading a vertex. The ids resolve to names by catalog join.
    """
    edge_table = sql.Identifier(graphname, "ag_edge").as_string(None)
    return f"""
        SELECT sl.labname AS label,
               el.labname AS relationship_type,
               tl.labname AS end_label
        FROM (
            SELECT DISTINCT graphid_labid(e.start) AS start_labid,
                            graphid_labid(e.id)    AS edge_labid,
                            graphid_labid(e."end") AS end_labid
            FROM {edge_table} e
        ) d
        JOIN pg_catalog.ag_graph g  ON g.graphname = %(graph)s
        JOIN pg_catalog.ag_label el ON el.graphid = g.oid AND el.labid = d.edge_labid
        JOIN pg_catalog.ag_label sl ON sl.graphid = g.oid AND sl.labid = d.start_labid
        JOIN pg_catalog.ag_label tl ON tl.graphid = g.oid AND tl.labid = d.end_labid
    """


def _unique_and_indexed(indexes: List[Any], constraints: List[Any]) -> tuple[dict, dict]:
    """Per label, which properties carry an index and which are asserted unique.

    Both are read, because they are kept in different places: a property index is an index, and
    a uniqueness assertion is an exclusion constraint that the property-index view filters out.
    Reading only one of them mislabels the other: a graph loaded through
    ``CREATE UNIQUE PROPERTY INDEX`` reported its key as unindexed, and one loaded through
    ``ASSERT ... IS UNIQUE`` would report the same key as having no constraint.
    """
    indexed: dict[str, set[str]] = {}
    unique: dict[str, set[str]] = {}
    for index in indexes:
        names = index_properties(index.definition) or ()
        indexed.setdefault(index.label, set()).update(names)
        if index.unique:
            unique.setdefault(index.label, set()).update(names)
    for constraint in constraints:
        for found in _ASSERTED_UNIQUE.findall(constraint.definition or ""):
            keys = {part.strip().strip('"') for part in found.split(",") if part.strip()}
            indexed.setdefault(constraint.label, set()).update(keys)
            if constraint.unique:
                unique.setdefault(constraint.label, set()).update(keys)
    return indexed, unique


def _shape_properties(
    shapes: tuple, indexed: set[str], unique: set[str]
) -> Dict[str, Any]:
    """A label's properties, each with its type and what is known about how it is stored."""
    return {
        shape.name: {
            "type": shape.kind,
            "declared": shape.declared,
            "indexed": shape.name in indexed,
            "unique": shape.name in unique,
        }
        for shape in shapes
    }


async def _describe_graph(
    pool: AsyncConnectionPool, graphname: str, sample: int, timeout: Optional[float]
) -> Dict[str, Any]:
    """What is in the graph, in the shape a prompt wants: one entry per vertex label.

    Counts, relationships, indexes and constraints are exact. Property types come from a
    bounded sample of each label, except where a property has a column of its own, which the
    catalog answers exactly.

    Nothing here scans the graph unless it has to. The labels and their counts come from the
    catalogs, the properties from ``sample`` rows per label, and the triples from the catalog
    the planner keeps -- falling back to a scan of the edges only when nothing has gathered
    that catalog since the last write to the graph.
    """
    async with get_pool_connection(pool) as conn:
        try:
            # Must precede any snapshot-taking statement in the transaction.
            await conn.execute("SET TRANSACTION READ ONLY")
            if timeout is not None:
                await conn.execute(
                    sql.SQL("SET LOCAL statement_timeout = {}").format(
                        sql.Literal(int(timeout * 1000))
                    )
                )
            description = await conn.describe(sample=sample, graph=graphname)
            indexes = await conn.indexes(graph=graphname)
            constraints = await conn.constraints(graph=graphname)
            triples = [
                (t.start, t.edge, t.end, t.edge_count) for t in description.triples
            ]
            if not description.meta_gathered:
                async with conn.cursor() as cur:
                    await cur.execute(
                        _relationship_scan_query(graphname), {"graph": graphname}
                    )
                    triples = [(*row, None) for row in await cur.fetchall()]
        finally:
            # A read-only transaction has nothing to commit, and what it *can* do is SET --
            # which on a pooled connection would be inherited by whoever borrows it next.
            await conn.rollback()

    indexed, unique = _unique_and_indexed(indexes, constraints)
    kinds = {label.name: label.kind for label in description.labels}
    builtin = {label.name for label in description.labels if label.is_builtin}

    # A label holding nothing is left out. It describes no data, and a graph that has been
    # written to for a while accumulates labels nothing was ever stored under -- which is
    # context spent on saying that there is nothing to say.
    schema: Dict[str, Any] = {
        label.name: {
            "type": "node",
            "count": description.counts.get(label.name, 0),
            "properties": _shape_properties(
                description.properties.get(label.name, ()),
                indexed.get(label.name, set()),
                unique.get(label.name, set()),
            ),
        }
        for label in description.labels
        if label.is_vertex
        and not label.is_builtin
        and description.counts.get(label.name, 0) > 0
    }

    for start, edge, end, edge_count in triples:
        if start in builtin or end in builtin or kinds.get(edge) != "e":
            continue
        entry = schema.setdefault(start, {"type": "node", "count": 0, "properties": {}})
        relationships = entry.setdefault("relationships", {})
        found = relationships.setdefault(
            edge,
            {
                "direction": "OUT",
                "labels": [],
                "properties": _shape_properties(
                    description.properties.get(edge, ()),
                    indexed.get(edge, set()),
                    unique.get(edge, set()),
                ),
            },
        )
        # One relationship type can reach more than one label; keep them all.
        if end not in found["labels"]:
            found["labels"].append(end)
        if edge_count is not None:
            found["count"] = found.get("count", 0) + edge_count
    return schema


async def _execute_write(
    pool: AsyncConnectionPool, graphname: str, query: str, params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run a write, read what it changed in the same transaction, and commit.

    The counters live on the connection and describe the last write on it, so the reading has
    to happen before anything else runs there -- which is what ``counts_`` does, inside the
    same transaction as the write.

    **Nothing raises once the commit has landed.** A failure after it is a failure to describe
    work the database has already kept, and reporting that as a failed write tells the caller
    the opposite of what happened: a ``CREATE`` was seen to persist while the tool said it had
    not run, and a model told that retries a write that already applied. So the rows changed
    are read first, the commit is last, and anything that goes wrong after it is logged and
    reported as a write that succeeded and could not be summarised.
    """
    set_path = sql.SQL("SET LOCAL graph_path = {}").format(sql.Identifier(graphname))
    async with get_pool_connection(pool) as conn:
        try:
            await conn.execute(set_path)
            # `None` rather than an empty mapping: psycopg reads `%` as a placeholder marker
            # whenever parameters are given, and a Cypher literal may hold one.
            result = await conn.execute_query(
                query, jsonb_params(params) or None, counts_=True
            )
            counts = result.counts
            stats: Dict[str, Any] = {
                "insertedvertices": counts.inserted_vertices,
                "insertededges": counts.inserted_edges,
                "deletedvertices": counts.deleted_vertices,
                "deletededges": counts.deleted_edges,
                "updatedproperties": counts.updated_properties,
            }
            rows = [
                {key: as_builtins(value) for key, value in zip(result.keys, record)}
                for record in result.records
            ]
        except BaseException:
            await conn.rollback()
            raise
        await conn.commit()
    try:
        stats["rows"] = [value_sanitize(row) for row in rows]
        stats["row_count"] = len(rows)
    except Exception as exc:  # pragma: no cover - shaping a value the server sent
        logger.error("Write committed; its result could not be shaped: %s", exc)
        stats["rows"] = []
        stats["note"] = "the write was committed; its returned rows could not be rendered"
    return stats


def _bounded(
    payload: Dict[str, Any], rows: List[Any], token_limit: Optional[int]
) -> Dict[str, Any]:
    """The response, holding as many whole rows as the budget allows.

    The rows are measured before the document is built, so what comes back is always JSON that
    parses. Cutting the finished document instead ends it inside a string or a brace, and a
    model handed that has been given nothing it can read -- it cannot even see which rows it
    received.
    """
    if not token_limit:
        payload["rows"] = rows
        return payload
    envelope = dict(payload, rows=[])
    kept, dropped = fit_rows(
        rows, token_limit, reserve=count_tokens(json.dumps(envelope, default=str))
    )
    payload["rows"] = kept
    payload["row_count"] = len(kept)
    if dropped:
        payload["rows_omitted"] = dropped
        payload["token_limit"] = token_limit
    return payload


async def _walk_result(
    pool: AsyncConnectionPool,
    graphname: str,
    query: str,
    params: Optional[Dict[str, Any]],
    timeout: Optional[float],
    keep: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Read a whole result once, keeping the first ``keep`` rows and counting all of them.

    A server-side cursor, so the rows stay on the server and arrive a chunk at a time. What
    this replaces is asking for the same result once per page: every page re-runs the query and
    throws away the rows before it, which costs the sum of the offsets. Measured over 200,000
    rows, walking in pages of 1,000 took 7.90 s and one streamed walk took 0.70 s.
    """
    # A server-side cursor holds the query where a subquery goes, and the grammar keeps three
    # clauses for the top of a statement -- so those are refused here, by name, rather than
    # producing a syntax error about a word the caller did place correctly.
    if (clause := unwrappable_clause(query)) is not None:
        raise ValueError(
            f"a whole-result walk reads the query as a subquery, and {clause} is a clause the "
            f"grammar accepts only at the top of a statement. Ask for a page instead, or end "
            f"the query with a RETURN that has no {clause} before it."
        )
    kept: List[Dict[str, Any]] = []
    total = 0
    async with get_pool_connection(pool) as conn:
        # A stream takes its cursor from the connection, so the columns are named there. Put
        # back afterwards: the connection goes on to serve somebody else's call.
        previous_factory = conn.row_factory
        conn.row_factory = namedtuple_row
        try:
            await conn.execute("SET TRANSACTION READ ONLY")
            if timeout is not None:
                await conn.execute(
                    sql.SQL("SET LOCAL statement_timeout = {}").format(
                        sql.Literal(int(timeout * 1000))
                    )
                )
            await conn.execute(
                sql.SQL("SET LOCAL graph_path = {}").format(sql.Identifier(graphname))
            )
            async for record in conn.stream(
                query, jsonb_params(params) or None, size=STREAM_CHUNK
            ):
                total += 1
                if len(kept) < keep:
                    kept.append(record_to_dict(record))
        finally:
            conn.row_factory = previous_factory
            await conn.rollback()
    return kept, total


def create_mcp_server(
    pool: AsyncConnectionPool,
    graphname: str,
    namespace: str = "",
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
    schema_sample: int = DEFAULT_SCHEMA_SAMPLE,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_page_size: int = MAX_PAGE_SIZE,
    gql_clauses: bool = False,
) -> FastMCP:
    """Create the FastMCP server with the schema / read / write / plan / health tools.

    ``gql_clauses`` says the server understands the GQL surface, which decides what the read
    and write tools tell a model they accept. 2.17 has none of those clauses, so advertising
    them there would be advertising syntax errors; :func:`main` asks the connection rather than
    assuming.
    """
    mcp = FastMCP("mcp-agensgraph-cypher")
    prefix = format_namespace(namespace)
    sample = max(1, int(schema_sample))
    default_page = max(1, min(int(page_size), int(max_page_size)))
    max_page = max(1, int(max_page_size))

    write_spellings = (
        "CREATE or its GQL spelling INSERT, MERGE, SET, REMOVE, DELETE"
        if gql_clauses
        else "CREATE, MERGE, SET, REMOVE, DELETE"
    )
    read_dialect = (
        "MATCH/RETURN, and the GQL reading clauses this server understands: LET, FILTER, "
        "NEXT, FINISH, FOR ... IN, OFFSET as a synonym for SKIP, RETURN ALL, OPTIONAL CALL "
        "and CALL ... YIELD"
        if gql_clauses
        else "MATCH/RETURN"
    )

    @mcp.tool(
        name=prefix + "get_agensgraph_schema",
        annotations=ToolAnnotations(
            title="Get AgensGraph Schema",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_agensgraph_schema() -> Dict[str, Any]:
        """List node labels, their properties, and the relationships between them.

        One entry per node label: its exact node count, its properties with the type each was
        found holding, and the relationship types leaving it with the labels they reach and the
        properties those relationships carry. A property says whether it carries an index and
        whether it is asserted unique.

        Property types are read from a bounded sample of each label rather than from every row,
        except where a property has a column of its own, which the catalog answers exactly.
        Counts, relationships and indexes are exact.
        """
        try:
            schema = await _describe_graph(
                pool, graphname, sample, float(read_timeout)
            )
            return schema
        except Exception as e:
            raise _failure("Schema read failed", e) from None

    @mcp.tool(
        name=prefix + "read_agensgraph_cypher",
        annotations=ToolAnnotations(
            title="Read AgensGraph Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_agensgraph_cypher(
        query: str = Field(
            ...,
            description=f"The Cypher query to execute. Reads only: {read_dialect}.",
        ),
        params: Optional[Dict[str, Any]] = Field(
            None, description="Parameters to pass to the Cypher query."
        ),
        limit: int = Field(
            default_page,
            ge=1,
            description=(
                f"Max rows to return in this page (default {default_page}); values above "
                f"{max_page} are clamped to {max_page}. The response echoes the effective limit."
            ),
        ),
        offset: int = Field(
            0,
            ge=0,
            description=(
                "Rows to skip -- use with `next_offset` to page through results. Skipping is "
                "not free: the rows before the page are produced and discarded, so the cost "
                "grows with the offset. Measured over 50,000 nodes: offset 0 took 0.062 s, "
                "10,000 took 1.19 s and 40,000 took 6.21 s. To reach rows deep in a result, "
                "narrow the query or order by a key and filter on the last one you saw; to "
                "read the whole result, ask for `walk` instead of paging."
            ),
        ),
        walk: bool = Field(
            False,
            description=(
                "Read the whole result in one pass instead of a page of it, and report the "
                "exact total row count. The rows come back a chunk at a time from a cursor on "
                "the server, so the result is never materialised in full here, and `limit` "
                "still bounds how many are returned. Paging through the same result costs the "
                "sum of the offsets: measured over 200,000 rows, pages of 1,000 took 7.90 s "
                "and one walk took 0.70 s."
            ),
        ),
    ) -> Dict[str, Any]:
        """Execute a read-only Cypher query and return one page of results.

        Runs in a read-only transaction, so the database rejects any write even if the query
        slips past the statement check. Results are paginated: at most `limit` rows are
        returned, and the response's `has_more` / `next_offset` tell you whether and how to
        fetch the next page. Returns a JSON object:
        `{"rows": [...], "row_count", "offset", "limit", "has_more", "next_offset"}`.

        A response is bounded by a token budget. When rows are left out for it, `rows_omitted`
        says how many -- whole rows are dropped, never part of one, so what comes back always
        parses.
        """
        _refuse_unless_one_statement(query)
        if writable_counters(query):
            raise ToolError(
                f"This tool only reads. Use the write tool for a statement that changes the "
                f"graph -- {write_spellings}."
            )
        page_limit = min(max(1, int(limit)), max_page)
        page_offset = max(0, int(offset))
        statement = quote_identifiers(query)
        try:
            if walk:
                rows, total = await _walk_result(
                    pool,
                    graphname,
                    statement,
                    params,
                    float(read_timeout),
                    page_limit,
                )
                payload: Dict[str, Any] = {
                    "row_count": len(rows),
                    "total_rows": total,
                    "offset": 0,
                    "limit": page_limit,
                    # The whole result was read, so what is left over is what `limit` and the
                    # token budget kept back. `next_offset` is empty because asking again with
                    # an offset is the thing this call exists to avoid.
                    "has_more": total > len(rows),
                    "next_offset": None,
                }
            else:
                rows, has_more = await run_paginated_query(
                    pool,
                    graphname,
                    statement,
                    params=params,
                    read_only=True,
                    timeout=float(read_timeout),
                    limit=page_limit,
                    offset=page_offset,
                )
                payload = {
                    "row_count": len(rows),
                    "offset": page_offset,
                    "limit": page_limit,
                    "has_more": has_more,
                    "next_offset": page_offset + page_limit if has_more else None,
                }
            return _bounded(payload, [value_sanitize(row) for row in rows], token_limit)
        except ValueError as e:
            raise ToolError(str(e)) from None
        except Exception as e:
            raise _failure("Read query failed", e) from None

    @mcp.tool(
        name=prefix + "explain_agensgraph_cypher",
        annotations=ToolAnnotations(
            title="Explain AgensGraph Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def explain_agensgraph_cypher(
        query: str = Field(..., description="The Cypher statement to plan."),
        analyze: bool = Field(
            False,
            description=(
                "Also run the statement and report actual timings. It runs in a read-only "
                "transaction, so a statement that would write is refused by the database."
            ),
        ),
    ) -> List[Dict[str, Any]]:
        """Show how AgensGraph would run a Cypher statement.

        Without `analyze` the statement is planned and not executed. With it the statement is
        executed to collect real timings, inside a read-only transaction either way -- so a
        statement that writes is refused by the server rather than by a reading of its text.
        Measured: `EXPLAIN (ANALYZE) INSERT ...` is refused with 25006 and leaves no row, and
        the Cypher writes are refused the same way.
        """
        _refuse_unless_one_statement(query)
        try:
            rows = await run_query(
                pool,
                graphname,
                explain_statement(quote_identifiers(query), analyze).as_string(),
                read_only=True,
                timeout=float(read_timeout),
            )
            return next(iter(rows[0].values())) if rows else []
        except Exception as e:
            raise _failure("Could not plan that statement", e) from None

    @mcp.tool(
        name=prefix + "recommend_property_indexes",
        annotations=ToolAnnotations(
            title="Recommend AgensGraph Property Indexes",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def recommend_property_indexes(
        query: str = Field(..., description="The Cypher query to advise on."),
        params: Optional[Dict[str, Any]] = Field(
            None,
            description=(
                "Parameters the query is run with. Give them: a list bound as a parameter "
                "plans as a jsonb containment test rather than as an index lookup, so advice "
                "about a query with parameters is advice about a different plan without them."
            ),
        ),
    ) -> Dict[str, Any]:
        """Suggest property indexes and rewrites for a query, from its plan.

        Reports the DDL to consider; it never runs it. Recommendations are reasoned from
        the plan and the label catalogs rather than costed against a built index, and the
        response says so.
        """
        _refuse_unless_one_statement(query)
        try:
            graph_param = {"graph": graphname}
            plan_rows = await run_query(
                pool,
                graphname,
                explain_statement(quote_identifiers(query), False).as_string(),
                params,
                read_only=True,
                timeout=float(read_timeout),
            )
            plan = next(iter(plan_rows[0].values()))
            labels = await run_query(
                pool, graphname, label_stats_query(), graph_param, read_only=True
            )
            indexes = await run_query(
                pool, graphname, existing_indexes_query(), graph_param, read_only=True
            )
            findings = analyze_plan(
                plan, relname_to_label(labels), indexed_properties(indexes)
            )
            payload = format_findings(findings)
            payload["existing_indexes"] = indexes
            return payload
        except Exception as e:
            raise _failure("Could not advise on that query", e) from None

    @mcp.tool(
        name=prefix + "agensgraph_health",
        annotations=ToolAnnotations(
            title="AgensGraph Health",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def agensgraph_health() -> Dict[str, Any]:
        """Report cache hit ratio, unused indexes, vacuum backlog and connection use.

        Each check stands on its own: one that needs an extension the server does not have
        reports that instead of failing the others.
        """
        report: Dict[str, Any] = {}
        try:
            extensions = await run_query(
                pool,
                graphname,
                extensions_query(),
                read_only=True,
            )
            report["extensions"] = {r["name"]: r["installed"] for r in extensions}
        except Exception:
            report["extensions"] = {}
        for name, statement in health_queries().items():
            try:
                report[name] = await run_query(
                    pool, graphname, statement, read_only=True
                )
            except Exception as e:
                report[name] = {"error": str(e)}
        for name in ("pgstattuple", "pg_buffercache"):
            if not report.get("extensions", {}).get(name):
                report[name] = missing_extension_note(name)
        return report

    @mcp.tool(
        name=prefix + "top_cypher_queries",
        annotations=ToolAnnotations(
            title="Top AgensGraph Cypher Queries",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def top_cypher_queries(
        limit: int = Field(20, ge=1, le=200, description="How many statements to return."),
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """The Cypher statements costing the most total time, from pg_stat_statements.

        Literals appear as parameters because Cypher is normalised the same way SQL is.
        """
        try:
            rows = await run_query(
                pool,
                graphname,
                top_cypher_queries_query(),
                {"limit": int(limit)},
                read_only=True,
            )
            return rows
        except Exception:
            return missing_extension_note("pg_stat_statements")

    # A read-only server does not register the write tool.
    if not read_only:

        @mcp.tool(
            name=prefix + "write_agensgraph_cypher",
            description=(
                "Execute a Cypher statement that changes the graph and return what it "
                "changed.\n\n"
                f"Takes one statement, and one that writes: {write_spellings}. The reply "
                "carries the five counters the statement is answerable for -- a counter this "
                "statement cannot be held to is reported as null rather than as nought -- and "
                "the rows the statement returned. A reply is a reply about work that was "
                "committed; a failure means nothing was."
            ),
            annotations=ToolAnnotations(
                title="Write AgensGraph Cypher",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )
        async def write_agensgraph_cypher(
            query: str = Field(..., description="The Cypher query to execute."),
            params: Dict[str, Any] = Field(
                default_factory=dict, description="Parameters to pass to the Cypher query."
            ),
        ) -> Dict[str, Any]:
            _refuse_unless_one_statement(query)
            if not writable_counters(query):
                raise ToolError(
                    "This tool is for a statement that changes the graph; use the read tool "
                    "for one that only reads."
                )
            try:
                stats = await _execute_write(pool, graphname, quote_identifiers(query), params)
            except Exception as e:
                raise _failure("Write query failed", e) from None
            return stats

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    graphname: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: Optional[str] = None,
    port: Optional[int] = None,
    path: Optional[str] = None,
    allow_origins: Optional[List[str]] = None,
    allowed_hosts: Optional[List[str]] = None,
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
) -> None:
    """Open the pool, bootstrap the graph, and serve over the chosen transport.

    Nothing is installed in the database. Asked what a JSON value holds, this used to create a
    plpgsql function in whatever database it was pointed at, on every start; the driver reads
    the same thing with the built-in ``jsonb_typeof`` and names the type in Python, and a full
    start leaves ``pg_proc`` and ``pg_class`` the size they were.
    """
    logger.info("Starting MCP AgensGraph Cypher Server")
    schema_sample = int(os.getenv("AGENSGRAPH_SCHEMA_SAMPLE", DEFAULT_SCHEMA_SAMPLE))
    page_size = int(os.getenv("AGENSGRAPH_PAGE_SIZE", DEFAULT_PAGE_SIZE))
    max_page_size = int(os.getenv("AGENSGRAPH_MAX_PAGE_SIZE", MAX_PAGE_SIZE))

    pool = create_pool(build_dsn(db_url, username, password, database))
    try:
        await pool.open()
        logger.info("Connection pool opened")
        await ensure_graph(pool, graphname)
        gql_clauses = await server_has_gql_clauses(pool)

        mcp = create_mcp_server(
            pool, graphname, namespace, read_timeout, token_limit, read_only,
            schema_sample, page_size, max_page_size, gql_clauses,
        )
        await run_server(
            mcp,
            transport=transport,
            host=host,
            port=port,
            path=path,
            allow_origins=allow_origins or [],
            allowed_hosts=allowed_hosts or [],
            server_name="AgensGraph Cypher MCP",
        )
    finally:
        await pool.close()
        logger.info("Connection pool closed")


async def server_has_gql_clauses(pool: AsyncConnectionPool) -> bool:
    """Whether this server understands the GQL clauses, from the version it announced.

    Asked once, at startup, because it decides what the tools tell a model they accept and a
    tool description is written when the tool is registered. The version arrives in the startup
    packet, so this costs no statement.
    """
    async with get_pool_connection(pool) as conn:
        found = bool(conn.capabilities.has_gql_clauses())
        await conn.rollback()
    return found
