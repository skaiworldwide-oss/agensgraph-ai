"""AgensGraph Cypher MCP server.

Three tools — schema introspection plus read and write Cypher — built on the
shared ``mcp_agensgraph_common`` core (connection pool, identifier-quoted graph
bootstrap, read-only-transaction enforcement, vertex/edge result parsing, transport).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

import psycopg
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations
from psycopg import sql
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool
from pydantic import Field

from mcp_agensgraph_common.results import record_to_dict, truncate_to_tokens, value_sanitize
from mcp_agensgraph_common.config import format_namespace
from mcp_agensgraph_common.connection import (
    build_dsn,
    create_pool,
    ensure_graph,
    get_pool_connection,
    jsonb_params,
    run_paginated_query,
    run_query,
)
from mcp_agensgraph_common.safety import is_write_query, quote_identifiers, quote_label
from mcp_agensgraph_common.transport import run_server

logger = logging.getLogger("mcp_agensgraph_cypher")

# Default cap on nodes sampled *per label* when introspecting the schema (bounds cost on
# large graphs); overridable via AGENSGRAPH_SCHEMA_SAMPLE.
DEFAULT_SCHEMA_SAMPLE = 1000

# Read-result pagination: default page size and hard ceiling on rows per call, so an
# unbounded query can't flood the agent's context or the server's memory. Overridable
# via AGENSGRAPH_PAGE_SIZE / AGENSGRAPH_MAX_PAGE_SIZE.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

# Helper SQL function used by the schema query (created once at startup).
SQL_TYPEOF_FUNCTION = r"""
CREATE OR REPLACE FUNCTION typeof(element jsonb)
RETURNS text AS $$
DECLARE
    elem_type text;
BEGIN
    elem_type := jsonb_typeof(element);
    IF elem_type = 'number' THEN
        IF element::text ~ '^\d+$' THEN
            RETURN 'INTEGER';
        ELSIF element::text ~ '^\d+\.\d+$' THEN
            RETURN 'FLOAT';
        ELSE
            RETURN 'NUMBER';
        END IF;
    ELSE
        CASE UPPER(elem_type)
            WHEN 'OBJECT' THEN RETURN 'MAP';
            WHEN 'ARRAY' THEN RETURN 'LIST';
            ELSE RETURN UPPER(elem_type);
        END CASE;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


def _vertex_labels_query() -> str:
    """SQL listing the graph's vertex labels.

    ``ag_vertex`` is the inheritance parent every vertex label hangs off, not a label of
    its own.
    """
    return """
        SELECT l.labname AS labname
        FROM pg_catalog.ag_label l
        JOIN pg_catalog.ag_graph g ON g.oid = l.graphid
        WHERE g.graphname = %(graph)s
          AND l.labkind = 'v'
          AND l.labname <> 'ag_vertex'
        ORDER BY l.labname
    """


def _attributes_query(labels: List[str], sample: int) -> str:
    """Cypher reporting each label's property names and types, sampling per label.

    One query part per label, each carrying its own ``LIMIT``, so the sample bounds every
    label.
    """
    parts = []
    for label in labels:
        try:
            quoted = quote_label(label)
        except ValueError:
            # A label the Cypher quoting cannot express: describe the rest of the graph
            # rather than fail the whole tool.
            logger.warning("Skipping label in schema introspection: %r", label)
            continue
        parts.append(
            f"MATCH (s:{quoted})\n"
            "WITH s, keys(s) AS keys, properties(s) AS props\n"
            f"LIMIT {int(sample)}\n"
            "UNWIND keys AS key\n"
            "RETURN label(s) AS label,\n"
            "       jsonb_object_agg(key::text, typeof(props[key])) AS attributes"
        )
    return "\nUNION ALL\n".join(parts)


def _node_counts_query(graphname: str) -> str:
    """SQL counting nodes per label, exactly.

    A ``graphid`` carries the label id of the row it identifies, so grouping on it counts
    per label without reading a property.
    """
    vertex_table = sql.Identifier(graphname, "ag_vertex").as_string(None)
    return f"""
        SELECT l.labname AS label,
               c.count    AS count
        FROM (
            SELECT graphid_labid(v.id) AS labid, count(*) AS count
            FROM {vertex_table} v
            GROUP BY 1
        ) c
        JOIN pg_catalog.ag_graph g  ON g.graphname = %(graph)s
        JOIN pg_catalog.ag_label l  ON l.graphid = g.oid AND l.labid = c.labid
    """


def _relationships_query(graphname: str) -> str:
    """SQL reporting the graph's (start label, type, end label) triples.

    A ``graphid`` carries the label id of the row it identifies, so an edge already names
    both of its endpoint labels: ``start``, ``end`` and ``id`` off the edge tables give the
    triples without reading a vertex. The ids resolve to names by catalog join.
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


# Every property key the graph declares UNIQUE, read once per introspection. A constraint
# names the key it covers only inside its rendered definition.
_UNIQUE_PROPERTY_KEYS_QUERY = r"""
    SELECT DISTINCT m[1] AS key_name
    FROM pg_catalog.pg_constraint r
    JOIN pg_catalog.ag_label l ON r.conrelid = l.relid
    JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
    CROSS JOIN LATERAL regexp_matches(
        pg_catalog.ag_get_graphconstraintdef(r.oid),
        '\(([^()]+)\)\s+IS\s+UNIQUE', 'gi'
    ) AS m
    WHERE g.graphname = %(graph)s
      AND r.contype IN ('c', 'x')
"""


def _transform_schema_format(
    attributes: List[Dict],
    relationships: List[Dict],
    counts: List[Dict],
    unique_keys: set[str],
) -> Dict:
    """Shape the raw schema rows into ``{Label: {type, count, properties, relationships}}``."""
    schema: Dict[str, Any] = {}
    count_map = {c["label"]: c["count"] for c in counts}

    rel_map: Dict[str, Dict[str, Any]] = {}
    for row in relationships:
        label = row["label"]
        rel_type = row["relationship_type"]
        end_label = row["end_label"]
        if not (label and rel_type and end_label):
            continue
        entry = rel_map.setdefault(label, {}).setdefault(
            rel_type.upper(), {"direction": "OUT", "labels": []}
        )
        target = end_label.capitalize()
        # One relationship type can reach more than one label; keep them all.
        if target not in entry["labels"]:
            entry["labels"].append(target)

    for record in attributes:
        label = record["label"]
        label_key = label.capitalize()

        properties = {}
        for prop_name, prop_type in (record["attributes"] or {}).items():
            properties[prop_name] = {
                "type": prop_type,
                "indexed": prop_name in unique_keys,
            }

        schema[label_key] = {
            "type": "node",
            "count": count_map.get(label, 0),
            "properties": properties,
        }
        if label in rel_map:
            schema[label_key]["relationships"] = rel_map[label]

    return schema


async def _ensure_helper_functions(pool: AsyncConnectionPool, graphname: str) -> None:
    """Create the schema-introspection helper function once (idempotent)."""
    await run_query(pool, graphname, SQL_TYPEOF_FUNCTION)


async def _execute_write(
    pool: AsyncConnectionPool, graphname: str, query: str, params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run a write query and read its stats in the SAME transaction.

    ``get_last_graph_write_stats()`` reports the most recent write on the current
    connection, so the write and the stats read must share one connection/tx. If
    the stats function is unavailable, fall back to a generic success result.
    """
    set_path = sql.SQL("SET LOCAL graph_path = {}").format(sql.Identifier(graphname))
    async with get_pool_connection(pool) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cur:
            try:
                await cur.execute(set_path)
                bound = jsonb_params(params)
                if bound:
                    await cur.execute(query, bound)
                else:
                    await cur.execute(query)
                try:
                    await cur.execute(
                        "SELECT * FROM get_last_graph_write_stats() AS counters"
                    )
                    stats_rows = await cur.fetchall()
                    stats = record_to_dict(stats_rows[0]) if stats_rows else {}
                except psycopg.Error:
                    stats = {"status": "success"}
                await conn.commit()
            except psycopg.Error:
                await conn.rollback()
                raise
    return stats


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
) -> FastMCP:
    """Create the FastMCP server with the schema / read / write tools."""
    mcp = FastMCP("mcp-agensgraph-cypher")
    prefix = format_namespace(namespace)
    sample = max(1, int(schema_sample))
    default_page = max(1, min(int(page_size), int(max_page_size)))
    max_page = max(1, int(max_page_size))

    count_query = _node_counts_query(graphname)

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
    async def get_agensgraph_schema() -> list[ToolResult]:
        """List node labels, their properties (with types/indexing), and relationships.

        Properties are inferred from a sample of nodes per label; node counts, the
        relationships, and which properties are unique are exact.
        """
        try:
            graph_param = {"graph": graphname}
            labels = [
                row["labname"]
                for row in await run_query(
                    pool, graphname, _vertex_labels_query(), graph_param, read_only=True
                )
            ]
            attributes_query = _attributes_query(labels, sample)
            attributes = (
                await run_query(pool, graphname, attributes_query, read_only=True)
                if attributes_query
                else []
            )
            relationships = await run_query(
                pool, graphname, _relationships_query(graphname), graph_param, read_only=True
            )
            counts = await run_query(
                pool, graphname, count_query, graph_param, read_only=True
            )
            unique_keys = {
                row["key_name"]
                for row in await run_query(
                    pool, graphname, _UNIQUE_PROPERTY_KEYS_QUERY, graph_param, read_only=True
                )
            }
            schema = _transform_schema_format(
                attributes, relationships, counts, unique_keys
            )
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(schema, default=str))]
            )
        except Exception as e:
            logger.error("Error retrieving schema: %s", e)
            raise ToolError("Failed to retrieve schema. See server logs for details.")

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
        query: str = Field(..., description="The Cypher query to execute (read-only: MATCH/RETURN)."),
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
            0, ge=0, description="Rows to skip — use with `next_offset` to page through results."
        ),
    ) -> list[ToolResult]:
        """Execute a read-only Cypher query and return one page of results.

        Runs in a read-only transaction, so the database rejects any write even if the
        query slips past the keyword check. Results are paginated: at most `limit` rows
        are returned, and the response's `has_more` / `next_offset` tell you whether and
        how to fetch the next page. Returns a JSON object:
        `{"rows": [...], "row_count", "offset", "limit", "has_more", "next_offset"}`.
        """
        if is_write_query(query):
            raise ToolError("Only read (MATCH/RETURN) queries are allowed by this tool.")
        page_limit = min(max(1, int(limit)), max_page)
        page_offset = max(0, int(offset))
        try:
            rows, has_more = await run_paginated_query(
                pool,
                graphname,
                quote_identifiers(query),
                params=params,
                read_only=True,
                timeout=float(read_timeout),
                limit=page_limit,
                offset=page_offset,
            )
            sanitized = [value_sanitize(el) for el in rows]
            payload = {
                "rows": sanitized,
                "row_count": len(sanitized),
                "offset": page_offset,
                "limit": page_limit,
                "has_more": has_more,
                "next_offset": page_offset + page_limit if has_more else None,
            }
            results_json = json.dumps(payload, default=str)
            if token_limit:
                results_json = truncate_to_tokens(results_json, token_limit)
            return ToolResult(content=[TextContent(type="text", text=results_json)])
        except Exception as e:
            logger.error("Error executing read query: %s\n%s\nparams=%s", e, query, params)
            raise ToolError("Read query failed. See server logs for details.")

    @mcp.tool(
        name=prefix + "write_agensgraph_cypher",
        annotations=ToolAnnotations(
            title="Write AgensGraph Cypher",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        enabled=not read_only,
    )
    async def write_agensgraph_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Dict[str, Any] = Field(
            default_factory=dict, description="Parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """Execute a write Cypher query (CREATE/MERGE/SET/DELETE/REMOVE) and return stats."""
        if not is_write_query(query):
            raise ToolError("This tool is for write queries; use the read tool for MATCH/RETURN.")
        try:
            stats = await _execute_write(pool, graphname, quote_identifiers(query), params)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(stats, default=str))]
            )
        except Exception as e:
            logger.error("Error executing write query: %s\n%s\nparams=%s", e, query, params)
            raise ToolError("Write query failed. See server logs for details.")

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
    """Open the pool, bootstrap the graph + helpers, and serve over the chosen transport."""
    logger.info("Starting MCP AgensGraph Cypher Server")
    schema_sample = int(os.getenv("AGENSGRAPH_SCHEMA_SAMPLE", DEFAULT_SCHEMA_SAMPLE))
    page_size = int(os.getenv("AGENSGRAPH_PAGE_SIZE", DEFAULT_PAGE_SIZE))
    max_page_size = int(os.getenv("AGENSGRAPH_MAX_PAGE_SIZE", MAX_PAGE_SIZE))

    pool = create_pool(build_dsn(db_url, username, password, database))
    try:
        await pool.open()
        logger.info("Connection pool opened")
        await ensure_graph(pool, graphname)
        await _ensure_helper_functions(pool, graphname)

        mcp = create_mcp_server(
            pool, graphname, namespace, read_timeout, token_limit, read_only,
            schema_sample, page_size, max_page_size,
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
