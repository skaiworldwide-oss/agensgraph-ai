"""AgensGraph MCP Server - Main server implementation."""

import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Pattern
from urllib.parse import urlparse

import psycopg  # type: ignore
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool, PoolTimeout  # type: ignore
from pydantic import Field
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .utils import _quote_identifiers, _truncate_string_to_tokens, _value_sanitize

logger = logging.getLogger("mcp_agensgraph_cypher")

# Regex patterns for parsing AgensGraph vertex and edge formats
VERTEX_REGEX: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
EDGE_REGEX: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

# Write query detection pattern
WRITE_QUERY_PATTERN = re.compile(
    r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", re.IGNORECASE
)

# SQL function to check if a property has unique constraint
SQL_PROPERTY_CONSTRAINT_FUNCTION = """
CREATE OR REPLACE FUNCTION property_has_unique_constraint(key_name TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    found BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint r
        JOIN pg_catalog.ag_label l ON r.conrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        WHERE g.graphname = current_setting('graph_path')
        AND r.contype IN ('c', 'x')
        AND pg_catalog.ag_get_graphconstraintdef(r.oid) ILIKE '%(' || key_name || ') IS UNIQUE%'
    ) INTO found;

    RETURN found;
END;
$$ LANGUAGE plpgsql;
"""

# SQL function to determine JSON type
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

def _format_namespace(namespace: str) -> str:
    """Format namespace with trailing dash if not empty."""
    if not namespace:
        return ""
    return namespace if namespace.endswith("-") else f"{namespace}-"


def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return WRITE_QUERY_PATTERN.search(query) is not None


@asynccontextmanager
async def get_pool_connection(
    pool: AsyncConnectionPool, timeout: Optional[float] = None
):
    """
    Get a connection from the pool with workaround for psycopg_pool bug.

    This context manager ensures proper connection lifecycle:
    - Acquires connection from pool
    - Yields connection for use
    - Returns connection to pool on exit
    """
    try:
        connection = await pool.getconn(timeout=timeout)
    except PoolTimeout:
        # Workaround for psycopg_pool bug
        await pool._add_connection(None)
        connection = await pool.getconn(timeout=timeout)

    try:
        async with connection:
            yield connection
    finally:
        await pool.putconn(connection)


def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
    """
    Convert an AgensGraph query record to a dictionary.

    Parses vertex and edge formats from AgensGraph:
    - Vertex: label[id]{properties}
    - Edge: label[id][start_id, end_id]{properties}

    Args:
        record: A named tuple from an AgensGraph query result

    Returns:
        Dictionary with field names as keys and parsed values
    """
    result = {}
    vertices = {}

    # First pass: Build vertex mapping for edge construction
    for field_name in record._fields:
        value = getattr(record, field_name)
        if isinstance(value, str):
            vertex_match = VERTEX_REGEX.match(value)
            if vertex_match:
                label, vertex_id, properties = vertex_match.groups()
                vertices[str(vertex_id)] = json.loads(properties)

    # Second pass: Parse all fields
    for field_name in record._fields:
        value = getattr(record, field_name)

        if isinstance(value, str):
            vertex_match = VERTEX_REGEX.match(value)
            edge_match = EDGE_REGEX.match(value)

            if vertex_match:
                result[field_name] = json.loads(vertex_match.group(3))
            elif edge_match:
                label, edge_id, start_id, end_id, properties = edge_match.groups()
                result[field_name] = (
                    vertices.get(start_id, {}),
                    label,
                    vertices.get(end_id, {}),
                )
            else:
                result[field_name] = value
        else:
            result[field_name] = value

    return result


def _transform_schema_format(records: List[Dict], counts: List[Dict]) -> Dict:
    """
    Transform raw schema data into structured dictionary format.

    Args:
        records: List of schema records with label, attributes, and relationships
        counts: List of count records with label and count

    Returns:
        Dictionary with capitalized labels as keys and schema info as values
    """
    schema = {}
    count_map = {c["label"]: c["count"] for c in counts}

    for record in records:
        label = record["label"]
        label_key = label.capitalize()

        # Transform properties
        properties = {}
        for prop_name, prop_type_str in record["attributes"].items():
            indexed = "unique indexed" in prop_type_str
            clean_type = prop_type_str.replace(" unique indexed", "")

            properties[prop_name] = {"type": clean_type, "indexed": indexed}

        # Transform relationships
        relationships = {}
        for rel_type, target_label in record.get("relationships", {}).items():
            if rel_type and target_label:
                relationships[rel_type.upper()] = {
                    "direction": "OUT",
                    "labels": [target_label.capitalize()],
                }

        schema[label_key] = {
            "type": "node",
            "count": count_map.get(label, 0),
            "properties": properties,
        }

        if relationships:
            schema[label_key]["relationships"] = relationships

    return schema

async def _execute_query(
    pool: AsyncConnectionPool,
    graph_name: str,
    query: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query on AgensGraph and return results.

    Args:
        pool: Connection pool (should already be opened)
        graph_name: Name of the graph to query
        query: Cypher query to execute
        params: Optional query parameters
        timeout: Optional query timeout in seconds

    Returns:
        List of dictionaries representing query results

    Raises:
        psycopg.Error: On database errors
    """
    async with get_pool_connection(pool) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            try:
                # Set statement timeout if specified
                if timeout is not None:
                    timeout_ms = int(timeout * 1000)
                    await cursor.execute(f"SET statement_timeout = {timeout_ms}")
                else:
                    await cursor.execute("SET statement_timeout = 0")

                # Set graph path and execute query
                await cursor.execute(f"SET graph_path = {graph_name}")

                if params:
                    await cursor.execute(query, params)
                else:
                    await cursor.execute(query)

                await conn.commit()

            except psycopg.Error as e:
                await conn.rollback()
                logger.error(f"Database error executing query: {e}\n{query}\n{params}")
                raise

            # Fetch results
            try:
                data = await cursor.fetchall()
            except psycopg.ProgrammingError:
                # Query doesn't return data (e.g., CREATE, SET)
                data = []

            # Convert to dictionaries
            if not data:
                return []
            return [_record_to_dict(record) for record in data]


async def _ensure_helper_functions(pool: AsyncConnectionPool, graph_name: str) -> None:
    """
    Ensure helper SQL functions exist in the database.

    Creates property_has_unique_constraint and typeof functions if needed.
    These are idempotent operations (CREATE OR REPLACE).

    Args:
        pool: Connection pool
        graph_name: Name of the graph
    """
    await _execute_query(pool, graph_name, SQL_PROPERTY_CONSTRAINT_FUNCTION)
    await _execute_query(pool, graph_name, SQL_TYPEOF_FUNCTION)

def create_mcp_server(
    pool: AsyncConnectionPool,
    graphname: str,
    namespace: str = "",
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
) -> FastMCP:
    """
    Create and configure the FastMCP server with AgensGraph tools.

    Args:
        pool: AsyncConnectionPool for database connections
        graphname: Name of the AgensGraph graph to use
        namespace: Optional namespace prefix for tool names
        read_timeout: Timeout in seconds for read queries
        token_limit: Optional limit on response tokens
        read_only: If True, disable write operations

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP("mcp-agensgraph-cypher")
    namespace_prefix = _format_namespace(namespace)
    allow_writes = not read_only

    @mcp.tool(
        name=namespace_prefix + "get_agensgraph_schema",
        annotations=ToolAnnotations(
            title="Get AgensGraph Schema",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_agensgraph_schema() -> list[ToolResult]:
        """
        List all nodes, their attributes and relationships in the AgensGraph database.

        Returns detailed schema information including:
        - Node labels and counts
        - Properties with types and indexing info
        - Relationships between nodes
        """
        schema_query = """
            SELECT label, attributes, relationships - 'non' AS relationships FROM (
                MATCH (start_node)
                WITH start_node,
                    keys(start_node) AS keys,
                    properties(start_node) AS props
                OPTIONAL MATCH (start_node)-[r]->(end_node)
                WITH DISTINCT
                    label(start_node) as label,
                    type(r) AS relationship_type,
                    label(end_node) AS end_label,
                    keys, props

                UNWIND keys AS key

                RETURN
                    label,
                    jsonb_object_agg(
                        key::text,
                        CASE
                            WHEN property_has_unique_constraint(key) THEN typeof(props[key]) || ' unique indexed'
                            ELSE typeof(props[key])
                        END
                    ) AS attributes,
                    jsonb_object_agg(
                        CASE
                            WHEN relationship_type IS NOT NULL THEN relationship_type::text
                            ELSE 'non'
                        END,
                        end_label
                    ) AS relationships
            )t
        """

        count_query = """
            MATCH (n)
            RETURN label(n) AS label, count(n) AS count
        """

        try:
            # Ensure helper functions exist
            await _ensure_helper_functions(pool, graphname)

            # Execute schema and count queries
            records = await _execute_query(pool, graphname, schema_query)
            counts = await _execute_query(pool, graphname, count_query)

            # Transform to structured format
            schema = _transform_schema_format(records, counts)
            results_json = json.dumps(schema, default=str)

            logger.debug(f"Schema query returned {len(schema)} labels")
            return ToolResult(content=[TextContent(type="text", text=results_json)])

        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            raise ToolError(f"Unexpected Error: {e}")

    @mcp.tool(
        name=namespace_prefix + "read_agensgraph_cypher",
        annotations=ToolAnnotations(
            title="Read AgensGraph Cypher",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_agensgraph_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[Dict[str, Any]] = Field(
            None, description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """
        Execute a read-only Cypher query on the AgensGraph database.

        Only MATCH queries are allowed. The query is subject to timeout restrictions.
        """
        if _is_write_query(query):
            raise ValueError("Only MATCH queries are allowed for read operations")

        # Quote identifiers to preserve case sensitivity
        query = _quote_identifiers(query)

        try:
            results = await _execute_query(
                pool, graphname, query, params=params, timeout=float(read_timeout)
            )

            # Sanitize and format results
            sanitized_results = [_value_sanitize(el) for el in results]
            results_json = json.dumps(sanitized_results, default=str)

            # Apply token limit if specified
            if token_limit:
                results_json = _truncate_string_to_tokens(results_json, token_limit)

            logger.debug(f"Read query returned {len(results)} rows")
            return ToolResult(content=[TextContent(type="text", text=results_json)])

        except Exception as e:
            logger.error(f"Error executing read query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    @mcp.tool(
        name=namespace_prefix + "write_agensgraph_cypher",
        annotations=ToolAnnotations(
            title="Write AgensGraph Cypher",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        enabled=allow_writes,
    )
    async def write_agensgraph_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Dict[str, Any] = Field(
            dict(), description="The parameters to pass to the Cypher query."
        ),
    ) -> list[ToolResult]:
        """
        Execute a write Cypher query on the AgensGraph database.

        Allows CREATE, MERGE, SET, DELETE, REMOVE operations.
        Returns statistics about the changes made.
        """
        if not _is_write_query(query):
            raise ValueError("Only write queries are allowed for write operations")

        # Quote identifiers to preserve case sensitivity
        query = _quote_identifiers(query)

        try:
            # Execute the write query
            await _execute_query(pool, graphname, query, params)

            # Get write statistics
            stats = await _execute_query(
                pool, graphname, "SELECT * FROM get_last_graph_write_stats() AS counters"
            )

            stats_json = json.dumps(stats[0], default=str)
            logger.debug(f"Write query stats: {stats_json}")

            return ToolResult(content=[TextContent(type="text", text=stats_json)])

        except Exception as e:
            logger.error(f"Error executing write query: {e}\n{query}\n{params}")
            raise ToolError(f"Error: {e}\n{query}\n{params}")

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    graphname: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    allow_origins: List[str] = [],
    allowed_hosts: List[str] = [],
    read_timeout: int = 30,
    token_limit: Optional[int] = None,
    read_only: bool = False,
) -> None:
    """
    Main entry point for the AgensGraph MCP server.

    Args:
        db_url: Database connection URL
        username: Database username
        password: Database password
        database: Database name
        graphname: Graph name to use
        transport: Transport protocol (stdio, sse, or http)
        namespace: Optional tool namespace prefix
        host: Server host for HTTP/SSE transport
        port: Server port for HTTP/SSE transport
        path: Server path for HTTP/SSE transport
        allow_origins: Allowed origins for CORS
        allowed_hosts: Allowed hosts for DNS rebinding protection
        read_timeout: Timeout for read queries in seconds
        token_limit: Optional token limit for responses
        read_only: If True, disable write operations
    """
    logger.info("Starting MCP AgensGraph Server")

    # Parse database URL and create connection string
    parsed = urlparse(db_url)
    connection_string = (
        f"postgresql://{username}:{password}@{parsed.hostname}:{parsed.port}/{database}"
    )

    # Initialize connection pool
    pool = AsyncConnectionPool(connection_string, open=False)

    try:
        await pool.open()
        logger.info("Database connection pool opened successfully")

        # Ensure graph exists
        async with get_pool_connection(pool) as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cursor:
                try:
                    await cursor.execute(f"CREATE GRAPH IF NOT EXISTS {graphname}")
                    await conn.commit()
                    logger.info(f"Graph '{graphname}' ensured to exist")
                except psycopg.Error as e:
                    logger.error(f"Database error creating graph: {e}")
                    sys.exit(1)

        # Configure middleware for HTTP/SSE transports
        custom_middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            ),
            Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts),
        ]

        # Create MCP server
        mcp = create_mcp_server(
            pool, graphname, namespace, read_timeout, token_limit, read_only
        )

        # Run server with specified transport
        match transport:
            case "http":
                logger.info(
                    f"Running AgensGraph Cypher MCP Server with HTTP transport on {host}:{port}{path}"
                )
                await mcp.run_http_async(
                    host=host,
                    port=port,
                    path=path,
                    middleware=custom_middleware,
                    stateless_http=True,
                )
            case "stdio":
                logger.info("Running AgensGraph Cypher MCP Server with stdio transport")
                await mcp.run_stdio_async()
            case "sse":
                logger.info(
                    f"Running AgensGraph Cypher MCP Server with SSE transport on {host}:{port}{path}"
                )
                await mcp.run_http_async(
                    host=host,
                    port=port,
                    path=path,
                    middleware=custom_middleware,
                    transport="sse",
                )
            case _:
                error_msg = f"Invalid transport: {transport} | Must be 'stdio', 'sse', or 'http'"
                logger.error(error_msg)
                raise ValueError(error_msg)

    finally:
        # Ensure pool is properly closed
        if pool:
            await pool.close()
            logger.info("Database connection pool closed")


if __name__ == "__main__":
    main()
