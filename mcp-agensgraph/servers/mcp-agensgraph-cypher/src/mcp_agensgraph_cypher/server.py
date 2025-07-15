import json
import logging
import re
import sys
from typing import Any, Literal, Optional, List, Dict, Pattern, NamedTuple
import psycopg  # type: ignore
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool, PoolTimeout  # type: ignore
from contextlib import asynccontextmanager

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import Field

property_has_unique_constraint_function = """
    CREATE OR REPLACE FUNCTION property_has_unique_constraint(
        key_name TEXT
    ) RETURNS BOOLEAN AS $$
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

typeof = r"""
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

logger = logging.getLogger("mcp_agensgraph_cypher")

vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

def _format_namespace(namespace: str) -> str:
    if namespace:
        if namespace.endswith("-"):
            return namespace
        else:
            return namespace + "-"
    else:
        return ""
    
async def _query(
    agensgraph_driver,
    graph_name,
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a provided query on the Agensgraph database and return the results.

    Parameters:
    -----------

        - query (str): The Cypher query to be executed against the database.
        - params (Optional[Dict[str, Any]]): Optional parameters to be used in the query.
            (default None)

    Returns:
    --------

        - List[Dict[str, Any]]: A list of dictionaries representing the result set of the
            query.
    """
    await agensgraph_driver.open()

    # execute the query, rolling back on an error
    async with get_pool_connection(agensgraph_driver) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as curs:
            try:
                await curs.execute(f'SET graph_path = "%s"', {graph_name})
                await curs.execute(query)
                await conn.commit()
            except psycopg.Error as e:
                await conn.rollback()
                logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            try:
                data = await curs.fetchall()
            except psycopg.ProgrammingError:
                data = []  # Handle queries that don’t return data
            if data is None:
                result = []
            # decode records
            else:
                result = [_record_to_dict(d) for d in data]

            return result

def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
    """
    Convert a record returned from an agensgraph query to a dictionary

    Args:
        record (): a record from an agensgraph query result

    Returns:
        Dict[str, Any]: a dictionary representation of the record where
            the dictionary key is the field name and the value is the
            value converted to a python type
    """
    # result holder
    d = {}

    # prebuild a mapping of vertex_id to vertex mappings to be used
    # later to build edges
    vertices = {}
    for k in record._fields:
        v = getattr(record, k)

        # records comes back label[id]{properties} which must be parsed
        if isinstance(v, str):
            vertex = vertex_regex.match(v)
            if vertex:
                label, vertex_id, properties = vertex.groups()
                properties = json.loads(properties)
                vertices[str(vertex_id)] = properties

    # iterate returned fields and parse appropriately
    for k in record._fields:
        v = getattr(record, k)

        if isinstance(v, str):
            vertex = vertex_regex.match(v)
            edge = edge_regex.match(v)

            if vertex:
                d[k] = json.loads(vertex.group(3))
            elif edge:
                elabel, edge_id, start_id, end_id, properties = edge.groups()
                d[k] = (
                    vertices.get(start_id, {}),
                    elabel,
                    vertices.get(end_id, {}),
                )
            else:
                d[k] = v

        else:
            d[k] = v

    return d

def _is_write_query(query: str) -> bool:
    """Check if the query is a write query."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )


def create_mcp_server(agensgraph_driver: AsyncConnectionPool, graphname: str, namespace: str = "", host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp: FastMCP = FastMCP("mcp-agensgraph-cypher", dependencies=["psycopg", "psycopg-pool", "psycopg[binary,pool]", "pydantic"], host=host, port=port)

    async def get_agensgraph_schema():
        """
        List all node, their attributes and their relationships
        to other nodes in the agensgraph graph.
        """

        get_schema_query = """
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

        try:
            await _query(agensgraph_driver, graphname, property_has_unique_constraint_function)
            await _query(agensgraph_driver, graphname, typeof)

            records = await _query(agensgraph_driver, graphname, get_schema_query)
            results_json_str = json.dumps(records, default=str)

            logger.debug(f"Read query returned {len(results_json_str)} rows")

            return types.CallToolResult(content=[types.TextContent(type="text", text=results_json_str)])

        except Exception as e:
            logger.error(f"Database error retrieving schema: {e}")
            return types.CallToolResult(
                isError=True, 
                content=[types.TextContent(type="text", text=f"Error: {e}")]
            )

    async def read_agensgraph_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(
            None, description="The parameters to pass to the Cypher query."
        ),
    ):
        """Execute a read Cypher query on the agensgraph database."""

        try:
            if _is_write_query(query):
                raise ValueError("Only MATCH queries are allowed for read-query")
        
            records = await _query(agensgraph_driver, graphname, query, params)
            results_json_str = json.dumps(records, default=str)

            logger.debug(f"Read query returned {len(results_json_str)} rows")
            
            return types.CallToolResult(content=[types.TextContent(type="text", text=results_json_str)])

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            return types.CallToolResult(
                isError=True, 
                content=[
                types.TextContent(type="text", text=f"Error: {e}\n{query}\n{params}")
            ]
            )

    async def write_agensgraph_cypher(
        query: str = Field(..., description="The Cypher query to execute."),
        params: Optional[dict[str, Any]] = Field(
            None, description="The parameters to pass to the Cypher query."
        ),
    ):
        """Execute a write Cypher query on the agensgraph database."""

        try:
            if not _is_write_query(query):
                raise ValueError("Only write queries are allowed for write-query")
            
            await _query(agensgraph_driver, graphname, query, params)
            counters = await _query(
                agensgraph_driver,
                "test",
                "SELECT * FROM get_last_graph_write_stats() AS counters"
            )
            counters_json_str = json.dumps(counters[0], default=str)

            logger.debug(f"Write query affected {counters_json_str}")

            return types.CallToolResult(content=[types.TextContent(type="text", text=counters_json_str)])

        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}\n{params}")
            return types.CallToolResult(
                isError=True, 
                content=[
                types.TextContent(type="text", text=f"Error: {e}\n{query}\n{params}")
            ]
            )

    namespace_prefix = _format_namespace(namespace)
    
    mcp.add_tool(get_agensgraph_schema, name=namespace_prefix+"get_agensgraph_schema")
    mcp.add_tool(read_agensgraph_cypher, name=namespace_prefix+"read_agensgraph_cypher")
    mcp.add_tool(write_agensgraph_cypher, name=namespace_prefix+"write_agensgraph_cypher")

    return mcp


async def main(
    db_name: str,
    username: str,
    password: str,
    graphname: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    transport: Literal["stdio", "sse"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    
    if not db_name or not username or not password or not graphname:
        raise ValueError("Environment variables AGENSGRAPH_DB, AGENSGRAPH_USERNAME, AGENSGRAPH_PASSWORD, and AGENSGRAPH_GRAPH_NAME must be set.")

    logger.info("Starting MCP agensgraph Server")

    db_url = f"postgresql://{username}:{password}@{db_host}:{db_port}/{db_name}"
    agensgraph_driver = AsyncConnectionPool(db_url, open=False)
    await agensgraph_driver.open()

    # execute the query, rolling back on an error
    async with get_pool_connection(agensgraph_driver) as conn:
        async with conn.cursor(row_factory=namedtuple_row) as curs:
            try:
                await curs.execute(f'CREATE GRAPH IF NOT EXISTS "%s"', {graphname})
                await curs.execute(f'SET graph_path = "%s"', {graphname})
                await conn.commit()
            except psycopg.Error as e:
                logger.error(f"Database error creating graph: {e}")
                sys.exit(1)

    mcp = create_mcp_server(agensgraph_driver, graphname, namespace, host, port)

    match transport:
        case "stdio":
            logger.info("Running AgensGraph Cypher MCP Server with stdio transport...")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(f"Running AgensGraph Cypher MCP Server with SSE transport on {host}:{port}...")
            await mcp.run_sse_async()
        case _:
            logger.error(f"Invalid transport: {transport} | Must be either 'stdio' or 'sse'")
            raise ValueError(f"Invalid transport: {transport} | Must be either 'stdio' or 'sse'")

@asynccontextmanager
async def get_pool_connection(agensgraph_driver: AsyncConnectionPool, timeout: Optional[float] = None):
    """Workaround for a psycopg_pool bug"""
    try:
        connection = await agensgraph_driver.getconn(timeout=timeout)
    except PoolTimeout:
        await agensgraph_driver._add_connection(None)  # workaround...
        connection = await agensgraph_driver.getconn(timeout=timeout)

    try:
        async with connection:
            yield connection
    finally:
        await agensgraph_driver.putconn(connection)

if __name__ == "__main__":
    main()
