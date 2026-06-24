import json
import logging
import os
from typing import List, Literal, Optional

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, ToolAnnotations
from psycopg import sql
from pydantic import Field

from mcp_agensgraph_common.connection import (
    build_dsn,
    create_pool,
    ensure_graph,
    get_pool_connection,
)
from mcp_agensgraph_common.config import format_namespace
from mcp_agensgraph_common.transport import run_server

from .agensgraph_memory import (
    AgensGraphMemory,
    Entity,
    ObservationAddition,
    ObservationDeletion,
    Relation,
)

# Set up logging
logger = logging.getLogger("mcp_agensgraph_memory")
logger.setLevel(logging.INFO)

# Default cap on entities returned by read_graph / search_memories, so a memory that
# has grown large can't flood the caller's context. Overridable via
# AGENSGRAPH_MEMORY_LIMIT; the response's `truncated` flag signals when it bit.
DEFAULT_MEMORY_LIMIT = 1000

jsonb_to_string = r"""
    CREATE OR REPLACE FUNCTION jsonb_to_string(j jsonb, sep text DEFAULT ', ')
    RETURNS text AS $$
    SELECT                           
    CASE                   
        WHEN jsonb_typeof(j) = 'array' THEN (
        SELECT string_agg(value::text, sep)
        FROM jsonb_array_elements_text(j)
        )                                                                                                                   
        WHEN jsonb_typeof(j) = 'object' THEN (
        SELECT string_agg(key || '=' || value, sep)
        FROM jsonb_each_text(j)
        )
        ELSE j::text
    END;
    $$ LANGUAGE sql IMMUTABLE
"""


def create_mcp_server(
    memory: AgensGraphMemory,
    namespace: str = "",
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
) -> FastMCP:
    """Create an MCP server instance for memory management."""

    namespace_prefix = format_namespace(namespace)
    mcp: FastMCP = FastMCP("mcp-agensgraph-memory")
    default_limit = max(1, int(memory_limit))

    @mcp.tool(
        name=namespace_prefix + "read_graph",
        annotations=ToolAnnotations(
            title="Read Graph",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def read_graph(
        limit: int = Field(
            default_limit,
            ge=1,
            description=(
                f"Max entities to return (default {default_limit}). If the memory has "
                "more, the response's `truncated` flag is set — narrow with "
                "search_memories. Relations reference entities by name, so the result "
                "stays coherent even when capped."
            ),
        ),
    ) -> ToolResult:
        """Read the knowledge graph (entities + relationships) from memory.

        Returns up to `limit` entities and the relationships touching them. Use this
        for an overview; for a large memory, prefer search_memories to narrow.

        Returns:
            KnowledgeGraph: { "entities": [...], "relations": [...], "truncated": bool }

        Example response:
        {
            "entities": [
                {"name": "John Smith", "type": "person", "observations": ["Works at SKAI Worldwide"]},
                {"name": "SKAI Worldwide Inc", "type": "company", "observations": ["Graph database company"]}
            ],
            "relations": [
                {"source": "John Smith", "target": "SKAI Worldwide Inc", "relationType": "WORKS_AT"}
            ],
            "truncated": false
        }
        """
        logger.info("MCP tool: read_graph")
        try:
            result = await memory.read_graph(limit=max(1, int(limit)))
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            logger.error(f"Error reading full knowledge graph: {e}")
            raise ToolError(f"Error reading full knowledge graph: {e}")

    @mcp.tool(
        name=namespace_prefix + "create_entities",
        annotations=ToolAnnotations(
            title="Create Entities",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def create_entities(
        entities: list[Entity] = Field(
            ...,
            description="List of entities to create with name, type, and observations",
        ),
    ) -> ToolResult:
        """Create multiple new entities in the knowledge graph.

        Creates new memory entities with their associated observations. If an entity with the same name
        already exists, this operation will merge the observations with existing ones.


        Returns:
            list[Entity]: The created entities with their final state

        Example call:
        {
            "entities": [
                {
                    "name": "Alice Johnson",
                    "type": "person",
                    "observations": ["Software engineer", "Lives in Seattle", "Enjoys hiking"]
                },
                {
                    "name": "Microsoft",
                    "type": "company",
                    "observations": ["Technology company", "Headquartered in Redmond, WA"]
                }
            ]
        }
        """
        logger.info(f"MCP tool: create_entities ({len(entities)} entities)")
        try:
            entity_objects = [Entity.model_validate(entity) for entity in entities]
            result = await memory.create_entities(entity_objects)
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text=json.dumps([e.model_dump() for e in result])
                    )
                ],
                structured_content={"result": result},
            )
        except Exception as e:
            logger.error(f"Error creating entities: {e}")
            raise ToolError(f"Error creating entities: {e}")

    @mcp.tool(
        name=namespace_prefix + "create_relations",
        annotations=ToolAnnotations(
            title="Create Relations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def create_relations(
        relations: list[Relation] = Field(
            ..., description="List of relations to create between existing entities"
        ),
    ) -> ToolResult:
        """Create multiple new relationships between existing entities in the knowledge graph.

        Creates directed relationships between entities that already exist. Both source and target
        entities must already be present in the graph. Use descriptive relationship types.

        Returns:
            list[Relation]: The created relationships

        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Microsoft",
                    "relationType": "WORKS_AT"
                },
                {
                    "source": "Alice Johnson",
                    "target": "Seattle",
                    "relationType": "LIVES_IN"
                }
            ]
        }
        """
        logger.info(f"MCP tool: create_relations ({len(relations)} relations)")
        try:
            relation_objects = [
                Relation.model_validate(relation) for relation in relations
            ]
            result = await memory.create_relations(relation_objects)
            return ToolResult(
                content=[
                    TextContent(
                        type="text", text=json.dumps([r.model_dump() for r in result])
                    )
                ],
                structured_content={"result": result},
            )
        except Exception as e:
            logger.error(f"Error creating relations: {e}")
            raise ToolError(f"Error creating relations: {e}")

    @mcp.tool(
        name=namespace_prefix + "add_observations",
        annotations=ToolAnnotations(
            title="Add Observations",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def add_observations(
        observations: list[ObservationAddition] = Field(
            ..., description="List of observations to add to existing entities"
        ),
    ) -> ToolResult:
        """Add new observations/facts to existing entities in the knowledge graph.

        Appends new observations to entities that already exist. The entity must be present
        in the graph before adding observations. Each observation should be a distinct fact.

        Returns:
            list[dict]: Details about the added observations including entity name and new facts

        Example call:
        {
            "observations": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Promoted to Senior Engineer", "Completed AWS certification"]
                },
                {
                    "entityName": "Microsoft",
                    "observations": ["Launched new AI products", "Stock price increased 15%"]
                }
            ]
        }
        """
        logger.info(f"MCP tool: add_observations ({len(observations)} additions)")
        try:
            observation_objects = [
                ObservationAddition.model_validate(obs) for obs in observations
            ]
            result = await memory.add_observations(observation_objects)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content={"result": result},
            )
        except Exception as e:
            logger.error(f"Error adding observations: {e}")
            raise ToolError(f"Error adding observations: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_entities",
        annotations=ToolAnnotations(
            title="Delete Entities",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_entities(
        entityNames: list[str] = Field(
            ..., description="List of exact entity names to delete permanently"
        ),
    ) -> ToolResult:
        """Delete entities and all their associated relationships from the knowledge graph.

        Permanently removes entities from the graph along with all relationships they participate in.
        This is a destructive operation that cannot be undone. Entity names must match exactly.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "entityNames": ["Old Company", "Outdated Person"]
        }

        Warning: This will delete the entities and ALL relationships they're involved in.
        """
        logger.info(f"MCP tool: delete_entities ({len(entityNames)} entities)")
        try:
            await memory.delete_entities(entityNames)
            return ToolResult(
                content=[
                    TextContent(type="text", text="Entities deleted successfully")
                ],
                structured_content={"result": "Entities deleted successfully"},
            )
        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            raise ToolError(f"Error deleting entities: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_observations",
        annotations=ToolAnnotations(
            title="Delete Observations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_observations(
        deletions: list[ObservationDeletion] = Field(
            ..., description="List of specific observations to remove from entities"
        ),
    ) -> ToolResult:
        """Delete specific observations from existing entities in the knowledge graph.

        Removes specific observation texts from entities. The observation text must match exactly
        what is stored. The entity will remain but the specified observations will be deleted.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "deletions": [
                {
                    "entityName": "Alice Johnson",
                    "observations": ["Old job title", "Outdated phone number"]
                },
                {
                    "entityName": "Microsoft",
                    "observations": ["Former CEO information"]
                }
            ]
        }

        Note: Observation text must match exactly (case-sensitive) to be deleted.
        """
        logger.info(f"MCP tool: delete_observations ({len(deletions)} deletions)")
        try:
            deletion_objects = [
                ObservationDeletion.model_validate(deletion) for deletion in deletions
            ]
            await memory.delete_observations(deletion_objects)
            return ToolResult(
                content=[
                    TextContent(type="text", text="Observations deleted successfully")
                ],
                structured_content={"result": "Observations deleted successfully"},
            )
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            raise ToolError(f"Error deleting observations: {e}")

    @mcp.tool(
        name=namespace_prefix + "delete_relations",
        annotations=ToolAnnotations(
            title="Delete Relations",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def delete_relations(
        relations: list[Relation] = Field(
            ..., description="List of specific relationships to delete from the graph"
        ),
    ) -> ToolResult:
        """Delete specific relationships between entities in the knowledge graph.

        Removes relationships while keeping the entities themselves. The source, target, and
        relationship type must match exactly for deletion. This only affects the relationships,
        not the entities they connect.

        Returns:
            str: Success confirmation message

        Example call:
        {
            "relations": [
                {
                    "source": "Alice Johnson",
                    "target": "Old Company",
                    "relationType": "WORKS_AT"
                },
                {
                    "source": "John Smith",
                    "target": "Former City",
                    "relationType": "LIVES_IN"
                }
            ]
        }

        Note: All fields (source, target, relationType) must match exactly for deletion.
        """
        logger.info(f"MCP tool: delete_relations ({len(relations)} relations)")
        try:
            relation_objects = [
                Relation.model_validate(relation) for relation in relations
            ]
            await memory.delete_relations(relation_objects)
            return ToolResult(
                content=[
                    TextContent(type="text", text="Relations deleted successfully")
                ],
                structured_content={"result": "Relations deleted successfully"},
            )
        except Exception as e:
            logger.error(f"Error deleting relations: {e}")
            raise ToolError(f"Error deleting relations: {e}")

    @mcp.tool(
        name=namespace_prefix + "search_memories",
        annotations=ToolAnnotations(
            title="Search Memories",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def search_memories(
        query: str = Field(
            ...,
            description="Search query to find entities by name, type, or observations",
        ),
        limit: int = Field(
            default_limit,
            ge=1,
            description=(
                f"Max matching entities to return (default {default_limit}); the "
                "response's `truncated` flag is set if there are more."
            ),
        ),
    ) -> ToolResult:
        """Search for entities in the knowledge graph using text search.

        Searches across entity names, types, and observations.
        Returns matching entities (up to `limit`) and their connections. Supports partial matches.

        Returns:
            KnowledgeGraph: { "entities": [...], "relations": [...], "truncated": bool }

        Example call:
        {
            "query": "engineer software"
        }

        This searches for entities containing "engineer" or "software" in their name, type, or observations.
        """
        logger.info(f"MCP tool: search_memories ('{query}')")
        try:
            result = await memory.search_memories(query, limit=max(1, int(limit)))
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise ToolError(f"Error searching memories: {e}")

    @mcp.tool(
        name=namespace_prefix + "find_memories_by_name",
        annotations=ToolAnnotations(
            title="Find Memories by Name",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def find_memories_by_name(
        names: list[str] = Field(
            ..., description="List of exact entity names to retrieve"
        ),
    ) -> ToolResult:
        """Find specific entities by their exact names.

        Retrieves entities that exactly match the provided names, along with all their
        relationships and connected entities. Use this when you know the exact entity names.

        Returns:
            KnowledgeGraph: Subgraph containing the specified entities and their relationships

        Example call:
        {
            "names": ["Alice Johnson", "Microsoft", "Seattle"]
        }

        This retrieves the entities with exactly those names plus their connections.
        """
        logger.info(f"MCP tool: find_memories_by_name ({len(names)} names)")
        try:
            result = await memory.find_memories_by_name(names)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            logger.error(f"Error finding memories by name: {e}")
            raise ToolError(f"Error finding memories by name: {e}")

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
) -> None:
    """Open the pool, bootstrap the graph + helpers, and serve over the chosen transport."""
    logger.info("Starting AgensGraph MCP Memory Server")

    pool = create_pool(build_dsn(db_url, username, password, database))
    try:
        await pool.open()
        logger.info("Connection pool opened")
        await ensure_graph(pool, graphname)

        # Create the jsonb_to_string helper used by fulltext search (idempotent).
        async with get_pool_connection(pool) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(jsonb_to_string)
            await conn.commit()

        memory = AgensGraphMemory(pool, graphname)
        await memory.create_fulltext_index()
        logger.info("AgensGraphMemory initialized")

        memory_limit = int(os.getenv("AGENSGRAPH_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT))
        mcp = create_mcp_server(memory, namespace, memory_limit)
        await run_server(
            mcp,
            transport=transport,
            host=host,
            port=port,
            path=path,
            allow_origins=allow_origins or [],
            allowed_hosts=allowed_hosts or [],
            server_name="AgensGraph Memory MCP",
        )
    finally:
        await pool.close()
        logger.info("Connection pool closed")
