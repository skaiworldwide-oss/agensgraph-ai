import json
import logging
import os
from typing import List, Literal, Optional

from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent, ToolAnnotations
from pydantic import Field

from mcp_agensgraph_common.config import format_namespace
from mcp_agensgraph_common.connection import build_dsn, check_role_cannot_run_programs
from mcp_agensgraph_common.transport import run_server

from .agensgraph_memory import (
    BY_NAME,
    BY_RECENCY,
    MAX_LIMIT,
    AgensGraphMemory,
    Entity,
    ObservationAddition,
    ObservationDeletion,
    Relation,
)
from .bootstrap import bootstrap, ensure_graph, make_pool

# Set up logging
logger = logging.getLogger("mcp_agensgraph_memory")
logger.setLevel(logging.INFO)

# Default cap on entities returned by read_graph / search_memories, so a memory that
# has grown large can't flood the caller's context. Overridable via
# AGENSGRAPH_MEMORY_LIMIT up to MAX_LIMIT; the response's `truncated` flag signals when it bit.
DEFAULT_MEMORY_LIMIT = 1000


def memory_limit_from_env(default: int = DEFAULT_MEMORY_LIMIT) -> int:
    """The configured page size, bounded, and unaffected by a value that is not a number."""
    given = os.getenv("AGENSGRAPH_MEMORY_LIMIT")
    if given is None:
        return default
    try:
        return max(1, min(int(given), MAX_LIMIT))
    except ValueError:
        logger.warning("AGENSGRAPH_MEMORY_LIMIT is not a number; using %d", default)
        return default


def tool_error(doing: str, exc: Exception) -> ToolError:
    """What a caller is told when something failed, and what the log is told.

    A database failure's own message is not passed on. PostgreSQL puts row data in a failure's
    DETAIL -- a uniqueness failure names the value, a type failure quotes the parameter -- and
    a tool result goes to a model and from there wherever the conversation goes. The SQLSTATE
    is passed on, because it says what kind of failure it was and carries nothing else.

    The log gets everything, including the DETAIL the driver keeps off the message.
    """
    if isinstance(exc, ValueError):
        # This server's own reading of the request. Saying what was wrong with it is how the
        # caller writes a call that works.
        logger.info("Refused a %s request: %s", doing, exc)
        return ToolError(f"{doing}: {exc}")
    detail = getattr(getattr(exc, "diag", None), "message_detail", None)
    logger.error("Error while %s: %s%s", doing, exc, f" DETAIL: {detail}" if detail else "")
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate:
        return ToolError(f"{doing} failed on the database (SQLSTATE {sqlstate}).")
    return ToolError(f"{doing} failed. The server log has the detail.")


def create_mcp_server(
    memory: AgensGraphMemory,
    namespace: str = "",
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
) -> FastMCP:
    """Create an MCP server instance for memory management."""

    namespace_prefix = format_namespace(namespace)
    mcp: FastMCP = FastMCP("mcp-agensgraph-memory")
    default_limit = max(1, min(int(memory_limit), MAX_LIMIT))

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
            le=MAX_LIMIT,
            description=(
                f"Max entities to return (default {default_limit}, most {MAX_LIMIT}). If "
                "the memory has more, the response's `truncated` flag is set — narrow with "
                "search_memories."
            ),
        ),
        order: str = Field(
            BY_NAME,
            description=(
                f"'{BY_NAME}' returns the page in name order, '{BY_RECENCY}' returns what was "
                f"written most recently first. Name is the default because it is the order the "
                f"memory is indexed in; asking for recency sorts instead."
            ),
        ),
    ) -> ToolResult:
        """Read a page of the knowledge graph (entities + relationships) from memory.

        Returns `limit` entities in the order asked for, and the relationships whose **both** ends
        are among them — so no relationship names an entity that is not in the response. Use
        this for an overview; for a large memory, prefer search_memories to narrow.

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
            result = await memory.read_graph(limit=limit, order=order)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            raise tool_error("reading the knowledge graph", e) from e

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
        """Create entities in the knowledge graph, merging into any that already exist.

        An entity whose name is already in the memory keeps its observations and gains the ones
        given; its type is set to the one given. Calling this twice with the same entity leaves
        one entity holding both sets of observations.

        Returns:
            list[Entity]: the entities as they now stand, read back from the memory

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
            raise tool_error("creating entities", e) from e

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
        """Create directed relationships between entities that are already in the memory.

        A relationship whose source or target is not in the memory is not written and comes
        back under `skipped`; create the entities first. The type is stored upper-cased, so
        `works_at` and `WORKS_AT` name the same relationship.

        Returns:
            dict: { "created": [Relation, ...], "skipped": [Relation, ...] }

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
            payload = {
                key: [r.model_dump() for r in value] for key, value in result.items()
            }
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(payload))],
                structured_content={"result": payload},
            )
        except Exception as e:
            raise tool_error("creating relations", e) from e

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
        """Add observations to entities that are already in the knowledge graph.

        An observation the entity already holds is not stored again, and an entity that is not
        in the memory comes back with `found` false — create it first. Each observation should
        be a distinct, standalone fact.

        Returns:
            list[dict]: per entity, { "entityName", "addedObservations", "found" }

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
            raise tool_error("adding observations", e) from e

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
        """Delete entities and all their relationships from the knowledge graph.

        Permanently removes entities along with every relationship they take part in. Entity
        names must match exactly; a name that is not in the memory comes back under `notFound`
        rather than being reported as a deletion.

        Returns:
            dict: { "deleted": [...], "notFound": [...], "deletedRelations": int }

        Example call:
        {
            "entityNames": ["Old Company", "Outdated Person"]
        }

        Warning: This will delete the entities and ALL relationships they're involved in.
        """
        logger.info(f"MCP tool: delete_entities ({len(entityNames)} entities)")
        try:
            result = await memory.delete_entities(entityNames)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content={"result": result},
            )
        except Exception as e:
            raise tool_error("deleting entities", e) from e

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
        """Delete specific observations from entities in the knowledge graph.

        The observation text must match exactly what is stored. The entity stays; only the
        observations named are removed, and the ones actually removed come back in the result.

        Returns:
            list[dict]: per entity, { "entityName", "deletedObservations", "found" }

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
            result = await memory.delete_observations(deletion_objects)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content={"result": result},
            )
        except Exception as e:
            raise tool_error("deleting observations", e) from e

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

        Removes relationships while keeping the entities. Source and target must match
        exactly; the type is matched upper-cased, so the case it is written in does not
        matter. The result says how many relationships were actually removed.

        Returns:
            dict: { "requested": int, "deletedRelations": int }

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
        """
        logger.info(f"MCP tool: delete_relations ({len(relations)} relations)")
        try:
            relation_objects = [
                Relation.model_validate(relation) for relation in relations
            ]
            result = await memory.delete_relations(relation_objects)
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(result))],
                structured_content={"result": result},
            )
        except Exception as e:
            raise tool_error("deleting relations", e) from e

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
            description=(
                "Words to search for across entity names, types and observations. Every word "
                "must match; the words are stemmed, so 'engineers' matches 'engineering'. "
                "'*' asks for everything rather than for a word."
            ),
        ),
        limit: int = Field(
            default_limit,
            ge=1,
            le=MAX_LIMIT,
            description=(
                f"Max matching entities to return (default {default_limit}, most "
                f"{MAX_LIMIT}); the response's `truncated` flag is set if there are more."
            ),
        ),
        order: str = Field(
            BY_NAME,
            description=(
                f"'{BY_NAME}' returns the matches in name order, '{BY_RECENCY}' returns what "
                f"was written most recently first."
            ),
        ),
    ) -> ToolResult:
        """Search the knowledge graph by words in an entity's name, type or observations.

        **Every** word given must match — the words are joined with AND, not OR. They are
        stemmed rather than matched as prefixes, so "engineers" finds "engineering" but "eng"
        finds neither, and a query of only stop words ("the", "of") matches nothing because
        the English dictionary drops them. A query of `"*"` asks for everything, which is
        read_graph by another name.

        Returns the matching entities, up to `limit`, and the relationships whose both ends
        are among them.

        Returns:
            KnowledgeGraph: { "entities": [...], "relations": [...], "truncated": bool }

        Example call:
        {
            "query": "engineer software"
        }

        This finds entities whose name, type or observations contain both "engineer" and
        "software".
        """
        logger.info("MCP tool: search_memories")
        try:
            result = await memory.search_memories(query, limit=limit, order=order)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            raise tool_error("searching memories", e) from e

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
        limit: int = Field(
            default_limit,
            ge=1,
            le=MAX_LIMIT,
            description=f"Max names to look up (default {default_limit}, most {MAX_LIMIT}).",
        ),
    ) -> ToolResult:
        """Find entities by their exact names, with what they are connected to.

        Returns the named entities, every relationship touching them in either direction, and
        the entities at the other end of those relationships — so nothing in the result points
        at something the result does not contain. Use this when you know the exact names.

        Returns:
            KnowledgeGraph: the named entities plus their neighbours and relationships

        Example call:
        {
            "names": ["Alice Johnson", "Microsoft", "Seattle"]
        }
        """
        logger.info(f"MCP tool: find_memories_by_name ({len(names)} names)")
        try:
            result = await memory.find_memories_by_name(names, limit=limit)
            return ToolResult(
                content=[TextContent(type="text", text=result.model_dump_json())],
                structured_content=result,
            )
        except Exception as e:
            raise tool_error("finding memories by name", e) from e

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
    allow_server_programs: bool = False,
    adopt_existing_graph: bool = False,
) -> None:
    """Open the pool, make what the graph needs, and serve over the chosen transport."""
    logger.info("Starting AgensGraph MCP Memory Server")

    dsn = build_dsn(db_url, username, password, database)
    await ensure_graph(dsn, graphname)
    pool = make_pool(dsn, graphname)
    try:
        await pool.open()
        # Before anything is served: a role that can run a command on the server's host makes
        # every read tool below a claim this server cannot keep.
        await check_role_cannot_run_programs(
            pool, allow_server_programs=allow_server_programs
        )
        await pool.wait()
        logger.info("Connection pool opened")
        # Not best-effort. Without the labels and the unique index, every write makes
        # duplicates and every search answers from an index that is not there, and a server
        # that swallowed the failure would report itself healthy while doing both.
        await bootstrap(pool, graphname, adopt=adopt_existing_graph)

        memory_limit = memory_limit_from_env()
        memory = AgensGraphMemory(pool, graphname, max_limit=MAX_LIMIT)
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


__all__ = [
    "AgensGraphMemory",
    "create_mcp_server",
    "main",
    "memory_limit_from_env",
    "tool_error",
]
