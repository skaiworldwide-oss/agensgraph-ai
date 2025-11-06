import logging
from typing import Any, Dict, List

from psycopg import AsyncConnection
from psycopg.rows import namedtuple_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger("mcp_agensgraph_memory")
logger.setLevel(logging.INFO)


# Models for our knowledge graph
class Entity(BaseModel):
    """Represents a memory entity in the knowledge graph.

    Example:
    {
        "name": "John Smith",
        "type": "person",
        "observations": ["Works at SKAI Worldwide", "Lives in San Francisco", "Expert in graph databases"]
    }
    """

    name: str = Field(
        description="Unique identifier/name for the entity. Should be descriptive and specific.",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc", "San Francisco"],
    )
    type: str = Field(
        description="Category or classification of the entity. Common types: 'person', 'company', 'location', 'concept', 'event'",
        min_length=1,
        examples=["person", "company", "location", "concept", "event"],
    )
    observations: List[str] = Field(
        description="List of facts, observations, or notes about this entity. Each observation should be a complete, standalone fact.",
        examples=[
            ["Works at SKAI Worldwide", "Lives in San Francisco"],
            ["Headquartered in Sweden", "Graph database company"],
        ],
    )


class Relation(BaseModel):
    """Represents a relationship between two entities in the knowledge graph.

    Example:
    {
        "source": "John Smith",
        "target": "SKAI Worldwide Inc",
        "relationType": "WORKS_AT"
    }
    """

    source: str = Field(
        description="Name of the source entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    target: str = Field(
        description="Name of the target entity (must match an existing entity name exactly)",
        min_length=1,
        examples=["SKAI Worldwide Inc", "San Francisco"],
    )
    relationType: str = Field(
        description="Type of relationship between source and target. Use descriptive, uppercase names with underscores.",
        min_length=1,
        examples=["WORKS_AT", "LIVES_IN", "MANAGES", "COLLABORATES_WITH", "LOCATED_IN"],
    )


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph containing entities and their relationships."""

    entities: List[Entity] = Field(
        description="List of all entities in the knowledge graph", default=[]
    )
    relations: List[Relation] = Field(
        description="List of all relationships between entities", default=[]
    )


class ObservationAddition(BaseModel):
    """Request to add new observations to an existing entity.

    Example:
    {
        "entityName": "John Smith",
        "observations": ["Recently promoted to Senior Engineer", "Speaks fluent German"]
    }
    """

    entityName: str = Field(
        description="Exact name of the existing entity to add observations to",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    observations: List[str] = Field(
        description="New observations/facts to add to the entity. Each should be unique and informative.",
        min_length=1,
    )


class ObservationDeletion(BaseModel):
    """Request to delete specific observations from an existing entity.

    Example:
    {
        "entityName": "John Smith",
        "observations": ["Old job title", "Outdated contact info"]
    }
    """

    entityName: str = Field(
        description="Exact name of the existing entity to remove observations from",
        min_length=1,
        examples=["John Smith", "SKAI Worldwide Inc"],
    )
    observations: List[str] = Field(
        description="Exact observation texts to delete from the entity (must match existing observations exactly)",
        min_length=1,
    )


def get_pool_connection(pool: AsyncConnectionPool):
    """Context manager for getting a connection from the pool."""
    return pool.connection()


class AgensGraphMemory:
    def __init__(self, connection_pool: AsyncConnectionPool, graphname: str):
        self.pool = connection_pool
        self.graphname = graphname

    async def _execute_cypher(
        self, conn: AsyncConnection, cypher_query: str, params: dict = None
    ):
        """Execute a Cypher query within AgensGraph."""
        async with conn.cursor(row_factory=namedtuple_row) as cursor:
            # Set graph path
            await cursor.execute(f"SET graph_path = {self.graphname}")

            # Execute the Cypher query
            if params:
                await cursor.execute(cypher_query, params)
            else:
                await cursor.execute(cypher_query)

            # Try to fetch results
            try:
                results = await cursor.fetchall()
                return results
            except Exception:
                # Query might not return results (INSERT, DELETE, etc.)
                return []

    async def create_fulltext_index(self):
        """Create a fulltext search index for entities if it doesn't exist.

        Uses PostgreSQL's text search (tsvector) for fulltext search capabilities.
        Creates a GIN property index on the tsvector for efficient searching.
        Also ensures the Memory VLABEL exists.
        """
        try:
            async with get_pool_connection(self.pool) as conn:
                async with conn.cursor(row_factory=namedtuple_row) as cursor:
                    # Set graph path
                    await cursor.execute(f"SET graph_path = {self.graphname}")

                    # Ensure Memory VLABEL exists
                    await cursor.execute('CREATE VLABEL IF NOT EXISTS "Memory"')

                    # Create GIN property index on tsvector for fulltext search
                    # This indexes name, type, and all observations combined
                    await cursor.execute("""
                        CREATE PROPERTY INDEX IF NOT EXISTS memory_fulltext_idx
                        ON "Memory"
                        USING gin
                        (
                            (
                                setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
                                setweight(to_tsvector('english', coalesce(type, '')), 'B') ||
                                setweight(to_tsvector('english', coalesce(jsonb_to_string(observations, ' '), '')), 'C')
                            )
                        )
                    """)
                    await conn.commit()
            logger.info(
                "Created Memory VLABEL and fulltext search property index using tsvector"
            )
        except Exception as e:
            # Index might already exist, which is fine
            logger.debug(f"Fulltext property index creation: {e}")

    async def load_graph(self, filter_query: str = None):
        """Load the entire knowledge graph from AgensGraph."""
        logger.info("Loading knowledge graph from AgensGraph")

        async with get_pool_connection(self.pool) as conn:
            # Build the filter condition using PostgreSQL fulltext search
            if filter_query and filter_query != "*":
                # Use tsvector and tsquery for fulltext search
                # Searches across name, type, and observations with weights
                filter_condition = """
                    WHERE (
                        setweight(to_tsvector('english', coalesce(entity.name, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(entity.type, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(jsonb_to_string(entity.observations, ' '), '')), 'C')
                    ) @@ plainto_tsquery('english', %(query)s)
                """
                params = {"query": filter_query}
            else:
                filter_condition = ""
                params = {}

            # Query to get all matching entities
            entity_query = f"""
                MATCH (entity:"Memory")
                {filter_condition}
                RETURN entity.name AS name, entity.type AS type, entity.observations AS observations
            """
            entities_data = await self._execute_cypher(conn, entity_query, params)

            entities = []
            entity_names = []
            for record in entities_data:
                entities.append(
                    Entity(
                        name=record.name,
                        type=record.type,
                        observations=record.observations or [],
                    )
                )
                entity_names.append(record.name)

            # Query to get all relationships for these entities
            relations = []
            if entity_names:
                rel_query = """
                    MATCH (source:"Memory")-[r]->(target:"Memory")
                    WHERE source.name <@ %(names)s OR target.name <@ %(names)s
                    RETURN source.name AS source, target.name AS target, label(r) AS "relationType"
                """

                relations_data = await self._execute_cypher(
                    conn, rel_query, {"names": Jsonb(entity_names)}
                )

                for record in relations_data:
                    relations.append(
                        Relation(
                            source=record.source,
                            target=record.target,
                            relationType=record.relationType,
                        )
                    )

            await conn.commit()

            logger.debug(f"Loaded entities: {entities}")
            logger.debug(f"Loaded relations: {relations}")

            return KnowledgeGraph(entities=entities, relations=relations)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        """Create multiple new entities in the knowledge graph."""
        logger.info(f"Creating {len(entities)} entities")

        async with get_pool_connection(self.pool) as conn:
            for entity in entities:
                # Create/update the entity
                # Note: We store the type as a property, not as a separate label
                query = """
                    MERGE (e:"Memory" {name: %(name)s})
                    SET e.type = %(type)s, e.observations = %(observations)s
                """

                await self._execute_cypher(
                    conn,
                    query,
                    {
                        "name": Jsonb(entity.name),
                        "type": Jsonb(entity.type),
                        "observations": Jsonb(entity.observations),
                    },
                )

            await conn.commit()

        return entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """Create multiple new relations between entities."""
        logger.info(f"Creating {len(relations)} relations")

        async with get_pool_connection(self.pool) as conn:
            for relation in relations:
                # create the relationship
                query = f"""
                    MATCH (fromNode:"Memory"), (toNode:"Memory")
                    WHERE fromNode.name = %(source)s AND toNode.name = %(target)s
                    MERGE (fromNode)-[r:"{relation.relationType}"]->(toNode)
                """

                await self._execute_cypher(
                    conn,
                    query,
                    {
                        "source": Jsonb(relation.source),
                        "target": Jsonb(relation.target),
                    },
                )

            await conn.commit()

        return relations

    async def add_observations(
        self, observations: List[ObservationAddition]
    ) -> List[Dict[str, Any]]:
        """Add new observations to existing entities."""
        logger.info(f"Adding observations to {len(observations)} entities")

        results = []
        async with get_pool_connection(self.pool) as conn:
            for obs in observations:
                # Get existing observations
                get_query = """
                    MATCH (e:"Memory" {name: %(name)s})
                    RETURN e.observations AS observations
                """

                existing_data = await self._execute_cypher(
                    conn, get_query, {"name": Jsonb(obs.entityName)}
                )

                if existing_data:
                    existing_obs = existing_data[0].observations or []
                    # Filter out observations that already exist
                    new_obs = [o for o in obs.observations if o not in existing_obs]

                    if new_obs:
                        # Update with new observations using Cypher list concatenation
                        update_query = """
                            MATCH (e:"Memory" {name: %(name)s})
                            SET e.observations = coalesce(e.observations, []) + %(new_obs)s
                        """

                        await self._execute_cypher(
                            conn,
                            update_query,
                            {"name": Jsonb(obs.entityName), "new_obs": Jsonb(new_obs)},
                        )

                        results.append(
                            {"entityName": obs.entityName, "addedObservations": new_obs}
                        )
                    else:
                        results.append(
                            {"entityName": obs.entityName, "addedObservations": []}
                        )

            await conn.commit()

        return results

    async def delete_entities(self, entity_names: List[str]) -> None:
        """Delete multiple entities and their associated relations."""
        logger.info(f"Deleting {len(entity_names)} entities")

        async with get_pool_connection(self.pool) as conn:
            for name in entity_names:
                query = """
                    MATCH (e:"Memory" {name: %(name)s})
                    DETACH DELETE e
                """

                await self._execute_cypher(conn, query, {"name": Jsonb(name)})

            await conn.commit()

        logger.info(f"Successfully deleted {len(entity_names)} entities")

    async def delete_observations(self, deletions: List[ObservationDeletion]) -> None:
        """Delete specific observations from entities."""
        logger.info(f"Deleting observations from {len(deletions)} entities")

        async with get_pool_connection(self.pool) as conn:
            for deletion in deletions:
                # Get existing observations
                get_query = """
                    MATCH (e:"Memory" {name: %(name)s})
                    RETURN e.observations AS observations
                """

                existing_data = await self._execute_cypher(
                    conn, get_query, {"name": Jsonb(deletion.entityName)}
                )

                if existing_data:
                    existing_obs = existing_data[0].observations or []
                    # Filter out observations to delete
                    remaining_obs = [
                        o for o in existing_obs if o not in deletion.observations
                    ]

                    # Update with remaining observations (no casting needed in Cypher)
                    update_query = """
                        MATCH (e:"Memory" {name: %(name)s})
                        SET e.observations = %(remaining_obs)s
                    """

                    await self._execute_cypher(
                        conn,
                        update_query,
                        {
                            "name": Jsonb(deletion.entityName),
                            "remaining_obs": Jsonb(remaining_obs),
                        },
                    )

            await conn.commit()

        logger.info(f"Successfully deleted observations from {len(deletions)} entities")

    async def delete_relations(self, relations: List[Relation]) -> None:
        """Delete multiple relations from the graph."""
        logger.info(f"Deleting {len(relations)} relations")

        async with get_pool_connection(self.pool) as conn:
            for relation in relations:
                query = f"""
                    MATCH (source:"Memory")-[r:"{relation.relationType}"]->(target:"Memory")
                    WHERE source.name = %(source)s AND target.name = %(target)s
                    DELETE r
                """

                await self._execute_cypher(
                    conn,
                    query,
                    {
                        "source": Jsonb(relation.source),
                        "target": Jsonb(relation.target),
                    },
                )

            await conn.commit()

        logger.info(f"Successfully deleted {len(relations)} relations")

    async def read_graph(self) -> KnowledgeGraph:
        """Read the entire knowledge graph."""
        return await self.load_graph()

    async def search_memories(self, query: str) -> KnowledgeGraph:
        """Search for memories based on a query."""
        logger.info(f"Searching for memories with query: '{query}'")
        return await self.load_graph(query)

    async def find_memories_by_name(self, names: List[str]) -> KnowledgeGraph:
        """Find specific memories by their names."""
        logger.info(f"Finding {len(names)} memories by name")

        async with get_pool_connection(self.pool) as conn:
            # Get entities
            entity_query = """
                MATCH (e:"Memory")
                WHERE e.name <@ %(names)s
                RETURN e.name AS name, e.type AS type, e.observations AS observations
            """

            entities_data = await self._execute_cypher(
                conn, entity_query, {"names": Jsonb(names)}
            )

            entities = []
            for record in entities_data:
                entities.append(
                    Entity(
                        name=record.name,
                        type=record.type,
                        observations=record.observations or [],
                    )
                )

            # Get relations for found entities
            relations = []
            if entities:
                rel_query = """
                    MATCH (source:"Memory")-[r]->(target:"Memory")
                    WHERE source.name <@ %(names)s OR target.name <@ %(names)s
                    RETURN source.name AS source, target.name AS target, label(r) AS relationType
                """

                relations_data = await self._execute_cypher(
                    conn, rel_query, {"names": Jsonb(names)}
                )

                for record in relations_data:
                    relations.append(
                        Relation(
                            source=record.source,
                            target=record.target,
                            relationType=record.relationType,
                        )
                    )

            await conn.commit()

        logger.info(f"Found {len(entities)} entities and {len(relations)} relations")
        return KnowledgeGraph(entities=entities, relations=relations)
