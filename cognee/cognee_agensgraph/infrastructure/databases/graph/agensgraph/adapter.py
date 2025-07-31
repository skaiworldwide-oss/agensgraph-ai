"""Agensgraph Adapter for Graph Database"""

import json, re
from cognee.shared.logging_utils import get_logger, ERROR
import asyncio
from textwrap import dedent
from typing import Optional, Any, List, Dict, Type, Tuple, Union, NamedTuple, Pattern
from contextlib import asynccontextmanager
from uuid import UUID
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.modules.storage.utils import JSONEncoder
from cognee.infrastructure.databases.exceptions.exceptions import NodesetFilterNotSupportedError
from .metrics import *

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout

logger = get_logger("AgensgraphAdapter", level=ERROR)
BASE_LABEL = "__Node__"

# Since we do not support multiple labels, we will maintain the extra labels as a list
# This function will be used in queries to append new labels to the existing list
# and ensure that the labels are unique
append_label_function = """
    CREATE OR REPLACE FUNCTION append_label(labels jsonb, new_label text) 
    RETURNS jsonb AS $$
    BEGIN
        IF labels IS NULL OR jsonb_typeof(labels) <> 'array' THEN
            labels := '[]'::jsonb;
        END IF;

        IF NOT labels @> to_jsonb(new_label) THEN
            RETURN labels || jsonb_build_array(new_label);
        ELSE
            RETURN labels;
        END IF;
    END;
    $$ LANGUAGE plpgsql;

"""

get_labels_function = """
    CREATE OR REPLACE FUNCTION get_labels(entity vertex)
    RETURNS jsonb AS $$
    DECLARE
        existing_labels jsonb;
        entity_label text;
    BEGIN
        existing_labels := entity.properties -> 'labels';
        IF existing_labels IS NULL OR jsonb_typeof(existing_labels) <> 'array' THEN
            existing_labels := '[]'::jsonb;
        END IF;

        entity_label := trim(both '"' from label(entity)::text);

        IF entity_label <> 'ag_vertex' AND NOT existing_labels @> to_jsonb(entity_label) THEN
            existing_labels := existing_labels || to_jsonb(entity_label);
        END IF;

        RETURN existing_labels;
    END;
    $$ LANGUAGE plpgsql;

"""

label_catalog = """
    CREATE TABLE IF NOT EXISTS label_catalog (
        graph_id oid PRIMARY KEY,
        labels jsonb DEFAULT '[]'::jsonb
    );

"""

track_labels = """
    CREATE OR REPLACE FUNCTION track_labels()
    RETURNS TRIGGER AS $$
    DECLARE
        graphid OID := {}::oid;
        new_labels JSONB;
    BEGIN
        INSERT INTO label_catalog (graph_id, labels)
        VALUES (graphid, '[]'::jsonb)
        ON CONFLICT (graph_id) DO NOTHING;

        IF NEW.properties ? 'labels' THEN
            new_labels := NEW.properties->'labels';
            new_labels := (
                SELECT jsonb_agg(elems)
                FROM jsonb_array_elements_text(new_labels) AS elems
                WHERE elems NOT IN ('__Node__')
            );
        ELSE
            new_labels := '[]'::jsonb;
        END IF;

        UPDATE label_catalog
        SET labels = (
            SELECT jsonb_agg(DISTINCT elems)
            FROM jsonb_array_elements(COALESCE(labels, '[]'::jsonb) || COALESCE(new_labels, '[]'::jsonb)) AS elems
        )
        WHERE graph_id = graphid;

        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

"""

label_trigger = """
    CREATE OR REPLACE TRIGGER trigger_track_labels
    AFTER INSERT OR UPDATE ON "{}"."{}"
    FOR EACH ROW
    EXECUTE FUNCTION track_labels();

"""

get_label_name_function = """
    CREATE OR REPLACE FUNCTION get_label_name(gid graphid)
    RETURNS text
    LANGUAGE SQL
    AS $$
        SELECT l.labname
        FROM ag_label l
        JOIN ag_graph g ON g.oid = l.graphid
        WHERE l.labid = graphid_labid(gid) AND
              g.graphname = current_setting('graph_path');
    $$;

"""

class AgensgraphQueryException(Exception):
    """Exception for the Agensgraph queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details

class AgensgraphAdapter(GraphDBInterface):
    """
    Handles interaction with a Agensgraph database through various graph operations.

    Public methods include:
    - get_session
    - query
    - has_node
    - add_node
    - add_nodes
    - extract_node
    - extract_nodes
    - delete_node
    - delete_nodes
    - has_edge
    - has_edges
    - add_edge
    - add_edges
    - get_edges
    - get_disconnected_nodes
    - get_predecessors
    - get_successors
    - get_neighbours
    - get_connections
    - remove_connection_to_predecessors_of
    - remove_connection_to_successors_of
    - delete_graph
    - serialize_properties
    - get_model_independent_graph_data
    - get_graph_data
    - get_nodeset_subgraph
    - get_filtered_graph_data
    - get_node_labels_string
    - get_relationship_labels_string
    - get_graph_metrics
    """

    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: Optional[str] = None,
        graph_database_password: Optional[str] = None,
        driver: Optional[Any] = None,
    ):
        self.driver = None
        self.driver_lock = asyncio.Lock()
        self.driver = AsyncConnectionPool(graph_database_url, open=False)
        self.graph_name = "cognee"
        self.graph_id = None

        return None
    
    async def initialize(self):
        """
        Initialize the Agensgraph storage
        """
        if self.driver is None:
            raise AgensgraphQueryException("Agensgraph driver is not initialized")

        # create graph and set graph_path
        async with self.driver_lock:
            try:
                await self.driver.open()
            except psycopg.errors.InvalidSchemaName as e:
                raise AgensgraphQueryException(
                    f"Failed to open connection to Agensgraph: {str(e)}"
                ) from e

        async with self.get_pool_connection() as conn:
            async with conn.cursor() as curs:
                try:
                    await curs.execute(f'CREATE GRAPH IF NOT EXISTS "{self.graph_name}"')
                    await curs.execute(f"SELECT oid from ag_graph WHERE graphname = '{self.graph_name}'")
                    self.graph_id = (await curs.fetchone())[0]
                    await curs.execute(f'SET graph_path = "{self.graph_name}"')
                    await curs.execute(f'CREATE VLABEL IF NOT EXISTS base')
                    await curs.execute(f'CREATE ELABEL IF NOT EXISTS "DIRECTED"')
                    await curs.execute(f'CREATE PROPERTY INDEX IF NOT EXISTS base_entity_idx ON base (entity_id)')
                    await curs.execute(append_label_function)
                    await curs.execute(get_labels_function)
                    await curs.execute(track_labels.format(self.graph_id))
                    await curs.execute(label_catalog.format(self.graph_name, BASE_LABEL))
                    await curs.execute(get_label_name_function)
                    await conn.commit()
                except (
                    psycopg.errors.InvalidSchemaName,
                    psycopg.errors.UniqueViolation,
                ):
                    await conn.rollback()
                    logger.warning(
                        f"Graph {self.graph_name} already exists or could not be created."
                    )
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        f"Error initializing graph {self.graph_name}: {str(e)}"
                    ) from e
            
        logger.info(f"Agensgraph storage initialized for graph: {self.graph_name}")


    @asynccontextmanager
    async def get_pool_connection(self, timeout: Optional[float] = None):
        """Workaround for a psycopg_pool bug"""

        try:
            connection = await self.driver.getconn(timeout=timeout)
        except PoolTimeout:
            await self.driver._add_connection(None)  # workaround...
            connection = await self.driver.getconn(timeout=timeout)

        try:
            async with connection:
                yield connection
        finally:
            await self.driver.putconn(connection)

    async def query(
        self,
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
        await self.driver.open()

        # execute the query, rolling back on an error
        async with self.get_pool_connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as curs:
                try:
                    await curs.execute(f'SET graph_path = {self.graph_name}')
                    await curs.execute(query, params)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        {
                            "message": f"Error executing graph query: {query}",
                            "detail": str(e),
                        }
                    ) from e
                try:
                    data = await curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []  # Handle queries that don’t return data
                if data is None:
                    result = []
                # decode records
                else:
                    result = [self._record_to_dict(d) for d in data]

                return result

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the specified ID exists in the database.

        Parameters:
        -----------

            - node_id (str): The ID of the node to check for existence.

        Returns:
        --------

            - bool: True if the node exists, otherwise False.
        """
        results = self.query(sql.SQL(
            """
                MATCH (n:{BASE_LABEL})
                WHERE n.id = %(node_id)s
                WITH COUNT(n) AS nodes
                RETURN nodes > 0 AS node_exists
            """).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
            {"node_id": Jsonb(node_id)}
        )
        return results[0]["node_exists"] if len(results) > 0 else False

    async def add_node(self, node: DataPoint):
        """
        Add a new node to the database based on the provided DataPoint object.

        Parameters:
        -----------

            - node (DataPoint): An instance of DataPoint representing the node to add.

        Returns:
        --------

            The result of the query execution, typically the ID of the added node.
        """
        query = """
            MERGE (node: {label} {{id: %(node_id)s}})
            ON CREATE SET node += %(properties)s, node.updated_at = now(), node.labels = append_label(node.labels, node_label)
            ON MATCH SET node += %(properties)s, node.updated_at = now(), node.labels = append_label(node.labels, node_label)
            RETURN ID(node) AS internal_id, node.id AS nodeId
            """

        params = {
            "node_id": str(node.id),
            "node_label": type(node).__name__,
            "properties": Jsonb(self.serialize_properties(node.model_dump()))
        }

        return await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            params,
        )

    @record_graph_changes
    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        """
        Add multiple nodes to the database in a single query.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint instances representing the nodes to
              add.

        Returns:
        --------

            - None: None
        """
        nodes = [
            {
                "node_id": str(node.id),
                "label": type(node).__name__,
                "properties": self.serialize_properties(node.model_dump()),
            }
            for node in nodes
        ]

        query = """
        UNWIND %(nodes)s AS node
        MERGE (n: {label} {{id: node.node_id}})
        ON CREATE SET n += node.properties, n.updated_at = now(), n.labels = append_label(n.labels, node.label)
        ON MATCH SET n += node.properties, n.updated_at = now(), n.labels = append_label(n.labels, node.label)
        RETURN ID(n) AS internal_id, n.id AS nodeId
        """

        results = await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            {"nodes": Jsonb(nodes)},
        )
        return results

    async def extract_node(self, node_id: str):
        """
        Retrieve a single node from the database by its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to retrieve.

        Returns:
        --------

            The node represented as a dictionary, or None if it does not exist.
        """
        results = await self.extract_nodes([node_id])

        return results[0] if len(results) > 0 else None

    async def extract_nodes(self, node_ids: List[str]):
        """
        Retrieve multiple nodes from the database by their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of IDs for the nodes to retrieve.

        Returns:
        --------

            A list of nodes represented as dictionaries.
        """
        query = """
        UNWIND %(node_ids)s AS id
        MATCH (node:{label} {{id: id}})
        RETURN node"""

        params = {"node_ids": Jsonb(node_ids)}

        results = await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            params,
        )

        return [result["node"] for result in results]

    async def delete_node(self, node_id: str):
        """
        Remove a node from the database identified by its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to delete.

        Returns:
        --------

            The result of the query execution, typically indicating success or failure.
        """
        query = """
        MATCH (node: {label} {{id: %(node_id)s}})
        DETACH DELETE node
        """
        params = {"node_id": Jsonb(node_id)}

        return await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            params
        )

    async def delete_nodes(self, node_ids: list[str]) -> None:
        """
        Delete multiple nodes from the database using their IDs.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of the nodes to delete.

        Returns:
        --------

            - None: None
        """
        query = """
        UNWIND %(node_ids)s AS id
        MATCH (node:{label} {{id: id}})
        DETACH DELETE node"""

        params = {"node_ids": Jsonb(node_ids)}

        return await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            params,
        )

    async def has_edge(self, from_node: UUID, to_node: UUID, edge_label: str) -> bool:
        """
        Check if an edge exists between two nodes with the specified IDs and edge label.

        Parameters:
        -----------

            - from_node (UUID): The ID of the node from which the edge originates.
            - to_node (UUID): The ID of the node to which the edge points.
            - edge_label (str): The label of the edge to check for existence.

        Returns:
        --------

            - bool: True if the edge exists, otherwise False.
        """
        query = """
            MATCH (from_node: {BASE_LABEL})-[:{edge_label}]->(to_node: {BASE_LABEL})
            WHERE from_node.id = %(from_node)s AND to_node.id = %(to_node)s
            WITH COUNT(relationship) AS relationships
            RETURN relationships > 0 AS edge_exists
        """

        params = {
            "from_node_id": str(from_node),
            "to_node_id": str(to_node),
        }

        edge_exists = await self.query(
            sql.SQL(query).format(
                BASE_LABEL=sql.Identifier(BASE_LABEL),
                edge_label=sql.Identifier(edge_label)
            ), params
        )
        return edge_exists

    async def has_edges(self, edges):
        """
        Check if multiple edges exist based on provided edge criteria.

        Parameters:
        -----------

            - edges: A list of edge specifications to check for existence.

        Returns:
        --------

            A list of boolean values indicating the existence of each edge.
        """
        edges = [
            {
                "from_node": str(edge[0]),
                "to_node": str(edge[1]),
                "relationship_name": edge[2],
            }
            for edge in edges
        ]
        query = """
            UNWIND %(edges)s AS edge
            MATCH (a)-[r]->(b)
            WHERE id(a)::jsonb = edge.from_node AND id(b)::jsonb = edge.to_node AND type(r) = edge.relationship_name
            RETURN edge.from_node AS from_node, edge.to_node AS to_node, edge.relationship_name AS relationship_name
        """

        params = {"edges": Jsonb(edges)}

        results = await self.query(query, params)
        return results

    async def add_edge(
        self,
        from_node: UUID,
        to_node: UUID,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = {},
    ):
        """
        Create a new edge between two nodes with specified properties.

        Parameters:
        -----------

            - from_node (UUID): The ID of the source node of the edge.
            - to_node (UUID): The ID of the target node of the edge.
            - relationship_name (str): The type/label of the edge to create.
            - edge_properties (Optional[Dict[str, Any]]): A dictionary of properties to assign
              to the edge. (default {})

        Returns:
        --------

            The result of the query execution, typically indicating the created edge.
        """
        query = """
            MATCH (from_node :{BASE_LABEL} {{id: %(from_node)s}}),
                  (to_node :{BASE_LABEL} {{id: %(to_node)s}})
            MERGE (from_node)-[r:{relationship_name}]->(to_node)
            ON CREATE SET r += %(properties)s, r.updated_at = now()
            ON MATCH SET r += %(properties)s, r.updated_at = now()
            RETURN r
            """

        params = {
            "from_node": Jsonb(str(from_node)),
            "to_node": Jsonb(str(to_node)),
            "properties": Jsonb(self.serialize_properties(edge_properties)),
        }

        return await self.query(
            sql.SQL(query).format(
                BASE_LABEL=sql.Identifier(BASE_LABEL),
                relationship_name=sql.Identifier(relationship_name),
            ), params
        )

    @record_graph_changes
    async def add_edges(self, edges: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """
        Add multiple edges between nodes in a single query.

        Parameters:
        -----------

            - edges (list[tuple[str, str, str, dict[str, Any]]]): A list of tuples where each
              tuple contains edge details to add.

        Returns:
        --------

            - None: None
        """
        # TODO: Optimize this
        for edge in edges:
            await self.add_edge(edge[0], 
                                edge[1],
                                edge[2],
                                {**(edge[3] if edge[3] else {}),
                                 "source_node_id": str(edge[0]),
                                 "target_node_id": str(edge[1])})

    async def get_edges(self, node_id: str):
        """
        Retrieve all edges connected to a specified node.

        Parameters:
        -----------

            - node_id (str): The ID of the node for which edges are retrieved.

        Returns:
        --------

            A list of edges connecting to the specified node, represented as tuples of details.
        """
        query = """
        MATCH (n: {BASE_LABEL} {{id: %(node_id)s}})-[r]-(m)
        RETURN n, r, m
        """

        results = await self.query(
            sql.SQL(query).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
            {"node_id": Jsonb(node_id)}
        )

        return [
            (result["n"]["id"], result["m"]["id"], {"relationship_name": result["r"][1]})
            for result in results
        ]

    async def get_disconnected_nodes(self) -> list[str]:
        """
        Find and return nodes that are not connected to any other nodes in the graph.

        Returns:
        --------

            - list[str]: A list of IDs of disconnected nodes.
        """
        # return await self.query(
        #     "MATCH (node) WHERE NOT (node)<-[:*]-() RETURN node.id as id",
        # )
        query = """
        // Step 1: Collect all nodes
        MATCH (n)
        WITH COLLECT(n) AS nodes

        // Step 2: Find all connected components
        WITH nodes
        UNWIND nodes AS startNode
        MATCH path = (startNode)-[*]-(connectedNode)
        WITH startNode, COLLECT(DISTINCT connectedNode) AS component

        // Step 3: Aggregate components
        WITH COLLECT(component) AS components

        // Step 4: Identify the largest connected component
        UNWIND components AS component
        WITH component
        ORDER BY SIZE(component) DESC
        LIMIT 1
        WITH component AS largestComponent

        // Step 5: Find nodes not in the largest connected component
        MATCH (n)
        WHERE NOT n IN largestComponent
        RETURN COLLECT(ID(n)) AS ids
        """

        results = await self.query(query)
        return results[0]["ids"] if len(results) > 0 else []

    async def get_predecessors(self, node_id: str, edge_label: str = None) -> list[str]:
        """
        Retrieve the predecessor nodes of a specified node based on an optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node whose predecessors are to be retrieved.
            - edge_label (str): Optional edge label to filter predecessors. (default None)

        Returns:
        --------

            - list[str]: A list of predecessor node IDs.
        """
        if edge_label is not None:
            query = """
            MATCH (node: {BASE_LABEL})<-[r:{edge_label}]-(predecessor)
            WHERE node.id = %(node_id)s
            RETURN predecessor
            """

            results = await self.query(
                sql.SQL(query).format(
                    BASE_LABEL=sql.Identifier(BASE_LABEL),
                    edge_label=sql.Identifier(edge_label),
                ),
                {"node_id": Jsonb(node_id)}
            )

            return [result["predecessor"] for result in results]
        else:
            query = """
            MATCH (node: {BASE_LABEL})<-[r]-(predecessor)
            WHERE node.id = %(node_id)s
            RETURN predecessor
            """

            results = await self.query(
                sql.SQL(query).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
                {"node_id": Jsonb(node_id)}
            )

            return [result["predecessor"] for result in results]

    async def get_successors(self, node_id: str, edge_label: str = None) -> list[str]:
        """
        Retrieve the successor nodes of a specified node based on an optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node whose successors are to be retrieved.
            - edge_label (str): Optional edge label to filter successors. (default None)

        Returns:
        --------

            - list[str]: A list of successor node IDs.
        """
        if edge_label is not None:
            query = """
            MATCH (node: {BASE_LABEL})-[r:{edge_label}]->(successor)
            WHERE node.id = %(node_id)s
            RETURN successor
            """

            results = await self.query(
                sql.SQL(query).format(
                    BASE_LABEL=sql.Identifier(BASE_LABEL),
                    edge_label=sql.Identifier(edge_label),
                ),
                {"node_id": Jsonb(node_id)}
            )

            return [result["successor"] for result in results]
        else:
            query = """
            MATCH (node: {BASE_LABEL})-[r]->(successor)
            WHERE node.id = %(node_id)s
            RETURN successor
            """

            results = await self.query(
                sql.SQL(query).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
                {"node_id": Jsonb(node_id)}
            )

            return [result["successor"] for result in results]

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all neighbors of a specified node, including all directly connected nodes.

        Parameters:
        -----------

            - node_id (str): The ID of the node for which neighbors are retrieved.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of neighboring nodes represented as dictionaries.
        """
        return await self.get_neighbours(node_id)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single node based on its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to retrieve.

        Returns:
        --------

            - Optional[Dict[str, Any]]: The requested node as a dictionary, or None if it does
              not exist.
        """
        query = """
        MATCH (node: {BASE_LABEL} {{id: %(node_id)s}})
        RETURN node
        """
        results = await self.query(
            sql.SQL(query).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
            {"node_id": Jsonb(node_id)}
        )
        return results[0]["node"] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple nodes based on their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of node IDs to retrieve.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of nodes represented as dictionaries.
        """
        query = """
        UNWIND %(node_ids)s AS id
        MATCH (node:{label} {{id: id}})
        RETURN node
        """
        results = await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL)),
            {"node_ids": Jsonb(node_ids)},
        )
        return [result["node"] for result in results]

    async def get_connections(self, node_id: UUID) -> list:
        """
        Retrieve all connections (predecessors and successors) for a specified node.

        Parameters:
        -----------

            - node_id (UUID): The ID of the node for which connections are retrieved.

        Returns:
        --------

            - list: A list of connections represented as tuples of details.
        """
        predecessors_query = """
        MATCH (node:{BASE_LABEL})<-[relation]-(neighbour)
        WHERE node.id = %(node_id)s
        RETURN neighbour, relation, node
        """
        successors_query = """
        MATCH (node:{BASE_LABEL})-[relation]->(neighbour)
        WHERE node.id = %(node_id)s
        RETURN node, relation, neighbour
        """

        predecessors, successors = await asyncio.gather(
            self.query(
                sql.SQL(predecessors_query).format(
                    BASE_LABEL=sql.Identifier(BASE_LABEL)
                ), {"node_id": Jsonb(str(node_id))}
            ),
            self.query(
                sql.SQL(successors_query).format(
                    BASE_LABEL=sql.Identifier(BASE_LABEL)
                ), {"node_id": Jsonb(str(node_id))}
            )
        )

        connections = []

        for neighbour in predecessors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        for neighbour in successors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        return connections

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        """
        Remove connections (edges) to all predecessors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None
        """
        query = """
        UNWIND %(node_ids)s AS id
        MATCH (node:{label1} {{id:id}})-[r:{label2}]->(predecessor:{label3})
        DELETE r;
        """

        params = {"node_ids": Jsonb(node_ids)}

        return await self.query(
            sql.SQL(query).format(
                label1=sql.Identifier(BASE_LABEL),
                label2=sql.Identifier(edge_label),
                label3=sql.Identifier(BASE_LABEL)
            ), params
        )

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        """
        Remove connections (edges) to all successors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None
        """
        query = """
        UNWIND %(node_ids)s AS id
        MATCH (node:{label1} {{id:id}})<-[r:{label2}]-(successor:{label3})
        DELETE r;
        """

        params = {"node_ids": Jsonb(node_ids)}

        return await self.query(
            sql.SQL(query).format(
                label1=sql.Identifier(BASE_LABEL),
                label2=sql.Identifier(edge_label),
                label3=sql.Identifier(BASE_LABEL)
            ), params
        )

    async def delete_graph(self):
        """
        Delete all nodes and edges from the graph database.

        Returns:
        --------

            The result of the query execution, typically indicating success or failure.
        """
        query = """MATCH (node:{label})
                DETACH DELETE node;"""

        return await self.query(
            sql.SQL(query).format(label=sql.Identifier(BASE_LABEL))
        )

    def serialize_properties(self, properties=dict()):
        """
        Convert properties of a node or edge into a serializable format suitable for storage.

        Parameters:
        -----------

            - properties: A dictionary of properties to serialize, defaults to an empty
              dictionary. (default dict())

        Returns:
        --------

            A dictionary with serialized property values.
        """
        serialized_properties = {}

        for property_key, property_value in properties.items():
            if isinstance(property_value, UUID):
                serialized_properties[property_key] = str(property_value)
                continue

            serialized_properties[property_key] = property_value

        return serialized_properties

    async def get_model_independent_graph_data(self):
        """
        Retrieve the basic graph data without considering the model specifics, returning nodes
        and edges.

        Returns:
        --------

            A tuple of nodes and edges data.
        """
        query_nodes = "MATCH (n) RETURN collect(properties(n)) AS nodes"
        nodes = await self.query(query_nodes)

        query_edges = "MATCH (n)-[r]->(m) RETURN collect([properties(n), properties(r), properties(m)]) AS edges"
        edges = await self.query(query_edges)

        return (nodes, edges)
    
    async def project_entire_graph(self, graph_name="cognee"):
        logger.warning(
            "Agensgraph does not support in-memory graph projection. "
        )

    async def get_graph_data(self):
        """
        Retrieve comprehensive data about nodes and relationships within the graph.

        Returns:
        --------

            A tuple containing two lists: nodes and edges with their properties.
        """
        query = "MATCH (n) RETURN ID(n) AS id, get_labels(n) AS labels, properties(n) AS properties"

        result = await self.query(query)
        nodes = [
            (
                record["properties"]["id"],
                record["properties"],
            )
            for record in result
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = await self.query(query)
        edges = [
            (
                record["properties"]["source_node_id"],
                record["properties"]["target_node_id"],
                record["type"],
                record["properties"],
            )
            for record in result
        ]

        return (nodes, edges)

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """
        Retrieve a subgraph based on specified node names and type, including their
        relationships.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of names for nodes to filter the subgraph.

        Returns:
        --------

            - Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]}: A tuple
              containing nodes and edges in the requested subgraph.
        """
        query = """
        UNWIND %(names)s AS wantedName
        MATCH (n)
        WHERE n.name = wantedName AND n.labels @> %(label)s
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[]-(nbr)
        WITH collect(DISTINCT properties(n)) AS prim, collect(DISTINCT properties(nbr)) AS nbrs
        WITH prim + nbrs AS nodelist
        UNWIND nodelist AS node
        WITH collect(DISTINCT node) AS nodes, collect(DISTINCT node.id) AS node_ids
        MATCH (a)-[r]-(b)
        WHERE a.id IN node_ids AND b.id IN node_ids
        WITH nodes, collect(DISTINCT r) AS rels
        RETURN
          [n IN nodes |
             { id: n.id,
                properties: n }] AS "rawNodes",
          [r IN rels  |
             { type: get_label_name(r.id::graphid),
                properties: r.properties }] AS "rawRels"
        """

        result = await self.query(query, {"names": Jsonb(node_name), "label": Jsonb(node_type.__name__)})
        if not result:
            return [], []

        raw_nodes = result[0]["rawNodes"]
        raw_rels = result[0]["rawRels"]

        nodes = [(n["properties"]["id"], n["properties"]) for n in raw_nodes]
        edges = [
            (
                r["properties"]["source_node_id"],
                r["properties"]["target_node_id"],
                r["type"],
                r["properties"],
            )
            for r in raw_rels
        ]

        return nodes, edges

    async def get_filtered_graph_data(self, attribute_filters):
        """
        Fetch nodes and edges filtered by specific attribute criteria.

        Parameters:
        -----------

            - attribute_filters: A list of dictionaries representing attributes and associated
              values for filtering.

        Returns:
        --------

            A tuple containing filtered nodes and edges based on the specified criteria.
        """
        where_clauses = []
        for attribute, values in attribute_filters[0].items():
            values_str = ", ".join(
                f"'{value}'" if isinstance(value, str) else str(value) for value in values
            )
            where_clauses.append(f"n.{attribute} IN [{values_str}]")

        where_clause = " AND ".join(where_clauses)

        query_nodes = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN ID(n) AS id, get_labels(n) AS labels, properties(n) AS properties
        """
        result_nodes = await self.query(query_nodes)

        nodes = [
            (
                record["id"],
                record["properties"],
            )
            for record in result_nodes
        ]

        query_edges = f"""
        MATCH (n)-[r]->(m)
        WHERE {where_clause} AND {where_clause.replace("n.", "m.")}
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result_edges = await self.query(query_edges)

        edges = [
            (
                record["source"],
                record["target"],
                record["type"],
                record["properties"],
            )
            for record in result_edges
        ]

        return (nodes, edges)

    async def graph_exists(self, graph_name="cognee"):
        """
        Check if a graph with a given name exists in the database.

        Parameters:
        -----------

            - graph_name: The name of the graph to check for existence, defaults to 'cognee'.
              (default 'cognee')

        Returns:
        --------

            True if the graph exists, otherwise False.
        """
        query = "SELECT 1 FROM ag_graph WHERE graphname = %(graph_name)s"
        result = await self.query(query, {"graph_name": graph_name})
        if (len(result) > 0):
            return True
        
        return False

    async def get_node_labels_string(self):
        """
        Fetch all node labels from the database and return them as a formatted string.

        Returns:
        --------

            A formatted string of node labels.
        """
        node_labels_query = "SELECT labels FROM label_catalog WHERE graph_id = {self.graph_id}::oid"
        node_labels_result = await self.query(node_labels_query)
        node_labels = node_labels_result[0]["labels"] if node_labels_result else []

        if not node_labels:
            raise ValueError("No node labels found in the database")

        node_labels_str = "[" + ", ".join(f"'{label}'" for label in node_labels) + "]"
        return node_labels_str

    async def get_relationship_labels_string(self):
        """
        Fetch all relationship types from the database and return them as a formatted string.

        Returns:
        --------

            A formatted string of relationship types.
        """
        relationship_types_query = f"""
        SELECT collect(labname) FROM ag_label
        WHERE graphid = {self.graph_id}::oid AND
              labkind = 'e' AND
              labname <> 'ag_edge'
        """
        relationship_types_result = await self.query(relationship_types_query)
        relationship_types = (
            relationship_types_result[0]["relationships"] if relationship_types_result else []
        )

        if not relationship_types:
            raise ValueError("No relationship types found in the database.")

        relationship_types_undirected_str = (
            "{"
            + ", ".join(f"{rel}" + ": {orientation: 'UNDIRECTED'}" for rel in relationship_types)
            + "}"
        )
        return relationship_types_undirected_str

    async def drop_graph(self, graph_name="cognee"):
        """
        Drop an existing graph from the database based on its name.

        Parameters:
        -----------

            - graph_name: The name of the graph to drop, defaults to 'cognee'. (default
              'cognee')
        """
        drop_query = f"DROP GRAPH IF EXISTS {graph_name} CASCADE"
        await self.query(drop_query)

    async def get_graph_metrics(self, include_optional=False):
        """
        Retrieve metrics related to the graph such as number of nodes, edges, and connected
        components.

        Parameters:
        -----------

            - include_optional: Specify whether to include optional metrics; defaults to False.
              (default False)

        Returns:
        --------

            A dictionary containing graph metrics, both mandatory and optional based on the
            input flag.
        """
        nodes, edges = await self.get_model_independent_graph_data()
        num_nodes = len(nodes[0]["nodes"])
        num_edges = len(edges[0]["edges"])

        mandatory_metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
            "edge_density": await get_edge_density(self),
            "num_selfloops": await count_self_loops(self),
        }

        if include_optional:
            logger.error(
                "Optional metrics are not implemented in AgensgraphAdapter yet."
            )

        return mandatory_metrics

    async def get_document_subgraph(self, content_hash: str):
        """
        Retrieve a subgraph related to a document identified by its content hash, including
        related entities and chunks.

        Parameters:
        -----------

            - content_hash (str): The hash identifying the document whose subgraph should be
              retrieved.

        Returns:
        --------

            The subgraph data as a dictionary, or None if not found.
        """
        query = """
        MATCH (doc)
        WHERE (get_labels(doc) @> 'TextDocument'::jsonb OR
               get_labels(doc) @> 'PdfDocument'::jsonb) AND
               doc.name = 'text_' + %(content_hash)s

        OPTIONAL MATCH (doc)<-[:is_part_of]-(chunk)
        WHERE get_labels(chunk) @> 'DocumentChunk'::jsonb
        OPTIONAL MATCH (chunk)-[:contains]->(entity)
        WHERE get_labels(entity) @> 'Entity'::jsonb
        AND NOT (
            SELECT EXISTS (
                MATCH (entity)<-[:contains]-(otherChunk)-[:is_part_of]->(otherDoc)
                WHERE get_labels(otherChunk) @> 'DocumentChunk'::jsonb AND
                      ANY(label IN get_labels(doc) WHERE label IN ['TextDocument', 'PdfDocument'])
                AND otherDoc.id <> doc.id
                RETURN 1
            )
        )
        OPTIONAL MATCH (chunk)<-[:made_from]-(made_node)
        WHERE get_labels(made_node) @> 'TextSummary'::jsonb
        OPTIONAL MATCH (entity)-[:is_a]->(type)
        WHERE get_labels(type) @> 'EntityType'::jsonb
        AND NOT (
            SELECT EXISTS (
                MATCH (type)<-[:is_a]-(otherEntity)<-[:contains]-(otherChunk)-[:is_part_of]->(otherDoc)
                WHERE get_labels(otherEntity) @> 'Entity'::jsonb AND
                      get_labels(otherChunk) @> 'DocumentChunk'::jsonb AND
                      (get_labels(otherDoc) @> 'TextDocument'::jsonb OR
                       get_labels(otherDoc) @> 'PdfDocument'::jsonb)
                AND otherDoc.id <> doc.id
                RETURN 1
            )
        )

        RETURN
            collect(DISTINCT properties(doc)) as document,
            collect(DISTINCT properties(chunk)) as chunks,
            collect(DISTINCT properties(entity)) as orphan_entities,
            collect(DISTINCT properties(made_node)) as made_from_nodes,
            collect(DISTINCT properties(type)) as orphan_types
        """
        result = await self.query(query, {"content_hash": Jsonb(content_hash)})
        return result[0] if result else None

    async def get_degree_one_nodes(self, node_type: str):
        """
        Fetch nodes of a specified type that have exactly one connection.

        Parameters:
        -----------

            - node_type (str): The type of nodes to retrieve, must be 'Entity' or 'EntityType'.

        Returns:
        --------

            A list of nodes with exactly one connection of the specified type.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        query = f"""
        MATCH (n)
        WHERE get_labels(n) @> '{node_type}'::jsonb
        WITH n, (SELECT count(1) FROM (MATCH (n)-[]-() return 1)t) as count
        WHERE count=1
        RETURN n
        """
        result = await self.query(query)
        return [record["n"] for record in result] if result else []

    @staticmethod
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
                vertex = AgensgraphAdapter.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensgraphAdapter.vertex_regex.match(v)
                edge = AgensgraphAdapter.edge_regex.match(v)

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

    @staticmethod
    def escape_str(val: str) -> str:
        return val.replace("'", "''").replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        for k, v in properties.items():
            if isinstance(v, str):
                v_escaped = AgensgraphAdapter.escape_str(v)
                prop = f'"{k}": \'{v_escaped}\''
            else:
                prop = f'"{k}": {v}'
            props.append(prop)

        if id is not None and "id" not in properties:
            id_val = AgensgraphAdapter.escape_str(v)(id) if isinstance(id, str) else id
            props.append(f"id: '{id_val}'" if isinstance(id, str) else f"id: {id_val}")

        return "{" + ", ".join(props) + "}"

    async def _query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        await self.driver.open()

        # execute the query, rolling back on an error
        async with self.get_pool_connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as curs:
                try:
                    await curs.execute(f'SET graph_path = {self.graph_name}')
                    await curs.execute(query)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        {
                            "message": f"Error executing graph query: {query}",
                            "detail": str(e),
                        }
                    ) from e
                try:
                    data = await curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []  # Handle queries that don’t return data
                if data is None:
                    result = []
                # decode records
                else:
                    result = [self._record_to_dict(d) for d in data]

                return result
