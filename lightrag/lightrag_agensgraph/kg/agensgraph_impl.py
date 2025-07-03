import asyncio
import inspect
import re, json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union, final
import pipmaster as pm
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from typing import Any, List, Dict, Optional, Tuple, NamedTuple, Pattern

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import logger
from lightrag.base import BaseGraphStorage
try:
    from lightrag.constants import GRAPH_FIELD_SEP
except ImportError:
    from lightrag.prompt import GRAPH_FIELD_SEP

if sys.platform.startswith("win"):
    import asyncio.windows_events

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if not pm.is_installed("psycopg-pool"):
    pm.install("psycopg-pool")
    pm.install("psycopg[binary,pool]")

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import psycopg  # type: ignore
from psycopg.rows import namedtuple_row  # type: ignore
from psycopg_pool import AsyncConnectionPool, PoolTimeout  # type: ignore


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


@final
@dataclass
class AgensgraphStorage(BaseGraphStorage):
    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with Agensgraph in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        DB = os.environ["AGENSGRAPH_DB"].replace("\\", "\\\\").replace("'", "\\'")
        USER = os.environ["AGENSGRAPH_USER"].replace("\\", "\\\\").replace("'", "\\'")
        PASSWORD = (
            os.environ["AGENSGRAPH_PASSWORD"]
            .replace("\\", "\\\\")
            .replace("'", "\\'")
        )
        HOST = os.environ["AGENSGRAPH_HOST"].replace("\\", "\\\\").replace("'", "\\'")
        PORT = os.environ.get("AGENSGRAPH_PORT", "8529")
        self.graph_name = namespace or os.environ.get("AGENSGRAPH_GRAPHNAME", "lightrag")

        connection_string = f"dbname='{DB}' user='{USER}' password='{PASSWORD}' host='{HOST}' port={PORT}"

        self._driver = AsyncConnectionPool(connection_string, open=False)

        return None

    async def initialize(self):
        """
        Initialize the Agensgraph storage by creating a connection pool.
        """
        if self._driver is None:
            raise AgensgraphQueryException("Agensgraph driver is not initialized")

        # create graph and set graph_path

        async with self._driver_lock:
            try:
                await self._driver.open()
            except psycopg.errors.InvalidSchemaName as e:
                raise AgensgraphQueryException(
                    f"Failed to open connection to Agensgraph: {str(e)}"
                ) from e

        async with self._get_pool_connection() as conn:
            async with conn.cursor() as curs:
                try:
                    await curs.execute(f"CREATE GRAPH IF NOT EXISTS {self.graph_name}")
                    await curs.execute(f'SET graph_path = {self.graph_name}')
                    await curs.execute(f'CREATE VLABEL IF NOT EXISTS base')
                    await curs.execute(f'CREATE ELABEL IF NOT EXISTS "DIRECTED"')
                    await curs.execute(f'CREATE PROPERTY INDEX IF NOT EXISTS base_entity_idx ON base (entity_id)')
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

    async def finalize(self):
        """Close the Agensgraph driver and release all resources"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self) -> None:
        # Agensgraph handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (n:base {{entity_id: '{self.escape_str(node_id)}'}})
                RETURN true AS node_exists LIMIT 1
                """
        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["node_exists"],
        )

        return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (a:base {{entity_id: '{self.escape_str(source_node_id)}'}})-[r]-(b:base {{entity_id: '{self.escape_str(target_node_id)}'}})
                RETURN true AS edgeExists LIMIT 1
                """
        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["edgeExists"],
        )
        return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (n:base {{entity_id: '{self.escape_str(node_id)}'}})
                RETURN n
                """
        records = await self._query(query)
        if records:
            # warn if there are multiple records returned
            if len(records) > 1:
                logger.warning(
                    "Multiple nodes found for entity_id '%s'. Returning first result.",
                    node_id,
                )
            node_dict = records[0]["n"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query,
                node_dict,
            )
            # Return the node properties as a dictionary
            return node_dict

        return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        query = f"""
                UNWIND {node_ids} AS id
                MATCH (n:base {{entity_id: id}})
                RETURN n.entity_id AS entity_id, n
                """
        records = await self._query(query)
        nodes = {}
        if records:
            for record in records:
                entity_id = record["entity_id"]
                node_dict = record["n"]
                logger.debug(
                    "{%s}: query: {%s}, result: {%s}",
                    inspect.currentframe().f_code.co_name,
                    query,
                    node_dict,
                )
                # Return a dictionary with entity_id as key
                nodes[entity_id] = node_dict
            return nodes
        return None

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (n:base {{entity_id: '{self.escape_str(node_id)}'}})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
                """
        record = (await self._query(query))[0]
        if record:
            edge_count = int(record["degree"])
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_count,
            )
            return edge_count
        else:
            logger.warning(f"No node found with label '{self.escape_str(node_id)}'")
            return 0

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.

        Args:
            node_ids: List of node labels (entity_id values) to look up.

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        query = f"""
                UNWIND {node_ids} AS id
                MATCH (n:base {{entity_id: id}})
                OPTIONAL MATCH (n)-[r]-()
                RETURN n.entity_id AS entity_id, count(r) AS degree
                """
        records = (await self._query(query))

        if records:
            degrees = {}
            for record in records:
                entity_id = record["entity_id"]
                degree = int(record["degree"])
                degrees[entity_id] = degree
                logger.debug(
                    "{%s}: query: {%s}, result: {%s}",
                    inspect.currentframe().f_code.co_name,
                    query,
                    degrees,
                )
            # For any node_id that did not return a record, set degree to 0.
            for nid in node_ids:
                if nid not in degrees:
                    logger.warning(f"No node found with label '{nid}'")
                    degrees[nid] = 0
            return degrees
        else:
            logger.warning("No nodes found for the provided labels.")
            return {nid: 0 for nid in node_ids}

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            "{%s}:query:src_Degree+trg_degree:result:{%s}",
            inspect.currentframe().f_code.co_name,
            degrees,
        )
        return degrees
    
    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Collect unique node IDs from all edge pairs.
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go.
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair.
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        
        logger.debug(
            "{%s}:query:edge_degrees_batch:result:{%s}",
            inspect.currentframe().f_code.co_name,
            edge_degrees,
        )
        return edge_degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Edge properties if found, default properties if not found or on error

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (start:base {{entity_id: '{self.escape_str(source_node_id)}'}})-[r]-("end":base {{entity_id: '{self.escape_str(target_node_id)}'}})
                RETURN properties(r) as edge_properties
                """
        records = await self._query(query)

        if records:
            if len(records) > 1:
                logger.warning(
                    "Multiple edges found between '%s' and '%s'. Returning first result.",
                    self.escape_str(source_node_id),
                    self.escape_str(target_node_id),
                )
            edge_result = records[0]["edge_properties"]

            required_keys = {
                "weight": 0.0,
                "source_id": None,
                "description": None,
                "keywords": None,
            }
            for key, default_value in required_keys.items():
                if key not in edge_result:
                    edge_result[key] = default_value
                    logger.warning(
                        f"Edge between {self.escape_str(source_node_id)} and {self.escape_str(target_node_id)} "
                        f"missing {key}, using default: {default_value}"
                    )
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_result,
            )
            return edge_result
        else:
            logger.warning(
                "No edge found between '%s' and '%s'. Returning default properties.",
                self.escape_str(source_node_id),
                self.escape_str(target_node_id),
            )
            # Return None when no edge found
            return None

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        query = f"""
                UNWIND {pairs} AS pair
                MATCH (start:base {{entity_id: pair.src}})-[r:"DIRECTED"]-("end":base {{entity_id: pair.tgt}})
                RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges
                """
        records = await self._query(query)
        edges_dict = {}
        if records:
            for record in records:
                src = record["src_id"]
                tgt = record["tgt_id"]
                edges = record["edges"]
                if edges and len(edges) > 0:
                    edge_props = edges[0]  # choose the first if multiple exist
                    # Ensure required keys exist with defaults
                    for key, default in {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_props:
                            edge_props[key] = default
                    edges_dict[(src, tgt)] = edge_props
                else:
                    edges_dict[(src, tgt)] = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edges_dict,
            )
            return edges_dict
        else:
            logger.warning("No edges found for the provided pairs.")
            return edges_dict
    
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            Exception: If there is an error executing the query
        """
        query = f"""
                MATCH (n:base {{entity_id: '{self.escape_str(source_node_id)}'}})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                WHERE connected.entity_id IS NOT NULL
                RETURN n, r, connected
                """
        results = await self._query(query)
        if results:
            edges = []
            for record in results:
                source_node = record["n"] if record["n"] else None
                connected_node = record["connected"] if record["connected"] else None

                if not source_node or not connected_node:
                    continue

                source_label = (
                    source_node.get("entity_id")
                    if source_node.get("entity_id")
                    else None
                )
                target_label = (
                    connected_node.get("entity_id")
                    if connected_node.get("entity_id")
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))
        else:
            logger.warning(f"No edges found for node with label '{source_node_id}'")
            return None

        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            edges,
        )
        return edges

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using UNWIND.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        # Query to get both outgoing and incoming edges
        query = f"""
                UNWIND {node_ids} AS id
                MATCH (n:base {{entity_id: id}})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                RETURN id AS queried_id, n.entity_id AS node_entity_id,
                        connected.entity_id AS connected_entity_id,
                        startNode(r).entity_id AS start_entity_id
                """
        records = await self._query(query)

        # Initialize the dictionary with empty lists for each node ID
        edges_dict = {node_id: [] for node_id in node_ids}

        if records:
            for record in records:
                queried_id = record["queried_id"]
                node_entity_id = record["node_entity_id"]
                connected_entity_id = record["connected_entity_id"]
                start_entity_id = record["start_entity_id"]

                # Skip if either node is None
                if not node_entity_id or not connected_entity_id:
                    continue

                # Determine the actual direction of the edge
                # If the start node is the queried node, it's an outgoing edge
                # Otherwise, it's an incoming edge
                if start_entity_id == node_entity_id:
                    # Outgoing edge: (queried_node -> connected_node)
                    edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                else:
                    # Incoming edge: (connected_node -> queried_node)
                    edges_dict[queried_id].append((connected_entity_id, node_entity_id))
        else:
            logger.warning("No edges found for the provided node IDs.")
            # If no edges found, return empty lists for each node ID
            return edges_dict
        
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            edges_dict,
        )
        return edges_dict
    
    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        query = f"""
                UNWIND {chunk_ids} AS chunk_id
                MATCH (n:base)
                WHERE n.source_id IS NOT NULL AND chunk_id <@ split(n.source_id, '{GRAPH_FIELD_SEP}')::jsonb
                RETURN DISTINCT n
                """

        results = await self._query(query)
        nodes = []
        for record in results:
            node_dict = record["n"]
            # Add node id (entity_id) to the dictionary for easier access
            node_dict["id"] = node_dict.get("entity_id")
            nodes.append(node_dict)
        return nodes
    
    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        query = f"""
                UNWIND {chunk_ids} AS chunk_id
                MATCH (a:base)-[r]-(b:base)
                WHERE r.source_id IS NOT NULL AND chunk_id <@ split(r.source_id, '{GRAPH_FIELD_SEP}')::jsonb
                RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
                """

        results = await self._query(query)
        edges = []
        for record in results:
            edge_properties = record["properties"]
            edge_properties["source"] = record["source"]
            edge_properties["target"] = record["target"]
            edges.append(edge_properties)
        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Agensgraph database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = self._format_properties(node_data)
        query = f"""
                MERGE (n:base {{entity_id: '{self.escape_str(node_id)}'}})
                SET n += {properties}
                """
        try:
            await self._query(query)
            logger.debug(
                "Upserted node with node_id '{%s}' and properties: {%s}",
                node_id,
                properties,
            )
        except Exception as e:
            logger.error("Error during upsert: {%s}", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        edge_properties = self._format_properties(edge_data)

        query = f"""
                MATCH (source:base {{entity_id: '{self.escape_str(source_node_id)}'}})
                WITH source
                MATCH (target:base {{entity_id: '{self.escape_str(target_node_id)}'}})
                MERGE (source)-[r:"DIRECTED"]-(target)
                SET r += {edge_properties}
                RETURN r, source, target
                """
        try:
            await self._query(query)
            logger.debug(
                "Upserted edge from '{%s}' to '{%s}' with properties: {%s}",
                source_node_id,
                target_node_id,
                edge_properties,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()
        visited_edge_pairs = set()
        queue = deque()

        # Step 1: Get starting nodes
        if node_label == "*":
            query = f"""
                    MATCH (n:base)
                    RETURN DISTINCT id(n) AS node_id, n
                    LIMIT {max_nodes}
                    """
            node_results = await self._query(query)
        else:
            query = f"""
                    MATCH (n:base {{entity_id: '{self.escape_str(node_label)}'}})
                    RETURN id(n) AS node_id, n
                    """
            node_results = await self._query(query)

        for record in node_results:
            node_data = record["n"]
            if not node_data.get("entity_id"):
                continue
            start_node = KnowledgeGraphNode(
                id=str(node_data["entity_id"]),
                labels=[node_data["entity_id"]],
                properties=node_data,
            )
            queue.append((start_node, None, 0))

        # Step 2: BFS traversal
        while queue and len(visited_nodes) < max_nodes:
            current_node, current_edge, current_depth = queue.popleft()

            if current_node.id in visited_nodes or current_depth > max_depth:
                continue

            result.nodes.append(current_node)
            visited_nodes.add(current_node.id)

            if current_edge and current_edge.id not in visited_edges:
                result.edges.append(current_edge)
                visited_edges.add(current_edge.id)

            if len(visited_nodes) >= max_nodes:
                result.is_truncated = True
                break

            # Step 3: Query neighbors
            query = f"""
            MATCH (a:base {{entity_id: '{self.escape_str(current_node.id)}'}})-[r]-(b)
            RETURN type(r) as rel_type, properties(r) as r, b, id(r) AS edge_id, id(b) AS target_id
            """
            records = await self._query(query)

            for record in records:
                rel_type = record["rel_type"]
                rel = record["r"]
                b_node = record["b"]
                edge_id = str(record["edge_id"])
                target_id = b_node.get("entity_id")

                if not target_id:
                    continue

                target_node = KnowledgeGraphNode(
                    id=str(target_id),
                    labels=[target_id],
                    properties=b_node,
                )

                target_edge = KnowledgeGraphEdge(
                    id=edge_id,
                    type=rel_type,
                    source=current_node.id,
                    target=target_id,
                    properties=rel,
                )

                sorted_pair = tuple(sorted([current_node.id, target_id]))
                if sorted_pair not in visited_edge_pairs:
                    if (
                        target_id in visited_nodes or
                        (target_id not in visited_nodes and current_depth < max_depth)
                    ):
                        result.edges.append(target_edge)
                        visited_edges.add(edge_id)
                        visited_edge_pairs.add(sorted_pair)

                if target_id not in visited_nodes and current_depth < max_depth:
                    queue.append((target_node, None, current_depth + 1))

        return result

    async def get_all_labels(self) -> list[str]:
        """Get all node labels in the database

        Returns:
            ["label1", "label2", ...]  # Alphabetically sorted label list
        """
        query = """
                MATCH (n:base)
                WHERE n.entity_id IS NOT NULL
                WITH DISTINCT n.entity_id AS label
                ORDER BY label
                RETURN collect(label) AS node_labels
                """
        results = await self._query(query)

        if not results:
            logger.warning("No labels found in the graph.")
            return []

        labels = results[0]["node_labels"]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            labels,
        )
        return labels

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """
        query = f"""
                MATCH (n:base {{entity_id: '{self.escape_str(node_id)}'}})
                DETACH DELETE n
                """
        try:
            await self._query(query)
            logger.debug(f"Deleted node with label '{self.escape_str(node_id)}'")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            query = f"""
                    MATCH (source:base {{entity_id: '{self.escape_str(source)}'}})-[r]-(target:base {{entity_id: '{self.escape_str(target)}'}})
                    DELETE r
                    """
            try:
                await self._query(query)
                logger.debug(
                    f"Deleted edge from '{self.escape_str(source)}' to '{self.escape_str(target)}'"
                )
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all nodes and relationships in the graph.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            query = """
                    MATCH (n)
                    DETACH DELETE n
                    """
            await self._query(query)
            logger.info(f"Successfully dropped all data from graph {self.graph_name}")
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            logger.error(f"Error dropping graph {self.graph_name}: {e}")
            return {"status": "error", "message": str(e)}

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
                vertex = AgensgraphStorage.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensgraphStorage.vertex_regex.match(v)
                edge = AgensgraphStorage.edge_regex.match(v)

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
                v_escaped = AgensgraphStorage.escape_str(v)
                prop = f'"{k}": \'{v_escaped}\''
            else:
                prop = f'"{k}": {v}'
            props.append(prop)

        if id is not None and "id" not in properties:
            id_val = AgensgraphStorage.escape_str(v)(id) if isinstance(id, str) else id
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
        await self._driver.open()

        # execute the query, rolling back on an error
        async with self._get_pool_connection() as conn:
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
                    result = [AgensgraphStorage._record_to_dict(d) for d in data]

                return result

    @asynccontextmanager
    async def _get_pool_connection(self, timeout: Optional[float] = None):
        """Workaround for a psycopg_pool bug"""

        try:
            connection = await self._driver.getconn(timeout=timeout)
        except PoolTimeout:
            await self._driver._add_connection(None)  # workaround...
            connection = await self._driver.getconn(timeout=timeout)

        try:
            async with connection:
                yield connection
        finally:
            await self._driver.putconn(connection)
