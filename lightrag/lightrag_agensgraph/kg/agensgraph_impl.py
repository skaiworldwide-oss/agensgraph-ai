"""
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, final

from psycopg import sql
from psycopg.types.json import Jsonb
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.base import BaseGraphStorage
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import logger

try:
    from lightrag.constants import GRAPH_FIELD_SEP
except ImportError:
    from lightrag.prompt import GRAPH_FIELD_SEP

from lightrag_agensgraph.kg._base import AgensgraphQueryException, _AgensStorageBase

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Max rows per UNWIND / OR-of-equalities batch.
CHUNK_SIZE = 1000


def _or_equalities(
    field: str, values: List[str], prefix: str
) -> Tuple[str, Dict[str, Any]]:
    """Build an index-friendly ``(field = v0 OR field = v1 ...)`` fragment.

    Only OR-of-equalities uses the ``entity_id`` btree index; ``IN`` / ``<@`` /
    UNWIND-variable lookups fall back to a sequential scan.
    """
    terms = []
    params: Dict[str, Any] = {}
    for i, value in enumerate(values):
        pname = f"{prefix}_{i}"
        params[pname] = Jsonb(value)
        terms.append(f"{field} = %({pname})s")
    return "(" + " OR ".join(terms) + ")", params


@final
@dataclass
class AgensgraphStorage(_AgensStorageBase, BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with Agensgraph in production")

    def __post_init__(self):
        # Preserve the original semantics: the graph name comes from the
        # storage namespace, falling back to AGENSGRAPH_GRAPHNAME. (Relational
        # stores isolate tenants by `workspace`; the graph store by graph name.)
        self.graph_name = self.namespace or os.environ.get(
            "AGENSGRAPH_GRAPHNAME", "lightrag"
        )
        self._graph_path = self.graph_name
        self._engine = None

    async def initialize(self):
        """Acquire the shared engine and bootstrap the graph (once)."""
        await self._acquire_engine()

        async def _ddl(cur):
            await cur.execute("CREATE VLABEL IF NOT EXISTS base")
            await cur.execute('CREATE ELABEL IF NOT EXISTS "DIRECTED"')
            await cur.execute(
                "CREATE PROPERTY INDEX IF NOT EXISTS base_entity_idx ON base (entity_id)"
            )

        await self._engine.ensure_graph(self.graph_name, _ddl)
        logger.info(f"AgensGraph storage initialized for graph: {self.graph_name}")

    async def finalize(self):
        await self._release_engine()

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
        query = """
                MATCH (n:base {entity_id: %(node_id)s})
                RETURN true AS node_exists LIMIT 1
                """
        records = await self._query(query, {"node_id": Jsonb(node_id)})
        # No row is returned when the node does not exist.
        return bool(records and records[0]["node_exists"])

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
        query = """
                MATCH (a:base {entity_id: %(source_node_id)s})-[r]-(b:base {entity_id: %(target_node_id)s})
                RETURN true AS "edgeExists" LIMIT 1
                """
        records = await self._query(query, {
            "source_node_id": Jsonb(source_node_id),
            "target_node_id": Jsonb(target_node_id),
        })
        # No row is returned when the edge does not exist.
        return bool(records and records[0]["edgeExists"])

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
        query = """
                MATCH (n:base {entity_id: %(node_id)s})
                RETURN n
                """
        records = await self._query(query, {"node_id": Jsonb(node_id)})
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
        query = """
                UNWIND %(node_ids)s AS id
                MATCH (n:base {entity_id: id})
                RETURN n.entity_id AS entity_id, n
                """
        records = await self._query(query, {"node_ids": Jsonb(node_ids)})
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
        query = """
                MATCH (n:base {entity_id: %(node_id)s})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
                """
        record = (await self._query(query, {"node_id": Jsonb(node_id)}))[0]
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
        query = """
                UNWIND %(node_ids)s AS id
                MATCH (n:base {entity_id: id})
                OPTIONAL MATCH (n)-[r]-()
                RETURN n.entity_id AS entity_id, count(r) AS degree
                """
        records = (await self._query(query, {"node_ids": Jsonb(node_ids)}))

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
        query = """
                MATCH (start:base {entity_id: %(source_node_id)s})-[r]-("end":base {entity_id: %(target_node_id)s})
                RETURN properties(r) as edge_properties
                """
        records = await self._query(query, {
            "source_node_id": Jsonb(source_node_id),
            "target_node_id": Jsonb(target_node_id),
        })

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
        query = """
                UNWIND %(pairs)s AS pair
                MATCH (start:base {entity_id: pair.src})-[r:"DIRECTED"]-("end":base {entity_id: pair.tgt})
                RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges
                """
        records = await self._query(query, {"pairs": Jsonb(pairs)})
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
        query = """
                MATCH (n:base {entity_id: %(source_node_id)s})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                WHERE connected.entity_id IS NOT NULL
                RETURN n, r, connected
                """
        results = await self._query(query, {"source_node_id": Jsonb(source_node_id)})
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
        query = """
                UNWIND %(node_ids)s AS id
                MATCH (n:base {entity_id: id})
                OPTIONAL MATCH (n)-[r]-(connected:base)
                RETURN id AS queried_id, n.entity_id AS node_entity_id,
                        connected.entity_id AS connected_entity_id,
                        startNode(r).entity_id AS start_entity_id
                """
        records = await self._query(query, {"node_ids": Jsonb(node_ids)})

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
        query = """
                UNWIND %(chunk_ids)s AS chunk_id
                MATCH (n:base)
                WHERE n.source_id IS NOT NULL AND chunk_id <@ split(n.source_id, {GRAPH_FIELD_SEP})::jsonb
                RETURN DISTINCT n
                """

        results = await self._query(sql.SQL(query).format(
            GRAPH_FIELD_SEP=sql.Literal(GRAPH_FIELD_SEP)
        ), {"chunk_ids": Jsonb(chunk_ids)})
        nodes = []
        for record in results:
            node_dict = record["n"]
            # Add node id (entity_id) to the dictionary for easier access
            node_dict["id"] = node_dict.get("entity_id")
            nodes.append(node_dict)
        return nodes
    
    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        query = """
                UNWIND %(chunk_ids)s AS chunk_id
                MATCH (a:base)-[r]-(b:base)
                WHERE r.source_id IS NOT NULL AND chunk_id <@ split(r.source_id, {GRAPH_FIELD_SEP})::jsonb
                RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
                """

        results = await self._query(sql.SQL(query).format(
            GRAPH_FIELD_SEP=sql.Literal(GRAPH_FIELD_SEP)
        ), {"chunk_ids": Jsonb(chunk_ids)})
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
        query = """
                MERGE (n:base {entity_id: %(node_id)s})
                SET n += %(node_data)s
                """
        try:
            await self._query(query, {
                "node_id": Jsonb(node_id),
                "node_data": Jsonb(node_data),
            })
            logger.debug(
                "Upserted node with node_id '{%s}' and properties: {%s}",
                node_id,
                node_data,
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
        query = """
                MATCH (source:base {entity_id: %(source_node_id)s})
                WITH source
                MATCH (target:base {entity_id: %(target_node_id)s})
                MERGE (source)-[r:"DIRECTED"]-(target)
                SET r += %(edge_data)s
                RETURN r, source, target
                """
        try:
            await self._query(query, {
                "source_node_id": Jsonb(source_node_id),
                "target_node_id": Jsonb(target_node_id),
                "edge_data": Jsonb(edge_data),
            })
            logger.debug(
                "Upserted edge from '{%s}' to '{%s}' with properties: {%s}",
                source_node_id,
                target_node_id,
                edge_data,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph as a KnowledgeGraph.

        For ``*`` the densest ``max_nodes`` nodes (by degree) and the edges among
        them are fetched in two queries. For a specific label a bounded BFS
        expands one frontier per depth (one query per level, via OR-of-equalities)
        instead of one query per node.

        Args:
            node_label: Label of the starting node, ``*`` means all nodes
            max_depth: Maximum BFS depth (Defaults to 3)
            max_nodes: Maximum nodes to return (Defaults to 1000)

        Returns:
            KnowledgeGraph with an ``is_truncated`` flag set when the node limit
            was hit.
        """
        result = KnowledgeGraph()
        nodes_by_id: Dict[str, KnowledgeGraphNode] = {}
        edges_by_id: Dict[str, KnowledgeGraphEdge] = {}

        def _add_node(props) -> Tuple[Optional[str], bool]:
            eid = props.get("entity_id") if props else None
            if eid is None:
                return None, False
            kid = str(eid)
            is_new = kid not in nodes_by_id
            if is_new:
                nodes_by_id[kid] = KnowledgeGraphNode(
                    id=kid, labels=[eid], properties=props
                )
            return kid, is_new

        def _add_edge(eid, rel_type, source, target, props) -> None:
            key = str(eid)
            s, t = str(source), str(target)
            if key in edges_by_id or s not in nodes_by_id or t not in nodes_by_id:
                return
            edges_by_id[key] = KnowledgeGraphEdge(
                id=key, type=rel_type, source=s, target=t, properties=props or {}
            )

        if node_label == "*":
            total = (
                await self._query(
                    "MATCH (n:base) WHERE n.entity_id IS NOT NULL RETURN count(n) AS c"
                )
            )[0]["c"]
            rows = await self._query(
                """
                MATCH (n:base) WHERE n.entity_id IS NOT NULL
                OPTIONAL MATCH (n)-[r]-()
                WITH n, count(r) AS deg
                ORDER BY deg DESC
                LIMIT %(limit)s
                RETURN n
                """,
                {"limit": max_nodes},
            )
            for r in rows:
                _add_node(r["n"])
            result.is_truncated = int(total) > max_nodes
            if nodes_by_id:
                erows = await self._query(
                    """
                    MATCH (a:base)-[r]-(b:base)
                    WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL
                    RETURN id(r) AS eid, type(r) AS rt, a.entity_id AS s,
                           b.entity_id AS t, properties(r) AS props
                    """
                )
                for er in erows:
                    _add_edge(er["eid"], er["rt"], er["s"], er["t"], er["props"])
        else:
            seed = await self._query(
                "MATCH (n:base {entity_id: %(label)s}) RETURN n",
                {"label": Jsonb(node_label)},
            )
            for r in seed:
                _add_node(r["n"])
            frontier = list(nodes_by_id.keys())
            depth = 0
            while frontier and depth < max_depth and len(nodes_by_id) < max_nodes:
                frag, params = _or_equalities("a.entity_id", frontier, "kg")
                rows = await self._query(
                    f"""
                    MATCH (a:base)-[r]-(b:base)
                    WHERE {frag} AND b.entity_id IS NOT NULL
                    RETURN id(r) AS eid, type(r) AS rt, a.entity_id AS s,
                           b.entity_id AS t, b, properties(r) AS props
                    """,
                    params,
                )
                next_frontier: List[str] = []
                for r in rows:
                    if len(nodes_by_id) >= max_nodes:
                        result.is_truncated = True
                        break
                    bid, is_new = _add_node(r["b"])
                    if bid is not None and is_new:
                        next_frontier.append(bid)
                for r in rows:
                    _add_edge(r["eid"], r["rt"], r["s"], r["t"], r["props"])
                frontier = next_frontier
                depth += 1

        result.nodes = list(nodes_by_id.values())
        result.edges = list(edges_by_id.values())
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
        query = """
                MATCH (n:base {entity_id: %(node_id)s})
                DETACH DELETE n
                """
        try:
            await self._query(query, {"node_id": Jsonb(node_id)})
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
        """Delete multiple nodes in chunked, index-backed batches.

        Args:
            nodes: List of node entity_ids to delete
        """
        if not nodes:
            return
        for start in range(0, len(nodes), CHUNK_SIZE):
            chunk = nodes[start : start + CHUNK_SIZE]
            frag, params = _or_equalities("n.entity_id", chunk, "rm")
            await self._query(
                f"MATCH (n:base) WHERE {frag} DETACH DELETE n", params
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges in chunked, index-backed batches.

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        if not edges:
            return
        for start in range(0, len(edges), CHUNK_SIZE):
            chunk = edges[start : start + CHUNK_SIZE]
            terms = []
            params: Dict[str, Any] = {}
            for i, (source, target) in enumerate(chunk):
                params[f"s_{i}"] = Jsonb(source)
                params[f"t_{i}"] = Jsonb(target)
                terms.append(f"(a.entity_id = %(s_{i})s AND b.entity_id = %(t_{i})s)")
            where = " OR ".join(terms)
            await self._query(
                f"MATCH (a:base)-[r]-(b:base) WHERE {where} DELETE r", params
            )

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
    def escape_str(val: str) -> str:
        return val.replace("'", "''").replace("\\", "\\\\").replace('"', '\\"')

    async def get_all_nodes(self) -> list[dict]:
        """Return the property dict of every node."""
        rows = await self._query(
            "MATCH (n:base) WHERE n.entity_id IS NOT NULL RETURN n"
        )
        return [r["n"] for r in rows]

    async def get_all_edges(self) -> list[dict]:
        """Return every edge once as {source, target, **properties}."""
        rows = await self._query(
            """
            MATCH (a:base)-[r]-(b:base)
            WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL
            RETURN id(r) AS eid, a.entity_id AS source, b.entity_id AS target,
                   properties(r) AS properties
            """
        )
        edges: list[dict] = []
        seen: set = set()
        for r in rows:
            # Undirected traversal yields each physical edge twice; dedupe by id.
            if r["eid"] in seen:
                continue
            seen.add(r["eid"])
            props = dict(r["properties"] or {})
            props["source"] = r["source"]
            props["target"] = r["target"]
            edges.append(props)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Return entity labels ordered by degree (most-connected first)."""
        rows = await self._query(
            """
            MATCH (n:base) WHERE n.entity_id IS NOT NULL
            OPTIONAL MATCH (n)-[r]-()
            WITH n.entity_id AS label, count(r) AS deg
            ORDER BY deg DESC, label ASC
            LIMIT %(limit)s
            RETURN collect(label) AS labels
            """,
            {"limit": limit},
        )
        return rows[0]["labels"] if rows else []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Case-insensitive substring search over entity labels."""
        pattern = "(?i).*" + re.escape(query) + ".*"
        rows = await self._query(
            """
            MATCH (n:base) WHERE n.entity_id IS NOT NULL AND n.entity_id =~ %(pattern)s
            WITH n.entity_id AS label
            ORDER BY label ASC
            LIMIT %(limit)s
            RETURN collect(label) AS labels
            """,
            {"pattern": Jsonb(pattern), "limit": limit},
        )
        return rows[0]["labels"] if rows else []

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Return the subset of node_ids that exist."""
        if not node_ids:
            return set()
        rows = await self._query(
            """
            UNWIND %(node_ids)s AS id
            MATCH (n:base {entity_id: id})
            RETURN n.entity_id AS entity_id
            """,
            {"node_ids": Jsonb(node_ids)},
        )
        return {r["entity_id"] for r in rows}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Upsert many nodes in UNWIND-batched MERGEs (index-backed)."""
        rows = [{"id": node_id, "props": data} for node_id, data in nodes]
        query = """
            UNWIND %(rows)s AS row
            MERGE (n:base {entity_id: row.id})
            SET n += row.props
            """
        for start in range(0, len(rows), CHUNK_SIZE):
            await self._query(query, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AgensgraphQueryException,)),
    )
    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Upsert many edges in one UNWIND-batched MERGE per chunk."""
        rows = [
            {"source_id": src, "target_id": tgt, "props": data}
            for src, tgt, data in edges
        ]
        query = """
            UNWIND %(rows)s AS row
            MATCH (source:base {entity_id: row.source_id})
            MATCH (target:base {entity_id: row.target_id})
            MERGE (source)-[r:"DIRECTED"]-(target)
            SET r += row.props
            """
        for start in range(0, len(rows), CHUNK_SIZE):
            await self._query(query, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])})
