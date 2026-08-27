"""AgensGraph graph adapter for cognee."""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple, Type
from uuid import UUID

import psycopg
from agensgraph import Edge, GraphId, Path, RetryPolicy, Vertex, to_builtins
from agensgraph.cypher import check_single_statement
from agensgraph.errors import safe_message
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import ERROR, get_logger
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ._engine import AgensEngine
from .metrics import count_self_loops, get_edge_density

logger = get_logger("AgensgraphAdapter", level=ERROR)
BASE_LABEL = "__Node__"
# Max rows per UNWIND / OR-of-equalities batch.
CHUNK_SIZE = 1000

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

# Neo4j's edge shorthand between two pattern nodes: (a)--(b), (a)-->(b), (a)<--(b), where
# what follows looks like a node: (b, (b:Label, (:Label, ({...} or ().
_EDGE_SHORTHAND = re.compile(r"\)\s*(<-|-)-(>?)\s*\((?=\s*(?:\w+\s*)?[:{)])")
_LITERALS = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"")


def spell_out_edges(statement: str) -> str:
    """Rewrite ``(a)--(b)`` as ``(a)-[]-(b)``, and the two arrow forms likewise.

    In AgensGraph ``--`` starts a comment, so the rest of such a statement is dropped
    and it fails as incomplete. Between two pattern nodes ``--`` cannot mean a comment
    in any statement that runs, so only that form is rewritten, and only outside string
    literals and quoted identifiers. Language models write this shorthand in most
    statements they generate.
    """
    blanked = _LITERALS.sub(lambda m: " " * len(m.group(0)), statement)
    out = []
    last = 0
    for m in _EDGE_SHORTHAND.finditer(blanked):
        left, arrow = m.group(1), m.group(2)
        out.append(statement[last : m.start()])
        out.append(")" + ("<-[]-" if left == "<-" else "-[]-" + arrow) + "(")
        last = m.end()
    out.append(statement[last:])
    return "".join(out)


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

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: Optional[str] = None,
        graph_database_password: Optional[str] = None,
        driver: Optional[Any] = None,
        *,
        query_read_only: bool = True,
        query_allow_server_programs: bool = True,
        retry_attempts: int = 6,
    ):
        conninfo = graph_database_url
        credentials = {}
        if graph_database_username:
            credentials["user"] = graph_database_username
        if graph_database_password:
            credentials["password"] = graph_database_password
        if credentials:
            conninfo = make_conninfo(conninfo, **credentials)
        self.conninfo = conninfo
        self.graph_name = "cognee"
        self.graph_id = None
        self._engine: Optional[AgensEngine] = None
        # query() runs statements written by a user or by a language model. It refuses
        # writes unless the caller turns this off.
        self.query_read_only = query_read_only
        # A read-only transaction does not stop a superuser from running a program on
        # the server, so the driver refuses one for such a role unless told to go ahead.
        # Development setups run as a superuser, so the default is to go ahead.
        self.query_allow_server_programs = query_allow_server_programs
        self.retry_policy = RetryPolicy(attempts=retry_attempts)

    async def initialize(self):
        """Connect once and create what the graph needs. Cheap when called again."""
        self._engine = AgensEngine.get(self.conninfo, graph_name=self.graph_name)
        await self._engine.setup_once(f"graph:{self.graph_name}", self._bootstrap)
        self.graph_id = self._engine.graph_id

    async def _bootstrap(self, conn) -> None:
        await conn.execute(f'CREATE VLABEL IF NOT EXISTS "{BASE_LABEL}"')
        await conn.execute('CREATE ELABEL IF NOT EXISTS "DIRECTED"')
        await conn.execute(
            f'CREATE PROPERTY INDEX IF NOT EXISTS base_id_idx ON "{BASE_LABEL}" (id)'
        )
        # name is the lookup key for nodeset / document subgraph queries.
        await conn.execute(
            f'CREATE PROPERTY INDEX IF NOT EXISTS base_name_idx ON "{BASE_LABEL}" (name)'
        )
        await conn.execute(append_label_function)
        await conn.execute(get_labels_function)
        await conn.execute(track_labels.format(self._engine.graph_id))
        await conn.execute(label_catalog)
        await conn.execute(get_label_name_function)

    async def finalize(self):
        """Close this event loop's pool. The next call opens a new one."""
        if self._engine is not None:
            await self._engine.aclose()

    # ---- statements ----

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a statement written by a user or by a language model.

        One statement at a time, and read-only unless ``query_read_only`` is off.
        """
        if isinstance(query, str):
            query = spell_out_edges(query)
        if not params:
            params = None
            if isinstance(query, str):
                check_single_statement(query)
        if not self.query_read_only:
            return await self._write(query, params)

        async def attempt():
            async with self._engine.connection() as conn:
                async with conn.read_only_transaction(
                    allow_server_programs=self.query_allow_server_programs
                ):
                    return await self._execute(conn, query, params)

        return await self._run_with_retry(attempt, wrote=False)

    async def _read(self, query, params=None) -> List[Dict[str, Any]]:
        async def attempt():
            async with self._engine.connection() as conn:
                return await self._execute(conn, query, params)

        return await self._run_with_retry(attempt, wrote=False)

    async def _write(self, query, params=None) -> List[Dict[str, Any]]:
        async def attempt():
            async with self._engine.connection() as conn:
                return await self._execute(conn, query, params)

        return await self._run_with_retry(attempt, wrote=True)

    async def _run_with_retry(self, attempt, *, wrote: bool):
        """Run ``attempt`` again while the driver says the failure was timing.

        Concurrent writers merging onto the same keys fail each other for the moment
        one of them takes to commit. A merge that loses such a race under a uniqueness
        constraint is reported as an exclusion violation (23P01), so it is judged as the
        unique violation it is.
        """
        number = 0
        while True:
            try:
                result = await attempt()
            except psycopg.Error as exc:
                number += 1
                decision = self.retry_policy.decide(
                    exc, number=number, wrote=wrote, merging=wrote
                )
                if not decision.retry and isinstance(exc, errors.ExclusionViolation):
                    decision = self.retry_policy.decide(
                        errors.UniqueViolation(), number=number, wrote=wrote, merging=True
                    )
                if not decision.retry:
                    logger.error("AgensGraph statement failed: %s", safe_message(exc))
                    raise
                await asyncio.sleep(decision.delay)
            else:
                self.retry_policy.succeeded()
                return result

    @staticmethod
    async def _execute(conn, query, params) -> List[Dict[str, Any]]:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query, params)
            if cur.description is None:
                return []
            rows = await cur.fetchall()
        return [AgensgraphAdapter._convert_row(row) for row in rows]

    @staticmethod
    def _convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Turn the driver's graph values into what cognee expects.

        A vertex becomes its property map. An edge becomes
        ``(start properties, label, end properties)``, with the endpoints taken from the
        vertices returned in the same row.
        """
        vertices: Dict[GraphId, Dict[str, Any]] = {}
        for value in row.values():
            if isinstance(value, Vertex):
                vertices[value.id] = value.properties
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Vertex):
                        vertices[item.id] = item.properties
        return {key: AgensgraphAdapter._convert(value, vertices) for key, value in row.items()}

    @staticmethod
    def _convert(value: Any, vertices: Dict[GraphId, Dict[str, Any]]) -> Any:
        if isinstance(value, Vertex):
            return value.properties
        if isinstance(value, Edge):
            return (vertices.get(value.start, {}), value.label, vertices.get(value.end, {}))
        if isinstance(value, GraphId):
            return str(value)
        if isinstance(value, Path):
            return to_builtins(value)
        if isinstance(value, list):
            return [AgensgraphAdapter._convert(item, vertices) for item in value]
        return value

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
        results = await self._read(sql.SQL(
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
        # Delegate to the batched path: it Jsonb-wraps the row (so the id and
        # label bind as agtype) and MERGEs on the indexed id. The previous
        # single-node query bound node_id as a bare string (invalid agtype) and
        # referenced an undefined `node_label` Cypher variable — both broken.
        return await self.add_nodes([node])

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

        results = await self._write(
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

        results = await self._read(
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

        return await self._write(
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

        return await self._write(
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
            MATCH (from_node: {BASE_LABEL})-[r:{edge_label}]->(to_node: {BASE_LABEL})
            WHERE from_node.id = %(from_node)s AND to_node.id = %(to_node)s
            WITH COUNT(r) AS relationships
            RETURN relationships > 0 AS edge_exists
        """

        params = {
            "from_node": Jsonb(str(from_node)),
            "to_node": Jsonb(str(to_node)),
        }

        results = await self._read(
            sql.SQL(query).format(
                BASE_LABEL=sql.Identifier(BASE_LABEL),
                edge_label=sql.Identifier(edge_label)
            ), params
        )
        return results[0]["edge_exists"] if results else False

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

        results = await self._read(query, params)
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

        return await self._write(
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
        if not edges:
            return
        # The relationship type must be a literal in MERGE, so batch per type;
        # each batch is one UNWIND query (endpoints matched via the indexed id).
        by_rel: dict[str, list[dict]] = {}
        for source, target, relationship_name, props in edges:
            by_rel.setdefault(relationship_name, []).append(
                {
                    "from_node": str(source),
                    "to_node": str(target),
                    "properties": self.serialize_properties(
                        {
                            **(props if props else {}),
                            "source_node_id": str(source),
                            "target_node_id": str(target),
                        }
                    ),
                }
            )

        query = """
            UNWIND %(rows)s AS row
            MATCH (from_node :{BASE_LABEL} {{id: row.from_node}}),
                  (to_node :{BASE_LABEL} {{id: row.to_node}})
            MERGE (from_node)-[r:{relationship_name}]->(to_node)
            ON CREATE SET r += row.properties, r.updated_at = now()
            ON MATCH SET r += row.properties, r.updated_at = now()
            """
        for relationship_name, rows in by_rel.items():
            formatted = sql.SQL(query).format(
                BASE_LABEL=sql.Identifier(BASE_LABEL),
                relationship_name=sql.Identifier(relationship_name),
            )
            for start in range(0, len(rows), CHUNK_SIZE):
                await self._write(
                    formatted, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])}
                )

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

        results = await self._read(
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
        # AgensGraph's Cypher rejects `//` comments and is pathologically slow on
        # unbounded variable-length `[*]` traversals, so we don't compute connected
        # components here. Per the docstring, "disconnected" = isolated nodes (no
        # incident edges) — a cheap single-hop degree check returning the cognee
        # `id` property (not the internal vertex id).
        query = (
            'MATCH (n:"__Node__") '
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS deg "
            "WHERE deg = 0 "
            "RETURN COLLECT(n.id) AS ids"
        )
        results = await self._read(query)
        return results[0]["ids"] if results else []

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

            results = await self._read(
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

            results = await self._read(
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

            results = await self._read(
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

            results = await self._read(
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
        query = """
            MATCH (n: {BASE_LABEL} {{id: %(node_id)s}})-[r]-(m: {BASE_LABEL})
            RETURN DISTINCT m
        """
        results = await self._read(
            sql.SQL(query).format(BASE_LABEL=sql.Identifier(BASE_LABEL)),
            {"node_id": Jsonb(node_id)},
        )
        return [result["m"] for result in results]

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
        results = await self._read(
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
        results = await self._read(
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
            self._read(
                sql.SQL(predecessors_query).format(
                    BASE_LABEL=sql.Identifier(BASE_LABEL)
                ), {"node_id": Jsonb(str(node_id))}
            ),
            self._read(
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

        return await self._write(
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

        return await self._write(
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

        return await self._write(
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
        nodes = await self._read(query_nodes)

        query_edges = "MATCH (n)-[r]->(m) RETURN collect([properties(n), properties(r), properties(m)]) AS edges"
        edges = await self._read(query_edges)

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

        result = await self._read(query)
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
        result = await self._read(query)
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
        # Run the node and edge selection as two queries (like `get_graph_data`).
        # A single combined query — collect the node set, then `MATCH (a)-[r]-(b)`
        # among it — plans on AgensGraph as a scan over every edge in the graph and
        # hangs at scale (and AgensGraph won't let an UNWIND-bound vertex anchor a
        # later MATCH). Split apart, each part is index-backed: the node match is
        # anchored on the indexed `name`, and the edge match uses the `id` index on
        # both endpoints.

        # 1) the named nodes of the requested type plus their 1-hop neighbours.
        nodes_query = """
        UNWIND %(names)s AS wantedName
        MATCH (n:"__Node__" {name: wantedName})
        WHERE n.labels @> %(label)s
        OPTIONAL MATCH (n)-[]-(nbr)
        RETURN collect(DISTINCT properties(n)) AS centers,
               collect(DISTINCT properties(nbr)) AS nbrs
        """
        result = await self._read(
            nodes_query, {"names": Jsonb(node_name), "label": Jsonb(node_type.__name__)}
        )
        if not result:
            return [], []

        by_id = {}
        for prop in (result[0]["centers"] or []) + (result[0]["nbrs"] or []):
            if prop:
                by_id[prop["id"]] = prop
        if not by_id:
            return [], []
        node_ids = list(by_id)

        # 2) the edges among that node set (index lookup on both endpoint ids).
        edges_query = """
        MATCH (a:"__Node__")-[r]->(b:"__Node__")
        WHERE a.id IN %(ids)s AND b.id IN %(ids)s
        RETURN TYPE(r) AS type, properties(r) AS properties
        """
        edge_result = await self._read(edges_query, {"ids": Jsonb(node_ids)})

        nodes = [(prop["id"], prop) for prop in by_id.values()]
        edges = [
            (
                record["properties"]["source_node_id"],
                record["properties"]["target_node_id"],
                record["type"],
                record["properties"],
            )
            for record in edge_result
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
        result_nodes = await self._read(query_nodes)

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
        result_edges = await self._read(query_edges)

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
        result = await self._read(query, {"graph_name": graph_name})
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
        node_labels_result = await self._read(node_labels_query)
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
        relationship_types_result = await self._read(relationship_types_query)
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
        """Drop the graph and everything in it."""
        await self._write(
            sql.SQL("DROP GRAPH IF EXISTS {} CASCADE").format(sql.Identifier(graph_name))
        )
        if graph_name == self.graph_name:
            self._engine.forget_graph()

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
        result = await self._read(query, {"content_hash": Jsonb(content_hash)})
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
        result = await self._read(query)
        return [record["n"] for record in result] if result else []
