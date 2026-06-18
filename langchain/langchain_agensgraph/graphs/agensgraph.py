'''
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
'''

from __future__ import annotations

import json
import re
import time
from contextlib import asynccontextmanager, contextmanager
from hashlib import md5
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from langchain_agensgraph.engine import AgensEngine

from langchain_agensgraph.graphs.graph_document import GraphDocument
from langchain_agensgraph.graphs.graph_store import GraphStore
from functools import wraps

import psycopg
from psycopg import sql

typeof_function = r"""
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

node_properties_query = f"""
    MATCH (a)
    UNWIND keys(properties(a)) AS prop
    WITH label(a) as label, prop, properties(a)[prop] AS value
    WHERE value IS NOT NULL
    WITH
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values
    RETURN label, COLLECT(DISTINCT {{'property': property, type: typeof(values[0])}}) as props;
"""

edge_properties_query = f"""
    MATCH ()-[e]->()
    WITH type(e) as label, properties(e) as properties
    UNWIND keys(properties) AS prop
    WITH label, prop, properties[prop] AS value
    WHERE value IS NOT NULL
    WITH
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values
    RETURN label, COLLECT(DISTINCT {{'property': property, type: typeof(values[0])}}) as props;
"""

triple_query = f"""
    MATCH (start_node)-[r]->(end_node)
    WITH labels(start_node) AS startlbls, type(r) AS relationship_type, labels(end_node) AS endlbls
    UNWIND startlbls AS start_label
    UNWIND endlbls AS end_label
    RETURN DISTINCT {{start: start_label, type: relationship_type, end: end_label}} AS output;
"""

LIST_LIMIT = 128
"""List-valued result properties longer than this are dropped when sanitize=True."""


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("langchain-agensgraph")
        except PackageNotFoundError:
            return "dev"
    except Exception:
        return "dev"


def _with_application_name(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``conf`` with ``application_name`` set if not provided."""
    out = dict(conf)
    if not out.get("application_name"):
        out["application_name"] = f"langchain-agensgraph/{_package_version()}"
    return out


def _sanitize_value(value: Any, *, list_limit: int = LIST_LIMIT) -> Any:
    """Recursively drop oversized lists from a result value.

    A list with more than
    ``list_limit`` elements is removed entirely (returned as ``None`` at the
    leaf, or omitted from a dict), so embeddings and other large arrays don't
    end up serialized into an LLM prompt.
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            sv = _sanitize_value(v, list_limit=list_limit)
            if sv is not None or v is None:
                out[k] = sv
        return out
    if isinstance(value, list):
        if len(value) > list_limit:
            return None
        cleaned = [_sanitize_value(v, list_limit=list_limit) for v in value]
        return cleaned
    return value


def require_psycopg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psycopg
        except ImportError:
            raise ImportError(
                "Could not import psycopg python package. "
                "Please install it with `pip install psycopg`."
            )
        return func(*args, **kwargs)
    return wrapper

def execute_query(cursor, query, params = {}, error_message = "Error executing graph query"):
    try:
        cursor.execute(query, params)
    except psycopg.Error as e:
        raise AgensQueryException(
            {
                "message": error_message,
                "detail": str(e),
            }
        )

class AgensQueryException(Exception):
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


class AgensGraph(GraphStore):
    """
    Agensgraph wrapper for graph operations.

    Args:
        graph_name (str): the name of the graph to connect to or create
        conf (Dict[str, Any]): the pgsql connection config passed directly
            to psycopg.connect
        create (bool): if True and graph doesn't exist, attempt to create it

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    # precompiled regex for checking chars in graph labels and
    # identifying record as vertex or edge
    label_regex: Pattern = re.compile("[^0-9a-zA-Z]+")
    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

    @require_psycopg
    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        create: bool = False,
        schema_cache_ttl: float = 0.0,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        engine: Optional["AgensEngine"] = None,
    ) -> None:
        """Create a new Agensgraph Graph instance.

        Args:
            graph_name: Name of the AgensGraph graph to use.
            conf: psycopg connection kwargs.
            create: Create the graph if it does not exist.
            schema_cache_ttl: When > 0, ``refresh_schema`` short-circuits if it
                ran within the last ``schema_cache_ttl`` seconds. Set to 0
                (default) to disable caching.
            timeout: Default per-query statement timeout in seconds. ``None``
                disables it. Can be overridden per call via ``query(..., timeout=)``.
            sanitize: When True, list-valued result properties longer than
                ``LIST_LIMIT`` (128) are stripped from query results so large
                arrays (e.g. embeddings) do not flood an LLM context.
            engine: Optional :class:`~langchain_agensgraph.engine.AgensEngine`.
                When provided, ``query``/``aquery`` borrow pooled connections so
                concurrent callers don't serialize on one connection. A dedicated
                connection is still kept for init, schema introspection, and
                transactional ``add_graph_documents``. When ``None`` the behavior
                is identical to a single dedicated connection.
        """

        self.graph_name = graph_name
        self._engine = engine
        # Keep the conf for lazy async-connection creation (see ``_aconn_get``).
        # Tag the connection so it is identifiable in pg_stat_activity.
        self._conf = _with_application_name(conf)
        self.connection = psycopg.connect(**self._conf)
        self.schema_cache_ttl = schema_cache_ttl
        self.timeout = timeout
        self.sanitize = sanitize
        self._schema_refreshed_at: float = 0.0
        self._server_version: Optional[Tuple[int, int, int]] = None
        self._has_meta_extension: bool = False
        self._aconn: Optional[psycopg.AsyncConnection] = None

        with self._get_cursor() as curs:
            # check if graph with name graph_name exists
            graph_id_query = "SELECT oid as graphid FROM ag_graph WHERE graphname = %(graphname)s;"
            params = {"graphname": graph_name}

            execute_query(curs, graph_id_query, params, "Error checking for existing graph")
            data = curs.fetchone()

            # if graph doesn't exist and create is True, create it
            if data is None:
                if create:
                    execute_query(curs, sql.SQL('CREATE GRAPH IF NOT EXISTS {graphname}').format(
                        graphname=sql.Identifier(graph_name)), error_message="Error creating graph")
                    self.connection.commit()
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                execute_query(curs, graph_id_query, params, "Error fetching graph id after creation")
                data = curs.fetchone()

            # store graph id and refresh the schema
            self.graphid = data.graphid


            # set the graph path to the current graph and declare some functions
            execute_query(curs, sql.SQL('SET graph_path = {graphname}').format(
                graphname=sql.Identifier(graph_name)), error_message="Error setting graph path")
            execute_query(curs, typeof_function)
            self.connection.commit()

        self._detect_capabilities()
        self.refresh_schema()

    @require_psycopg
    def _detect_capabilities(self) -> None:
        """Probe AgensGraph version and the ``meta`` extension.

        Sets ``self._server_version`` to a 3-tuple (major, minor, patch) and
        ``self._has_meta_extension`` to a bool. Both default to a safe
        ``(0,0,0)`` / ``False`` if the probe itself fails — so the integration
        always falls back to behaviors that work on older AgensGraph builds.
        """
        try:
            rows = self.query("SELECT version() AS v")
            ver_str = rows[0]["v"] if rows else ""
            match = re.search(r"AgensGraph\s+(\d+)\.(\d+)\.(\d+)", ver_str)
            if match:
                self._server_version = (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
        except Exception:
            pass
        try:
            rows = self.query(
                "SELECT 1 AS ok FROM pg_extension WHERE extname = 'meta'"
            )
            self._has_meta_extension = bool(rows)
        except Exception:
            self._has_meta_extension = False

    @require_psycopg
    def _meta_vertex_labels(self) -> List[str]:
        """List vertex labels for ``self.graph_name`` via the ``meta`` extension.

        Falls back to scanning ``ag_label`` if the extension isn't installed.
        """
        if self._has_meta_extension:
            rows = self.query(
                "SELECT label_name FROM meta.vertex_labels(%(g)s::name)",
                {"g": self.graph_name},
            )
            return [r["label_name"] for r in rows]
        # Fallback for older AgensGraph: read the catalog directly.
        rows = self.query(
            "SELECT l.labname AS label_name FROM ag_label l "
            "JOIN ag_graph g ON l.graphid = g.oid "
            "WHERE g.graphname = %(g)s::name AND l.labkind = 'v'",
            {"g": self.graph_name},
        )
        return [r["label_name"] for r in rows]

    @require_psycopg
    def _meta_edge_labels(self) -> List[str]:
        """List edge labels via the ``meta`` extension or catalog fallback."""
        if self._has_meta_extension:
            rows = self.query(
                "SELECT label_name FROM meta.edge_labels(%(g)s::name)",
                {"g": self.graph_name},
            )
            return [r["label_name"] for r in rows]
        rows = self.query(
            "SELECT l.labname AS label_name FROM ag_label l "
            "JOIN ag_graph g ON l.graphid = g.oid "
            "WHERE g.graphname = %(g)s::name AND l.labkind = 'e'",
            {"g": self.graph_name},
        )
        return [r["label_name"] for r in rows]

    @require_psycopg
    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self.connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    @contextmanager
    def _acquire(self) -> "Iterator[psycopg.Connection]":
        """Yield the connection ``query`` should run on.

        Uses a pooled connection from the engine when one is configured and we
        are not inside a wrapping transaction (``add_graph_documents``); falls
        back to the dedicated connection otherwise. The fallback path is
        byte-for-byte the pre-engine behavior.
        """
        if self._engine is not None and not getattr(self, "_in_transaction", False):
            with self._engine.connection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            yield self.connection

    @asynccontextmanager
    async def _aacquire(self) -> "AsyncIterator[psycopg.AsyncConnection]":
        """Async sibling of :meth:`_acquire`."""
        if self._engine is not None:
            async with self._engine.aconnection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            yield await self._aconn_get()

    @require_psycopg
    def _get_triples(self) -> List[Dict[str, str]]:
        """
        Get a set of distinct relationship types (as a list of dicts) in the graph
        to be used as context by an llm.

        Returns:
            List[Dict[str, str]]: relationships as a list of dicts in the format
                "{'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}"
        """

        triple_schema = []
        with self._get_cursor() as curs:
            execute_query(curs, triple_query)
            rows = curs.fetchall()
            triple_schema = [row.output for row in rows]
        
        return triple_schema

    @staticmethod
    def _format_triples(triples: List[Dict[str, str]]) -> List[str]:
        """
        Convert a list of relationships from dictionaries to formatted strings
        to be better readable by an llm

        Args:
            triples (List[Dict[str,str]]): a list relationships in the form
                {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

        Returns:
            List[str]: a list of relationships in the form
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """
        triple_template = '(:"{start}")-[:"{type}"]->(:"{end}")'
        triple_schema = [triple_template.format(**triple) for triple in triples]

        return triple_schema

    @require_psycopg
    def _get_node_properties(self) -> List[Dict[str, Any]]:
        """
        Fetch a list of available node properties by node label to be used
        as context for an llm

        Args:
            n_labels (List[str]): a list of node labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of node labels and
                their corresponding properties in the form
                "{
                    'labels': <node_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """

        node_properties = []
        with self._get_cursor() as curs:
            execute_query(curs, node_properties_query)
            rows = curs.fetchall()
            
            for row in rows:
                node_properties.append(
                    {
                        "labels": row.label,
                        "properties": row.props
                    }
                )

        return node_properties

    @require_psycopg
    def _get_edge_properties(self) -> List[Dict[str, Any]]:
        """
        Fetch a list of available edge properties by edge label to be used
        as context for an llm

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of edge labels
                and their corresponding properties in the form
                "{
                    'labels': <edge_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """
        edge_properties = []
        with self._get_cursor() as curs:
            execute_query(curs, edge_properties_query)
            rows = curs.fetchall()
            
            for row in rows:
                edge_properties.append(
                    {
                        "type": row.label,
                        "properties": row.props
                    }
                )

        return edge_properties

    def refresh_schema(self, *, force: bool = False) -> None:
        """Refresh the graph schema information.

        When ``schema_cache_ttl > 0`` and a refresh happened more recently than
        that, this is a no-op unless ``force=True``. Useful for read-mostly
        applications where the schema changes infrequently.
        """

        if (
            not force
            and self.schema_cache_ttl > 0
            and (time.monotonic() - self._schema_refreshed_at) < self.schema_cache_ttl
        ):
            return

        # fetch graph schema information
        node_properties = self._get_node_properties()
        edge_properties = self._get_edge_properties()
        triple_schema = self._get_triples()

        # update the formatted string representation
        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {self._format_triples(triple_schema)}
        """

        # update the dictionary representation
        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in edge_properties},
            "relationships": triple_schema,
            "metadata": {
                "agensgraph_version": self._server_version,
                "has_meta_extension": self._has_meta_extension,
            },
        }
        self._schema_refreshed_at = time.monotonic()

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

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
                vertex = AgensGraph.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensGraph.vertex_regex.match(v)
                edge = AgensGraph.edge_regex.match(v)

                if vertex:
                    d[k] = json.loads(vertex.group(3))

                # convert edge from id-label->id by replacing id with node information
                # we only do this if the vertex was also returned in the query
                # resolve the edge endpoints to their node property maps
                elif edge:
                    elabel, edge_id, start_id, end_id, properties = edge.groups()
                    d[k] = (
                        vertices.get(start_id, {}),
                        elabel,
                        vertices.get(end_id, {}),
                    )
                else:
                    try:
                        d[k] = json.loads(v)
                    except json.JSONDecodeError:
                        d[k] = v

            else:
                d[k] = v

        return d

    @require_psycopg
    def query(
        self,
        query: str,
        params: dict = {},
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and
        converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query
            timeout (Optional[float]): statement timeout in seconds for this
                call. Falls back to the instance ``timeout``. ``None`` disables.

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        # execute the query, rolling back on an error
        in_txn = getattr(self, "_in_transaction", False)
        effective_timeout = timeout if timeout is not None else self.timeout
        with self._acquire() as conn:
            with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    if effective_timeout is not None:
                        # SET LOCAL is scoped to the current (implicit) transaction
                        # and auto-resets on commit/rollback.
                        curs.execute(
                            "SET LOCAL statement_timeout = %s",
                            (int(effective_timeout * 1000),),
                        )
                    curs.execute(query, params)
                    if not in_txn:
                        conn.commit()
                except psycopg.Error as e:
                    if not in_txn:
                        conn.rollback()
                    raise AgensQueryException(
                        {
                            "message": "Error executing graph query: {}".format(query),
                            "detail": str(e),
                        }
                    )
                try:
                    data = curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []  # Handle queries that don’t return data

                if data is None:
                    result = []
                # convert to dictionaries
                else:
                    result = [self._record_to_dict(d) for d in data]

                if self.sanitize:
                    result = [_sanitize_value(row) for row in result]

                return result

    # ---------- v0.2.0: async surface ----------

    async def _aconn_get(self) -> psycopg.AsyncConnection:
        """Return a lazily-opened ``AsyncConnection`` bound to ``graph_path``.

        Built on demand so users that never call any ``a*`` method do not
        pay the cost of a second TCP/socket connection.
        """
        if self._aconn is None or self._aconn.closed:
            self._aconn = await psycopg.AsyncConnection.connect(**self._conf)
            async with self._aconn.cursor() as cur:
                await cur.execute(
                    sql.SQL("SET graph_path = {n}").format(
                        n=sql.Identifier(self.graph_name)
                    )
                )
            await self._aconn.commit()
        return self._aconn

    async def aquery(
        self, query: str, params: dict = {}, timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`query`.

        Uses a pooled connection when an engine is configured, otherwise a
        dedicated lazily-opened ``AsyncConnection``.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        async with self._aacquire() as conn:
            async with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as cur:
                try:
                    if effective_timeout is not None:
                        await cur.execute(
                            "SET LOCAL statement_timeout = %s",
                            (int(effective_timeout * 1000),),
                        )
                    await cur.execute(query, params)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensQueryException(
                        {
                            "message": "Error executing graph query: {}".format(query),
                            "detail": str(e),
                        }
                    )
                try:
                    data = await cur.fetchall()
                except psycopg.ProgrammingError:
                    data = []
                result = [self._record_to_dict(d) for d in (data or [])]
                if self.sanitize:
                    result = [_sanitize_value(row) for row in result]
                return result

    async def aclose(self) -> None:
        """Close the async connection (if any). Safe to call multiple times."""
        if self._aconn is not None and not self._aconn.closed:
            await self._aconn.close()
            self._aconn = None

    def close(self) -> None:
        """Close the sync connection. Idempotent.

        Note: if an async connection was opened it should be closed with
        :meth:`aclose` from within an event loop; ``close`` only closes the
        synchronous connection.
        """
        if getattr(self, "connection", None) is not None and not self.connection.closed:
            self.connection.close()

    def __enter__(self) -> "AgensGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "AgensGraph":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
        self.close()

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
        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace("'", "\\'")

        props = []
        # wrap property key in double quotes to escape
        for k, v in properties.items():
            if isinstance(v, str):
                prop = f'"{k}": \'{_esc(v)}\''
            else:
                prop = f'"{k}": {v}'
            props.append(prop)
        if id is not None and "id" not in properties:
            if isinstance(id, str):
                props.append(f"id: '{_esc(id)}'")
            else:
                props.append(f"id: {id}")
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def clean_graph_labels(label: str) -> str:
        """
        remove any disallowed characters from a label and replace with '_'

        Args:
            label (str): the original label

        Returns:
            str: the sanitized version of the label
        """
        return re.sub(AgensGraph.label_regex, "_", label)

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        insert a list of graph documents into the graph

        Args:
            graph_documents (List[GraphDocument]): the list of documents to be inserted
            include_source (bool): if True add nodes for the sources
                with MENTIONS edges to the entities they mention

        Returns:
            None

        All statements for one ``add_graph_documents`` call commit atomically:
        if any insert (label DDL, node, edge) fails partway through, the entire
        batch is rolled back so the graph never holds orphan nodes/edges.
        """
        # query for inserting nodes
        node_insert_query = (
            """
            MERGE (n:{label} {{id: %(id)s}})
            SET n = %(properties)s;
            """
            if not include_source
            else """
            MERGE (n:{label} %(properties)s)
            MERGE (d:{doc_label} %(d_properties)s)
            MERGE (d)-[:"MENTIONS"]->(n)
        """
        )

        # query for inserting edges
        edge_insert_query = """
            MERGE ("from":{f_label} %(f_properties)s)
            MERGE ("to":{t_label} %(t_properties)s)
            MERGE ("from")-[:{r_label} %(r_properties)s]->("to")
        """
        # iterate docs and insert them — wrapped in a single transaction so a
        # failure halfway through can never leave orphan nodes behind.
        self._in_transaction = True
        try:
            with self.connection.transaction():
                self._add_graph_documents_inner(
                    graph_documents, include_source,
                    node_insert_query, edge_insert_query,
                )
        finally:
            self._in_transaction = False
        return

    def _add_graph_documents_inner(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool,
        node_insert_query: str,
        edge_insert_query: str,
    ) -> None:
        for doc in graph_documents:
            # if we are adding sources, create an id for the source
            if include_source:
                if not doc.source.metadata.get("id"):
                    doc.source.metadata["id"] = md5(
                        doc.source.page_content.encode("utf-8")
                    ).hexdigest()

            # insert entity nodes
            for node in doc.nodes:
                # Ensure that the label used in merge exists (due to bug in agensgraph)
                self.query(sql.SQL('CREATE VLABEL IF NOT EXISTS {label}').format(
                    label=sql.Identifier(AgensGraph.clean_graph_labels(node.type))
                ))
                if include_source:
                    self.query(sql.SQL('CREATE VLABEL IF NOT EXISTS {label}').format(
                        label=sql.Identifier(AgensGraph.clean_graph_labels(doc.source.type))
                    ))
                    self.query(sql.SQL('CREATE ELABEL IF NOT EXISTS {label}').format(
                        label=sql.Identifier("MENTIONS")
                    ))

                node.properties["id"] = node.id
                if include_source:
                    query = sql.SQL(node_insert_query).format(
                        label=sql.Identifier(AgensGraph.clean_graph_labels(node.type)),
                        doc_label=sql.Identifier(AgensGraph.clean_graph_labels(doc.source.type)),
                    )
                    params = {
                        'properties': json.dumps(node.properties),
                        'd_properties': json.dumps(doc.source.metadata),
                        'id': json.dumps(node.id)
                    }
                else:
                    query = sql.SQL(node_insert_query).format(
                        label=sql.Identifier(AgensGraph.clean_graph_labels(node.type))
                    )
                    params = {
                        'properties': json.dumps(node.properties),
                        'id': json.dumps(node.id)
                    }
                self.query(query, params)

            # insert relationships
            for edge in doc.relationships:

                edge.source.properties["id"] = edge.source.id
                edge.target.properties["id"] = edge.target.id
                inputs = {
                    "f_label": AgensGraph.clean_graph_labels(edge.source.type),
                    "f_properties": json.dumps(edge.source.properties),
                    "t_label": AgensGraph.clean_graph_labels(edge.target.type),
                    "t_properties": json.dumps(edge.target.properties),
                    "r_label": AgensGraph.clean_graph_labels(edge.type).upper(),
                    "r_properties": json.dumps(edge.properties),
                }

                # Ensure that the label used in merge exists (due to bug in agensgraph)
                self.query(sql.SQL('CREATE VLABEL IF NOT EXISTS {f_label}').format(
                    f_label=sql.Identifier(inputs["f_label"])
                ))
                self.query(sql.SQL('CREATE VLABEL IF NOT EXISTS {t_label}').format(
                    t_label=sql.Identifier(inputs["t_label"])
                ))
                self.query(sql.SQL('CREATE ELABEL IF NOT EXISTS {r_label}').format(
                    r_label=sql.Identifier(inputs["r_label"])
                ))

                query = sql.SQL(edge_insert_query).format(
                    f_label=sql.Identifier(inputs["f_label"]),
                    t_label=sql.Identifier(inputs["t_label"]),
                    r_label=sql.Identifier(inputs["r_label"]),
                )
                params = {
                    'f_properties': inputs["f_properties"],
                    't_properties': inputs["t_properties"],
                    'r_properties': inputs["r_properties"]
                }
                self.query(query, params)
