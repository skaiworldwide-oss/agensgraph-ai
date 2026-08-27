"""AgensGraph graph adapter for cognee."""

import asyncio
import hashlib
import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type
from uuid import UUID

import psycopg
from agensgraph import (
    DesiredIndex,
    DesiredLabel,
    Edge,
    GraphId,
    Path,
    RetryPolicy,
    Unique,
    Vertex,
    to_builtins,
)
from agensgraph.cypher import check_single_statement
from agensgraph.errors import safe_message
from agensgraph.introspect import MAX_IDENTIFIER
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import ERROR, get_logger
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row, tuple_row
from psycopg.types.json import Jsonb

from ._engine import AgensEngine
from .metrics import graph_metrics

logger = get_logger("AgensgraphAdapter", level=ERROR)

# Every node is on a label named after its DataPoint class, and every such label is a
# child of this one, so a match on it reads every node.
#
# Label names are lower case. Cypher folds an unquoted identifier to lower case, so
# MATCH (e:Entity) looks for a label named entity; a language model writing Cypher for
# cognee's natural-language search, and most people, write labels unquoted. A class
# name is kept as written in the type property, which is what cognee reads.
BASE_LABEL = "__node__"

# The classes cognee's own queries name. Declared up front so those queries never meet
# a label that does not exist yet.
KNOWN_CLASSES = (
    "TextDocument",
    "PdfDocument",
    "DocumentChunk",
    "Entity",
    "EntityType",
    "TextSummary",
    "NodeSet",
)

# Rows per statement for UNWIND-bound lists.
CHUNK_SIZE = 1000

# How many recently written node ids are remembered with their label. An edge whose
# endpoints were written recently is matched on their labels directly, one index probe
# per endpoint, instead of through the parent label, which probes every child label.
REMEMBERED_IDS = 200_000

# Up to this many values are written out as an OR of equalities. From here on one bound
# list is used. Writing values out is faster for a few (15.1 ms against 22.8 ms for one)
# and grows by about 14 ms per value; a bound list stays flat.
OR_CHAIN_LIMIT = 20

_PROPERTY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

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


def bounded_name(*parts: str) -> str:
    """A name that fits in an identifier and stays distinct after truncation.

    An identifier is 63 bytes. A DataPoint class name can be longer, and the server
    truncates a longer label name instead of refusing it, so two names that agree for
    63 bytes would become one label. A name that fits is used as it is; a longer one
    keeps what fits plus a hash of the whole.
    """
    name = "_".join(parts)
    encoded = name.encode()
    if len(encoded) <= MAX_IDENTIFIER:
        return name
    digest = hashlib.blake2b(encoded, digest_size=8).hexdigest()
    head = encoded[: MAX_IDENTIFIER - len(digest) - 1].decode("utf-8", "ignore")
    return f"{head}_{digest}"


def label_for(name: str) -> str:
    """The label a DataPoint class, or a relationship, is stored on."""
    return bounded_name(name).lower()


KNOWN_LABELS = tuple(label_for(c) for c in KNOWN_CLASSES)


def _unique_id_name(label: str) -> str:
    return bounded_name(label, "unique_id")


def _name_index_name(label: str) -> str:
    return bounded_name(label, "name_idx")


class AgensgraphAdapter(GraphDBInterface):
    """
    Cognee's graph store on AgensGraph.

    Each DataPoint class is a label under ``__Node__``. Every label carries a uniqueness
    constraint on ``id`` and an index on ``name``, which is what cognee looks nodes up by.
    Edges are on a label named after the relationship and carry ``source_node_id``,
    ``target_node_id`` and ``relationship_name`` as properties.
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
        # id -> label of nodes this process wrote recently, oldest first.
        self._recent: "OrderedDict[str, str]" = OrderedDict()
        self._edge_labels: Set[str] = set()

    def _remember(self, node_id: str, label: str) -> None:
        recent = self._recent
        if node_id in recent:
            recent.move_to_end(node_id)
        recent[node_id] = label
        while len(recent) > REMEMBERED_IDS:
            recent.popitem(last=False)

    # ---- lifecycle ----

    async def initialize(self):
        """Connect once and create what the graph needs. Cheap when called again."""
        self._engine = AgensEngine.get(self.conninfo, graph_name=self.graph_name)
        await self._engine.setup_once(f"graph:{self.graph_name}", self._bootstrap)
        self.graph_id = self._engine.graph_id

    async def _bootstrap(self, conn) -> None:
        await self._declare_on(conn, [BASE_LABEL, *KNOWN_LABELS])
        self._engine.labels = await self._established_labels(conn)

    async def finalize(self):
        """Close this event loop's pool. The next call opens a new one."""
        if self._engine is not None:
            await self._engine.aclose()

    # ---- labels ----

    async def _established_labels(self, conn) -> Set[str]:
        """The node labels that exist and carry their own uniqueness on id."""
        labels = await conn.labels(graph=self.graph_name)
        constraints = await conn.constraints(graph=self.graph_name)
        unique = {c.label for c in constraints if c.unique}
        return {
            label.name
            for label in labels
            if label.kind == "v" and label.parent == BASE_LABEL and label.name in unique
        }

    async def _declare_on(self, conn, labels: List[str]) -> None:
        """Create the labels, their constraints and their indexes, one writer at a time.

        Reconciling reads what exists and then creates what is missing, which is two
        steps; the lock makes concurrent writers take them one after another. It lasts
        as long as the transaction.
        """
        children = [label for label in labels if label != BASE_LABEL]
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (int(self._engine.graph_id),))
            desired = []
            if BASE_LABEL in labels:
                desired.append(DesiredLabel(BASE_LABEL, "v"))
            desired.extend(DesiredLabel(label, "v", BASE_LABEL) for label in children)
            await conn.ensure_labels(desired, graph=self.graph_name)
            # A constraint on the parent does not reach a child, so every label has one.
            await conn.ensure_constraints(
                [Unique(label, "id", _unique_id_name(label)) for label in labels],
                graph=self.graph_name,
            )
            await conn.ensure_indexes(
                [DesiredIndex(label, ("name",), name=_name_index_name(label)) for label in labels],
                graph=self.graph_name,
            )

    async def _ensure_labels(self, labels: Iterable[str]) -> None:
        wanted = sorted(set(labels) - self._engine.labels)
        if not wanted:
            return
        async with self._engine.connection() as conn:
            await self._declare_on(conn, wanted)
            self._engine.labels = await self._established_labels(conn)

    async def _label_exists(self, label: str) -> bool:
        if label in self._engine.labels:
            return True
        async with self._engine.connection() as conn:
            self._engine.labels = await self._established_labels(conn)
        return label in self._engine.labels

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

    async def _read(self, query, params=None, *, by_index: bool = False) -> List[Dict[str, Any]]:
        async def attempt():
            async with self._engine.connection() as conn:
                return await self._execute(conn, query, params, by_index=by_index)

        return await self._run_with_retry(attempt, wrote=False)

    async def _read_raw(self, query, params=None) -> List[tuple]:
        """Rows as tuples, with no conversion of graph values.

        For the reads that return only property maps and ids. Converting each row
        costs about 4 µs, which is 400 ms over the 100,000 rows of a whole-graph read.
        """

        async def attempt():
            async with self._engine.connection() as conn:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(query, params)
                    return await cur.fetchall() if cur.description is not None else []

        return await self._run_with_retry(attempt, wrote=False)

    async def _read_raw_by_index(self, query, params=None) -> List[tuple]:
        """Tuple rows for a statement that matches nodes from a bound list."""

        async def attempt():
            async with self._engine.connection() as conn:
                async with conn.transaction():
                    async with conn.pipeline():
                        await conn.execute("SET LOCAL enable_seqscan = off")
                        cur = conn.cursor(row_factory=tuple_row)
                        await cur.execute(query, params)
                    rows = await cur.fetchall() if cur.description is not None else []
                    await cur.close()
                    return rows

        return await self._run_with_retry(attempt, wrote=False)

    async def _write(self, query, params=None, *, by_index: bool = False) -> List[Dict[str, Any]]:
        async def attempt():
            async with self._engine.connection() as conn:
                return await self._execute(conn, query, params, by_index=by_index)

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
    async def _execute(conn, query, params, *, by_index: bool = False) -> List[Dict[str, Any]]:
        """Run one statement and convert its rows.

        ``by_index`` is for a statement that matches nodes from a bound list. The
        planner sizes the list at 100 rows and, past a few thousand nodes, joins it by
        hashing every node's property map instead of probing the unique index once per
        value; that read is what the index exists to avoid. The setting is local to the
        transaction and the four statements go out in one flush.
        """
        if not by_index:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, params)
                if cur.description is None:
                    return []
                rows = await cur.fetchall()
            return [AgensgraphAdapter._convert_row(row) for row in rows]
        async with conn.transaction():
            async with conn.pipeline():
                await conn.execute("SET LOCAL enable_seqscan = off")
                cur = conn.cursor(row_factory=dict_row)
                await cur.execute(query, params)
            rows = await cur.fetchall() if cur.description is not None else []
            await cur.close()
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

    @staticmethod
    def _equalities(field: str, values: List[Any], key: str) -> Tuple[str, str, Dict[str, Any]]:
        """A predicate ``field = one of values`` and the parameters it binds.

        Returns ``(prelude, predicate, params)``. A few values are written out as an OR
        of equalities; more become one UNWIND-bound list in the prelude. ``IN`` against a
        bound list is not used: it is jsonb containment and never uses an index.
        """
        if len(values) <= OR_CHAIN_LIMIT:
            params = {f"{key}_{i}": Jsonb(v) for i, v in enumerate(values)}
            predicate = " OR ".join(f"{field} = %({name})s" for name in params)
            return "", f"({predicate})", params
        return (
            f"UNWIND %({key})s AS {key}_value ",
            f"{field} = {key}_value",
            {key: Jsonb(list(values))},
        )

    @staticmethod
    def _sql_table(graph: str, table: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(graph), sql.Identifier(table))

    # ---- nodes ----

    async def has_node(self, node_id: str) -> bool:
        results = await self._read(
            sql.SQL(
                "MATCH (n:{base} {{id: %(node_id)s}}) RETURN count(n) > 0 AS node_exists"
            ).format(base=sql.Identifier(BASE_LABEL)),
            {"node_id": Jsonb(str(node_id))},
        )
        return bool(results[0]["node_exists"]) if results else False

    async def add_node(self, node: DataPoint):
        return await self.add_nodes([node])

    @record_graph_changes
    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        """Write nodes, each on the label of its class, merging on id.

        One node per id. cognee derives an id from a name, so an Entity and an EntityType
        with the same name share an id; cognee's own adapters keep one node for it and
        take the class from the last write. So a node whose id already exists on another
        label is updated where it is, and its type property records the class written.
        """
        rows_by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for node in nodes:
            node_id = str(node.id)
            rows_by_id[node_id] = (
                label_for(type(node).__name__),
                self.serialize_properties(node.model_dump()),
            )
        if not rows_by_id:
            return
        # Where do these ids live already? Remembered ids cost nothing; the rest are
        # looked up in one statement per 1,000 through the labels' unique indexes.
        placed: Dict[str, str] = {}
        unknown = []
        for node_id in rows_by_id:
            label = self._recent.get(node_id)
            if label is None:
                unknown.append(node_id)
            else:
                placed[node_id] = label
        if unknown:
            statement = sql.SQL(
                "UNWIND %(ids)s AS wanted MATCH (n:{base} {{id: wanted}}) RETURN wanted, label(n)"
            ).format(base=sql.Identifier(BASE_LABEL))
            for start in range(0, len(unknown), CHUNK_SIZE):
                rows = await self._read_raw_by_index(
                    statement, {"ids": Jsonb(unknown[start : start + CHUNK_SIZE])}
                )
                for node_id, label in rows:
                    placed[node_id] = label

        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for node_id, (label, props) in rows_by_id.items():
            target = placed.get(node_id, label)
            by_label.setdefault(target, []).append({"id": node_id, "props": props})
            self._remember(node_id, target)
        await self._ensure_labels(by_label)
        for label, rows in by_label.items():
            statement = sql.SQL(
                "UNWIND %(rows)s AS row "
                "MERGE (n:{label} {{id: row.id}}) "
                "SET n += row.props"
            ).format(label=sql.Identifier(label))
            for start in range(0, len(rows), CHUNK_SIZE):
                await self._write(
                    statement, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])}, by_index=True
                )

    async def extract_node(self, node_id: str):
        results = await self.extract_nodes([node_id])
        return results[0] if results else None

    async def extract_nodes(self, node_ids: List[str]):
        return await self.get_nodes(node_ids)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        results = await self._read(
            sql.SQL("MATCH (node:{base} {{id: %(node_id)s}}) RETURN node").format(
                base=sql.Identifier(BASE_LABEL)
            ),
            {"node_id": Jsonb(str(node_id))},
        )
        return results[0]["node"] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
        results = await self._read(
            sql.SQL(
                "UNWIND %(node_ids)s AS wanted MATCH (node:{base} {{id: wanted}}) RETURN node"
            ).format(base=sql.Identifier(BASE_LABEL)),
            {"node_ids": Jsonb([str(i) for i in node_ids])},
            by_index=True,
        )
        return [result["node"] for result in results]

    async def delete_node(self, node_id: str):
        return await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: list[str]) -> None:
        if not node_ids:
            return
        ids = [str(i) for i in node_ids]
        statement = sql.SQL(
            "UNWIND %(node_ids)s AS wanted MATCH (node:{base} {{id: wanted}}) DETACH DELETE node"
        ).format(base=sql.Identifier(BASE_LABEL))
        for start in range(0, len(ids), CHUNK_SIZE):
            await self._write(
                statement, {"node_ids": Jsonb(ids[start : start + CHUNK_SIZE])}, by_index=True
            )

    # ---- reads around one node ----
    #
    # A pattern with two nodes matched through the parent label joins two sets of child
    # labels; the planner estimates tens of thousands of rows for a ten-row answer and
    # runs it in parallel workers, about 10 ms. Anchoring the node in Cypher and reading
    # its edges in SQL by graphid takes under 1 ms: a graphid names its table, and the
    # edge tables carry indexes on both endpoints.

    ANCHOR = 'SELECT gid FROM (MATCH (n:"' + BASE_LABEL + '" {{id: {param}}}) RETURN id(n) AS gid) t'

    def _around(self, node_param: str, relationship: Optional[str], select: str) -> sql.Composed:
        """SQL for the edges at one node in both directions, joined to both endpoints.

        ``select`` sees ``e`` (the edge), ``s`` and ``t`` (start and end vertex) and
        ``a.gid`` (the anchor's graphid).
        """
        anchor = sql.SQL(self.ANCHOR.format(param=f"%({node_param})s"))
        edges = self._sql_table(
            self.graph_name, relationship if relationship is not None else "ag_edge"
        )
        vertices = self._sql_table(self.graph_name, "ag_vertex")
        return sql.SQL(
            "WITH a AS ({anchor}) "
            "SELECT {select} FROM a, {edges} e "
            "JOIN {vertices} s ON s.id = e.start JOIN {vertices} t ON t.id = e.\"end\" "
            "WHERE e.start = a.gid "
            "UNION ALL "
            "SELECT {select} FROM a, {edges} e "
            "JOIN {vertices} s ON s.id = e.start JOIN {vertices} t ON t.id = e.\"end\" "
            "WHERE e.\"end\" = a.gid AND e.start <> a.gid"
        ).format(anchor=anchor, select=sql.SQL(select), edges=edges, vertices=vertices)

    @staticmethod
    def _relationship_of(row_label: str, properties: Dict[str, Any]) -> str:
        return properties.get("relationship_name") or row_label.split(".")[-1].strip('"')

    # ---- edges ----

    async def has_edge(self, from_node: UUID, to_node: UUID, edge_label: str) -> bool:
        if not await self._label_exists_any(label_for(edge_label)):
            return False
        rows = await self._read_raw(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM {edges} e WHERE e.start = ({a}) AND e.\"end\" = ({b}))"
            ).format(
                edges=self._sql_table(self.graph_name, label_for(edge_label)),
                a=sql.SQL(self.ANCHOR.format(param="%(from_node)s")),
                b=sql.SQL(self.ANCHOR.format(param="%(to_node)s")),
            ),
            {"from_node": Jsonb(str(from_node)), "to_node": Jsonb(str(to_node))},
        )
        return bool(rows[0][0]) if rows else False

    async def _label_exists_any(self, label: str) -> bool:
        """Whether a label of either kind exists; an edge label appears when first written."""
        if label in self._engine.labels or label in self._edge_labels:
            return True
        async with self._engine.connection() as conn:
            self._edge_labels = {
                lab.name for lab in await conn.labels(graph=self.graph_name) if lab.kind == "e"
            }
        return label in self._edge_labels

    async def has_edges(self, edges) -> List[Tuple[str, str, str]]:
        """The edges in ``edges`` that exist, as ``(from_id, to_id, relationship_name)``.

        The ids are resolved to graphids in one statement, then the edge tables are read
        by endpoint pairs through their indexes.
        """
        if not edges:
            return []
        wanted = [(str(e[0]), str(e[1]), e[2]) for e in edges]
        ids = sorted({i for pair in wanted for i in pair[:2]})
        gid_of: Dict[str, GraphId] = {}
        statement = sql.SQL(
            "UNWIND %(ids)s AS wanted MATCH (n:{base} {{id: wanted}}) RETURN wanted, id(n) AS gid"
        ).format(base=sql.Identifier(BASE_LABEL))
        for start in range(0, len(ids), CHUNK_SIZE):
            rows = await self._read_raw_by_index(statement, {"ids": Jsonb(ids[start : start + CHUNK_SIZE])})
            for wanted_id, gid in rows:
                gid_of[wanted_id] = self._graphid(gid)
        pairs = [(gid_of[f], gid_of[t], rel) for f, t, rel in wanted if f in gid_of and t in gid_of]
        if not pairs:
            return []
        rows = await self._read_raw(
            sql.SQL(
                "SELECT w.f, w.t, w.rel FROM unnest(%(f)s, %(t)s, %(rel)s::text[]) AS w(f, t, rel) "
                "WHERE EXISTS (SELECT 1 FROM {edges} e WHERE e.start = w.f AND e.\"end\" = w.t "
                "AND (e.properties->>'relationship_name' = w.rel OR e.tableoid::regclass::text = "
                "quote_ident(%(graph)s) || '.' || quote_ident(lower(w.rel))))"
            ).format(edges=self._sql_table(self.graph_name, "ag_edge")),
            {"f": [p[0] for p in pairs], "t": [p[1] for p in pairs], "rel": [p[2] for p in pairs],
             "graph": self.graph_name},
        )
        id_of = {gid: node_id for node_id, gid in gid_of.items()}
        return [(id_of[self._graphid(f)], id_of[self._graphid(t)], rel) for f, t, rel in rows]

    async def add_edge(
        self,
        from_node: UUID,
        to_node: UUID,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = None,
    ):
        return await self.add_edges([(from_node, to_node, relationship_name, edge_properties or {})])

    @record_graph_changes
    async def add_edges(self, edges: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """Write edges, merging on the endpoints and the relationship.

        Every edge carries ``source_node_id``, ``target_node_id`` and ``relationship_name``
        as properties; the whole-graph read relies on that.
        """
        if not edges:
            return
        # Grouped by relationship and by the labels of the endpoints when both were
        # written by this process recently; otherwise the endpoints are found through
        # the parent label.
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        recent = self._recent
        for source, target, relationship_name, props in edges:
            from_node, to_node = str(source), str(target)
            a_label = recent.get(from_node, BASE_LABEL)
            b_label = recent.get(to_node, BASE_LABEL)
            groups.setdefault((relationship_name, a_label, b_label), []).append(
                {
                    "from_node": from_node,
                    "to_node": to_node,
                    "properties": self.serialize_properties(
                        {
                            **(props or {}),
                            "source_node_id": from_node,
                            "target_node_id": to_node,
                            "relationship_name": relationship_name,
                        }
                    ),
                }
            )
        for (relationship_name, a_label, b_label), rows in groups.items():
            self._edge_labels.add(label_for(relationship_name))
            statement = sql.SQL(
                "UNWIND %(rows)s AS row "
                "MATCH (a:{a_label} {{id: row.from_node}}), (b:{b_label} {{id: row.to_node}}) "
                "MERGE (a)-[r:{rel}]->(b) "
                "SET r += row.properties"
            ).format(
                a_label=sql.Identifier(a_label),
                b_label=sql.Identifier(b_label),
                rel=sql.Identifier(label_for(relationship_name)),
            )
            for start in range(0, len(rows), CHUNK_SIZE):
                await self._write(
                    statement, {"rows": Jsonb(rows[start : start + CHUNK_SIZE])}, by_index=True
                )

    async def get_edges(self, node_id: str):
        rows = await self._read_raw(
            self._around("node_id", None, "s.properties->>'id', t.properties->>'id', "
                         "e.tableoid::regclass::text, e.properties"),
            {"node_id": Jsonb(str(node_id))},
        )
        return [
            (s_id, t_id, {"relationship_name": self._relationship_of(label, props)})
            for s_id, t_id, label, props in rows
        ]

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        rows = await self._read_raw(
            self._around("node_id", None,
                         "CASE WHEN e.start = a.gid THEN t.id ELSE s.id END, "
                         "CASE WHEN e.start = a.gid THEN t.properties ELSE s.properties END"),
            {"node_id": Jsonb(str(node_id))},
        )
        seen: Dict[Any, Dict[str, Any]] = {}
        for gid, props in rows:
            seen.setdefault(gid, props)
        return list(seen.values())

    async def _one_direction(self, node_id: str, edge_label: Optional[str], incoming: bool) -> list:
        anchor = sql.SQL(self.ANCHOR.format(param="%(node_id)s"))
        edges = self._sql_table(
            self.graph_name, label_for(edge_label) if edge_label is not None else "ag_edge"
        )
        vertices = self._sql_table(self.graph_name, "ag_vertex")
        here, there = ('e."end"', "e.start") if incoming else ("e.start", 'e."end"')
        statement = sql.SQL(
            "WITH a AS ({anchor}) SELECT v.properties FROM a, {edges} e "
            "JOIN {vertices} v ON v.id = " + there + " WHERE " + here + " = a.gid"
        ).format(anchor=anchor, edges=edges, vertices=vertices)
        rows = await self._read_raw(statement, {"node_id": Jsonb(str(node_id))})
        return [p for (p,) in rows]

    async def get_predecessors(self, node_id: str, edge_label: str = None) -> list:
        return await self._one_direction(node_id, edge_label, incoming=True)

    async def get_successors(self, node_id: str, edge_label: str = None) -> list:
        return await self._one_direction(node_id, edge_label, incoming=False)

    async def get_connections(self, node_id: UUID) -> list:
        """Every edge at the node, as ``(start properties, {relationship_name}, end properties)``."""
        rows = await self._read_raw(
            self._around("node_id", None, "s.properties, e.tableoid::regclass::text, e.properties, t.properties"),
            {"node_id": Jsonb(str(node_id))},
        )
        return [
            (s_props, {"relationship_name": self._relationship_of(label, e_props)}, t_props)
            for s_props, label, e_props, t_props in rows
        ]

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        await self._write(
            sql.SQL(
                "UNWIND %(node_ids)s AS wanted "
                "MATCH (node:{base} {{id: wanted}})-[r:{rel}]->(:{base}) DELETE r"
            ).format(base=sql.Identifier(BASE_LABEL), rel=sql.Identifier(label_for(edge_label))),
            {"node_ids": Jsonb([str(i) for i in node_ids])},
            by_index=True,
        )

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        await self._write(
            sql.SQL(
                "UNWIND %(node_ids)s AS wanted "
                "MATCH (node:{base} {{id: wanted}})<-[r:{rel}]-(:{base}) DELETE r"
            ).format(base=sql.Identifier(BASE_LABEL), rel=sql.Identifier(label_for(edge_label))),
            {"node_ids": Jsonb([str(i) for i in node_ids])},
            by_index=True,
        )

    # ---- the graph as a whole ----

    async def delete_graph(self):
        """Delete every node and edge. The labels stay."""
        await self._write(
            sql.SQL("MATCH (node:{base}) DETACH DELETE node").format(base=sql.Identifier(BASE_LABEL))
        )

    def serialize_properties(self, properties=None):
        serialized = {}
        for key, value in (properties or {}).items():
            serialized[key] = str(value) if isinstance(value, UUID) else value
        return serialized

    async def get_model_independent_graph_data(self):
        nodes = await self._read(
            sql.SQL("MATCH (n:{base}) RETURN collect(properties(n)) AS nodes").format(
                base=sql.Identifier(BASE_LABEL)
            )
        )
        edges = await self._read(
            sql.SQL(
                "MATCH (n:{base})-[r]->(m:{base}) "
                "RETURN collect([properties(n), properties(r), properties(m)]) AS edges"
            ).format(base=sql.Identifier(BASE_LABEL))
        )
        return (nodes, edges)

    async def project_entire_graph(self, graph_name="cognee"):
        logger.warning("Agensgraph does not support in-memory graph projection.")

    async def get_graph_data(self):
        """Every node as ``(id, properties)`` and every edge as
        ``(source_id, target_id, relationship_name, properties)``.

        Read as Cypher; reading the label tables in SQL measured the same (1,337 ms
        against 1,404 ms on 26,564 nodes and 72,944 edges), so the graph form stays.
        """
        node_rows = await self._read_raw(
            sql.SQL("MATCH (n:{base}) RETURN properties(n)").format(base=sql.Identifier(BASE_LABEL))
        )
        nodes = self._one_per_id((p["id"], p) for (p,) in node_rows if "id" in p)

        edge_rows = await self._read_raw("MATCH ()-[r]->() RETURN label(r), properties(r)")
        edges = []
        skipped = 0
        for kind, p in edge_rows:
            if "source_node_id" in p and "target_node_id" in p:
                edges.append((p["source_node_id"], p["target_node_id"], kind, p))
            else:
                skipped += 1
        if skipped:
            logger.warning("%d edges without endpoint ids in their properties were left out", skipped)
        return (nodes, edges)

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """The named nodes of one class, their neighbours, and the edges among them.

        The names are matched on the class's label through its name index. From there
        everything is read by graphid: the nodes through their tables' primary keys and
        the edges through the edge tables' endpoint indexes. A graphid names its table,
        so neither read touches any other label.
        """
        label = label_for(node_type.__name__)
        if not node_name or not await self._label_exists(label):
            return [], []

        prelude, predicate, params = self._equalities("n.name", node_name, "name")
        result = await self._read(
            sql.SQL(
                prelude + "MATCH (n:{label}) WHERE " + predicate + " "
                "OPTIONAL MATCH (n)-[]-(nbr) "
                "RETURN collect(DISTINCT id(n)) AS centers, collect(DISTINCT id(nbr)) AS nbrs"
            ).format(label=sql.Identifier(label)),
            params,
        )
        if not result:
            return [], []
        graphids = {
            self._graphid(g) for g in (result[0]["centers"] or []) + (result[0]["nbrs"] or []) if g
        }
        if not graphids:
            return [], []
        gids = list(graphids)

        node_rows = await self._read_raw(
            sql.SQL("SELECT properties FROM {vertices} WHERE id = ANY(%(gids)s)").format(
                vertices=self._sql_table(self.graph_name, "ag_vertex")
            ),
            {"gids": gids},
        )
        nodes = self._one_per_id((p["id"], p) for (p,) in node_rows if "id" in p)

        edge_rows = await self._read_raw(
            sql.SQL(
                "SELECT properties FROM {edges} "
                'WHERE start = ANY(%(gids)s) AND "end" = ANY(%(gids)s)'
            ).format(edges=self._sql_table(self.graph_name, "ag_edge")),
            {"gids": gids},
        )
        edges = [
            (p["source_node_id"], p["target_node_id"], p.get("relationship_name"), p)
            for (p,) in edge_rows
            if "source_node_id" in p and "target_node_id" in p
        ]
        return nodes, edges

    @staticmethod
    def _one_per_id(nodes: Iterable[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
        """Keep the first node for each id; cognee's in-memory graph refuses a second."""
        seen: Dict[str, dict] = {}
        dropped = 0
        for node_id, props in nodes:
            if node_id in seen:
                dropped += 1
            else:
                seen[node_id] = props
        if dropped:
            logger.warning("%d nodes shared an id with another node and were left out", dropped)
        return list(seen.items())

    @staticmethod
    def _graphid(value: Any) -> GraphId:
        if isinstance(value, GraphId):
            return value
        labid, locid = str(value).split(".")
        return GraphId(labid=int(labid), locid=int(locid))

    async def get_filtered_graph_data(self, attribute_filters):
        """Nodes whose attributes take one of the given values, and the edges among them."""
        predicates = []
        params: Dict[str, Any] = {}
        preludes = []
        for i, (attribute, values) in enumerate(attribute_filters[0].items()):
            if not _PROPERTY_NAME.match(attribute):
                raise ValueError(f"not a property name: {attribute!r}")
            for alias in ("n", "m"):
                prelude, predicate, bound = self._equalities(
                    f"{alias}.{attribute}", list(values), f"{alias}_f{i}"
                )
                if alias == "n":
                    preludes.append(prelude)
                    n_pred = predicate
                else:
                    preludes.append(prelude)
                    m_pred = predicate
                params.update(bound)
            predicates.append((n_pred, m_pred))
        n_where = " AND ".join(p[0] for p in predicates)
        m_where = " AND ".join(p[1] for p in predicates)
        n_prelude = "".join(p for p in preludes[0::2])
        m_prelude = "".join(p for p in preludes[1::2])

        node_rows = await self._read(
            sql.SQL(
                n_prelude + "MATCH (n:{base}) WHERE " + n_where + " RETURN properties(n) AS properties"
            ).format(base=sql.Identifier(BASE_LABEL)),
            {k: v for k, v in params.items() if k.startswith("n_")},
        )
        nodes = [(row["properties"]["id"], row["properties"]) for row in node_rows]

        edge_rows = await self._read(
            sql.SQL(
                n_prelude + m_prelude + "MATCH (n:{base})-[r]->(m:{base}) "
                "WHERE " + n_where + " AND " + m_where + " "
                "RETURN label(r) AS type, properties(r) AS properties"
            ).format(base=sql.Identifier(BASE_LABEL)),
            params,
        )
        edges = [
            (p["source_node_id"], p["target_node_id"], row["type"], p)
            for row in edge_rows
            for p in (row["properties"],)
        ]
        return (nodes, edges)

    async def graph_exists(self, graph_name="cognee"):
        result = await self._read(
            "SELECT 1 FROM ag_graph WHERE graphname = %(graph_name)s", {"graph_name": graph_name}
        )
        return len(result) > 0

    async def drop_graph(self, graph_name="cognee"):
        """Drop the graph and everything in it."""
        await self._write(
            sql.SQL("DROP GRAPH IF EXISTS {} CASCADE").format(sql.Identifier(graph_name))
        )
        if graph_name == self.graph_name:
            self._engine.forget_graph()

    async def get_graph_metrics(self, include_optional=False):
        """Counts and structure of the graph; see :func:`metrics.graph_metrics`."""
        return await graph_metrics(self, include_optional=include_optional)

    # ---- what cognee's delete reads ----

    async def get_document_subgraph(self, content_hash: str):
        """A document, its chunks, and what only this document brought in.

        Entities, summaries and types that another document also uses stay out of the
        result, so deleting this document does not take them away.
        """
        statement = sql.SQL(
            "MATCH (doc:{base} {{name: %(name)s}}) "
            "WHERE label(doc) IN [{text_doc}, {pdf_doc}] "
            "OPTIONAL MATCH (doc)<-[:is_part_of]-(chunk:{chunk}) "
            "OPTIONAL MATCH (chunk)-[:contains]->(entity:{entity}) "
            "WHERE NOT (SELECT EXISTS ("
            "  MATCH (entity)<-[:contains]-(oc:{chunk})-[:is_part_of]->(od:{base}) "
            "  WHERE label(od) IN [{text_doc}, {pdf_doc}] AND od.id <> doc.id "
            "  RETURN 1)) "
            "OPTIONAL MATCH (chunk)<-[:made_from]-(made_node:{summary}) "
            "OPTIONAL MATCH (entity)-[:is_a]->(type:{etype}) "
            "WHERE NOT (SELECT EXISTS ("
            "  MATCH (type)<-[:is_a]-(oe:{entity})<-[:contains]-(oc2:{chunk})-[:is_part_of]->(od2:{base}) "
            "  WHERE label(od2) IN [{text_doc}, {pdf_doc}] AND od2.id <> doc.id "
            "  RETURN 1)) "
            "RETURN collect(DISTINCT properties(doc)) AS document, "
            "collect(DISTINCT properties(chunk)) AS chunks, "
            "collect(DISTINCT properties(entity)) AS orphan_entities, "
            "collect(DISTINCT properties(made_node)) AS made_from_nodes, "
            "collect(DISTINCT properties(type)) AS orphan_types"
        ).format(
            base=sql.Identifier(BASE_LABEL),
            chunk=sql.Identifier(label_for("DocumentChunk")),
            entity=sql.Identifier(label_for("Entity")),
            summary=sql.Identifier(label_for("TextSummary")),
            etype=sql.Identifier(label_for("EntityType")),
            text_doc=sql.Literal(label_for("TextDocument")),
            pdf_doc=sql.Literal(label_for("PdfDocument")),
        )
        result = await self._read(statement, {"name": Jsonb(f"text_{content_hash}")})
        if not result or not result[0]["document"]:
            return None
        return {key: [p for p in value if p] for key, value in result[0].items()}

    async def get_degree_one_nodes(self, node_type: str):
        """Nodes of ``node_type`` with exactly one edge."""
        if node_type not in ("Entity", "EntityType"):
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")
        # Plain SQL over the label's table: the edge tables carry indexes on start and
        # on end, so the two counts are two index probes per node.
        rows = await self._read_raw(
            sql.SQL(
                "SELECT v.properties FROM {label} v "
                "WHERE (SELECT count(*) FROM {edges} e WHERE e.start = v.id) "
                "    + (SELECT count(*) FROM {edges} e WHERE e.\"end\" = v.id) = 1"
            ).format(
                label=self._sql_table(self.graph_name, label_for(node_type)),
                edges=self._sql_table(self.graph_name, "ag_edge"),
            )
        )
        return [p for (p,) in rows]

    async def get_disconnected_nodes(self) -> list[str]:
        """The ids of nodes with no edge at all."""
        rows = await self._read_raw(
            sql.SQL(
                "SELECT v.properties->>'id' AS id FROM {vertices} v "
                "WHERE NOT EXISTS (SELECT 1 FROM {edges} e WHERE e.start = v.id) "
                "  AND NOT EXISTS (SELECT 1 FROM {edges} e WHERE e.\"end\" = v.id)"
            ).format(
                vertices=self._sql_table(self.graph_name, "ag_vertex"),
                edges=self._sql_table(self.graph_name, "ag_edge"),
            )
        )
        return [node_id for (node_id,) in rows if node_id is not None]
