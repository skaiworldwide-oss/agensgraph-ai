"""
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import agensgraph
import psycopg
from agensgraph import (
    AsyncConnection,
    Capabilities,
    Connection,
    Edge,
    Jsonb,
    Path,
    SparseVector,
    Vector,
    Vertex,
)
from agensgraph.errors import CapabilityError
from agensgraph.untrusted import check_untrusted

from .._core.component import Component
from .._core.engine import AgensEngine, acquire
from .._core.names import bounded, checked, companion_schema, ident, printed, qualified
from .._core.program import Call, InTransaction, Pipeline, Program, Statement, mappings, scalar
from .._core.settings import NESTED_LOOPS
from .._core.setup import Setup, graph_exists, made_graph
from ..errors import AgensQueryError, DuplicateKeys, UnsafeCypherError
from .graph_document import GraphDocument

LIST_LIMIT = 128
DEFAULT_SCHEMA_SAMPLE = 100
ENHANCED_SCHEMA_SAMPLE = 1000
BASE_ENTITY_LABEL = "__Entity__"
SOURCE_EDGE = "MENTIONS"

# Each label's planner estimate, summed over the label and every label inheriting from it, as a
# match on the label reads them all; a table reached by two paths of inheritance is summed once.
ESTIMATES = (
    "WITH RECURSIVE labels AS (SELECT l.labname::text AS name, l.labkind::text AS kind, l.relid "
    "FROM ag_label l JOIN ag_graph g ON g.oid = l.graphid WHERE g.graphname = %s), "
    "tree (name, relid) AS (SELECT name, relid FROM labels UNION "
    "SELECT t.name, i.inhrelid FROM tree t JOIN pg_inherits i ON i.inhparent = t.relid) "
    "SELECT l.name, l.kind, sum(greatest(c.reltuples, 0))::bigint FROM labels l "
    "JOIN tree t ON t.name = l.name JOIN pg_class c ON c.oid = t.relid GROUP BY l.name, l.kind"
)

# The most rows counted for a label with no estimate, so a large label not analysed yet costs a
# bounded read and shows at least this many until it is.
COUNT_LIMIT = 10_000


def _sanitize_value(value: Any, *, list_limit: int = LIST_LIMIT) -> Any:
    """A result value with every list longer than ``list_limit`` dropped, key and all.

    Embeddings and other large arrays would otherwise be written into an LLM's prompt. A graph
    element is described by its identity, label and properties, and only the properties, the one
    part that can hold a large list, are walked.
    """
    if isinstance(value, Vertex | Edge):
        return _sanitize_value(agensgraph.to_builtins(value), list_limit=list_limit)
    if isinstance(value, Path):
        return {
            "vertices": [_sanitize_value(v, list_limit=list_limit) for v in value.vertices],
            "edges": [_sanitize_value(e, list_limit=list_limit) for e in value.edges],
        }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            kept = _sanitize_value(item, list_limit=list_limit)
            if kept is not None or item is None:
                out[key] = kept
        return out
    if isinstance(value, Vector):
        # A vector is not a list, though it compares equal to one, so the rule below misses it.
        return None if len(value) > list_limit else list(value)
    if isinstance(value, SparseVector):
        return None if len(value) > list_limit else value.to_dict()
    if isinstance(value, list):
        if len(value) > list_limit:
            return None
        return [_sanitize_value(item, list_limit=list_limit) for item in value]
    return value


class AgensGraph(Component):
    """A graph for LangChain: Cypher in, rows out, and a schema to show a model.

    ``engine`` is an :class:`AgensEngine` or a conninfo string (sharing one engine per string).
    Nothing is sent when the graph is made. The graph must exist unless ``create_graph=True``.

    ``query`` runs Cypher the application wrote; ``query_untrusted`` runs Cypher a model wrote,
    through the driver's ``run_untrusted``, on the engine's ``untrusted`` connections. The schema
    is read on first use, or at construction with ``refresh_schema=True``, and kept for
    ``schema_cache_ttl`` seconds; its counts are the planner's estimates.
    """

    _what = "graph"

    def __init__(
        self,
        engine: AgensEngine | str,
        *,
        graph_name: str,
        create_graph: bool = False,
        setup: bool = True,
        schema_cache_ttl: float = 60.0,
        timeout: float | None = None,
        sanitize: bool = False,
        enhanced_schema: bool = False,
        include_indexes: bool = True,
        refresh_schema: bool = False,
    ) -> None:
        self._engine, self._owns_engine = acquire(engine)
        self.graph_name = checked(graph_name, "graph name")
        self.create_graph = create_graph
        self.schema_cache_ttl = schema_cache_ttl
        self.timeout = timeout
        self.sanitize = sanitize
        self.enhanced_schema = enhanced_schema
        self.include_indexes = include_indexes
        self._setup_wanted = setup
        self.schema = ""
        self.structured_schema: dict[str, Any] = {}
        self._schema_read_at: float | None = None
        self._ensured: set[tuple[Any, ...]] = set()
        self._setup = Setup(
            self._engine,
            self.graph_name,
            lambda: graph_exists(self.graph_name),
            lambda: made_graph(self.graph_name, create=self.create_graph),
            key="graph",
            forgotten=self._forgotten,
        )
        if refresh_schema:
            self.refresh_schema()

    # -- what it is ----------------------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return self._engine.capabilities

    async def acapabilities(self) -> Capabilities:
        return await self._engine.acapabilities()

    # -- setup ---------------------------------------------------------------------------------

    def setup(self) -> None:
        """Make the graph if asked to and check it exists, now rather than on the first call."""
        self._ensure()

    async def asetup(self) -> None:
        await self._aensure()

    def _ensure(self) -> None:
        self._open()
        if self._setup_wanted:
            self._setup.ensure()

    async def _aensure(self) -> None:
        self._open()
        if self._setup_wanted:
            await self._setup.aensure()

    # -- the caller's Cypher -------------------------------------------------------------------

    def query(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run a statement and return its rows as mappings of column to value.

        Values are the driver's: a vertex is a :class:`agensgraph.Vertex`, a path a
        :class:`agensgraph.Path`; with ``sanitize=True`` they are plain mappings with long lists
        dropped. A refusal from the server raises :class:`AgensQueryError`.
        """
        self._ensure()
        try:
            return self._engine.run(
                lambda: self._query(query, params, timeout), graph=self.graph_name
            )
        except psycopg.Error as exc:
            raise _query_error(exc) from exc

    async def aquery(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        await self._aensure()
        try:
            return await self._engine.arun(
                lambda: self._query(query, params, timeout), graph=self.graph_name
            )
        except psycopg.Error as exc:
            raise _query_error(exc) from exc

    def query_many(
        self, statements: Sequence[tuple[str, Sequence[Any] | Mapping[str, Any] | None] | str]
    ) -> list[list[dict[str, Any]]]:
        """Run statements in one round trip; their rows, statement by statement.

        Outside :meth:`transaction` they share one implicit transaction: all apply or none.
        """
        self._ensure()
        try:
            return self._engine.run(lambda: self._many(statements), graph=self.graph_name)
        except psycopg.Error as exc:
            raise _query_error(exc) from exc

    async def aquery_many(
        self, statements: Sequence[tuple[str, Sequence[Any] | Mapping[str, Any] | None] | str]
    ) -> list[list[dict[str, Any]]]:
        await self._aensure()
        try:
            return await self._engine.arun(lambda: self._many(statements), graph=self.graph_name)
        except psycopg.Error as exc:
            raise _query_error(exc) from exc

    def _query(
        self, text: str, params: Any, timeout: float | None
    ) -> Program[list[dict[str, Any]]]:
        budget = timeout if timeout is not None else self.timeout
        settings = None if budget is None else {"statement_timeout": int(budget * 1000)}
        result = yield Statement(text, params, settings=settings)
        return self._rows(result)

    def _many(
        self, statements: Sequence[tuple[str, Any] | str]
    ) -> Program[list[list[dict[str, Any]]]]:
        steps = tuple(
            Statement(item) if isinstance(item, str) else Statement(item[0], item[1])
            for item in statements
        )
        if not steps:
            return []
        results = yield Pipeline(steps)
        return [self._rows(result) for result in results]

    def _rows(self, result: Any) -> list[dict[str, Any]]:
        rows = mappings(result)
        return [_sanitize_value(row) for row in rows] if self.sanitize else rows

    @contextmanager
    def transaction(self) -> Iterator[Connection[Any]]:
        """A transaction that every call through this graph's engine joins until the block ends."""
        with self._engine.transaction() as conn:
            yield conn

    @asynccontextmanager
    async def atransaction(self) -> AsyncIterator[AsyncConnection[Any]]:
        async with self._engine.atransaction() as conn:
            yield conn

    # -- a model's Cypher ----------------------------------------------------------------------

    def query_untrusted(
        self,
        text: str,
        *,
        max_rows: int = 100,
        timeout: float = 10.0,
        allow_writes: bool = False,
        unsafe_shared_role: bool = False,
    ) -> list[dict[str, Any]]:
        """Run Cypher a model wrote, and nothing it could not have meant to.

        The driver reads the text first and refuses anything that is not plainly one Cypher read,
        then runs it in a read-only transaction of its own that is always rolled back, reading at
        most ``max_rows`` rows, as a role that is not the application's (the engine's
        ``untrusted_conninfo``). ``allow_writes`` runs Cypher writes as the engine's
        ``writer_conninfo`` role, on 2.16 and later, and is never retried.

        Text the reading refuses raises :class:`UnsafeCypherError`, before a connection is taken;
        a refusal from the server raises :class:`AgensQueryError`.
        """
        _readable(text, max_rows, allow_writes)
        self._ensure()
        if allow_writes:
            _writes_checked(self.capabilities)
        role = self._engine.main_role
        try:
            return self._engine.run(
                lambda: self._untrusted(
                    text, max_rows, timeout, allow_writes, unsafe_shared_role, role
                ),
                graph=self.graph_name,
                kind="writer" if allow_writes else "untrusted",
            )
        except psycopg.Error as exc:
            if exc.sqlstate is None:
                raise
            raise _query_error(exc) from exc

    async def aquery_untrusted(
        self,
        text: str,
        *,
        max_rows: int = 100,
        timeout: float = 10.0,
        allow_writes: bool = False,
        unsafe_shared_role: bool = False,
    ) -> list[dict[str, Any]]:
        _readable(text, max_rows, allow_writes)
        await self._aensure()
        if allow_writes:
            _writes_checked(await self.acapabilities())
        role = await self._engine.amain_role()
        try:
            return await self._engine.arun(
                lambda: self._untrusted(
                    text, max_rows, timeout, allow_writes, unsafe_shared_role, role
                ),
                graph=self.graph_name,
                kind="writer" if allow_writes else "untrusted",
            )
        except psycopg.Error as exc:
            if exc.sqlstate is None:
                raise
            raise _query_error(exc) from exc

    def _untrusted(
        self,
        text: str,
        max_rows: int,
        timeout: float,
        allow_writes: bool,
        unsafe_shared_role: bool,
        app_role: str,
    ) -> Program[list[dict[str, Any]]]:
        answer = yield Call(
            "run_untrusted",
            (text,),
            {
                "max_rows": max_rows,
                "timeout": timeout,
                "allow_writes": allow_writes,
                "app_role": app_role,
                "schemas": [self.graph_name, companion_schema(self.graph_name)],
                "unsafe_shared_role": unsafe_shared_role,
            },
        )
        rows = [dict(zip(answer.columns, row, strict=True)) for row in answer.rows]
        return [_sanitize_value(row) for row in rows] if self.sanitize else rows

    # -- the schema ----------------------------------------------------------------------------

    def refresh_schema(self, *, force: bool = False) -> None:
        """Read the schema again, unless it was read within ``schema_cache_ttl`` seconds.

        Labels, their properties (from a sample of each label), which labels each edge label
        joins, the planner's estimate of each label's size and, with ``include_indexes``, the
        property indexes a query can use.
        """
        if not force and self._fresh():
            return
        self._ensure()
        description, estimates, exact, indexes = self._engine.run(
            self._describe, replay=True, graph=self.graph_name
        )
        examples: list[list[dict[str, Any]]] = []
        if self.enhanced_schema:
            examples = self._engine.run(
                lambda: self._examples(description), graph=self.graph_name, replay=True
            )
        self._build(description, estimates, exact, indexes, examples)

    async def arefresh_schema(self, *, force: bool = False) -> None:
        if not force and self._fresh():
            return
        await self._aensure()
        description, estimates, exact, indexes = await self._engine.arun(
            self._describe, replay=True, graph=self.graph_name
        )
        examples: list[list[dict[str, Any]]] = []
        if self.enhanced_schema:
            examples = await self._engine.arun(
                lambda: self._examples(description), graph=self.graph_name, replay=True
            )
        self._build(description, estimates, exact, indexes, examples)

    def _fresh(self) -> bool:
        return (
            self._schema_read_at is not None
            and self.schema_cache_ttl > 0
            and time.monotonic() - self._schema_read_at < self.schema_cache_ttl
        )

    @property
    def get_schema(self) -> str:
        """The schema as the text a prompt carries, read first if it never has been."""
        if self._schema_read_at is None:
            self.refresh_schema()
        return self.schema

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """The schema as a mapping: ``node_props``, ``rel_props``, ``relationships``, ``counts``,
        ``indexes``, ``metadata``."""
        if self._schema_read_at is None:
            self.refresh_schema()
        return self.structured_schema

    def _describe(self) -> Program[tuple[Any, dict[str, tuple[str, int]], dict[str, int], Any]]:
        sample = ENHANCED_SCHEMA_SAMPLE if self.enhanced_schema else DEFAULT_SCHEMA_SAMPLE
        # The triple catalog is filled by a gather, which the driver runs only when the server
        # says the catalog is not current, so a graph that has not changed pays a read.
        description = yield Call(
            "describe",
            kwargs={"sample": sample, "refresh": True, "counts": False, "graph": self.graph_name},
        )
        found = yield Statement(ESTIMATES, (self.graph_name,))
        estimates = {str(name): (str(kind), int(rows)) for name, kind, rows in found.records}
        # A label never analysed, or analysed while empty, has no estimate worth showing; it is
        # counted instead, up to COUNT_LIMIT rows.
        unknown = [
            name
            for name, (_, rows) in estimates.items()
            if rows <= 0 and not name.startswith("ag_")
        ]
        exact: dict[str, int] = {}
        if unknown:
            counted = yield Pipeline(
                tuple(
                    Statement(
                        "SELECT count(*) FROM (SELECT FROM "
                        f"{qualified(self.graph_name, name)} LIMIT {COUNT_LIMIT}) AS kept"
                    )
                    for name in unknown
                )
            )
            exact = {
                name: int(scalar(result)) for name, result in zip(unknown, counted, strict=True)
            }
        indexes = None
        if self.include_indexes:
            indexes = yield Call("indexes", kwargs={"graph": self.graph_name})
        return description, estimates, exact, indexes

    def _examples(self, description: Any) -> Program[list[list[dict[str, Any]]]]:
        labels = [label for label in description.labels if not label.is_builtin]
        if not labels:
            return []
        statements = tuple(
            Statement(
                f"MATCH ()-[n:{ident(label.name)}]->() WITH n LIMIT %s RETURN properties(n)"
                if label.kind == "e"
                else f"MATCH (n:{ident(label.name)}) WITH n LIMIT %s RETURN properties(n)",
                (25,),
            )
            for label in labels
        )
        results = yield Pipeline(statements)
        return [[row[0] or {} for row in result.records] for result in results]

    def _build(
        self,
        description: Any,
        estimates: dict[str, tuple[str, int]],
        exact: dict[str, int],
        indexes: Any,
        examples: list[list[dict[str, Any]]],
    ) -> None:
        counts = {
            name: exact.get(name, rows)
            for name, (_, rows) in estimates.items()
            if not name.startswith("ag_")
        }
        kinds = {label.name: label.kind for label in description.labels}
        node_props: dict[str, list[dict[str, Any]]] = {}
        rel_props: dict[str, list[dict[str, Any]]] = {}
        for label, shapes in description.properties.items():
            # A label that holds nothing is left out: it exists in the catalogs, but telling a
            # model about it invites a query that matches nothing.
            if counts.get(label, 0) <= 0:
                continue
            entries = [{"property": shape.name, "type": shape.kind} for shape in shapes]
            (rel_props if kinds.get(label) == "e" else node_props)[label] = entries
        if examples:
            sampled = [label for label in description.labels if not label.is_builtin]
            for label, rows in zip(sampled, examples, strict=True):
                section = rel_props if label.kind == "e" else node_props
                for entry in section.get(label.name, []):
                    seen: list[Any] = []
                    for row in rows:
                        value = _sanitize_value(row.get(entry["property"]))
                        if value is not None and value not in seen and len(seen) < 3:
                            seen.append(value)
                    if seen:
                        entry["examples"] = seen
        structured: dict[str, Any] = {
            "node_props": node_props,
            "rel_props": rel_props,
            "relationships": [
                {"start": t.start, "type": t.edge, "end": t.end}
                for t in description.triples
                if counts.get(t.edge, 0) > 0
            ],
            "counts": {name: rows for name, rows in counts.items() if rows > 0},
            "indexes": _index_facts(indexes) if indexes is not None else [],
            "metadata": {
                "agensgraph_version": self.capabilities.reported,
                "meta_gathered": description.meta_gathered,
            },
        }
        self.structured_schema = structured
        self.schema = _rendered(structured)
        self._schema_read_at = time.monotonic()

    # -- writing graph documents ---------------------------------------------------------------

    def add_graph_documents(
        self,
        graph_documents: Sequence[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,  # noqa: N803 - LangChain's spelling
        *,
        batch_size: int = 100,
        dedupe_edges: bool = True,
    ) -> None:
        """Write the nodes and relationships of graph documents, merging on node ids.

        A node is a vertex of its type's label, found again by its ``id``; a relationship is an
        edge of its type, found again by its two ends, so writing a document twice changes
        nothing the second time. ``include_source`` writes each source document as a
        ``Document`` vertex with a ``MENTIONS`` edge to every node it yielded;
        ``baseEntityLabel`` makes every node label inherit ``__Entity__``, so
        ``MATCH (n:"__Entity__")`` finds them all. ``dedupe_edges`` keeps one edge per pair and
        type with a unique index, so concurrent writers cannot duplicate one; pass False for a
        graph that holds several edges of a type between one pair.

        Labels and indexes are made first, under the graph's schema lock; then each batch of
        ``batch_size`` documents is one round trip and one implicit transaction. The caller's
        documents are not changed.
        """
        self._ensure()
        for start in range(0, len(graph_documents), max(1, batch_size)):
            vertices, edges, sources = _collect(
                graph_documents[start : start + batch_size], include_source
            )
            inheriting = _entities(vertices, sources) if baseEntityLabel else ()
            wanted = self._wanted(vertices, edges, inheriting, dedupe_edges)
            if wanted is not None:
                self._engine.run(lambda: self._labels(*wanted), kind="maintenance")
                self._made(wanted)
            self._engine.run(
                lambda: self._write(vertices, edges),
                graph=self.graph_name,
                replay=True,
                merging=True,
            )

    async def aadd_graph_documents(
        self,
        graph_documents: Sequence[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,  # noqa: N803 - LangChain's spelling
        *,
        batch_size: int = 100,
        dedupe_edges: bool = True,
    ) -> None:
        await self._aensure()
        for start in range(0, len(graph_documents), max(1, batch_size)):
            vertices, edges, sources = _collect(
                graph_documents[start : start + batch_size], include_source
            )
            inheriting = _entities(vertices, sources) if baseEntityLabel else ()
            wanted = self._wanted(vertices, edges, inheriting, dedupe_edges)
            if wanted is not None:
                await self._engine.arun(lambda: self._labels(*wanted), kind="maintenance")
                self._made(wanted)
            await self._engine.arun(
                lambda: self._write(vertices, edges),
                graph=self.graph_name,
                replay=True,
                merging=True,
            )

    def ensure_labels(
        self, vertices: Sequence[str] = (), edges: Sequence[str] = (), *, base_entity: bool = False
    ) -> None:
        """Make vertex and edge labels that do not exist, each vertex label with a unique index
        on ``id``; with ``base_entity``, every vertex label named inherits ``__Entity__``."""
        self._ensure()
        named = tuple(sorted(vertices))
        wanted = (named, tuple(sorted(edges)), named if base_entity else (), False)
        self._engine.run(lambda: self._labels(*wanted), kind="maintenance")

    async def aensure_labels(
        self, vertices: Sequence[str] = (), edges: Sequence[str] = (), *, base_entity: bool = False
    ) -> None:
        await self._aensure()
        named = tuple(sorted(vertices))
        wanted = (named, tuple(sorted(edges)), named if base_entity else (), False)
        await self._engine.arun(lambda: self._labels(*wanted), kind="maintenance")

    def _wanted(
        self,
        vertices: Mapping[str, Any],
        edges: Mapping[tuple[str, str, str], Any],
        inheriting: tuple[str, ...],
        dedupe_edges: bool,
    ) -> tuple[Any, ...] | None:
        """The labels of a batch this graph has not made yet, as :meth:`_labels` takes them, or
        None when it has made them all. Each label is remembered on its own, so a batch that only
        combines labels made before needs nothing made."""
        new_vertices = tuple(
            sorted(v for v in vertices if ("vertex", v, v in inheriting) not in self._ensured)
        )
        new_edges = tuple(
            sorted(
                e
                for e in {edge for _, edge, _ in edges}
                if ("edge", e, dedupe_edges) not in self._ensured
            )
        )
        if not new_vertices and not new_edges:
            return None
        return (
            new_vertices,
            new_edges,
            tuple(v for v in inheriting if v in new_vertices),
            dedupe_edges,
        )

    def _made(self, wanted: tuple[Any, ...]) -> None:
        """Remember the labels :meth:`_labels` made, once what made them is committed."""
        vertices, edges, inheriting, dedupe_edges = wanted
        made = {("vertex", v, v in inheriting) for v in vertices}
        made |= {("edge", e, dedupe_edges) for e in edges}
        self._engine.remember(lambda: self._ensured.update(made))

    def _labels(
        self,
        vertices: tuple[str, ...],
        edges: tuple[str, ...],
        inheriting: tuple[str, ...],
        dedupe_edges: bool,
    ) -> Program[None]:
        graph = self.graph_name

        def made() -> Program[None]:
            yield Call("graph", (graph,))
            known = yield Statement(
                "SELECT l.labname::text FROM ag_label l JOIN ag_graph g ON g.oid = l.graphid "
                "WHERE g.graphname = %s",
                (graph,),
            )
            existing = {row[0] for row in known.records}
            if inheriting and BASE_ENTITY_LABEL not in existing:
                yield Statement(f"CREATE VLABEL IF NOT EXISTS {ident(BASE_ENTITY_LABEL)}")
            for label in vertices:
                entity = label in inheriting
                if label not in existing:
                    inherit = f" INHERITS ({ident(BASE_ENTITY_LABEL)})" if entity else ""
                    yield Statement(f"CREATE VLABEL IF NOT EXISTS {ident(label)}{inherit}")
                elif entity:
                    inherits = yield Statement(
                        "SELECT EXISTS (SELECT 1 FROM pg_inherits i "
                        "WHERE i.inhrelid = to_regclass(%s) AND i.inhparent = to_regclass(%s))",
                        (printed(graph, label), printed(graph, BASE_ENTITY_LABEL)),
                    )
                    if not scalar(inherits):
                        yield Statement(
                            f"ALTER VLABEL {ident(label)} INHERIT {ident(BASE_ENTITY_LABEL)}"
                        )
                yield Statement(
                    f"CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS {ident(_index_name(label, 'id'))} "
                    f"ON {ident(label)} (id)"
                )
            for label in edges:
                if label not in existing:
                    yield Statement(f"CREATE ELABEL IF NOT EXISTS {ident(label)}")
                if dedupe_edges:
                    name = _index_name(label, "pair")
                    try:
                        yield Statement(
                            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident(name)} "
                            f'ON {qualified(graph, label)} (start, "end")'
                        )
                    except psycopg.errors.UniqueViolation as exc:
                        raise DuplicateKeys(
                            f"the edges of {label!r} already hold two between one pair, so one per "
                            "pair cannot be kept: delete the extras, or pass dedupe_edges=False"
                        ) from exc

        yield InTransaction(made(), lock=graph)

    def _write(
        self,
        vertices: Mapping[str, Mapping[Any, dict[str, Any]]],
        edges: Mapping[tuple[str, str, str], list[tuple[Any, Any, dict[str, Any]]]],
    ) -> Program[None]:
        statements: list[Statement] = []
        for label, rows in vertices.items():
            statements.append(
                Statement(
                    f"UNWIND %s AS r MERGE (n:{ident(label)} {{id: r.id}}) SET n += r.props",
                    (Jsonb([{"id": key, "props": props} for key, props in rows.items()]),),
                )
            )
        for (start, label, end), pairs in edges.items():
            with_props = any(props for _, _, props in pairs)
            statements.append(
                Statement(
                    f"UNWIND %s AS e MATCH (a:{ident(start)} {{id: e.s}}), "
                    f"(b:{ident(end)} {{id: e.t}}) MERGE (a)-[r:{ident(label)}]->(b)"
                    + (" SET r += e.p" if with_props else ""),
                    (Jsonb([{"s": s, "t": t, "p": props} for s, t, props in pairs]),),
                )
            )
        if not statements:
            return
        yield Pipeline(tuple(statements), settings=NESTED_LOOPS)

    # -- lifecycle -----------------------------------------------------------------------------

    def drop_graph(self, *, companions: bool = True) -> None:
        """Drop the graph and everything in it, and with ``companions`` its ``"<graph>_store"``
        schema: vectors, text, LangGraph state, chat history. Every component on the engine that
        set up on this graph sets up again on its next call."""
        self._open()
        self._engine.run(lambda: self._drop(companions), kind="maintenance")
        self._engine.forget_setups(self.graph_name)

    async def adrop_graph(self, *, companions: bool = True) -> None:
        self._open()
        await self._engine.arun(lambda: self._drop(companions), kind="maintenance")
        self._engine.forget_setups(self.graph_name)

    def _forgotten(self) -> None:
        self._schema_read_at = None
        self._ensured.clear()

    def _drop(self, companions: bool) -> Program[None]:
        def dropped() -> Program[None]:
            yield Statement(f"DROP GRAPH IF EXISTS {ident(self.graph_name)} CASCADE")
            if companions:
                yield Statement(
                    f"DROP SCHEMA IF EXISTS {ident(companion_schema(self.graph_name))} CASCADE"
                )

        yield InTransaction(dropped(), lock=self.graph_name)


def _readable(text: str, max_rows: int, allow_writes: bool) -> None:
    """Refuse model-written text the driver's reading refuses, without taking a connection."""
    if max_rows < 1:
        raise ValueError(f"at least one row has to be readable, got {max_rows}")
    try:
        check_untrusted(text, allow_writes=allow_writes)
    except ValueError as exc:
        raise UnsafeCypherError(str(exc)) from None


def _writes_checked(capabilities: Capabilities) -> None:
    """Refuse model-written writes on a server that does not check Cypher writes itself."""
    if not capabilities.has("cypher_write_enforced"):
        raise CapabilityError(
            "model-written writes need a server that checks Cypher writes against privileges and "
            f"READ ONLY itself, AgensGraph 2.16 or later; this server is {capabilities.reported}"
        )


def _query_error(exc: psycopg.Error) -> AgensQueryError:
    detail = exc.diag.message_detail if exc.diag else None
    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return AgensQueryError(
        message + (f" ({detail})" if detail else ""), sqlstate=getattr(exc, "sqlstate", None)
    )


def _collect(
    documents: Sequence[GraphDocument], include_source: bool
) -> tuple[
    dict[str, dict[Any, dict[str, Any]]],
    dict[tuple[str, str, str], list[tuple[Any, Any, dict[str, Any]]]],
    set[str],
]:
    """A batch grouped by label, each node once, the last mention's properties winning; and the
    labels of the source documents.

    A node is mentioned in a document's nodes, with its properties, and again as the end of a
    relationship, often bare; the mentions are merged so the second does not erase the first.
    Edges are grouped by their whole triple, so the statement writing them names both ends'
    labels and an end is found only in its own label.
    """
    vertices: dict[str, dict[Any, dict[str, Any]]] = {}
    edges: dict[tuple[str, str, str], list[tuple[Any, Any, dict[str, Any]]]] = {}
    sources: set[str] = set()

    def remember(label: str, key: Any, properties: Mapping[str, Any]) -> None:
        checked(label, "node type")
        known = vertices.setdefault(label, {}).setdefault(key, {})
        known.update({k: v for k, v in properties.items() if k != "id"})

    for document in documents:
        for node in document.nodes:
            remember(node.type, node.id, node.properties)
        for edge in document.relationships:
            remember(edge.source.type, edge.source.id, edge.source.properties)
            remember(edge.target.type, edge.target.id, edge.target.properties)
            checked(edge.type, "relationship type")
            edges.setdefault((edge.source.type, edge.type, edge.target.type), []).append(
                (edge.source.id, edge.target.id, dict(edge.properties))
            )
        source = document.source
        if include_source and source is not None:
            key = (
                source.metadata.get("id")
                or hashlib.md5(source.page_content.encode(), usedforsecurity=False).hexdigest()
            )
            label = source.type or "Document"
            sources.add(label)
            remember(label, key, {**source.metadata, "text": source.page_content})
            for node in document.nodes:
                edges.setdefault((label, SOURCE_EDGE, node.type), []).append((key, node.id, {}))
    return vertices, edges, sources


def _entities(vertices: Mapping[str, Any], sources: set[str]) -> tuple[str, ...]:
    """The labels that inherit ``__Entity__``: every node label, not the source documents'."""
    return tuple(sorted(label for label in vertices if label not in sources))


def _index_name(label: str, what: str) -> str:
    return bounded(f"{label}__{what}")


def _index_facts(indexes: Any) -> list[dict[str, Any]]:
    """The btree property indexes, as facts a query writer can act on.

    A predicate an index cannot serve returns the right rows and reads the whole label, and only
    the plan says so; telling a model which indexes exist lets it choose the spelling one serves.
    Vector and text indexes, and the vector store's ``__id__`` index, are left out.
    """
    facts = []
    for index in indexes:
        tail = index.definition.split(" USING ", 1)
        if len(tail) != 2:
            continue
        method, _, keys = tail[1].partition(" ")
        keys = keys.split(" WHERE ", 1)[0].strip()
        if method != "btree" or "__id__" in keys or not keys:
            continue
        facts.append({"label": index.label, "on": keys, "unique": index.unique})
    return facts


def _rendered(structured: Mapping[str, Any]) -> str:
    """The schema as the text a prompt carries."""
    triples = [
        '(:"{start}")-[:"{type}"]->(:"{end}")'.format(**rel) for rel in structured["relationships"]
    ]
    indexes = [
        '(:"{label}") ON {on}{unique}'.format(
            label=entry["label"], on=entry["on"], unique=" UNIQUE" if entry["unique"] else ""
        )
        for entry in structured.get("indexes", [])
    ]
    nodes = [
        {"labels": label, "properties": props} for label, props in structured["node_props"].items()
    ]
    rels = [
        {"type": label, "properties": props} for label, props in structured["rel_props"].items()
    ]
    parts = [
        "Node properties are the following:",
        str(nodes),
        "Relationship properties are the following:",
        str(rels),
        "The relationships are the following:",
        str(triples),
        "Element counts (estimates) are the following:",
        str(structured["counts"]),
    ]
    if indexes:
        parts += [
            "Property indexes are the following (and id(x) is always indexed, on every label):",
            str(indexes),
        ]
    return "\n".join(parts)
