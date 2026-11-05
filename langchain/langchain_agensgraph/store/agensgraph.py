"""A LangGraph ``BaseStore`` in the companion schema of an AgensGraph graph.

Items live in ``"<graph>_store".store`` -- one row per (namespace, key), the namespace as its
labels joined by ``.``, the value as ``jsonb`` -- and, with an index configured, their embeddings
in ``"<graph>_store".store_vectors``, one row per indexed field, deleted with the item. The layout
and its semantics follow PostgresStore from ``langgraph-checkpoint-postgres`` (MIT): a filter means
what it means to ``InMemoryStore``, a semantic search scores an item by its nearest field and fills
any shortfall with items that have no embedding, and an item's time to live can be refreshed when
it is read.

A batch is one round trip: its reads, then its writes, in one pipeline and so one implicit
transaction, after one embedding call for every text the batch needs embedded. Reads therefore see
the store as it was before the batch, and the last write to a key in a batch is the one kept.

Tables are made on first use, under the graph's schema lock, from an append-only list of
migrations whose progress is kept in the schema. The graph itself need not exist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from agensgraph import Distance as VectorDistance
from agensgraph import Jsonb, Vector
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
    validate_op_namespace,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore

from .._core.component import Component
from .._core.engine import AgensEngine, bind
from .._core.names import companion_schema, literal, qualified
from .._core.program import Pipeline, Program, Statement
from .._core.settings import EF_SEARCH_MAX
from .._core.setup import Migrations, Setup

logger = logging.getLogger(__name__)

Distance = Literal["cosine", "l2", "inner_product"]

_MEASURES: dict[str, tuple[VectorDistance, str]] = {
    # distance: pgvector's measure, and the score from the distance d
    "cosine": (VectorDistance.COSINE, "1 - d"),
    "l2": (VectorDistance.L2, "-d"),
    "inner_product": (VectorDistance.INNER_PRODUCT, "-d"),
}


class AgensIndexConfig(IndexConfig, total=False):
    """``IndexConfig`` with the index and the measure used for it.

    ``ann_index_config``: ``{"kind": "hnsw", "m": 16, "ef_construction": 64}``,
    ``{"kind": "ivfflat", "lists": 100}`` or ``{"kind": "flat"}`` for none, with
    ``"vector_type": "vector" | "halfvec"``. ``distance_type``: ``"cosine"`` (the default),
    ``"l2"`` or ``"inner_product"``.
    """

    ann_index_config: dict[str, Any]
    distance_type: Distance


@dataclass(frozen=True)
class _Index:
    """An index configuration, checked and resolved."""

    dims: int
    embeddings: Embeddings
    fields: list[tuple[str, Literal["$"] | list[str]]]
    vectors_per_item: int
    vector_type: str
    distance: str
    kind: str
    parameters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, config: Mapping[str, Any]) -> _Index:
        dims = int(config["dims"])
        if dims < 1:
            raise ValueError(f"dims must be positive, not {dims}")
        embeddings = ensure_embeddings(config.get("embed"))
        if embeddings is None:
            raise ValueError("an index needs embed=: an Embeddings, a function or provider:model")
        raw_fields = config.get("fields") or ["$"]
        if isinstance(raw_fields, str):
            raw_fields = [raw_fields]
        fields: list[tuple[str, Literal["$"] | list[str]]] = []
        count = 0
        for path in raw_fields:
            if path == "$":
                fields.append(("$", "$"))
                count += 1
            else:
                tokens = tokenize_path(path)
                fields.append((path, tokens))
                count += len(tokens)
        ann = dict(config.get("ann_index_config") or {})
        kind = str(ann.pop("kind", "hnsw"))
        if kind not in ("hnsw", "ivfflat", "flat"):
            raise ValueError(f"an index kind is hnsw, ivfflat or flat, not {kind!r}")
        vector_type = str(ann.pop("vector_type", "vector"))
        if vector_type not in ("vector", "halfvec"):
            raise ValueError(f"a vector type is vector or halfvec, not {vector_type!r}")
        distance = str(config.get("distance_type", "cosine"))
        if distance not in _MEASURES:
            raise ValueError(f"a distance is cosine, l2 or inner_product, not {distance!r}")
        allowed = {"hnsw": {"m", "ef_construction"}, "ivfflat": {"lists"}, "flat": set()}[kind]
        parameters = {}
        for name, value in ann.items():
            name = "lists" if name == "nlist" else name
            if name in allowed and int(value) > 0:
                parameters[name] = int(value)
        return cls(dims, embeddings, fields, count, vector_type, distance, kind, parameters)

    @property
    def operator(self) -> str:
        return _MEASURES[self.distance][0].value

    @property
    def operator_class(self) -> str:
        return _MEASURES[self.distance][0].operator_class_for(self.vector_type)

    @property
    def score(self) -> str:
        return _MEASURES[self.distance][1]

    @property
    def typed(self) -> str:
        """The embedding column's type, such as ``vector(1536)``."""
        return f"{self.vector_type}({self.dims})"


class _Tables:
    """The store's tables in a graph's companion schema, and the statements that use them."""

    def __init__(self, graph: str, index: _Index | None) -> None:
        schema = companion_schema(graph)
        self.schema = schema
        self.store = qualified(schema, "store")
        self.vectors = qualified(schema, "store_vectors")
        self.index = index
        self.migrations = [
            f"CREATE TABLE IF NOT EXISTS {self.store} (prefix text NOT NULL, key text NOT NULL, "
            "value jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), "
            "updated_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz, "
            "ttl_minutes double precision, PRIMARY KEY (prefix, key))",
            f"CREATE INDEX IF NOT EXISTS store_prefix_idx ON {self.store} "
            "USING btree (prefix text_pattern_ops)",
            f"CREATE INDEX IF NOT EXISTS store_expires_at_idx ON {self.store} (expires_at) "
            "WHERE expires_at IS NOT NULL",
        ]
        self.vector_migrations: list[str] = []
        if index is not None:
            self.vector_migrations.append(
                f"CREATE TABLE IF NOT EXISTS {self.vectors} (prefix text NOT NULL, "
                f"key text NOT NULL, field_name text NOT NULL, embedding {index.typed} NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now(), "
                "updated_at timestamptz NOT NULL DEFAULT now(), "
                "PRIMARY KEY (prefix, key, field_name), "
                f"FOREIGN KEY (prefix, key) REFERENCES {self.store} (prefix, key) "
                "ON DELETE CASCADE)"
            )
            if index.kind != "flat":
                options = ", ".join(f"{k} = {v}" for k, v in index.parameters.items())
                self.vector_migrations.append(
                    f"CREATE INDEX IF NOT EXISTS store_vectors_embedding_idx ON {self.vectors} "
                    f"USING {index.kind} (embedding {index.operator_class})"
                    + (f" WITH ({options})" if options else "")
                )

    @property
    def steps(self) -> dict[int, str]:
        """Every migration by its number, the vector table's from 1000 on, so either list can grow
        without renumbering the other."""
        steps = dict(enumerate(self.migrations))
        steps.update((1000 + number, step) for number, step in enumerate(self.vector_migrations))
        return steps

    # -- reads -----------------------------------------------------------------------------

    def get(self, ttl: TTLConfig | None) -> str:
        """Every key asked for, with its TTL refreshed where asked, in one statement."""
        alive = _alive("s", ttl)
        return (
            "WITH asked AS (SELECT * FROM unnest(%s::text[], %s::text[], %s::bool[]) "
            "AS a(prefix, key, refresh)), "
            f"refreshed AS (UPDATE {self.store} s "
            "SET expires_at = now() + make_interval(secs => s.ttl_minutes * 60) FROM asked a "
            "WHERE s.prefix = a.prefix AND s.key = a.key AND a.refresh "
            f"AND s.ttl_minutes IS NOT NULL{alive} RETURNING s.prefix) "
            "SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at "
            f"FROM {self.store} s JOIN asked a ON s.prefix = a.prefix AND s.key = a.key"
            f"{' WHERE ' + alive[5:] if alive else ''}"
        )

    def listing(self, where: str, ttl: TTLConfig | None, refresh: bool) -> str:
        """A search without a query: newest first."""
        alive = _alive("s", ttl)
        found = (
            "SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, NULL::float8 AS score "
            f"FROM {self.store} s WHERE {where}{alive} "
            "ORDER BY s.updated_at DESC, s.prefix, s.key LIMIT %s OFFSET %s"
        )
        return self._refreshing(found) if refresh else found

    def semantic(self, where: str, ttl: TTLConfig | None, refresh: bool) -> str:
        """A search with a query: each item scored by its nearest field, best first; then, when
        fewer than the limit are scored, items without an embedding, newest first, fill the
        shortfall. The fill's scan runs only when there is one: its limit is then zero, and the
        server reads nothing for it.

        Parameters: the query vector, the filter's, how many field rows to read, the limit, the
        offset, the filter's again, the limit again.
        """
        assert self.index is not None
        alive = _alive("s", ttl)
        found = (
            "SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, "
            f"{self.index.score.replace('d', 'best.d')} AS score FROM ("
            "SELECT DISTINCT ON (near.prefix, near.key) near.prefix, near.key, near.d FROM ("
            f"SELECT sv.prefix, sv.key, sv.embedding {self.index.operator} "
            f"%s::{self.index.typed} AS d "
            f"FROM {self.vectors} sv JOIN {self.store} s "
            "ON s.prefix = sv.prefix AND s.key = sv.key "
            f"WHERE {where}{alive} ORDER BY d LIMIT %s) near "
            "ORDER BY near.prefix, near.key, near.d) best "
            f"JOIN {self.store} s ON s.prefix = best.prefix AND s.key = best.key "
            "ORDER BY best.d, s.prefix, s.key LIMIT %s OFFSET %s"
        )
        filled = (
            "SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, NULL::float8 AS score "
            f"FROM {self.store} s WHERE {where}{alive} AND NOT EXISTS ("
            f"SELECT 1 FROM {self.vectors} sv WHERE sv.prefix = s.prefix AND sv.key = s.key) "
            "ORDER BY s.updated_at DESC, s.prefix, s.key "
            "LIMIT greatest(%s - (SELECT count(*) FROM found), 0)"
        )
        refreshed = f"{self._refreshed}, " if refresh else ""
        return (
            f"WITH found AS ({found}), {refreshed}filled AS ({filled}) "
            "SELECT * FROM found UNION ALL SELECT * FROM filled"
        )

    def _refreshing(self, found: str) -> str:
        return f"WITH found AS ({found}), {self._refreshed} SELECT * FROM found"

    @property
    def _refreshed(self) -> str:
        """A ``WITH`` clause's statement putting off the expiry of every item in ``found``."""
        return (
            f"refreshed AS (UPDATE {self.store} s "
            "SET expires_at = now() + make_interval(secs => s.ttl_minutes * 60) FROM found f "
            "WHERE s.prefix = f.prefix AND s.key = f.key AND s.ttl_minutes IS NOT NULL "
            "RETURNING s.prefix)"
        )

    def namespaces(self, conditions: Sequence[str], ttl: TTLConfig | None) -> str:
        """Distinct namespaces, cut to ``max_depth`` labels, in byte order."""
        where = list(conditions)
        if ttl and ttl.get("omit_expired"):
            where.append("(expires_at IS NULL OR expires_at > now())")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        return (
            'SELECT DISTINCT ON (cut COLLATE "C") cut FROM ('
            "SELECT CASE WHEN %s::integer IS NULL THEN prefix ELSE "
            "array_to_string((string_to_array(prefix, '.'))[1:%s::integer], '.') END AS cut "
            f"FROM {self.store}{clause}) n "
            'ORDER BY cut COLLATE "C" LIMIT %s OFFSET %s'
        )

    # -- writes ----------------------------------------------------------------------------

    @property
    def delete(self) -> str:
        return (
            f"DELETE FROM {self.store} s USING unnest(%s::text[], %s::text[]) AS d(prefix, key) "
            "WHERE s.prefix = d.prefix AND s.key = d.key"
        )

    @property
    def upsert(self) -> str:
        return (
            f"INSERT INTO {self.store} AS s "
            "(prefix, key, value, created_at, updated_at, expires_at, ttl_minutes) "
            "SELECT u.prefix, u.key, u.value, now(), now(), "
            "now() + make_interval(secs => u.ttl * 60), u.ttl "
            "FROM unnest(%s::text[], %s::text[], %s::jsonb[], %s::float8[]) "
            "AS u(prefix, key, value, ttl) "
            "ON CONFLICT (prefix, key) DO UPDATE SET value = EXCLUDED.value, "
            "updated_at = now(), expires_at = EXCLUDED.expires_at, "
            "ttl_minutes = EXCLUDED.ttl_minutes"
        )

    @property
    def unindex(self) -> str:
        return (
            f"DELETE FROM {self.vectors} sv USING unnest(%s::text[], %s::text[]) AS d(prefix, key) "
            "WHERE sv.prefix = d.prefix AND sv.key = d.key"
        )

    @property
    def index_rows(self) -> str:
        assert self.index is not None
        return (
            f"INSERT INTO {self.vectors} (prefix, key, field_name, embedding) "
            f"SELECT u.prefix, u.key, u.field, u.embedding::{self.index.typed} "
            "FROM unnest(%s::text[], %s::text[], %s::text[], %s::vector[]) "
            "AS u(prefix, key, field, embedding) "
            "ON CONFLICT (prefix, key, field_name) DO UPDATE "
            "SET embedding = EXCLUDED.embedding, updated_at = now()"
        )

    @property
    def sweep(self) -> str:
        return (
            f"WITH gone AS (SELECT prefix, key FROM {self.store} "
            "WHERE expires_at IS NOT NULL AND expires_at < now() LIMIT %s FOR UPDATE SKIP LOCKED) "
            f"DELETE FROM {self.store} s USING gone "
            "WHERE s.prefix = gone.prefix AND s.key = gone.key RETURNING 1"
        )


def _alive(alias: str, ttl: TTLConfig | None) -> str:
    """`` AND <alias> has not expired`` when expired items are to be left out of reads."""
    if ttl and ttl.get("omit_expired"):
        return f" AND ({alias}.expires_at IS NULL OR {alias}.expires_at > now())"
    return ""


# -- namespaces and filters ------------------------------------------------------------------


def _text(namespace: Sequence[str]) -> str:
    return ".".join(namespace)


def _namespace(text: str) -> tuple[str, ...]:
    return tuple(text.split(".")) if text else ()


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _within(prefix: Sequence[str], params: list[Any]) -> str:
    """The namespace itself or any namespace below it."""
    if not prefix:
        return "TRUE"
    path = _text(prefix)
    params += [path, _escape_like(path) + ".%"]
    return "(s.prefix = %s OR s.prefix LIKE %s ESCAPE '\\')"


def _match(path: Sequence[str], kind: str) -> str:
    """A pattern matching namespaces on whole labels, ``*`` standing for any one label."""
    body = r"\.".join("[^.]+" if part == "*" else re.escape(part) for part in path)
    return rf"(^|\.){body}\Z" if kind == "suffix" else rf"^{body}(\.|\Z)"


def _filter(where: Mapping[str, Any] | None, params: list[Any]) -> str:
    """A search filter with the meaning ``InMemoryStore`` gives it, over each item's value.

    A key's value is compared to the item's value at that key: a mapping of operators applies
    each (``$eq``, ``$ne``, and ``$gt``, ``$gte``, ``$lt``, ``$lte`` as numbers, a string that
    reads as a number counting as one, and a missing key matching none of them); any other
    mapping matches an object holding at least those keys, compared the same way; a list matches
    a list of the same length, element by element; anything else must be equal, ``None`` matching
    a missing key.
    """
    if not where:
        return "TRUE"
    return " AND ".join(
        _compared("s.value->" + _key(key), value, params) for key, value in where.items()
    )


def _key(key: object) -> str:
    if not isinstance(key, str) or "\x00" in key:
        raise ValueError(f"a filter key must be a string, not {key!r}")
    return literal(key)


def _compared(expr: str, wanted: Any, params: list[Any]) -> str:
    if isinstance(wanted, Mapping):
        if any(isinstance(k, str) and k.startswith("$") for k in wanted):
            return (
                "(" + " AND ".join(_operator(expr, op, v, params) for op, v in wanted.items()) + ")"
            )
        parts = [f"jsonb_typeof({expr}) = 'object'"]
        parts += [_compared(f"({expr})->{_key(k)}", v, params) for k, v in wanted.items()]
        return "(" + " AND ".join(parts) + ")"
    if isinstance(wanted, list | tuple):
        # A CASE, since SQL may evaluate either side of an AND first and the length of a
        # scalar is an error.
        parts = [
            f"(CASE WHEN jsonb_typeof({expr}) = 'array' "
            f"THEN jsonb_array_length({expr}) = {len(wanted)} ELSE false END)"
        ]
        parts += [_compared(f"({expr})->{i}", v, params) for i, v in enumerate(wanted)]
        return "(" + " AND ".join(parts) + ")"
    if wanted is None:
        return f"({expr} IS NULL OR {expr} = 'null'::jsonb)"
    params.append(Jsonb(wanted))
    return f"coalesce({expr} = %s::jsonb, false)"


def _operator(expr: str, op: str, value: Any, params: list[Any]) -> str:
    if op == "$eq":
        if value is None:
            return f"({expr} IS NULL OR {expr} = 'null'::jsonb)"
        params.append(Jsonb(value))
        return f"coalesce({expr} = %s::jsonb, false)"
    if op == "$ne":
        if value is None:
            return f"({expr} IS NOT NULL AND {expr} <> 'null'::jsonb)"
        params.append(Jsonb(value))
        return f"coalesce({expr} <> %s::jsonb, true)"
    comparisons = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
    if op in comparisons:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{op} compares numbers, not {value!r}")
        # A number, or a string that reads as one, as Python's float() reads it; anything else,
        # a missing key included, does not compare.
        params += [_NUMERIC_TEXT, float(value)]
        return (
            f"coalesce(CASE WHEN jsonb_typeof({expr}) = 'number' THEN ({expr})::numeric "
            f"WHEN jsonb_typeof({expr}) = 'string' AND ({expr} #>> '{{}}') ~ %s "
            f"THEN ({expr} #>> '{{}}')::numeric END {comparisons[op]} %s::numeric, false)"
        )
    raise ValueError(f"unsupported filter operator {op!r}")


_NUMERIC_TEXT = r"^\s*[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?\s*$"


# -- the shared core -------------------------------------------------------------------------


@dataclass
class _Plan:
    """The statements of one batch, and how each result becomes the batch's results."""

    statements: list[Statement] = field(default_factory=list)
    readers: list[Callable[[list[Any], list[Result]], None]] = field(default_factory=list)

    def read(self, statement: Statement, reader: Callable[[list[Any], list[Result]], None]) -> None:
        self.statements.append(statement)
        self.readers.append(reader)

    def write(self, statement: Statement) -> None:
        self.read(statement, lambda rows, results: None)


class _StoreCore(Component):
    """What both stores share: binding, setup, closing, and the program of a batch."""

    supports_ttl: bool = True
    _what = "store"

    def _init_core(
        self,
        engine: AgensEngine | str | None,
        graph: Any,
        graph_name: str | None,
        index: AgensIndexConfig | IndexConfig | None,
        ttl: TTLConfig | None,
        setup: bool,
    ) -> None:
        self._engine, self._owns_engine, self.graph_name = bind(engine, graph, graph_name)
        self.index_config = index
        self._index = _Index.of(index) if index else None
        self.embeddings: Embeddings | None = self._index.embeddings if self._index else None
        self.ttl_config = ttl
        self._tables = _Tables(self.graph_name, self._index)
        self._setup_wanted = setup
        migrations = Migrations(self._tables.schema, "store_migrations", self._tables.steps)
        self._setup = Setup(
            self._engine,
            self.graph_name,
            migrations.check,
            migrations.create,
            key=migrations.key,
        )

    async def asetup(self) -> None:
        """Make the store's tables now rather than on its first call."""
        self._open()
        await self._setup.aensure()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        listed = _grouped(ops)
        self._open()
        if self._setup_wanted:
            await self._setup.aensure()
        texts, fields = self._texts(listed)
        vectors = await self.embeddings.aembed_documents(texts) if texts and self.embeddings else []
        return await self._engine.arun(lambda: self._batch(listed, vectors, fields))

    async def asweep_ttl(self, *, batch_size: int = 1000) -> int:
        """Delete expired items, a batch at a time; how many were deleted."""
        self._open()
        await self._setup.aensure()
        return await self._engine.arun(lambda: self._sweep_all(batch_size))

    def cypher_subquery(self, namespace: str = "%s", *, nested: bool = False) -> str:
        """A SQL subquery a Cypher statement can hold, giving the items in a namespace as a list.

        Each element is a map of ``namespace`` (its labels), ``key``, ``value``, ``created_at``
        and ``updated_at``; expired items are left out. ``namespace`` is the SQL for the
        namespace's labels joined by ``.``: a placeholder, by default, or an expression.
        ``nested`` takes the namespaces under it as well. For example::

            graph.query(
                f"UNWIND [1] AS a WITH {store.cypher_subquery()} AS items UNWIND items AS i "
                'MATCH (p:"Person") WHERE p.name = i.value.name RETURN p, i.key AS key',
                ("users.prefs",),
            )
        """
        under = " OR starts_with(s.prefix, w.prefix || '.')" if nested else ""
        return (
            "(SELECT coalesce(jsonb_agg(jsonb_build_object('namespace', "
            "string_to_array(s.prefix, '.'), 'key', s.key, 'value', s.value, 'created_at', "
            "s.created_at, 'updated_at', s.updated_at) ORDER BY s.prefix, s.key), '[]') "
            f"FROM {self._tables.store} s, (SELECT ({namespace})::text AS prefix) w "
            f"WHERE (s.prefix = w.prefix{under}) "
            "AND (s.expires_at IS NULL OR s.expires_at > now()))"
        )

    # -- the batch -------------------------------------------------------------------------

    def _texts(self, ops: Sequence[Op]) -> tuple[list[str], list[tuple[str, str, str, int]]]:
        """Search queries to embed, and the fields of each indexed put: (prefix, key, field,
        position in the list of texts)."""
        texts: list[str] = []
        fields: list[tuple[str, str, str, int]] = []
        if self._index is None:
            return texts, fields
        for op in ops:
            if isinstance(op, SearchOp) and op.query:
                texts.append(op.query)
        for op in _last_puts(ops).values():
            if op.value is None or op.index is False:
                continue
            paths = (
                self._index.fields
                if op.index is None
                else [(path, tokenize_path(path)) for path in op.index]
            )
            for path, tokens in paths:
                found = get_text_at_path(op.value, tokens)
                for i, text in enumerate(found):
                    name = f"{path}.{i}" if len(found) > 1 else path
                    fields.append((_text(op.namespace), op.key, name, len(texts)))
                    texts.append(text)
        return texts, fields

    def _batch(
        self,
        ops: Sequence[Op],
        vectors: list[list[float]],
        fields: list[tuple[str, str, str, int]],
    ) -> Program[list[Result]]:
        results: list[Result] = [None] * len(ops)
        plan = _Plan()
        tables = self._tables
        ttl = self.ttl_config
        query_vectors = iter(vectors)

        gets = [(i, op) for i, op in enumerate(ops) if isinstance(op, GetOp)]
        if gets:
            refresh = bool(ttl and ttl.get("refresh_on_read"))

            def read_gets(rows: list[Any], out: list[Result]) -> None:
                found = {(r[0], r[1]): r for r in rows}
                for i, op in gets:
                    row = found.get((_text(op.namespace), op.key))
                    out[i] = None if row is None else _item(row)

            plan.read(
                Statement(
                    tables.get(ttl),
                    (
                        [_text(op.namespace) for _, op in gets],
                        [op.key for _, op in gets],
                        [bool(op.refresh_ttl) and refresh for _, op in gets],
                    ),
                ),
                read_gets,
            )

        for i, op in enumerate(ops):
            if isinstance(op, SearchOp):
                self._search(plan, i, op, query_vectors)
            elif isinstance(op, ListNamespacesOp):
                self._list(plan, i, op)

        puts = _last_puts(ops)
        if puts:
            self._put(plan, list(puts.values()), vectors, fields)

        if not plan.statements:
            return results
        rows = yield Pipeline(tuple(plan.statements))
        for reader, result in zip(plan.readers, rows, strict=True):
            reader(list(result.records), results)
        return results

    def _search(
        self, plan: _Plan, i: int, op: SearchOp, query_vectors: Iterator[list[float]]
    ) -> None:
        tables, ttl = self._tables, self.ttl_config
        refresh = bool(ttl and ttl.get("refresh_on_read")) and op.refresh_ttl
        params: list[Any] = []
        where = _within(op.namespace_prefix, params)
        where = f"{where} AND {_filter(op.filter, params)}"

        def read_found(rows: list[Any], out: list[Result]) -> None:
            out[i] = [_search_item(row) for row in rows]

        if op.query and self._index is not None:
            vector = Vector(np.asarray(next(query_vectors), dtype=np.float32))
            fetch = (op.offset + op.limit) * self._index.vectors_per_item * 2 + 1

            plan.read(
                Statement(
                    tables.semantic(where, ttl, refresh),
                    (vector, *params, fetch, op.limit, op.offset, *params, op.limit),
                    settings=self._search_settings(fetch),
                ),
                read_found,
            )
        else:
            plan.read(
                Statement(tables.listing(where, ttl, refresh), (*params, op.limit, op.offset)),
                read_found,
            )

    def _search_settings(self, fetch: int) -> dict[str, object] | None:
        if self._index is None or self._index.kind != "hnsw":
            return None
        return {
            "hnsw.ef_search": max(40, min(fetch, EF_SEARCH_MAX)),
            "hnsw.iterative_scan": "strict_order",
        }

    def _list(self, plan: _Plan, i: int, op: ListNamespacesOp) -> None:
        conditions = []
        params: list[Any] = [op.max_depth, op.max_depth]
        for condition in op.match_conditions or ():
            if condition.path:
                conditions.append("prefix ~ %s")
                params.append(_match(condition.path, condition.match_type))
        params += [op.limit, op.offset]

        def read(rows: list[Any], out: list[Result]) -> None:
            out[i] = [_namespace(row[0]) for row in rows]

        plan.read(Statement(self._tables.namespaces(conditions, self.ttl_config), params), read)

    def _put(
        self,
        plan: _Plan,
        puts: list[PutOp],
        vectors: list[list[float]],
        fields: list[tuple[str, str, str, int]],
    ) -> None:
        tables = self._tables
        deleted = [op for op in puts if op.value is None]
        stored = [op for op in puts if op.value is not None]
        if deleted:
            plan.write(
                Statement(
                    tables.delete,
                    ([_text(op.namespace) for op in deleted], [op.key for op in deleted]),
                )
            )
        if not stored:
            return
        plan.write(
            Statement(
                tables.upsert,
                (
                    [_text(op.namespace) for op in stored],
                    [op.key for op in stored],
                    [Jsonb(dict(cast(Mapping[str, Any], op.value))) for op in stored],
                    [None if op.ttl is None else float(op.ttl) for op in stored],
                ),
            )
        )
        if self._index is None:
            return
        plan.write(
            Statement(
                tables.unindex,
                ([_text(op.namespace) for op in stored], [op.key for op in stored]),
            )
        )
        if fields:
            plan.write(
                Statement(
                    tables.index_rows,
                    (
                        [f[0] for f in fields],
                        [f[1] for f in fields],
                        [f[2] for f in fields],
                        [Vector(np.asarray(vectors[f[3]], dtype=np.float32)) for f in fields],
                    ),
                )
            )

    def _sweep_all(self, batch_size: int) -> Program[int]:
        """Expired items deleted ``batch_size`` at a time, each batch its own statement, until a
        batch comes up short; how many were deleted."""
        total = 0
        while True:
            swept = yield Statement(self._tables.sweep, (batch_size,))
            total += len(swept.records)
            if len(swept.records) < batch_size:
                return total


def _last_puts(ops: Iterable[Op]) -> dict[tuple[tuple[str, ...], str], PutOp]:
    """The last put to each key: the one a batch keeps."""
    puts: dict[tuple[tuple[str, ...], str], PutOp] = {}
    for op in ops:
        if isinstance(op, PutOp):
            puts[(op.namespace, op.key)] = op
    return puts


def _item(row: Sequence[Any]) -> Item:
    prefix, key, value, created, updated = row[:5]
    return Item(
        value=_value(value),
        key=key,
        namespace=_namespace(prefix),
        created_at=created,
        updated_at=updated,
    )


def _search_item(row: Sequence[Any]) -> SearchItem:
    prefix, key, value, created, updated, score = row[:6]
    return SearchItem(
        namespace=_namespace(prefix),
        key=key,
        value=_value(value),
        created_at=created,
        updated_at=updated,
        score=None if score is None else float(score),
    )


def _value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else json.loads(value)


def _grouped(ops: Iterable[Op]) -> list[Op]:
    listed = list(ops)
    for op in listed:
        validate_op_namespace(op)
    return listed


# -- the stores ------------------------------------------------------------------------------


class AgensStore(_StoreCore, BaseStore):
    """A LangGraph store whose blocking and awaiting methods both talk to the server directly.

    ``index`` makes ``search(query=...)`` semantic; ``ttl`` gives items a time to live, refreshed
    on read with ``refresh_on_read`` and removed by :meth:`sweep_ttl` or the sweeper thread.
    """

    def __init__(
        self,
        engine: AgensEngine | str | None = None,
        *,
        graph: Any = None,
        graph_name: str | None = None,
        index: AgensIndexConfig | IndexConfig | None = None,
        ttl: TTLConfig | None = None,
        setup: bool = True,
    ) -> None:
        self._init_core(engine, graph, graph_name, index, ttl, setup)
        self._sweeper: threading.Thread | None = None
        self._stop = threading.Event()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conninfo: str, *, graph_name: str, **kwargs: Any
    ) -> Iterator[AgensStore]:
        """A store on an engine of its own, closed with the block."""
        with AgensEngine(conninfo) as engine:
            store = cls(engine, graph_name=graph_name, **kwargs)
            try:
                yield store
            finally:
                store.stop_ttl_sweeper()

    def setup(self) -> None:
        """Make the store's tables now rather than on its first call."""
        self._open()
        self._setup.ensure()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        listed = _grouped(ops)
        self._open()
        if self._setup_wanted:
            self._setup.ensure()
        texts, fields = self._texts(listed)
        vectors = self.embeddings.embed_documents(texts) if texts and self.embeddings else []
        return self._engine.run(lambda: self._batch(listed, vectors, fields))

    def sweep_ttl(self, *, batch_size: int = 1000) -> int:
        """Delete expired items, a batch at a time; how many were deleted."""
        self._open()
        self._setup.ensure()
        return self._engine.run(lambda: self._sweep_all(batch_size))

    def start_ttl_sweeper(self, sweep_interval_minutes: float | None = None) -> threading.Thread:
        """Sweep expired items every interval on a daemon thread, until stopped or closed."""
        if self._sweeper is not None and self._sweeper.is_alive():
            return self._sweeper
        interval = 60 * (
            sweep_interval_minutes or (self.ttl_config or {}).get("sweep_interval_minutes") or 5
        )
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.sweep_ttl()
                except Exception:  # noqa: BLE001 - a sweep that fails is tried again next time
                    logger.exception("sweeping expired store items failed")

        self._sweeper = threading.Thread(target=loop, name="agens-store-ttl", daemon=True)
        self._sweeper.start()
        return self._sweeper

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the sweeper; whether it has stopped."""
        if self._sweeper is None:
            return True
        self._stop.set()
        self._sweeper.join(timeout)
        stopped = not self._sweeper.is_alive()
        if stopped:
            self._sweeper = None
        return stopped

    def close(self) -> None:
        self.stop_ttl_sweeper()
        super().close()

    async def aclose(self) -> None:
        self.stop_ttl_sweeper()
        await super().aclose()


class AsyncAgensStore(_StoreCore, AsyncBatchedBaseStore):
    """A LangGraph store for an event loop: concurrent calls on the loop are batched together.

    Make it inside a running loop, or with :meth:`afrom_conn_string`. Its blocking methods may be
    called from another thread; on the loop's own thread they would wait on themselves, and are
    refused.
    """

    def __init__(
        self,
        engine: AgensEngine | str | None = None,
        *,
        graph: Any = None,
        graph_name: str | None = None,
        index: AgensIndexConfig | IndexConfig | None = None,
        ttl: TTLConfig | None = None,
        setup: bool = True,
    ) -> None:
        AsyncBatchedBaseStore.__init__(self)
        self._init_core(engine, graph, graph_name, index, ttl, setup)
        self._sweeper_task: asyncio.Task[None] | None = None

    @classmethod
    @asynccontextmanager
    async def afrom_conn_string(
        cls, conninfo: str, *, graph_name: str, **kwargs: Any
    ) -> AsyncIterator[AsyncAgensStore]:
        """A store on an engine of its own, whose pools close with the block."""
        engine = AgensEngine(conninfo)
        store = cls(engine, graph_name=graph_name, **kwargs)
        try:
            yield store
        finally:
            await store.stop_ttl_sweeper()
            await engine.aclose()

    async def setup(self) -> None:
        """Make the store's tables now rather than on its first call."""
        await self.asetup()

    async def sweep_ttl(self, *, batch_size: int = 1000) -> int:
        """Delete expired items, a batch at a time; how many were deleted."""
        return await self.asweep_ttl(batch_size=batch_size)

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: float | None = None
    ) -> asyncio.Task[None]:
        """Sweep expired items every interval in a task on this loop, until stopped."""
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return self._sweeper_task
        interval = 60 * (
            sweep_interval_minutes or (self.ttl_config or {}).get("sweep_interval_minutes") or 5
        )

        async def loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.sweep_ttl()
                except Exception:  # noqa: BLE001 - a sweep that fails is tried again next time
                    logger.exception("sweeping expired store items failed")

        self._sweeper_task = asyncio.create_task(loop())
        return self._sweeper_task

    async def stop_ttl_sweeper(self) -> bool:
        """Stop the sweeper; whether it has stopped."""
        task, self._sweeper_task = self._sweeper_task, None
        if task is None:
            return True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True

    def close(self) -> None:
        """Stop the sweeper, which a blocking call can only cancel, and close."""
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            self._sweeper_task = None
        super().close()

    async def aclose(self) -> None:
        await self.stop_ttl_sweeper()
        await super().aclose()


__all__ = ["AgensIndexConfig", "AgensStore", "AsyncAgensStore"]
