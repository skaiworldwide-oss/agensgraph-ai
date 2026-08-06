"""LangGraph ``BaseStore`` backed by AgensGraph.

Stores long-term, cross-thread memory as graph vertices, so a memory is an ordinary
vertex that can be linked to other memories and to domain data with ordinary edges::

    (:StoreItem {prefix, key, value, created_at, updated_at})

``BaseStore`` declares only ``batch``/``abatch`` as abstract and derives ``get``,
``put``, ``search``, ``delete`` and ``list_namespaces`` from them, so those two entry
points carry the whole implementation. Each batch issues one statement per kind of
operation it contains, whatever the number of items.

A namespace tuple is stored as a ``.``-joined path, which LangGraph's own rejection of
``.`` inside a namespace label makes lossless and reversible by ``split``. A composite
property index over ``(prefix, key)`` serves both point lookups and namespace scans.

Embeddings are held outside the property bag, in a narrow table in a companion schema
keyed by graphid and indexed with HNSW. A foreign key ties each row to its vertex and
cascades, so deleting a memory drops its embedding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from psycopg import sql
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import AgensGraph

# "." separates namespace labels; "/" is its byte-successor and closes a
# descendant range.
NS_SEP = "."
NS_SEP_NEXT = "/"

DEFAULT_LABEL = "StoreItem"
DEFAULT_VECTOR_TABLE = "item_vec"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def flatten_namespace(namespace: Tuple[str, ...]) -> str:
    """Join a namespace tuple into its indexed path form.

    A namespace that could not be encoded unambiguously is rejected.
    """
    if not namespace:
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in namespace:
        if not label:
            raise InvalidNamespaceError(
                f"Namespace labels cannot be empty strings. Got {namespace}"
            )
        if NS_SEP in label:
            raise InvalidNamespaceError(
                f"Invalid namespace label {label!r} in {namespace}. "
                f"Namespace labels cannot contain periods ('{NS_SEP}')."
            )
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError(
            f'Root label for namespace cannot be "langgraph". Got: {namespace}'
        )
    return NS_SEP.join(namespace)


def unflatten_namespace(prefix: str) -> Tuple[str, ...]:
    """Recover the namespace tuple from its path form."""
    return tuple(prefix.split(NS_SEP))


def _descendant_bounds(prefix: str) -> Tuple[str, str]:
    """Half-open range covering every strict descendant of ``prefix``."""
    return prefix + NS_SEP, prefix + NS_SEP_NEXT


class AgensStore(BaseStore):
    """Long-term memory for LangGraph agents, stored in an AgensGraph graph.

    Args:
        graph: An existing :class:`AgensGraph`. Supply this to share a connection pool
            with other components.
        conf: psycopg connection parameters, used when ``graph`` is not given.
        graph_name: Graph to create/use when constructing from ``conf``.
        label: Vertex label holding items.
        index: ``{"dims": int, "embed": Embeddings, "fields": [...]}`` to enable
            semantic search. Omit it and the store is a plain key/value store.
        promoted: Property names to mirror into typed generated columns, so filters and
            sorts on them compare in the column's own type. A promoted value is held in
            both the column and the property map, and native comparison can order values
            differently from jsonb comparison.
        vector_schema: Schema holding the embedding table. Must not be a graph schema.
    """

    supports_ttl = False

    def __init__(
        self,
        graph: Optional[AgensGraph] = None,
        *,
        conf: Optional[Dict[str, Any]] = None,
        graph_name: str = "store",
        label: str = DEFAULT_LABEL,
        index: Optional[Dict[str, Any]] = None,
        promoted: Optional[Sequence[str]] = None,
        vector_schema: Optional[str] = None,
    ) -> None:
        if graph is None:
            if conf is None:
                raise ValueError("AgensStore requires either `graph` or `conf`.")
            graph = AgensGraph(graph_name, conf, create=True)
        self._graph = graph
        self._label = label
        self._index = index
        self._promoted = tuple(promoted or ())
        self._vector_schema = vector_schema or f"{graph.graph_name}_store"
        self._setup()

    # ---- schema setup ----

    def _setup(self) -> None:
        """Create the label, its indexes, and (when indexing) the embedding table."""
        self._graph.query(self._create_label_cypher())
        # Every read filters on prefix, and a point lookup on both.
        self._graph.query(
            self._create_property_index_cypher(f"{self._label}_pk", ("prefix", "key"))
        )
        self._graph.query(
            self._create_property_index_cypher(f"{self._label}_prefix", ("prefix",))
        )
        if self._index:
            self._require_vector_extension()
            for stmt in self._create_vector_table_sql():
                self._graph.query(stmt)

    def _require_vector_extension(self) -> None:
        """Semantic search needs pgvector, which AgensGraph does not bundle."""
        rows = self._graph.query(
            "SELECT count(*) AS n FROM pg_extension WHERE extname = 'vector'"
        )
        if not rows or int(next(iter(rows[0].values()))) == 0:
            raise RuntimeError(
                "AgensStore(index=...) needs the pgvector extension, which is not "
                "installed in this database. Build it against this server's pg_config "
                "and run CREATE EXTENSION vector, or omit `index` to use the store "
                "without semantic search."
            )

    def _create_label_cypher(self) -> sql.Composed:
        if self._promoted:
            cols = sql.SQL(", ").join(
                sql.SQL("{c} text GENERATED").format(c=sql.Identifier(name))
                for name in self._promoted
            )
            return sql.SQL("CREATE VLABEL IF NOT EXISTS {l} ({cols})").format(
                l=sql.Identifier(self._label), cols=cols
            )
        return sql.SQL("CREATE VLABEL IF NOT EXISTS {l}").format(
            l=sql.Identifier(self._label)
        )

    def _create_property_index_cypher(
        self, name: str, props: Sequence[str]
    ) -> sql.Composed:
        return sql.SQL(
            "CREATE PROPERTY INDEX IF NOT EXISTS {name} ON {l} ({cols})"
        ).format(
            name=sql.Identifier(name),
            l=sql.Identifier(self._label),
            cols=sql.SQL(", ").join(sql.Identifier(p) for p in props),
        )

    def _create_vector_table_sql(self) -> List[sql.Composed]:
        """DDL for the embedding table.

        A graph schema holds only labels, so the table lives in a companion schema and
        refers back to the label by graphid, cascading on delete.
        """
        dims = int(self._index["dims"])  # type: ignore[index]
        vec = sql.Identifier(self._vector_schema, DEFAULT_VECTOR_TABLE)
        return [
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {s}").format(
                s=sql.Identifier(self._vector_schema)
            ),
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {vec} ("
                "  id graphid PRIMARY KEY REFERENCES {label} (id) ON DELETE CASCADE,"
                "  embedding vector({dims})"
                ")"
            ).format(
                vec=vec,
                label=sql.Identifier(self._graph.graph_name, self._label),
                dims=sql.SQL(str(dims)),
            ),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {vec} "
                "USING hnsw (embedding vector_cosine_ops)"
            ).format(
                name=sql.Identifier(f"{DEFAULT_VECTOR_TABLE}_hnsw"), vec=vec
            ),
        ]

    # ---- predicate builders ----

    def _key_predicate(
        self, pairs: Sequence[Tuple[str, str]], params: Dict[str, Any]
    ) -> sql.Composed:
        """An OR of ``(prefix = .. AND key = ..)`` equalities.

        Each term is an equality on both indexed properties, so the composite index
        answers every key in the batch.
        """
        terms = []
        for i, (prefix, key) in enumerate(pairs):
            params[f"p{i}"] = Jsonb(prefix)
            params[f"k{i}"] = Jsonb(key)
            terms.append(
                sql.SQL("(n.prefix = %({p})s AND n.key = %({k})s)").format(
                    p=sql.SQL(f"p{i}"), k=sql.SQL(f"k{i}")
                )
            )
        return sql.SQL(" OR ").join(terms)

    @staticmethod
    def _namespace_predicate_named(
        prefix: str, params: Dict[str, Any], tag: str
    ) -> sql.Composed:
        """Match a namespace and all of its descendants.

        ``.`` (0x2E) separates namespace labels and ``/`` (0x2F) follows it, so a
        namespace and everything beneath it occupy the contiguous range
        ``[p, p/)`` on the leading index column, which the planner can seek and read in
        order. The range also spans siblings whose next character sorts below ``.``,
        such as ``p!x``; the trailing ``prefix = p OR prefix >= p.`` excludes them.
        """
        lo, hi = _descendant_bounds(prefix)
        params[f"ns_p{tag}"] = Jsonb(prefix)
        params[f"ns_lo{tag}"] = Jsonb(lo)
        params[f"ns_hi{tag}"] = Jsonb(hi)
        return sql.SQL(
            "(n.prefix >= %({p})s AND n.prefix < %({hi})s "
            " AND (n.prefix = %({p})s OR n.prefix >= %({lo})s))"
        ).format(
            p=sql.SQL(f"ns_p{tag}"),
            lo=sql.SQL(f"ns_lo{tag}"),
            hi=sql.SQL(f"ns_hi{tag}"),
        )

    def _namespace_predicate(
        self, prefix: str, params: Dict[str, Any], suffix: str = ""
    ) -> sql.Composed:
        return self._namespace_predicate_named(prefix, params, suffix)

    @staticmethod
    def _value_path(path: Sequence[str]) -> sql.Composed:
        """A property reference into ``value``, e.g. ``n.value.a.b``."""
        return sql.SQL(".").join(
            [sql.SQL("n"), sql.SQL("value")] + [sql.Identifier(p) for p in path]
        )

    def _filter_predicate(
        self, flt: Dict[str, Any], params: Dict[str, Any]
    ) -> sql.Composed:
        """Filters over ``value``, including comparison operators and nested keys.

        A field maps to a scalar or list for equality, or to a map of ``$`` operators
        (``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``). A map without
        operators descends into the nested key of the same name. Every form is
        expressed in the query, so paging still happens in the database.
        """
        terms: List[sql.Composed] = []
        counter = [0]

        def emit(path: List[str], want: Any) -> None:
            if isinstance(want, dict) and any(k.startswith("$") for k in want):
                for op, operand in want.items():
                    terms.append(
                        self._operator_term(path, op, operand, params, counter)
                    )
                return
            if isinstance(want, dict):
                for key, nested in want.items():
                    emit(path + [key], nested)
                return
            terms.append(self._operator_term(path, "$eq", want, params, counter))

        for field, want in flt.items():
            emit([field], want)
        return sql.SQL(" AND ").join(terms)

    def _operator_term(
        self,
        path: List[str],
        op: str,
        operand: Any,
        params: Dict[str, Any],
        counter: List[int],
    ) -> sql.Composed:
        name = f"f{counter[0]}"
        counter[0] += 1
        params[name] = Jsonb(operand)
        ref = self._value_path(path)
        placeholder = sql.SQL(f"%({name})s")
        if op == "$eq":
            return sql.SQL("{ref} = {v}").format(ref=ref, v=placeholder)
        if op == "$ne":
            # An absent key is unequal to anything, so it satisfies $ne.
            return sql.SQL("({ref} IS NULL OR {ref} <> {v})").format(
                ref=ref, v=placeholder
            )
        comparison = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}.get(op)
        if comparison is None:
            raise ValueError(f"Unsupported operator: {op}")
        return sql.SQL("{ref} {cmp} {v}").format(
            ref=ref, cmp=sql.SQL(comparison), v=placeholder
        )

    @classmethod
    def _matches_filter(cls, value: Any, flt: Dict[str, Any]) -> bool:
        """Evaluate a filter in Python, for rows already fetched by another path."""
        return all(cls._compare(value.get(k) if isinstance(value, dict) else None, v)
                   for k, v in flt.items())

    @classmethod
    def _compare(cls, value: Any, want: Any) -> bool:
        if isinstance(want, dict):
            if any(k.startswith("$") for k in want):
                return all(cls._apply(value, op, o) for op, o in want.items())
            if not isinstance(value, dict):
                return False
            return all(cls._compare(value.get(k), v) for k, v in want.items())
        if isinstance(want, (list, tuple)):
            return (
                isinstance(value, (list, tuple))
                and len(value) == len(want)
                and all(cls._compare(v, w) for v, w in zip(value, want))
            )
        return value == want

    @staticmethod
    def _apply(value: Any, op: str, operand: Any) -> bool:
        if op == "$eq":
            return value == operand
        if op == "$ne":
            return value != operand
        if op in ("$gt", "$gte", "$lt", "$lte"):
            if value is None:
                return False
            left, right = float(value), float(operand)
            return {
                "$gt": left > right,
                "$gte": left >= right,
                "$lt": left < right,
                "$lte": left <= right,
            }[op]
        raise ValueError(f"Unsupported operator: {op}")

    # ---- row <-> item ----

    @staticmethod
    def _row_to_item(row: Dict[str, Any], cls: Any = Item, **extra: Any) -> Any:
        value = row["value"]
        if isinstance(value, str):  # a bag value round-trips as text in some shapes
            value = json.loads(value)
        return cls(
            value=value,
            key=row["key"],
            namespace=unflatten_namespace(row["prefix"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            **extra,
        )

    # ---- statements ----

    def _put_cypher(self) -> sql.Composed:
        """Set-based upsert. ``created_at`` survives an update; ``updated_at`` moves."""
        return sql.SQL(
            "UNWIND %(rows)s AS r "
            "MERGE (n:{l} {{prefix: r.prefix, key: r.key}}) "
            "  ON CREATE SET n.value = r.value, n.created_at = r.created_at, "
            "                n.updated_at = r.updated_at "
            "  ON MATCH  SET n.value = r.value, n.updated_at = r.updated_at "
            "RETURN id(n) AS id, r.prefix AS prefix, r.key AS key"
        ).format(l=sql.Identifier(self._label))

    def _delete_cypher(self, predicate: sql.Composed) -> sql.Composed:
        return sql.SQL("MATCH (n:{l}) WHERE {pred} DETACH DELETE n").format(
            l=sql.Identifier(self._label), pred=predicate
        )

    def _get_cypher(self, predicate: sql.Composed) -> sql.Composed:
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} "
            "RETURN n.prefix AS prefix, n.key AS key, n.value AS value, "
            "       n.created_at AS created_at, n.updated_at AS updated_at"
        ).format(l=sql.Identifier(self._label), pred=predicate)

    def _search_cypher(
        self, predicate: sql.Composed, limit: int, offset: int
    ) -> sql.Composed:
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} "
            "RETURN n.prefix AS prefix, n.key AS key, n.value AS value, "
            "       n.created_at AS created_at, n.updated_at AS updated_at "
            "ORDER BY n.prefix, n.key SKIP {off} LIMIT {lim}"
        ).format(
            l=sql.Identifier(self._label),
            pred=predicate,
            off=sql.SQL(str(int(offset))),
            lim=sql.SQL(str(int(limit))),
        )

    def _list_namespaces_cypher(
        self, predicate: Optional[sql.Composed], limit: int, offset: int
    ) -> sql.Composed:
        where = (
            sql.SQL("WHERE {p} ").format(p=predicate) if predicate is not None
            else sql.SQL("")
        )
        return sql.SQL(
            "MATCH (n:{l}) {where}RETURN DISTINCT n.prefix AS prefix "
            "ORDER BY prefix SKIP {off} LIMIT {lim}"
        ).format(
            l=sql.Identifier(self._label),
            where=where,
            off=sql.SQL(str(int(offset))),
            lim=sql.SQL(str(int(limit))),
        )

    def _vector_search_sql(self, over_fetch: int) -> sql.Composed:
        """Top-N by cosine distance from HNSW, then look the vertices up by id.

        The distance scan covers only the embedding table, and the ids it returns are
        joined to the label's own relation on its ``id`` primary key, so the property
        bag is read once per hit.
        """
        return sql.SQL(
            "SELECT m.properties ->> 'prefix' AS prefix, "
            "       m.properties ->> 'key' AS key, "
            "       m.properties -> 'value' AS value, "
            "       m.properties ->> 'created_at' AS created_at, "
            "       m.properties ->> 'updated_at' AS updated_at, "
            "       t.dist AS dist "
            "FROM (SELECT v.id, v.embedding <=> %(qvec)s::vector AS dist "
            "      FROM {vec} v ORDER BY dist LIMIT {n}) t "
            "JOIN {label} m ON m.id = t.id "
            "ORDER BY t.dist"
        ).format(
            vec=sql.Identifier(self._vector_schema, DEFAULT_VECTOR_TABLE),
            label=sql.Identifier(self._graph.graph_name, self._label),
            n=sql.SQL(str(int(over_fetch))),
        )

    def _upsert_vectors_sql(self, count: int) -> sql.Composed:
        values = sql.SQL(", ").join(
            sql.SQL("(%({i})s::graphid, %({e})s::vector)").format(
                i=sql.SQL(f"vid{i}"), e=sql.SQL(f"vec{i}")
            )
            for i in range(count)
        )
        return sql.SQL(
            "INSERT INTO {vec} (id, embedding) VALUES {vals} "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
        ).format(
            vec=sql.Identifier(self._vector_schema, DEFAULT_VECTOR_TABLE), vals=values
        )

    # ---- embedding ----

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embed = self._index["embed"]  # type: ignore[index]
        if hasattr(embed, "embed_documents"):
            return embed.embed_documents(texts)
        return [embed(t) for t in texts]  # type: ignore[operator]

    async def _aembed(self, texts: List[str]) -> List[List[float]]:
        embed = self._index["embed"]  # type: ignore[index]
        if hasattr(embed, "aembed_documents"):
            return await embed.aembed_documents(texts)
        return self._embed(texts)

    def _text_for(self, value: Dict[str, Any]) -> str:
        """The text an item is embedded from: the named fields, else the whole value."""
        fields = (self._index or {}).get("fields")
        if not fields:
            return json.dumps(value, default=str)
        parts = [str(value[f]) for f in fields if f in value]
        return " ".join(parts)

    # ---- op grouping ----

    @staticmethod
    def _group(ops: Iterable[Op]) -> Tuple[
        List[Tuple[int, GetOp]],
        List[Tuple[int, PutOp]],
        List[Tuple[int, PutOp]],
        List[Tuple[int, SearchOp]],
        List[Tuple[int, ListNamespacesOp]],
    ]:
        """Split ops by kind, keeping each op's index so results can be reordered."""
        gets: List[Tuple[int, GetOp]] = []
        puts: List[Tuple[int, PutOp]] = []
        deletes: List[Tuple[int, PutOp]] = []
        searches: List[Tuple[int, SearchOp]] = []
        lists: List[Tuple[int, ListNamespacesOp]] = []
        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                gets.append((i, op))
            elif isinstance(op, PutOp):
                (deletes if op.value is None else puts).append((i, op))
            elif isinstance(op, SearchOp):
                searches.append((i, op))
            elif isinstance(op, ListNamespacesOp):
                lists.append((i, op))
            else:  # pragma: no cover - guards a future op kind
                raise NotImplementedError(f"Unsupported operation: {type(op).__name__}")
        return gets, puts, deletes, searches, lists

    def _put_rows(self, puts: List[Tuple[int, PutOp]]) -> List[Dict[str, Any]]:
        now = _utcnow()
        rows = []
        for _, op in puts:
            rows.append(
                {
                    "prefix": flatten_namespace(op.namespace),
                    "key": op.key,
                    "value": op.value,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        return rows

    @staticmethod
    def _has_wildcard(op: ListNamespacesOp) -> bool:
        return any(
            any(part == "*" for part in cond.path)
            for cond in (op.match_conditions or ())
        )

    @classmethod
    def _match_predicate_sql(
        cls, op: ListNamespacesOp, params: Dict[str, Any]
    ) -> Optional[sql.Composed]:
        """Express wildcard-free match conditions in SQL.

        A prefix condition reuses the namespace range; a suffix condition matches on the
        path's tail. Conditions holding ``*`` are matched positionally by
        :meth:`_apply_wildcards` and yield no predicate here.
        """
        if not op.match_conditions or cls._has_wildcard(op):
            return None
        terms = []
        for i, cond in enumerate(op.match_conditions):
            joined = NS_SEP.join(cond.path)
            if cond.match_type == "prefix":
                terms.append(cls._namespace_predicate_named(joined, params, f"_mc{i}"))
            else:
                params[f"mc_s{i}"] = Jsonb(NS_SEP + joined)
                params[f"mc_x{i}"] = Jsonb(joined)
                terms.append(
                    sql.SQL(
                        "(n.prefix = %({x})s OR n.prefix ENDS WITH %({s})s)"
                    ).format(x=sql.SQL(f"mc_x{i}"), s=sql.SQL(f"mc_s{i}"))
                )
        return sql.SQL(" OR ").join(terms)

    @staticmethod
    def _wildcard_ok(prefix: str, path: Sequence[str], match_type: str) -> bool:
        """Match a namespace path against a pattern whose labels may be ``*``."""
        parts = prefix.split(NS_SEP)
        window = parts[: len(path)] if match_type == "prefix" else parts[-len(path):]
        if len(window) != len(path):
            return False
        return all(p == "*" or p == w for p, w in zip(path, window))

    def _apply_wildcards(
        self, prefixes: List[str], op: ListNamespacesOp
    ) -> List[str]:
        if not op.match_conditions:
            return prefixes
        wild = [c for c in op.match_conditions if any(p == "*" for p in c.path)]
        if not wild:
            return prefixes
        plain = [c for c in op.match_conditions if not any(p == "*" for p in c.path)]
        out = []
        for prefix in prefixes:
            if any(self._wildcard_ok(prefix, c.path, c.match_type) for c in wild):
                out.append(prefix)
            elif plain:
                out.append(prefix)
        return out

    @staticmethod
    def _truncate_depth(prefixes: List[str], max_depth: Optional[int]) -> List[str]:
        if max_depth is None:
            return prefixes
        seen, out = set(), []
        for prefix in prefixes:
            trimmed = NS_SEP.join(prefix.split(NS_SEP)[:max_depth])
            if trimmed not in seen:
                seen.add(trimmed)
                out.append(trimmed)
        return out

    # ---- sync API ----

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        ops = list(ops)
        results: List[Result] = [None] * len(ops)
        gets, puts, deletes, searches, lists = self._group(ops)

        if deletes:
            params: Dict[str, Any] = {}
            pairs = [
                (flatten_namespace(op.namespace), op.key) for _, op in deletes
            ]
            self._graph.query(
                self._delete_cypher(self._key_predicate(pairs, params)), params
            )
            for i, _ in deletes:
                results[i] = None

        if puts:
            rows = self._put_rows(puts)
            written = self._graph.query(self._put_cypher(), {"rows": Jsonb(rows)})
            if self._index:
                self._write_vectors(rows, written)
            for i, _ in puts:
                results[i] = None

        if gets:
            params = {}
            pairs = [(flatten_namespace(op.namespace), op.key) for _, op in gets]
            rows = self._graph.query(
                self._get_cypher(self._key_predicate(pairs, params)), params
            )
            found = {(r["prefix"], r["key"]): r for r in rows}
            for i, op in gets:
                row = found.get((flatten_namespace(op.namespace), op.key))
                results[i] = self._row_to_item(row) if row else None

        for i, op in searches:
            results[i] = self._search(op)

        for i, op in lists:
            results[i] = self._list_namespaces(op)

        return results

    def _write_vectors(
        self, rows: List[Dict[str, Any]], written: List[Dict[str, Any]]
    ) -> None:
        """Embed the written items and upsert them into the side table."""
        ids = {(r["prefix"], r["key"]): r["id"] for r in written}
        targets = [r for r in rows if (r["prefix"], r["key"]) in ids]
        if not targets:
            return
        vectors = self._embed([self._text_for(r["value"]) for r in targets])
        params: Dict[str, Any] = {}
        for i, (row, vec) in enumerate(zip(targets, vectors)):
            params[f"vid{i}"] = ids[(row["prefix"], row["key"])]
            params[f"vec{i}"] = json.dumps(vec)
        self._graph.query(self._upsert_vectors_sql(len(targets)), params)

    def _search(self, op: SearchOp) -> List[SearchItem]:
        prefix = flatten_namespace(op.namespace_prefix)
        if op.query and self._index:
            return self._semantic_search(op, prefix)
        params: Dict[str, Any] = {}
        predicate = self._namespace_predicate(prefix, params)
        if op.filter:
            predicate = sql.SQL("({ns}) AND ({f})").format(
                ns=predicate, f=self._filter_predicate(op.filter, params)
            )
        rows = self._graph.query(
            self._search_cypher(predicate, op.limit, op.offset), params
        )
        return [self._row_to_item(r, SearchItem, score=None) for r in rows]

    def _semantic_search(self, op: SearchOp, prefix: str) -> List[SearchItem]:
        """Rank by distance, then narrow to the namespace and filter.

        The distance index ranks the whole table, so candidates are over-fetched and
        narrowed afterwards.
        """
        over_fetch = max(op.limit + op.offset, 1) * 4
        qvec = self._embed([op.query])[0]  # type: ignore[list-item]
        rows = self._graph.query(
            self._vector_search_sql(over_fetch), {"qvec": json.dumps(qvec)}
        )
        lo, hi = _descendant_bounds(prefix)
        out: List[SearchItem] = []
        for row in rows:
            row_prefix = row["prefix"]
            if not (row_prefix == prefix or (lo <= row_prefix < hi)):
                continue
            value = row["value"]
            if isinstance(value, str):
                value = json.loads(value)
            if op.filter and not self._matches_filter(value, op.filter):
                continue
            out.append(
                self._row_to_item(
                    {**row, "value": value},
                    SearchItem,
                    score=1.0 - float(row["dist"]),
                )
            )
        return out[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        params: Dict[str, Any] = {}
        predicate = self._match_predicate_sql(op, params)
        # Depth truncation and wildcards collapse rows, so paginate after them.
        rows = self._graph.query(
            self._list_namespaces_cypher(predicate, 1_000_000, 0), params
        )
        prefixes = [r["prefix"] for r in rows]
        prefixes = self._apply_wildcards(prefixes, op)
        prefixes = self._truncate_depth(prefixes, op.max_depth)
        page = prefixes[op.offset : op.offset + op.limit]
        return [unflatten_namespace(p) for p in page]

    # ---- async API ----

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        ops = list(ops)
        results: List[Result] = [None] * len(ops)
        gets, puts, deletes, searches, lists = self._group(ops)

        if deletes:
            params: Dict[str, Any] = {}
            pairs = [(flatten_namespace(op.namespace), op.key) for _, op in deletes]
            await self._graph.aquery(
                self._delete_cypher(self._key_predicate(pairs, params)), params
            )

        if puts:
            rows = self._put_rows(puts)
            written = await self._graph.aquery(
                self._put_cypher(), {"rows": Jsonb(rows)}
            )
            if self._index:
                await self._awrite_vectors(rows, written)

        if gets:
            params = {}
            pairs = [(flatten_namespace(op.namespace), op.key) for _, op in gets]
            rows = await self._graph.aquery(
                self._get_cypher(self._key_predicate(pairs, params)), params
            )
            found = {(r["prefix"], r["key"]): r for r in rows}
            for i, op in gets:
                row = found.get((flatten_namespace(op.namespace), op.key))
                results[i] = self._row_to_item(row) if row else None

        for i, op in searches:
            results[i] = await self._asearch(op)

        for i, op in lists:
            results[i] = await self._alist_namespaces(op)

        return results

    async def _awrite_vectors(
        self, rows: List[Dict[str, Any]], written: List[Dict[str, Any]]
    ) -> None:
        ids = {(r["prefix"], r["key"]): r["id"] for r in written}
        targets = [r for r in rows if (r["prefix"], r["key"]) in ids]
        if not targets:
            return
        vectors = await self._aembed([self._text_for(r["value"]) for r in targets])
        params: Dict[str, Any] = {}
        for i, (row, vec) in enumerate(zip(targets, vectors)):
            params[f"vid{i}"] = ids[(row["prefix"], row["key"])]
            params[f"vec{i}"] = json.dumps(vec)
        await self._graph.aquery(self._upsert_vectors_sql(len(targets)), params)

    async def _asearch(self, op: SearchOp) -> List[SearchItem]:
        prefix = flatten_namespace(op.namespace_prefix)
        if op.query and self._index:
            over_fetch = max(op.limit + op.offset, 1) * 4
            qvec = (await self._aembed([op.query]))[0]
            rows = await self._graph.aquery(
                self._vector_search_sql(over_fetch), {"qvec": json.dumps(qvec)}
            )
            return self._narrow_semantic(rows, op, prefix)
        params: Dict[str, Any] = {}
        predicate = self._namespace_predicate(prefix, params)
        if op.filter:
            predicate = sql.SQL("({ns}) AND ({f})").format(
                ns=predicate, f=self._filter_predicate(op.filter, params)
            )
        rows = await self._graph.aquery(
            self._search_cypher(predicate, op.limit, op.offset), params
        )
        return [self._row_to_item(r, SearchItem, score=None) for r in rows]

    def _narrow_semantic(
        self, rows: List[Dict[str, Any]], op: SearchOp, prefix: str
    ) -> List[SearchItem]:
        lo, hi = _descendant_bounds(prefix)
        out: List[SearchItem] = []
        for row in rows:
            row_prefix = row["prefix"]
            if not (row_prefix == prefix or (lo <= row_prefix < hi)):
                continue
            value = row["value"]
            if isinstance(value, str):
                value = json.loads(value)
            if op.filter and not self._matches_filter(value, op.filter):
                continue
            out.append(
                self._row_to_item(
                    {**row, "value": value}, SearchItem, score=1.0 - float(row["dist"])
                )
            )
        return out[op.offset : op.offset + op.limit]

    async def _alist_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        params: Dict[str, Any] = {}
        predicate = self._match_predicate_sql(op, params)
        rows = await self._graph.aquery(
            self._list_namespaces_cypher(predicate, 1_000_000, 0), params
        )
        prefixes = [r["prefix"] for r in rows]
        prefixes = self._apply_wildcards(prefixes, op)
        prefixes = self._truncate_depth(prefixes, op.max_depth)
        page = prefixes[op.offset : op.offset + op.limit]
        return [unflatten_namespace(p) for p in page]

    # ---- lifecycle ----

    def close(self) -> None:
        self._graph.close()

    async def aclose(self) -> None:
        await self._graph.aclose()


__all__ = ["AgensStore", "flatten_namespace", "unflatten_namespace"]
