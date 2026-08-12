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
property index over ``(prefix, key)`` serves point lookups. Each item also stores the
namespaces containing it, so "everything under this namespace" is a containment test
over that list rather than a range over the path -- a range compares two strings, and
jsonb compares with the database's collation, which on a linguistically collated
database put a descendant outside its own parent's range.

Embeddings are held outside the property bag, in a narrow table in a companion schema
keyed by graphid and indexed with HNSW. A foreign key ties each row to its vertex and
cascades, so deleting a memory drops its embedding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agensgraph import Vector
from agensgraph.introspect import DesiredIndex
from agensgraph.vector import search_option_statements
from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.embed import get_text_at_path
from psycopg import sql
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import AgensGraph, checked_names

# "." separates namespace labels; "/" is its byte-successor.
NS_SEP = "."
NS_SEP_NEXT = "/"

DEFAULT_LABEL = "StoreItem"
DEFAULT_VECTOR_TABLE = "item_vec"

_EVERY_NAMESPACE = 1_000_000
"""How many distinct namespaces one read will take when it has to take them all.

Only when something after the statement collapses rows -- a wildcard matched by position,
a depth limit that shortens a namespace -- can the page not be taken by the server. The
rows are distinct namespaces rather than items, so this counts the namespaces a store
holds and not the memories in them.
"""


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
    """Half-open range covering every strict descendant of ``prefix``.

    For narrowing in Python, where a ``str`` compares by code point and ``.`` (0x2E) is
    therefore followed by ``/`` (0x2F). The database compares a namespace as jsonb under
    its default collation, which orders punctuation differently and takes no ``COLLATE``,
    so a query narrows with :func:`ancestors_of` instead.
    """
    return prefix + NS_SEP, prefix + NS_SEP_NEXT


def ancestors_of(prefix: str) -> List[str]:
    """Every namespace that contains this one, itself included.

    ``"users.alice.memories"`` yields ``["users", "users.alice",
    "users.alice.memories"]``.

    Stored beside the prefix so that "is a descendant of p" becomes "holds p among its
    ancestors" -- an equality inside a list rather than a comparison between two
    strings, which no collation reorders. A GIN property index over the list serves it.
    """
    parts = prefix.split(NS_SEP)
    return [NS_SEP.join(parts[: i + 1]) for i in range(len(parts))]


class AgensStore(BaseStore):
    """Long-term memory for LangGraph agents, stored in an AgensGraph graph.

    Args:
        graph: An existing :class:`AgensGraph`. Supply this to share a connection pool
            with other components.
        conf: psycopg connection parameters, used when ``graph`` is not given.
        graph_name: Graph to create/use when constructing from ``conf``. label: Vertex
        label holding items. index: ``{"dims": int, "embed": Embeddings, "fields":
        [...]}`` to enable
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
            # Nothing here reads the schema, and describing a graph counts every vertex
            # and every edge in it, for a component that never looks at the answer.
            graph = AgensGraph(
                graph_name, conf, create=True, refresh_schema=False
            )
            self._owns_graph = True
        else:
            self._owns_graph = False
        self._graph = graph
        self._label = label
        self._index = index
        self._promoted = tuple(promoted or ())
        self._vector_schema = vector_schema or f"{graph.graph_name}_store"
        # Whether this server's pgvector can be told to keep looking when a filter rejects
        # what the distance index returned. Established when the extension is checked for.
        self._iterative_scan = False
        self._setup()

    # ---- schema setup ----

    def _setup(self) -> None:
        """Create the label, its indexes, and (when indexing) the embedding table."""
        checked_names(
            label=self._label,
            vector_schema=self._vector_schema,
            **{f"promoted[{i}]": name for i, name in enumerate(self._promoted)},
        )
        if self._promoted:
            # Promoted columns are part of how the label is declared, so this one cannot go
            # through the shared path, which creates a label and nothing else.
            self._graph.query(self._create_label_cypher())
        else:
            self._graph.create_labels(vertices=(self._label,))
        self._graph.ensure_indexes(
            [
                # An item is one namespace and one key, and a put merges on the pair, so
                # the index over it is unique -- otherwise two writers both look, both
                # find nothing, and both create. A read narrowing by namespace alone is
                # served by its leading column, so there is nothing else to index.
                DesiredIndex(
                    label=self._label,
                    properties=("prefix", "key"),
                    unique=True,
                    name=f"{self._label}_pk",
                ),
                # A descendant search asks whether a namespace is among an item's
                # ancestors, and GIN is the access method for a containment test over a
                # list.
                DesiredIndex(
                    label=self._label,
                    properties=("ancestors",),
                    method="gin",
                    name=f"{self._label}_ancestors",
                ),
            ]
        )
        if self._index:
            self._require_vector_extension()
            # An embedding is sent as itself once the types are registered rather than as
            # its decimal spelling, which for 1,536 dimensions is 6,152 bytes against
            # 21,504 and leaves the server nothing to parse.
            self._graph.register_vectors()
            for stmt in self._create_vector_table_sql():
                self._graph.query(stmt)

    def _require_vector_extension(self) -> None:
        """Semantic search needs pgvector, which AgensGraph does not bundle.

        The driver reads the catalogs for this, so there is one answer to the question
        rather than a second query of this package's own asking it a slightly different
        way.
        """
        version = self._graph.connection.vector_version()
        if version is not None:
            # Whether a filtered search can be told to keep looking rather than to stop at
            # a fixed number of candidates. Asked once here rather than per search.
            self._iterative_scan = version >= (0, 8)
            return
        raise RuntimeError(
            "AgensStore(index=...) needs the pgvector extension, which is not created "
            "in this database. Run CREATE EXTENSION vector as a role that may -- and if "
            "the extension is not on the server, build it against this server's "
            "pg_config first -- or omit `index` to use the store without semantic "
            "search."
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

        Written out rather than asked for through the driver's ``vector_index``, which
        describes a property index over a vector held in an element's properties. These
        embeddings are not held there: an item's value is a document a caller gave, and
        putting a thousand floats beside it means every read of the item detoasts them.
        So they sit in a table of their own, keyed by the element's identity, and an index
        over a plain table is not a property index.
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
            # The namespace test, in a form the search that joins this table can use.
            #
            # A property index is written in terms of the graph's own property accessor,
            # and that accessor cannot be spelled in plain SQL -- so the index over
            # `ancestors` serves the Cypher reads and is invisible to the search here,
            # which has to be SQL because it joins the embedding table. With this one the
            # planner narrows to the rows that could match and ranks those, instead of
            # ranking the whole table.
            #
            # Written out rather than asked for through `ensure_indexes`, which describes
            # property indexes and cannot describe this one.
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {label} "
                "USING gin ((properties -> 'ancestors') jsonb_ops)"
            ).format(
                name=sql.Identifier(f"{self._label}_ancestors_json"),
                label=sql.Identifier(self._graph.graph_name, self._label),
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
            params[f"p{i}"] = prefix
            params[f"k{i}"] = key
            terms.append(
                sql.SQL("(n.prefix = %({p})s AND n.key = %({k})s)").format(
                    p=sql.SQL(f"p{i}"), k=sql.SQL(f"k{i}")
                )
            )
        # ORed: these are the keys of one batch, and a row matching any of them is
        # wanted.
        return sql.SQL(" OR ").join(terms)

    @staticmethod
    def _namespace_predicate_named(
        prefix: str, params: Dict[str, Any], tag: str
    ) -> sql.Composed:
        """Match a namespace and all of its descendants.

        An item stores every namespace that contains it, so this asks whether the one
        being searched for is among them. That is an equality inside a list, which no
        collation can reorder, and a GIN property index over the list answers it.
        """
        params[f"ns_p{tag}"] = Jsonb([prefix])
        return sql.SQL("(n.ancestors @> %({p})s)").format(p=sql.SQL(f"ns_p{tag}"))

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
        operators descends into the nested key of the same name. Every form is expressed
        in the query, so paging still happens in the database.
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
        # A scalar goes as itself: the driver sends a string as text, which is what a
        # property index on the key is on. Wrapped as jsonb it is compared as jsonb, so
        # the index no longer matches and the comparison reads the label instead. A list
        # or a map stays jsonb, being what it is compared to.
        params[name] = Jsonb(operand) if isinstance(operand, (list, dict)) else operand
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
            "                n.updated_at = r.updated_at, n.ancestors = r.ancestors "
            "  ON MATCH  SET n.value = r.value, n.updated_at = r.updated_at, "
            "                n.ancestors = r.ancestors "
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

    def _recall_options(self, over_fetch: int) -> Dict[str, Any]:
        """What the index has to be told to fill a page that is narrowed afterwards.

        The distance index ranks every embedding in the table and knows nothing about
        namespaces, so a search of one namespace looks at the nearest few overall and keeps
        whichever of them happen to belong to it. Where a namespace holds a small fraction
        of the store, the nearest few hold none of it and a page comes back short.

        So the index is told to keep going until the page is full rather than to look at a
        fixed number of candidates: ``iterative_scan`` re-enters the index as the filter
        rejects what it returns. ``strict_order`` rather than ``relaxed_order`` because the
        candidates are ranked and cut afterwards, so their order is the answer.

        Raising ``ef_search`` alone does not do it: forty is already the default, so
        setting forty sets nothing, and a page of ten asks for exactly that many
        candidates. It is still raised alongside, for the pages that ask for more.

        Iterative scanning arrived in pgvector 0.8. Below that there is nothing to set, and
        a search there is as good as the over-fetch makes it.
        """
        # pgvector takes 1..1000 for ef_search and refuses anything outside, so a large
        # enough page would have made the search fail rather than merely under-recall.
        options: Dict[str, Any] = {"hnsw.ef_search": min(max(over_fetch, 40), 1000)}
        if self._iterative_scan:
            options["hnsw.iterative_scan"] = "strict_order"
        return options

    def _vector_search_sql(self, over_fetch: int) -> sql.Composed:
        """The nearest candidates *within the namespace*, by distance.

        The namespace is part of the search rather than something applied to its results.
        Ranked globally and narrowed afterwards, a search of one user's memories in a store
        holding many users returns only whatever of theirs happens to fall in the global
        nearest few, which for a store of any size is little or none of it.

        The distance is still computed only over the embedding table; the join to the
        label carries the containment test that decides which rows count.
        """
        return sql.SQL(
            "SELECT m.properties ->> 'prefix' AS prefix, "
            "       m.properties ->> 'key' AS key, "
            "       m.properties -> 'value' AS value, "
            "       m.properties ->> 'created_at' AS created_at, "
            "       m.properties ->> 'updated_at' AS updated_at, "
            "       t.dist AS dist "
            "FROM (SELECT v.id, v.embedding <=> %(qvec)s AS dist "
            "      FROM {vec} v "
            "      JOIN {label} mm ON mm.id = v.id "
            "      WHERE mm.properties -> 'ancestors' @> %(ns)s "
            "      ORDER BY dist LIMIT {n}) t "
            "JOIN {label} m ON m.id = t.id "
            "ORDER BY t.dist"
        ).format(
            vec=sql.Identifier(self._vector_schema, DEFAULT_VECTOR_TABLE),
            label=sql.Identifier(self._graph.graph_name, self._label),
            n=sql.SQL(str(int(over_fetch))),
        )

    def _upsert_vectors_sql(self, count: int) -> sql.Composed:
        values = sql.SQL(", ").join(
            sql.SQL("(%({i})s::graphid, %({e})s)").format(
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
        # LangGraph's fields are JSON paths -- "metadata.title", "items[0]", "items[*]"
        # -- not top-level keys. Read as keys, a dotted field matched nothing and every
        # item was embedded from the empty string, which ranks them all alike.
        parts: List[str] = []
        for field in fields:
            try:
                parts.extend(
                    str(found) for found in get_text_at_path(value, field) if found
                )
            except Exception:
                if field in value:
                    parts.append(str(value[field]))
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
                    "prefix": (prefix := flatten_namespace(op.namespace)),
                    "key": op.key,
                    "value": op.value,
                    "created_at": now,
                    "updated_at": now,
                    # Every namespace containing this one, so that a descendant search
                    # is an equality inside a list rather than a comparison the
                    # database's collation gets a say in.
                    "ancestors": ancestors_of(prefix),
                }
            )
        return rows

    @staticmethod
    def _to_embed(
        rows: List[Dict[str, Any]], puts: List[Tuple[int, PutOp]]
    ) -> List[Dict[str, Any]]:
        """The rows of a batch whose text is to be embedded.

        ``index=False`` on a put asks for the item to be stored and not embedded, so it
        is left out here -- otherwise the caller is charged for an embedding they declined
        and the item turns up in results they meant it to stay out of.
        """
        return [row for row, (_, op) in zip(rows, puts) if op.index is not False]

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
        :meth:`_apply_match_conditions` and yield no predicate here.
        """
        if not op.match_conditions or cls._has_wildcard(op):
            return None
        terms = []
        for i, cond in enumerate(op.match_conditions):
            joined = NS_SEP.join(cond.path)
            if cond.match_type == "prefix":
                terms.append(cls._namespace_predicate_named(joined, params, f"_mc{i}"))
            else:
                params[f"mc_s{i}"] = NS_SEP + joined
                params[f"mc_x{i}"] = joined
                terms.append(
                    sql.SQL(
                        "(n.prefix = %({x})s OR n.prefix ENDS WITH %({s})s)"
                    ).format(x=sql.SQL(f"mc_x{i}"), s=sql.SQL(f"mc_s{i}"))
                )
        # ANDed: LangGraph's reference applies `all(...)` over the conditions, so a
        # namespace has to satisfy every one of them.
        return sql.SQL(" AND ").join(terms)

    @staticmethod
    def _condition_ok(prefix: str, condition: MatchCondition) -> bool:
        """Whether a namespace satisfies one condition, ``*`` matching any one label.

        A namespace shorter than the path cannot satisfy it, and the labels are compared
        from the front for a prefix and from the back for a suffix.
        """
        parts = prefix.split(NS_SEP)
        path = condition.path
        if len(parts) < len(path):
            return False
        if condition.match_type == "prefix":
            pairs = zip(parts, path)
        elif condition.match_type == "suffix":
            pairs = zip(reversed(parts), reversed(path))
        else:
            raise ValueError(f"Unsupported match type: {condition.match_type}")
        return all(want == "*" or part == want for part, want in pairs)

    def _apply_match_conditions(
        self, prefixes: List[str], op: ListNamespacesOp
    ) -> List[str]:
        """Keep the namespaces satisfying every condition.

        Read here rather than in the statement when any condition holds a ``*``, since a
        wildcard matches a label by position and there is no range for a prefix index to
        narrow. Every condition is applied, not only the ones holding a wildcard: the
        statement emitted none of them in that case, so all of them are still to be
        applied, and a namespace has to satisfy all of them rather than any.
        """
        if not op.match_conditions or not self._has_wildcard(op):
            return prefixes
        return [
            prefix
            for prefix in prefixes
            if all(self._condition_ok(prefix, c) for c in op.match_conditions)
        ]

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
            indexed = self._to_embed(rows, puts)
            # Embedded before anything is written, and outside the transaction, because
            # this is a call to something that is not the database. Written first, the
            # item would be committed and then the call made -- and a call that fails
            # leaves an item nothing can find, for as long as it exists.
            vectors = (
                self._embed([self._text_for(r["value"]) for r in indexed])
                if self._index and indexed
                else []
            )

            def write() -> List[Dict[str, Any]]:
                with self._graph.transaction():
                    written = self._graph.query(
                        self._put_cypher(), {"rows": Jsonb(rows)}
                    )
                    if vectors:
                        self._write_vectors(indexed, written, vectors)
                    return written

            self._graph.merging(write)
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

    def _vector_params(
        self,
        rows: List[Dict[str, Any]],
        written: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> Tuple[int, Dict[str, Any]]:
        """Pair each embedding with the element it belongs to, by the key it was put at."""
        ids = {(r["prefix"], r["key"]): r["id"] for r in written}
        params: Dict[str, Any] = {}
        count = 0
        for row, vec in zip(rows, vectors):
            held = ids.get((row["prefix"], row["key"]))
            if held is None:
                continue
            params[f"vid{count}"] = held
            params[f"vec{count}"] = Vector(vec)
            count += 1
        return count, params

    def _write_vectors(
        self,
        rows: List[Dict[str, Any]],
        written: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        """Put the embeddings in the side table, beside the items they belong to."""
        count, params = self._vector_params(rows, written, vectors)
        if count:
            self._graph.query(self._upsert_vectors_sql(count), params)

    def _ranked_by_distance(
        self, over_fetch: int, qvec: List[float], prefix: str
    ) -> List[Dict[str, Any]]:
        """The nearest candidates, with the index told how many to look at.

        The option and the search go on one connection, which is the only place the option
        means anything: sent through a pool they would land on different backends, and the
        search meant to look further would not. Inside a transaction so that it ends with
        the search rather than tuning whoever borrows the connection next.
        """
        options = self._recall_options(over_fetch)
        statement = self._vector_search_sql(over_fetch)
        params = {"qvec": Vector(qvec), "ns": Jsonb([prefix])}
        if not options:
            return self._graph.query(statement, params)
        with self._graph.transaction() as conn:
            # In one round trip rather than one each, since nothing reads their results.
            conn.pipeline_batch(
                [(one, None) for one in search_option_statements(options, local=True)]
            )
            return self._graph.query(statement, params)

    async def _aranked_by_distance(
        self, over_fetch: int, qvec: List[float], prefix: str
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`_ranked_by_distance`."""
        options = self._recall_options(over_fetch)
        statement = self._vector_search_sql(over_fetch)
        params = {"qvec": Vector(qvec), "ns": Jsonb([prefix])}
        if not options:
            return await self._graph.aquery(statement, params)
        async with self._graph.atransaction() as conn:
            await conn.pipeline_batch(
                [(one, None) for one in search_option_statements(options, local=True)]
            )
            return await self._graph.aquery(statement, params)

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
        ranked = self._in_namespace(
            self._ranked_by_distance(over_fetch, qvec, prefix), prefix
        )
        if not (op.filter and ranked):
            return self._semantic_items(ranked, None, op)
        statement, params = self._filter_query(ranked, op.filter)
        return self._semantic_items(ranked, self._graph.query(statement, params), op)

    @staticmethod
    def _in_namespace(
        rows: List[Dict[str, Any]], prefix: str
    ) -> List[Dict[str, Any]]:
        """The candidates under the namespace searched, itself included.

        Read here because the rows are already in hand and a namespace is compared as a
        whole label: an exact string against an exact string, which is the same question
        however it is asked.
        """
        lo, hi = _descendant_bounds(prefix)
        return [r for r in rows if r["prefix"] == prefix or lo <= r["prefix"] < hi]

    def _filter_query(
        self, ranked: List[Dict[str, Any]], flt: Dict[str, Any]
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Which of the candidates pass the filter, asked of the database.

        The same predicate a search without a query uses, so one filter means one answer.
        Read in Python it would be a second implementation of every operator, and a
        comparison is a different question there: ``$gt`` over a value stored as text
        orders by type class in jsonb and by magnitude in Python, so the two answered
        oppositely and which one a caller got depended on whether they passed a query.

        The candidates are named by the two properties a composite index covers, so this
        is a point lookup per candidate rather than a scan.
        """
        params: Dict[str, Any] = {}
        pairs = [(row["prefix"], row["key"]) for row in ranked]
        predicate = sql.SQL("({keys}) AND ({f})").format(
            keys=self._key_predicate(pairs, params),
            f=self._filter_predicate(flt, params),
        )
        return self._select_keys_cypher(predicate), params

    def _select_keys_cypher(self, predicate: sql.Composed) -> sql.Composed:
        """Only which items matched: the values are already in hand."""
        return sql.SQL(
            "MATCH (n:{l}) WHERE {pred} RETURN n.prefix AS prefix, n.key AS key"
        ).format(l=sql.Identifier(self._label), pred=predicate)

    def _semantic_items(
        self,
        ranked: List[Dict[str, Any]],
        kept: Optional[List[Dict[str, Any]]],
        op: SearchOp,
    ) -> List[SearchItem]:
        """The candidates that survived, in the order the distance put them in.

        ``kept`` names the ones that passed a filter, or is ``None`` when there was no
        filter to pass.
        """
        keys = None if kept is None else {(r["prefix"], r["key"]) for r in kept}
        out: List[SearchItem] = []
        for row in ranked:
            if keys is not None and (row["prefix"], row["key"]) not in keys:
                continue
            value = row["value"]
            if isinstance(value, str):
                value = json.loads(value)
            out.append(
                self._row_to_item(
                    {**row, "value": value},
                    SearchItem,
                    score=1.0 - float(row["dist"]),
                )
            )
        return out[op.offset : op.offset + op.limit]

    def _collapses_rows(self, op: ListNamespacesOp) -> bool:
        """Whether anything after the statement can turn several rows into fewer.

        A wildcard condition is matched by position and a depth limit shortens a
        namespace, and either can drop or merge rows -- so the page cannot be taken until
        they have been. Without them the rows the statement returns are the answer, and
        the server can take the page itself.
        """
        return op.max_depth is not None or self._has_wildcard(op)

    def _namespace_page(
        self,
        op: ListNamespacesOp,
        predicate: Optional[sql.Composed],
        params: Dict[str, Any],
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """The statement that reads the namespaces, paged where paging is sound."""
        if self._collapses_rows(op):
            # Everything, because what comes back is not yet what is being counted. The
            # rows are distinct namespaces rather than items, so this is the number of
            # namespaces there are and not the number of memories.
            return self._list_namespaces_cypher(predicate, _EVERY_NAMESPACE, 0), params
        return (
            self._list_namespaces_cypher(predicate, op.limit, op.offset),
            params,
        )

    def _namespaces_from(
        self, rows: List[Dict[str, Any]], op: ListNamespacesOp
    ) -> List[Tuple[str, ...]]:
        """The namespaces a read found, narrowed and paged if that is still to do."""
        prefixes = [r["prefix"] for r in rows]
        if self._collapses_rows(op):
            prefixes = self._apply_match_conditions(prefixes, op)
            prefixes = self._truncate_depth(prefixes, op.max_depth)
            prefixes = prefixes[op.offset : op.offset + op.limit]
        return [unflatten_namespace(p) for p in prefixes]

    def _list_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        params: Dict[str, Any] = {}
        predicate = self._match_predicate_sql(op, params)
        rows = self._graph.query(*self._namespace_page(op, predicate, params))
        return self._namespaces_from(rows, op)

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
            indexed = self._to_embed(rows, puts)
            # Embedded first, for the reason given in :meth:`batch`.
            vectors = (
                await self._aembed([self._text_for(r["value"]) for r in indexed])
                if self._index and indexed
                else []
            )
            async def write() -> None:
                async with self._graph.atransaction():
                    written = await self._graph.aquery(
                        self._put_cypher(), {"rows": Jsonb(rows)}
                    )
                    if vectors:
                        await self._awrite_vectors(indexed, written, vectors)

            await self._graph.amerging(write)
            for i, _ in puts:
                results[i] = None

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
        self,
        rows: List[Dict[str, Any]],
        written: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        """Async sibling of :meth:`_write_vectors`."""
        count, params = self._vector_params(rows, written, vectors)
        if count:
            await self._graph.aquery(self._upsert_vectors_sql(count), params)

    async def _asearch(self, op: SearchOp) -> List[SearchItem]:
        prefix = flatten_namespace(op.namespace_prefix)
        if op.query and self._index:
            over_fetch = max(op.limit + op.offset, 1) * 4
            qvec = (await self._aembed([op.query]))[0]
            ranked = self._in_namespace(
                await self._aranked_by_distance(over_fetch, qvec, prefix), prefix
            )
            if not (op.filter and ranked):
                return self._semantic_items(ranked, None, op)
            statement, params = self._filter_query(ranked, op.filter)
            return self._semantic_items(
                ranked, await self._graph.aquery(statement, params), op
            )
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

    async def _alist_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        params: Dict[str, Any] = {}
        predicate = self._match_predicate_sql(op, params)
        rows = await self._graph.aquery(*self._namespace_page(op, predicate, params))
        return self._namespaces_from(rows, op)

    # ---- lifecycle ----

    def close(self) -> None:
        """Close the graph this store opened for itself.

        A store given a ``graph=`` was handed one that belongs to somebody else -- very
        likely shared with a vector store and a checkpointer -- and closing it from here
        took their connection out from under them.
        """
        if self._owns_graph:
            self._graph.close()

    async def aclose(self) -> None:
        """Async sibling of :meth:`close`."""
        if self._owns_graph:
            await self._graph.aclose()
            self._graph.close()


__all__ = ["AgensStore", "flatten_namespace", "unflatten_namespace"]
