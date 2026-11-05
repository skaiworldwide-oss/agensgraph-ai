"""The statements a vector store sends, built from where its pieces live.

A chunk is a vertex of the store's label. Its property map holds the text, the metadata and the
caller's id under ``__id__``, which a unique property index keeps unique. Its embedding is a row of
the driver's vector table in the companion schema ``"<graph>_store"``, keyed by the vertex's id with
a cascading foreign key, so deleting the vertex deletes its vector. With a keyword search, a stored
``tsvector`` for each chunk is a row of a second table keyed the same way.

A write is one pipeline, so one round trip and one implicit transaction: the vertices are updated
where their properties changed and created where missing, then the vector and text rows are
upserted through a Cypher lookup of each vertex's id by ``__id__``. A search is plain SQL over the
vector table joined to the label by id, which is what lets a filter and an approximate index work
together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from agensgraph import VectorStorage

from .._core.names import (
    bounded,
    companion_schema,
    companion_table,
    ident,
    literal,
    printed,
    qualified,
)
from .config import KeywordConfig

ID_KEY = "__id__"


@dataclass(frozen=True)
class Layout:
    """Where one store's chunks, vectors and text live, and the statements that use them."""

    graph: str
    label: str
    text_property: str
    storage: VectorStorage
    keyword: KeywordConfig | None
    filter_keys: tuple[str, ...]

    # -- names -----------------------------------------------------------------------------

    @cached_property
    def store_schema(self) -> str:
        return companion_schema(self.graph)

    @cached_property
    def label_table(self) -> str:
        return qualified(self.graph, self.label)

    @cached_property
    def cypher_label(self) -> str:
        return ident(self.label)

    @cached_property
    def vector_table_name(self) -> str:
        name = getattr(self.storage.storage, "name", None)
        assert name is not None, "a vector store keeps its vectors in a table"
        return str(name)

    @cached_property
    def vector_table(self) -> str:
        return qualified(self.store_schema, self.vector_table_name)

    @cached_property
    def text_table_name(self) -> str:
        return companion_table(self.label, self.storage.property, "fts")

    @cached_property
    def text_table(self) -> str:
        return qualified(self.store_schema, self.text_table_name)

    @cached_property
    def column(self) -> str:
        return ident(self.storage.property)

    @cached_property
    def typed(self) -> str:
        return f"{self.storage.type}({self.storage.dimensions})"

    @cached_property
    def operator(self) -> str:
        return str(self.storage.distance.value)

    @cached_property
    def id_index(self) -> str:
        return bounded(f"{self.label}__id__")

    def filter_index(self, key: str) -> str:
        return bounded(f"{self.label}__meta__{key}")

    @cached_property
    def catch_all_index(self) -> str:
        return bounded(f"{self.label}__meta")

    @cached_property
    def text_index(self) -> str:
        return bounded(f"{self.text_table_name}_gin")

    # -- setup -----------------------------------------------------------------------------

    def required(self) -> list[tuple[str, str]]:
        """Every relation the store needs, as ``(schema, name)``, for one existence check."""
        names = [
            (self.graph, self.label),
            (self.graph, self.id_index),
            (self.graph, self.catch_all_index),
            (self.store_schema, self.vector_table_name),
        ]
        names += [(self.graph, self.filter_index(key)) for key in self.filter_keys]
        if self.keyword is not None:
            names += [
                (self.store_schema, self.text_table_name),
                (self.store_schema, self.text_index),
            ]
        return names

    @cached_property
    def create_label(self) -> str:
        return f"CREATE VLABEL IF NOT EXISTS {self.cypher_label}"

    @cached_property
    def create_id_index(self) -> str:
        return (
            f"CREATE UNIQUE PROPERTY INDEX IF NOT EXISTS {ident(self.id_index)} "
            f"ON {self.cypher_label} ({ID_KEY})"
        )

    @cached_property
    def create_schema(self) -> str:
        return f"CREATE SCHEMA IF NOT EXISTS {ident(self.store_schema)}"

    @cached_property
    def create_catch_all_index(self) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS {ident(self.catch_all_index)} ON {self.label_table} "
            "USING gin (properties jsonb_path_ops)"
        )

    def create_filter_index(self, key: str) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS {ident(self.filter_index(key))} ON {self.label_table} "
            f"((properties->{literal(key)}))"
        )

    @cached_property
    def create_text_table(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.text_table} ("
            f"id graphid PRIMARY KEY REFERENCES {self.label_table} (id) ON DELETE CASCADE, "
            "tsv tsvector NOT NULL)"
        )

    @cached_property
    def create_text_index(self) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS {ident(self.text_index)} ON {self.text_table} "
            "USING gin (tsv)"
        )

    # -- writes ----------------------------------------------------------------------------

    @cached_property
    def update_vertices(self) -> str:
        """Replace the properties of each vertex whose properties differ from the row's."""
        return (
            f"UNWIND %s AS row MATCH (c:{self.cypher_label} {{{ID_KEY}: row.{ID_KEY}}}) "
            "WITH c, row WHERE properties(c) <> row SET c = row"
        )

    @cached_property
    def create_vertices(self) -> str:
        """Create a vertex for each row whose key no vertex has."""
        return (
            f"UNWIND %s AS row OPTIONAL MATCH (c:{self.cypher_label} {{{ID_KEY}: row.{ID_KEY}}}) "
            f"WITH row, c WHERE c IS NULL CREATE (:{self.cypher_label} =row)"
        )

    @cached_property
    def _ids_by_key(self) -> str:
        return (
            f"WITH ids AS (UNWIND %s AS k MATCH (c:{self.cypher_label} {{{ID_KEY}: k}}) "
            "RETURN k AS k, id(c) AS id) "
        )

    @cached_property
    def upsert_vectors(self) -> str:
        """Write each key's vector, leaving one that is already the same untouched."""
        return (
            self._ids_by_key + f"INSERT INTO {self.vector_table} AS v (id, {self.column}) "
            f"SELECT ids.id, x.emb::{self.typed} FROM ids "
            "JOIN unnest(%s::text[], %s::vector[]) AS x(k, emb) ON ids.k = to_jsonb(x.k) "
            f"ON CONFLICT (id) DO UPDATE SET {self.column} = EXCLUDED.{self.column} "
            f"WHERE v.{self.column} IS DISTINCT FROM EXCLUDED.{self.column}"
        )

    @cached_property
    def upsert_text(self) -> str:
        """Write each key's tsvector, computed by the server from the text given."""
        return (
            self._ids_by_key + f"INSERT INTO {self.text_table} AS f (id, tsv) "
            "SELECT ids.id, to_tsvector(%s::regconfig, x.body) FROM ids "
            "JOIN unnest(%s::text[], %s::text[]) AS x(k, body) ON ids.k = to_jsonb(x.k) "
            "ON CONFLICT (id) DO UPDATE SET tsv = EXCLUDED.tsv "
            "WHERE f.tsv IS DISTINCT FROM EXCLUDED.tsv"
        )

    @cached_property
    def delete_by_key(self) -> str:
        return f"UNWIND %s AS k MATCH (c:{self.cypher_label} {{{ID_KEY}: k}}) DETACH DELETE c"

    def keys_where(self, where: str) -> str:
        return f"SELECT l.properties->>{literal(ID_KEY)} FROM {self.label_table} l WHERE {where}"

    @cached_property
    def vector_table_regclass(self) -> str:
        """The vector table's name as ``::regclass`` reads it, for a parameter."""
        return printed(self.store_schema, self.vector_table_name)

    FAILED_INDEX_BUILDS = (
        "SELECT i.indexrelid::regclass::text FROM pg_index i "
        "WHERE i.indrelid = %s::regclass AND NOT i.indisvalid "
        "AND NOT EXISTS (SELECT 1 FROM pg_stat_progress_create_index p "
        "WHERE p.index_relid = i.indexrelid)"
    )
    """The invalid indexes of the table named by the parameter that no session is building: what
    a concurrent build that failed leaves behind."""

    @cached_property
    def clear_tables(self) -> str:
        tables = [self.vector_table] + ([self.text_table] if self.keyword is not None else [])
        return "TRUNCATE " + ", ".join(tables)

    @cached_property
    def clear_vertices(self) -> str:
        return f"MATCH (c:{self.cypher_label}) DETACH DELETE c"

    # -- reads -----------------------------------------------------------------------------

    @cached_property
    def get_by_key(self) -> str:
        return f"UNWIND %s AS k MATCH (c:{self.cypher_label} {{{ID_KEY}: k}}) RETURN properties(c)"

    def nearest(self, where: str, *, embeddings: bool, exact: bool) -> str:
        """The ``%s`` nearest chunks to ``%s`` passing ``where``: properties, distance, the
        stored embedding when asked for, and the vertex's id.

        Ordered by distance alone, which the approximate index answers; ``exact`` adds the id as a
        tie-break, which no index answers, so the search reads every row.
        """
        extra = f", v.{self.column}" if embeddings else ""
        order = "distance, l.id" if exact else "distance"
        return (
            f"SELECT l.properties, v.{self.column} {self.operator} %s::{self.typed} AS distance"
            f"{extra}, l.id FROM {self.vector_table} v JOIN {self.label_table} l ON l.id = v.id "
            f"WHERE {where} ORDER BY {order} LIMIT %s"
        )

    def hybrid(self, where: str, *, exact: bool) -> str:
        """The vector and keyword candidates for a fused search, one row per chunk in either.

        Parameters: the query vector, the filter's parameters, the vector candidates' number, the
        regconfig, the query text, the filter's parameters again, the keyword candidates' number.
        Each row: properties, distance (or null), vector rank, keyword score (or null), keyword
        rank, the vertex's id.
        """
        order = "distance, l.id" if exact else "distance"
        return (
            "WITH vh AS (SELECT id, distance, row_number() OVER (ORDER BY distance, id) AS r FROM ("
            f"SELECT l.id, v.{self.column} {self.operator} %s::{self.typed} AS distance "
            f"FROM {self.vector_table} v JOIN {self.label_table} l ON l.id = v.id "
            f"WHERE {where} ORDER BY {order} LIMIT %s) s), "
            "kh AS (SELECT id, score, row_number() OVER (ORDER BY score DESC, id) AS r FROM ("
            "SELECT l.id, ts_rank_cd(f.tsv, q) AS score "
            f"FROM {self.text_table} f JOIN {self.label_table} l ON l.id = f.id, "
            "websearch_to_tsquery(%s::regconfig, %s) q "
            f"WHERE f.tsv @@ q AND {where} ORDER BY score DESC, l.id LIMIT %s) s) "
            "SELECT l.properties, vh.distance, vh.r, kh.score, kh.r, l.id "
            "FROM vh FULL OUTER JOIN kh ON vh.id = kh.id "
            f"JOIN {self.label_table} l ON l.id = coalesce(vh.id, kh.id)"
        )

    def bridge(self, retrieval_query: str) -> str:
        """The hits, matched again in Cypher, then the caller's retrieval query.

        ``node`` is the chunk's vertex and ``score`` the hit's score, handed on best first, as the
        search ranked them; the retrieval query reads them and returns ``text``, ``score`` and
        ``metadata``, and may return ``doc_id``.
        """
        return (
            f"UNWIND %s AS hit MATCH (node:{self.cypher_label}) "
            "WHERE id(node) = (hit.gid)::graphid "
            f"WITH node, hit.score AS score, hit.rank AS hit_order ORDER BY hit_order "
            f"{retrieval_query}"
        )

    @cached_property
    def count_vectors(self) -> str:
        return f"SELECT count(*) FROM {self.vector_table}"

    VECTOR_DIMENSIONS = (
        "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(%s) AND a.attname = %s AND NOT a.attisdropped"
    )
    """The declared type of an existing vector table's column, ``vector(1536)``, or no row."""


def keyword_text(properties: dict[str, object], fields: Sequence[str]) -> str:
    """The text a chunk's keyword search reads: the named properties' values, in order."""
    parts = []
    for field in fields:
        value = properties.get(field)
        if value is None:
            continue
        parts.append(value if isinstance(value, str) else str(value))
    return "\n".join(parts)
