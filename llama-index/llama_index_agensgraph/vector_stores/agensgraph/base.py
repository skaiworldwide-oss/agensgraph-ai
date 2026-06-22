from typing import (
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
import logging

import json, re
from contextlib import asynccontextmanager, contextmanager

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.filters import metadata_filters_to_cypher

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

# Maximum rows per UNWIND batch when ingesting nodes, so a large ``add`` does
# not ship one oversized parameter / build one huge server-side list.
CHUNK_SIZE = 1000

get_vector_index_info_function = r"""
    CREATE OR REPLACE FUNCTION ag_list_vector_indexes(
        index_name text DEFAULT NULL,
        node_label text DEFAULT NULL,
        embedding_node_property text DEFAULT NULL
    )
    RETURNS TABLE (
        name text,
        labelortype text,
        property text,
        entitytype text,
        dimensions int
    )
    LANGUAGE sql
    AS $$
        SELECT
            c.relname AS name,
            l.labname AS labelOrType,
            CASE
                WHEN indexdef ~ '\(+([a-zA-Z_][a-zA-Z0-9_]*)\)+::' THEN regexp_replace(indexdef, '.*\(+([a-zA-Z_][a-zA-Z0-9_]*)\)+::.*', '\1')
                ELSE NULL
            END AS property,
            CASE
                WHEN l.labkind = 'v' THEN 'NODE'
                WHEN l.labkind = 'e' THEN 'RELATIONSHIP'
                ELSE 'UNKNOWN'
            END AS entityType,
            CASE
                WHEN indexdef ~ 'vector\((\d+)\)' THEN (regexp_match(indexdef, 'vector\((\d+)\)'))[1]::int
                ELSE NULL
            END AS dimensions
        FROM
            pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        JOIN pg_catalog.ag_label l ON i.indrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        JOIN LATERAL pg_catalog.ag_get_propindexdef(c.oid) AS indexdef ON true
        WHERE
            g.graphname = current_setting('graph_path')
            AND i.indexprs IS NOT NULL
            AND indexdef ~ '::vector\(\d+\)'
            AND (
                index_name IS NULL AND node_label IS NULL AND embedding_node_property IS NULL
                OR (
                    (index_name IS NOT NULL AND c.relname = index_name)
                    OR (
                        node_label IS NOT NULL
                        AND embedding_node_property IS NOT NULL
                        AND l.labname = node_label
                        AND CASE
                            WHEN indexdef ~ '\(\(\(([^)]+)\)::' THEN regexp_replace(indexdef, '.*\(\(\(([^)]+)\)::.*', '\1')
                            ELSE NULL
                        END = embedding_node_property
                    )
                )
            )
    $$;


"""

get_keyword_index_info_function = r"""
    CREATE OR REPLACE FUNCTION ag_list_text_indexes(
        index_name text DEFAULT NULL,
        node_label text DEFAULT NULL,
        text_node_properties text[] DEFAULT NULL
    )
    RETURNS TABLE (
        name text,
        labelortype text,
        properties text[],
        entitytype text
    )
    LANGUAGE sql
    AS $$
    WITH extracted_props AS (
        SELECT
            c.relname AS name,
            l.labname AS labelOrType,
            ARRAY(
                SELECT
                    trim(both '"' from trim(m[1]))
                FROM
                    regexp_matches(indexdef, 'to_tsvector\((?:[^,]+),\s*([^)]+)\)', 'g') AS m
            ) AS props,
            CASE
                WHEN l.labkind = 'v' THEN 'NODE'
                WHEN l.labkind = 'e' THEN 'RELATIONSHIP'
                ELSE 'UNKNOWN'
            END AS entityType
        FROM
            pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        JOIN pg_catalog.ag_label l ON i.indrelid = l.relid
        JOIN pg_catalog.ag_graph g ON l.graphid = g.oid
        JOIN LATERAL pg_catalog.ag_get_propindexdef(c.oid) AS indexdef ON true
        WHERE
            g.graphname = current_setting('graph_path')
            AND i.indexprs IS NOT NULL
            AND indexdef ~ 'to_tsvector\('
    )
    SELECT
        name,
        labelOrType,
        props AS properties,
        entityType
    FROM
        extracted_props
    WHERE
        (
            (index_name IS NULL OR name = index_name)
            AND (node_label IS NULL OR labelOrType = node_label)
            AND (
                text_node_properties IS NULL OR
                array(SELECT unnest(props) ORDER BY 1) = array(SELECT unnest(text_node_properties) ORDER BY 1)
            )
        );
    $$;

"""

def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if variable is not null and raise error accordingly."""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")


def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists."""
    return sorted(lst, key=lambda x: x.get("name") != index_name)


def clean_params(params: List[BaseNode]) -> List[Dict[str, Any]]:
    """Convert BaseNode object to a dictionary to be imported into Agensgraph."""
    clean_params = []
    for record in params:
        text = record.get_content(metadata_mode=MetadataMode.NONE)
        embedding = record.get_embedding()
        id = record.node_id
        metadata = node_to_metadata_dict(record, remove_text=True, flat_metadata=False)
        # Remove redundant metadata information
        for k in ["document_id", "doc_id"]:
            del metadata[k]
        clean_params.append(
            {"text": text, "embedding": embedding, "id": id, "metadata": metadata}
        )
    return clean_params


def remove_lucene_chars(text: Optional[str]) -> Optional[str]:
    """Remove Lucene special characters."""
    if not text:
        return None
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


# Metadata-filter translation now lives in
# ``llama_index_agensgraph.filters.metadata_filters_to_cypher`` (shared with the
# property graph store and supporting all 14 FilterOperators).

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

_vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
_edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")

class AgensgraphVectorStore(BasePydanticVectorStore):
    """
    Agensgraph Vector Store.

    Examples:
        # `pip install TODO`


        ```python
        from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

        url = "postgresql://username:password@localhost:5432/dbname"
        embed_dim = 1536

        agensgraph_vector = AgensgraphVectorStore(url, embed_dim)
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    distance_strategy: str
    index_name: str
    keyword_index_name: str
    hybrid_search: bool
    node_label: str
    embedding_node_property: str
    text_node_property: str
    retrieval_query: str
    embedding_dimension: int

    _graph_name: Optional[str] = "vector_store"
    _support_metadata_filter: bool = PrivateAttr()
    _engine: Optional[AgensEngine] = PrivateAttr(default=None)
    _aconn: Optional[psycopg.AsyncConnection] = PrivateAttr(default=None)
    _url: str = PrivateAttr()

    def __init__(
        self,
        url: str,
        embedding_dimension: int,
        graph_name: Optional[str] = "vector_store",
        index_name: str = "vector",
        keyword_index_name: str = "keyword",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: str = "cosine",
        hybrid_search: bool = False,
        retrieval_query: str = "",
        engine: Optional[AgensEngine] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            distance_strategy=distance_strategy,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            hybrid_search=hybrid_search,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            text_node_property=text_node_property,
            retrieval_query=retrieval_query,
            embedding_dimension=embedding_dimension,
        )

        if distance_strategy not in ["cosine"]:
            raise ValueError("Only cosine distance strategy is supported for now")

        self._graph_name = graph_name
        # The engine (pool) is wired in only after setup completes: graph/index
        # creation must run on the dedicated connection, before the graph exists
        # (a pooled checkout would try to `SET graph_path` to a missing graph).
        self._engine = None
        self._aconn = None
        self._url = url

        # A dedicated connection is always held for setup/introspection. When an
        # engine is supplied, runtime queries instead check out a pooled
        # connection (see ``_acquire``); without one, every query uses this
        # dedicated connection -- the original single-connection behavior.
        try:
            self._connection = psycopg.connect(url)
        except psycopg.OperationalError as e:
            raise ValueError(f"Failed to connect to Agensgraph database: {e}")

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        # Create the graph and utility functions
        self.database_query(get_vector_index_info_function)
        self.database_query(get_keyword_index_info_function)
        self.database_query(sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
            sql.Identifier(self._graph_name)
        ))
        self.database_query(sql.SQL("SET graph_path = {}").format(
            sql.Literal(self._graph_name)
        ))

        self.verify_vector_support()

        # The `add` path MERGEs nodes by `id`; without a btree index on that
        # property every MERGE falls back to a sequential scan, making ingest
        # O(N^2). Create it unconditionally (IF NOT EXISTS) so it is present
        # even when the HNSW index already exists.
        self.create_id_index()

        index_already_exists = self.retrieve_existing_index()
        if not index_already_exists:
            self.create_new_index()
        if hybrid_search:
            fts_node_label = self.retrieve_existing_fts_index()
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                self.create_new_keyword_index()
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == self.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Setup is done; runtime queries may now use the pool.
        self._engine = engine

    def verify_label_existence(self) -> None:
        """Create label if it does not exist."""
        self.database_query(
            sql.SQL("CREATE VLABEL IF NOT EXISTS {}").format(
                sql.Identifier(self.node_label)
            )
        )

    @property
    def client(self) -> psycopg.Connection:
        return self._connection

    def create_property_index(
        self, property_name: str, label: Optional[str] = None
    ) -> None:
        """
        Create a btree property index on ``property_name`` for ``label``
        (defaults to this store's node label).

        Useful for metadata keys you filter on: a metadata-filtered vector
        search cannot use the HNSW index for the filter, so without a property
        index the filter degrades to a sequential scan. Indexing the filter key
        lets the planner pre-select matching rows via an index/bitmap scan.
        """
        target_label = label or self.node_label
        self.verify_label_existence()
        self.database_query(
            sql.SQL(
                "CREATE PROPERTY INDEX IF NOT EXISTS {index_name} ON {node_label} ({prop})"
            ).format(
                index_name=sql.Identifier(f"{target_label}_{property_name}_idx"),
                node_label=sql.Identifier(target_label),
                prop=sql.Identifier(property_name),
            )
        )

    def create_id_index(self) -> None:
        """
        Create btree property indexes on the MERGE key (`id`) and the delete
        key (`ref_doc_id`).

        ``add`` upserts nodes with ``MERGE (c:{label} {id: row.id})`` and
        ``delete`` matches on ``ref_doc_id``; without these indexes each such
        lookup performs a sequential scan (bulk ingest becomes O(N^2)).
        """
        self.create_property_index("id")
        self.create_property_index("ref_doc_id")

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in agensgraph.
        """
        self.verify_label_existence()
        index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
            ON {node_label} USING hnsw
            (({embedding_node_property}::vector({embedding_dimension})) vector_cosine_ops)"""

        self.database_query(
            sql.SQL(index_query).format(
                index_name=sql.Identifier(self.index_name),
                node_label=sql.Identifier(self.node_label),
                embedding_node_property=sql.Identifier(self.embedding_node_property),
                embedding_dimension=self.embedding_dimension
            )
        )

    def retrieve_existing_index(self) -> bool:
        """
        Check if the vector index exists in the Agensgraph database
        and returns its embedding dimension.

        This method queries the Agensgraph database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.

        """
        index_information = self.database_query(
            """SELECT * FROM ag_list_vector_indexes(index_name => %(index_name)s,
                                                    node_label => %(node_label)s,
                                                    embedding_node_property => %(embedding_node_property)s)
            """,
            params={
                "index_name": self.index_name,
                "node_label": self.node_label,
                "embedding_node_property": self.embedding_node_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelortype"]
            self.embedding_node_property = index_information[0]["property"]
            self.embedding_dimension = index_information[0]["dimensions"]

            return True
        except IndexError:
            return False

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """
        Check if the fulltext index exists in the Agensgraph database.

        This method queries the Agensgraph database for existing fts indexes
        with the specified name.

        Returns:
            (Tuple): keyword index information

        """
        index_information = self.database_query(
            """SELECT * FROM ag_list_text_indexes(index_name => %(index_name)s,
                                                  node_label => %(node_label)s,
                                                  text_node_properties => %(text_node_properties)s)
            """,
            params={
                "index_name": self.keyword_index_name,
                "node_label": self.node_label,
                "text_node_properties": [self.text_node_property],
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.keyword_index_name = index_information[0]["name"]
            self.text_node_property = index_information[0]["properties"][0]
            node_label = index_information[0]["labelortype"]
            return node_label
        except IndexError:
            return None

    def create_new_keyword_index(self, text_node_properties: List[str] = []) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new full text index in Agensgraph.
        """
        # make sure label exists
        self.verify_label_existence()
        node_props = text_node_properties or [self.text_node_property]

        fts_parts = [sql.SQL('(to_tsvector(\'english\', {}))').format(sql.Identifier(el)) for el in node_props]
        fts_index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
                             ON {node_label} USING gin ({expr})"""

        self.database_query(
            sql.SQL(fts_index_query).format(
                index_name=sql.Identifier(self.keyword_index_name),
                node_label=sql.Identifier(self.node_label),
                expr=sql.SQL(", ").join(fts_parts)
            )
        )

    def _build_add(
        self, nodes: List[BaseNode]
    ) -> Tuple[List[str], sql.Composed, List[Dict[str, Any]]]:
        """Build the ids, the formatted import query, and the cleaned rows."""
        ids = [r.node_id for r in nodes]
        import_query = """
            UNWIND %(data)s AS row
            MERGE (c:{label} {{id: row.id}})
            WITH c, row
            SET c.{embedding_node_property} = row.embedding,
                c.{text_node_property} = row.text
            SET c += row.metadata
        """

        formatted_query = sql.SQL(import_query).format(
            label=sql.Identifier(self.node_label),
            embedding_node_property=sql.Identifier(self.embedding_node_property),
            text_node_property=sql.Identifier(self.text_node_property),
        )
        return ids, formatted_query, clean_params(nodes)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        ids, formatted_query, rows = self._build_add(nodes)
        for start in range(0, len(rows), CHUNK_SIZE):
            batch = rows[start : start + CHUNK_SIZE]
            self.database_query(formatted_query, params={"data": Jsonb(batch)})

        return ids

    async def async_add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """True-async counterpart of :meth:`add`."""
        ids, formatted_query, rows = self._build_add(nodes)
        for start in range(0, len(rows), CHUNK_SIZE):
            batch = rows[start : start + CHUNK_SIZE]
            await self.adatabase_query(formatted_query, params={"data": Jsonb(batch)})

        return ids

    def _build_query(
        self, query: VectorStoreQuery
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build the formatted query SQL and parameters for a vector query."""
        # Filter only on IS NOT NULL. An `array_size(embedding) = dim` guard is
        # evaluated per row, which stops the planner from using the HNSW index
        # (forcing a sequential scan); the IS NOT NULL check plus the ::vector(N)
        # cast below already enforce a correctly-dimensioned embedding.
        base_index_query = (
                """MATCH (n:{label})
                WHERE n.{embedding_property} IS NOT NULL {filter_clause} """
        )

        base_cosine_query = """
            WITH n, n.{embedding_property}::vector({embedding_dimension}) <=> %(embedding)s::vector({embedding_dimension}) AS inv_score
            ORDER BY inv_score
            LIMIT %(k)s
            WITH n, 1 - inv_score AS score 
            """

        filter_params: Dict[str, Any] = {}
        filter_clause: sql.Composed = sql.SQL("")
        if query.filters:
            # Metadata filtering and hybrid doesn't work
            if self.hybrid_search:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )

            snippet, filter_params = metadata_filters_to_cypher(
                query.filters, alias="n"
            )
            filter_clause = sql.SQL("AND (") + snippet + sql.SQL(")")
            index_query = base_index_query + base_cosine_query
        else:
            # hybrid is handled in query()/aquery() (RRF over two top-level
            # queries), so a bare hybrid store with no query_str is plain vector.
            index_query = base_index_query + base_cosine_query

        index_query = index_query + " WITH *, n as node "
        default_retrieval = """
            RETURN node.{text_property} AS text, score, 
            node.id AS id, 
            node || jsonb_build_object({text_property_literal}, Null, 
            {embedding_property_literal}, Null, 'id', Null) AS metadata
        """

        index_query += self.retrieval_query or default_retrieval


        parameters = {
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }

        formatted_query = sql.SQL(index_query).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
            embedding_dimension=self.embedding_dimension,
            text_property_literal=sql.Literal(self.text_node_property),
            embedding_property_literal=sql.Literal(self.embedding_node_property),
            filter_clause=filter_clause,
        )
        return formatted_query, parameters

    @staticmethod
    def _results_to_query_result(
        results: List[Dict[str, Any]]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = metadata_dict_to_node(record["metadata"])
            node.set_content(str(record["text"]))
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _hybrid_modality_sql(self, modality: str) -> sql.Composed:
        """Top-level Cypher for one hybrid modality, returning (id, text, metadata)
        ordered by relevance. Kept top-level (not nested in a SQL sub-query) so the
        HNSW / full-text index is used; the modalities are fused in _rrf_fuse."""
        tail = """
            WITH n AS node
            RETURN node.{text_property} AS text, node.id AS id,
                node || jsonb_build_object({text_property_literal}, Null,
                    {embedding_property_literal}, Null, 'id', Null) AS metadata
        """
        if modality == "semantic":
            head = """
                MATCH (n:{label}) WHERE n.{embedding_property} IS NOT NULL
                WITH n, n.{embedding_property}::vector({embedding_dimension}) <=> %(embedding)s::vector({embedding_dimension}) AS d
                ORDER BY d LIMIT %(k)s
            """
        else:  # keyword
            head = """
                MATCH (n:{label})
                WHERE n.{text_property} IS NOT NULL AND
                      to_tsvector('english', n.{text_property}) @@ plainto_tsquery('english', %(query)s)
                WITH n, ts_rank_cd(to_tsvector('english', n.{text_property}), plainto_tsquery('english', %(query)s)) AS s
                ORDER BY s DESC LIMIT %(k)s
            """
        return sql.SQL(head + tail).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            embedding_dimension=self.embedding_dimension,
            text_property=sql.Identifier(self.text_node_property),
            text_property_literal=sql.Literal(self.text_node_property),
            embedding_property_literal=sql.Literal(self.embedding_node_property),
        )

    def _rrf_fuse(
        self, modalities: List[List[Dict[str, Any]]], k: int
    ) -> List[Dict[str, Any]]:
        """Reciprocal-rank-fusion of ranked per-modality result rows."""
        rc = 60  # RRF rank constant in 1/(rc+rank); 60 is the de-facto default
        scores: Dict[str, float] = {}
        data: Dict[str, Dict[str, Any]] = {}
        for rows in modalities:
            for rank, r in enumerate(rows):
                scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (rc + rank + 1)
                data.setdefault(r["id"], r)
        top = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        return [{**data[i], "score": scores[i]} for i in top]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if self.hybrid_search and query.query_str:
            params = {
                "k": query.similarity_top_k,
                "embedding": query.query_embedding,
                "query": remove_lucene_chars(query.query_str),
            }
            sem = self.database_query(self._hybrid_modality_sql("semantic"), params=params)
            kw = self.database_query(self._hybrid_modality_sql("keyword"), params=params)
            return self._results_to_query_result(
                self._rrf_fuse([sem, kw], query.similarity_top_k)
            )
        formatted_query, parameters = self._build_query(query)
        results = self.database_query(formatted_query, params=parameters)
        return self._results_to_query_result(results)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """True-async counterpart of :meth:`query`."""
        if self.hybrid_search and query.query_str:
            params = {
                "k": query.similarity_top_k,
                "embedding": query.query_embedding,
                "query": remove_lucene_chars(query.query_str),
            }
            sem = await self.adatabase_query(self._hybrid_modality_sql("semantic"), params=params)
            kw = await self.adatabase_query(self._hybrid_modality_sql("keyword"), params=params)
            return self._results_to_query_result(
                self._rrf_fuse([sem, kw], query.similarity_top_k)
            )
        formatted_query, parameters = self._build_query(query)
        results = await self.adatabase_query(formatted_query, params=parameters)
        return self._results_to_query_result(results)

    def _build_delete(self, ref_doc_id: str) -> Tuple[sql.Composed, Dict[str, Any]]:
        query = """
            MATCH (n:{label})
            WHERE n.ref_doc_id = %(id)s
            DETACH DELETE n
            """
        return (
            sql.SQL(query).format(label=sql.Identifier(self.node_label)),
            {"id": Jsonb(ref_doc_id)},
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        formatted_query, params = self._build_delete(ref_doc_id)
        self.database_query(formatted_query, params=params)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """True-async counterpart of :meth:`delete`."""
        formatted_query, params = self._build_delete(ref_doc_id)
        await self.adatabase_query(formatted_query, params=params)

    def _node_match_clause(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[MetadataFilters],
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build a ``WHERE ...`` fragment (possibly empty) matching node_ids/filters."""
        conds: List[sql.Composed] = []
        params: Dict[str, Any] = {}
        if node_ids:
            # OR-of-equalities rather than ``id IN [...]``: only the equality
            # form matches the ``id`` btree index (the planner serves it via a
            # BitmapOr index scan), whereas ``IN`` over a jsonb array always
            # falls back to a sequential scan.
            id_terms = []
            for i, nid in enumerate(node_ids):
                pname = f"node_id_{i}"
                params[pname] = Jsonb(nid)
                id_terms.append(sql.SQL("n.id = %({p})s").format(p=sql.SQL(pname)))
            conds.append(sql.SQL("(") + sql.SQL(" OR ").join(id_terms) + sql.SQL(")"))
        if filters:
            snippet, fparams = metadata_filters_to_cypher(filters, alias="n")
            conds.append(sql.SQL("(") + snippet + sql.SQL(")"))
            params.update(fparams)
        if conds:
            return sql.SQL("WHERE ") + sql.SQL(" AND ").join(conds), params
        return sql.SQL(""), params

    def _build_get_nodes(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[MetadataFilters],
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        where, params = self._node_match_clause(node_ids, filters)
        query = """
            MATCH (n:{label})
            {where}
            RETURN n.{text_property} AS text,
                   n.id AS id,
                   n || jsonb_build_object({text_property_literal}, Null,
                        {embedding_property_literal}, Null, 'id', Null) AS metadata
            """
        return (
            sql.SQL(query).format(
                label=sql.Identifier(self.node_label),
                text_property=sql.Identifier(self.text_node_property),
                text_property_literal=sql.Literal(self.text_node_property),
                embedding_property_literal=sql.Literal(self.embedding_node_property),
                where=where,
            ),
            params,
        )

    @staticmethod
    def _records_to_nodes(results: List[Dict[str, Any]]) -> List[BaseNode]:
        nodes: List[BaseNode] = []
        for record in results:
            node = metadata_dict_to_node(record["metadata"])
            node.set_content(str(record["text"]))
            nodes.append(node)
        return nodes

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes by id and/or metadata filters."""
        query, params = self._build_get_nodes(node_ids, filters)
        return self._records_to_nodes(self.database_query(query, params=params))

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """True-async counterpart of :meth:`get_nodes`."""
        query, params = self._build_get_nodes(node_ids, filters)
        return self._records_to_nodes(await self.adatabase_query(query, params=params))

    def _build_delete_nodes(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[MetadataFilters],
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        where, params = self._node_match_clause(node_ids, filters)
        query = "MATCH (n:{label}) {where} DETACH DELETE n"
        return (
            sql.SQL(query).format(
                label=sql.Identifier(self.node_label), where=where
            ),
            params,
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id and/or metadata filters."""
        query, params = self._build_delete_nodes(node_ids, filters)
        self.database_query(query, params=params)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """True-async counterpart of :meth:`delete_nodes`."""
        query, params = self._build_delete_nodes(node_ids, filters)
        await self.adatabase_query(query, params=params)

    def clear(self) -> None:
        """Delete all nodes for this store's label."""
        self.database_query(
            sql.SQL("MATCH (n:{label}) DETACH DELETE n").format(
                label=sql.Identifier(self.node_label)
            )
        )

    async def aclear(self) -> None:
        """True-async counterpart of :meth:`clear`."""
        await self.adatabase_query(
            sql.SQL("MATCH (n:{label}) DETACH DELETE n").format(
                label=sql.Identifier(self.node_label)
            )
        )

    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self._connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    def verify_vector_support(self) -> None:
        """
        Verify if the graph store supports vector operations
        """
        with self._get_cursor() as curs:
            try:
                curs.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self._connection.commit()
            except psycopg.Error:
                self._connection.rollback()
                raise ValueError(
                    """Vector extension not supported\nUnable to install pg_vector extension"""
                )

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
                vertex = _vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = _vertex_regex.match(v)
                edge = _edge_regex.match(v)

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

    def database_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and
        converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query (not used in this implementation)

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """

        # execute the query, rolling back on an error
        with self._acquire() as conn:
            with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    curs.execute(query, params)
                    conn.commit()
                except psycopg.Error as e:
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

                return result

    @contextmanager
    def _acquire(self) -> "Iterator[psycopg.Connection]":
        """Yield the connection ``database_query`` should run on.

        Uses a pooled connection from the engine when one is configured; falls
        back to the dedicated connection otherwise (the pre-engine behavior).
        """
        if self._engine is not None:
            with self._engine.connection(graph_path=self._graph_name) as conn:
                yield conn
        else:
            yield self._connection

    @asynccontextmanager
    async def _aacquire(self) -> "AsyncIterator[psycopg.AsyncConnection]":
        """Async sibling of :meth:`_acquire`."""
        if self._engine is not None:
            async with self._engine.aconnection(graph_path=self._graph_name) as conn:
                yield conn
        else:
            if self._aconn is None or self._aconn.closed:
                self._aconn = await psycopg.AsyncConnection.connect(self._url)
                async with self._aconn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("SET graph_path = {n}").format(
                            n=sql.Identifier(self._graph_name)
                        )
                    )
                await self._aconn.commit()
            yield self._aconn

    async def adatabase_query(
        self, query: str, params: dict = {}
    ) -> List[Dict[str, Any]]:
        """Async counterpart of :meth:`database_query` (true async I/O)."""
        async with self._aacquire() as conn:
            async with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    await curs.execute(query, params)
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
                    data = await curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []

                if data is None:
                    return []
                return [self._record_to_dict(d) for d in data]