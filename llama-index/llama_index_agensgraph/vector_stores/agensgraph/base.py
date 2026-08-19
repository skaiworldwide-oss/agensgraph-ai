from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)
import logging
import re

from contextlib import asynccontextmanager, contextmanager

import agensgraph
from agensgraph import Edge, Vertex
from agensgraph.vector import Vector
from agensgraph.errors import safe_message
from agensgraph.introspect import DesiredIndex
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.filters import metadata_filters_to_cypher
from llama_index_agensgraph.graph_stores.agensgraph.utils import AgensQueryException

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

# An expression index records its expression, and that is the only place the
# indexed property and the vector's width appear. The driver's element parser
# reads a plain index and returns nothing for one of these, so they are read
# here -- where the two SQL functions this replaces used to do the same regex
# work, in the caller's database, installed on every construction.
VECTOR_INDEX_EXPR = re.compile(
    r"""\(+ "?(?P<property>[A-Za-z_][A-Za-z0-9_]*)"? \)*
        ::vector\( (?P<dimensions>\d+) \)""",
    re.VERBOSE,
)
TEXT_INDEX_EXPR = re.compile(
    r"""to_tsvector\( \s* '[^']*' \s* , \s*
        "?(?P<property>[A-Za-z_][A-Za-z0-9_]*)"? \s* \)""",
    re.VERBOSE,
)


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

logger = logging.getLogger(__name__)


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
    _vectors_registered: bool = PrivateAttr(default=False)

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
            self._connection = agensgraph.Connection.connect(url)
        except psycopg.OperationalError as e:
            raise ValueError(f"Failed to connect to Agensgraph database: {e}")

        # With the vector types registered, an embedding travels as itself
        # instead of as its decimal spelling for the server to parse back.
        self._vectors_registered = self._connection.has_vectors()
        if self._vectors_registered:
            self._connection.register_vectors()

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
        self,
        property_name: str,
        label: Optional[str] = None,
        unique: bool = False,
    ) -> None:
        """Index ``property_name`` on ``label``, this store's node label by default.

        Useful for the metadata keys you filter on: a metadata-filtered vector
        search cannot use the HNSW index for the filter, so without a property
        index the filter reads every embedded node.

        ``unique`` also makes the index refuse a second node with the same value.
        The whole reconciliation runs in one transaction, because making an
        existing index unique means dropping it and building it again: if the
        rebuild is refused, a graph that had an index would otherwise be left
        with none, and every MERGE on that property would go back to reading the
        whole label.
        """
        target_label = label or self.node_label
        self.verify_label_existence()
        desired = DesiredIndex(
            target_label,
            (property_name,),
            unique,
            f"{target_label}_{property_name}_idx",
            "btree",
            None,
        )
        with self._acquire() as conn:
            try:
                with conn.transaction():
                    conn.ensure_indexes([desired], graph=self._graph_name)
            except psycopg.errors.UniqueViolation:
                if not unique:
                    raise
                # The label already holds two nodes of one id. Saying so is more
                # use than refusing to open the store, and the index that was
                # there is still there.
                logger.warning(
                    "%s already holds more than one node per %s, so that index "
                    "cannot enforce uniqueness; duplicates already written stay, "
                    "and concurrent writers can add more.",
                    target_label,
                    property_name,
                )
                with conn.transaction():
                    conn.ensure_indexes(
                        [desired._replace(unique=False)], graph=self._graph_name
                    )
            except psycopg.Error as e:
                raise AgensQueryException(
                    {
                        "message": f"Error indexing {target_label}.{property_name}",
                        "details": safe_message(e),
                    },
                    cause=e,
                ) from e

    def create_id_index(self) -> None:
        """
        Create btree property indexes on the MERGE key (`id`) and the delete
        key (`ref_doc_id`).

        ``add`` upserts nodes with ``MERGE (c:{label} {id: row.id})`` and
        ``delete`` matches on ``ref_doc_id``; without these indexes each such
        lookup performs a sequential scan (bulk ingest becomes O(N^2)).

        The one on ``id`` is unique. Two callers adding the same node at the same
        moment both find it missing and both create it, and the id is what every
        later read, update and delete of that node goes through.
        """
        self.create_property_index("id", unique=True)
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

    def _graph_indexes(self) -> List[Any]:
        """Every property index of this store's label, from the catalog."""
        with self._acquire() as conn:
            try:
                return conn.indexes(self.node_label, graph=self._graph_name)
            finally:
                try:
                    conn.commit()
                except psycopg.Error:
                    pass

    def retrieve_existing_index(self) -> bool:
        """Adopt the vector index already on this label, if there is one.

        Sets this store's index name, label, embedded property and dimension from
        it, so a store opened against an existing index agrees with what is there
        rather than trying to build a second one beside it.
        """
        found = []
        for index in self._graph_indexes():
            match = VECTOR_INDEX_EXPR.search(index.definition)
            if match is None:
                continue
            if index.name != self.index_name and (
                match.group("property") != self.embedding_node_property
            ):
                continue
            found.append(
                {
                    "name": index.name,
                    "labelortype": index.label,
                    "property": match.group("property"),
                    "dimensions": int(match.group("dimensions")),
                }
            )
        if not found:
            return False
        # The one named is the one meant; any other is a fallback.
        chosen = sort_by_index_name(found, self.index_name)[0]
        self.index_name = chosen["name"]
        self.node_label = chosen["labelortype"]
        self.embedding_node_property = chosen["property"]
        self.embedding_dimension = chosen["dimensions"]
        return True

    def retrieve_existing_fts_index(self) -> Optional[str]:
        """The label of the full-text index already on this store's label, if any."""
        found = []
        for index in self._graph_indexes():
            properties = TEXT_INDEX_EXPR.findall(index.definition)
            if not properties:
                continue
            if index.name != self.keyword_index_name and (
                properties != [self.text_node_property]
            ):
                continue
            found.append(
                {
                    "name": index.name,
                    "labelortype": index.label,
                    "properties": properties,
                }
            )
        if not found:
            return None
        chosen = sort_by_index_name(found, self.keyword_index_name)[0]
        self.keyword_index_name = chosen["name"]
        self.text_node_property = chosen["properties"][0]
        return chosen["labelortype"]

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

    def _rows_to_write(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """One flat property map per node, as it is stored.

        The metadata keys are stored beside ``text`` and ``embedding`` rather than
        under a key of their own -- which is what ``SET c += row.metadata`` did --
        so a metadata key named ``text`` still wins, as it did before.
        """
        rows = []
        for row in clean_params(nodes):
            rows.append(
                {
                    "id": row["id"],
                    self.text_node_property: row["text"],
                    self.embedding_node_property: row["embedding"],
                    **row["metadata"],
                }
            )
        return rows

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Write these nodes, keyed on their id.

        The statement this replaces created each node and then wrote its
        properties in two further passes, so 5,000 nodes left 5,000 insertions,
        5,000 updates and dead rows behind, and the HNSW index indexed every
        version. Keyed on ``id``, whose index is unique, the same 5,000 are
        5,000 insertions and nothing else: 15.4 s to 13.0 s, and 4.4 s inside
        :meth:`bulk_ingest`.
        """
        rows = self._rows_to_write(nodes)
        with self._acquire() as conn:
            try:
                conn.upsert_vertices(
                    self.node_label,
                    "id",
                    rows,
                    on_existing="update",
                    graph=self._graph_name,
                )
                conn.commit()
            except psycopg.Error as e:
                conn.rollback()
                raise AgensQueryException(
                    {
                        "message": f"Error writing {len(rows)} nodes",
                        "details": safe_message(e),
                    },
                    cause=e,
                ) from e
        return [row["id"] for row in rows]

    async def async_add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """True-async counterpart of :meth:`add`."""
        rows = self._rows_to_write(nodes)
        async with self._aacquire() as conn:
            try:
                await conn.upsert_vertices(
                    self.node_label,
                    "id",
                    rows,
                    on_existing="update",
                    graph=self._graph_name,
                )
                await conn.commit()
            except psycopg.Error as e:
                await conn.rollback()
                raise AgensQueryException(
                    {
                        "message": f"Error writing {len(rows)} nodes",
                        "details": safe_message(e),
                    },
                    cause=e,
                ) from e
        return [row["id"] for row in rows]

    @contextmanager
    def bulk_ingest(self) -> Iterator[None]:
        """Drop the vector index for the writes in this block and build it after.

        Keeping the index current costs more than the writing does: of the 15.4 s
        it took to write 5,000 nodes of 384 numbers, roughly three quarters was
        the index, and the same writes with the index built afterwards took 4.4 s
        -- 3.5x, and the gap widens as the label grows, because every insertion
        walks a graph that is getting bigger.

        **A search inside this block reads every row**, because for its duration
        the index is not there -- for everyone using the label, not only this
        caller. It is for loading a corpus, not for a store answering questions.
        If the process is killed inside the block the index stays dropped;
        constructing the store builds it again.
        """
        self._drop_vector_index()
        try:
            yield
        finally:
            self.create_new_index()

    def _drop_vector_index(self) -> None:
        with self._acquire() as conn:
            try:
                conn.execute(
                    sql.SQL("DROP PROPERTY INDEX IF EXISTS {}").format(
                        sql.Identifier(self.index_name)
                    )
                )
                conn.commit()
            except psycopg.Error as e:
                conn.rollback()
                raise AgensQueryException(
                    {
                        "message": f"Error dropping index {self.index_name}",
                        "details": safe_message(e),
                    },
                    cause=e,
                ) from e

    def _bind_embedding(self, embedding: Any) -> Any:
        """Bind an embedding as itself where the server can read one.

        A list sent as jsonb is written out as decimal text and parsed back: on
        1536 numbers that is 31 KB on the wire against 6 KB, and the whole query
        measured 1.35x slower. Without the vector types registered -- no vector
        extension -- it stays jsonb with a cast, which still works.
        """
        if embedding is None:
            return None
        if self._vectors_registered:
            return Vector(embedding)
        return Jsonb(list(embedding))

    def _embedding_placeholder(self) -> sql.SQL:
        """How a bound embedding is spelled in the statement.

        A vector needs no cast; a list of numbers in jsonb does, and a typmod has
        to be a literal, so it is written in rather than bound.
        """
        if self._vectors_registered:
            return sql.SQL("%(embedding)s")
        return sql.SQL("%(embedding)s::vector({})").format(
            sql.SQL(str(int(self.embedding_dimension)))
        )

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
            WITH n, n.{embedding_property}::vector({embedding_dimension}) <=> {bound_embedding} AS inv_score
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
            "embedding": self._bind_embedding(query.query_embedding),
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }

        formatted_query = sql.SQL(index_query).format(
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            text_property=sql.Identifier(self.text_node_property),
            embedding_dimension=self.embedding_dimension,
            bound_embedding=self._embedding_placeholder(),
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
                WITH n, n.{embedding_property}::vector({embedding_dimension}) <=> {bound_embedding} AS d
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
            bound_embedding=self._embedding_placeholder(),
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
                "embedding": self._bind_embedding(query.query_embedding),
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
                "embedding": self._bind_embedding(query.query_embedding),
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
        # An element comes back as an element. Matched out of its printed form, a
        # stored string that happens to read like one -- a chunk quoting a query
        # result, say -- became a vertex, and its properties overwrote the real
        # one returned in another field.
        vertices: Dict[Any, Dict[str, Any]] = {}
        for name in record._fields:
            value = getattr(record, name)
            if isinstance(value, Vertex):
                vertices[value.id] = dict(value.properties)

        d: Dict[str, Any] = {}
        for name in record._fields:
            value = getattr(record, name)
            if isinstance(value, Edge):
                d[name] = (
                    vertices.get(value.start, {}),
                    value.label,
                    vertices.get(value.end, {}),
                )
            elif isinstance(value, Vertex):
                d[name] = dict(value.properties)
            else:
                d[name] = value

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
                self._aconn = await agensgraph.AsyncConnection.connect(self._url)
                if self._vectors_registered:
                    await self._aconn.register_vectors()
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