import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager, contextmanager
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

import psycopg
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from psycopg import sql
from psycopg.types.json import Jsonb

import agensgraph
from agensgraph import Edge, RetryPolicy, Vertex
from agensgraph.errors import safe_message
from agensgraph.introspect import DesiredIndex
from agensgraph.vector import Distance, Vector, generated_column
from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.filters import metadata_filters_to_cypher
from llama_index_agensgraph.graph_stores.agensgraph.utils import (
    COMPLETE_FILTERED_SEARCH,
    AgensQueryException,
    aapply_search_options,
    apply_search_options,
    bounded_name,
    check_no_copy,
    known_search_options,
    lost_the_creation_race,
    query_failed,
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
# An index over a promoted column has no cast to read: the column is already a
# vector, so the definition is `USING hnsw (embedding vector_cosine_ops)` and the
# dimension is in the column's type instead. Without this spelling an existing
# index went unrecognised, so the dimension it was built for was never compared
# with the one asked for -- the store opened, and every query after it failed.
PROMOTED_INDEX_EXPR = re.compile(
    r"""USING \s+ \w+ \s* \(\s*
        "?(?P<property>[A-Za-z_][A-Za-z0-9_]*)"? \s+ \w*vector\w* """,
    re.VERBOSE,
)
VECTOR_TYPE_EXPR = re.compile(r"vector\(\s*(?P<dimensions>\d+)\s*\)")
TEXT_INDEX_EXPR = re.compile(
    r"""to_tsvector\( \s* '[^']*' \s* , \s*
        "?(?P<property>[A-Za-z_][A-Za-z0-9_]*)"? \s* \)""",
    re.VERBOSE,
)


# How a distance is asked for and how it is indexed. The operator in a query and
# the operator class of the index have to agree, or the index cannot serve the
# ordering and the search reads every row instead -- with nothing to say so.
DISTANCES = {
    "cosine": (Distance.COSINE, "vector_cosine_ops"),
    "l2": (Distance.L2, "vector_l2_ops"),
    "euclidean": (Distance.L2, "vector_l2_ops"),
    "inner_product": (Distance.INNER_PRODUCT, "vector_ip_ops"),
}


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
        `pip install llama-index-agensgraph`


        ```python
        from llama_index_agensgraph.vector_stores.agensgraph import (
            AgensgraphVectorStore,
        )

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
    promote_embedding: bool
    index_options: Dict[str, Any]
    search_options: Dict[str, Any]

    _graph_name: Optional[str] = "vector_store"
    _support_metadata_filter: bool = PrivateAttr()
    _engine: Optional[AgensEngine] = PrivateAttr(default=None)
    _aconn: Optional[psycopg.AsyncConnection] = PrivateAttr(default=None)
    _url: str = PrivateAttr()
    _vectors_registered: bool = PrivateAttr(default=False)
    _promoted: bool = PrivateAttr(default=False)
    _retry_policy: RetryPolicy = PrivateAttr()
    _apool_pool: Optional[Any] = PrivateAttr(default=None)
    _apool_loop: Optional[Any] = PrivateAttr(default=None)
    _apool_lock: Any = PrivateAttr(default=None)

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
        promote_embedding: bool = True,
        index_options: Optional[Dict[str, Any]] = None,
        search_options: Optional[Dict[str, Any]] = None,
        retry_attempts: int = 6,
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
            promote_embedding=promote_embedding,
            index_options=index_options or {},
            search_options=search_options or {},
        )

        if distance_strategy not in DISTANCES:
            raise ValueError(
                f"{distance_strategy!r} is not a distance this store measures; it "
                "can do " + ", ".join(sorted(DISTANCES))
            )

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
        # Two coroutines reaching the first await would each build a pool and the
        # second assignment would drop the first still holding its connections.
        self._apool_lock = asyncio.Lock()
        self._vectors_registered = self._connection.has_vectors()
        if self._vectors_registered:
            self._connection.register_vectors()
        # A column needs its width when it is declared, and a server that cannot
        # give a property one leaves the embedding in the map.
        if promote_embedding and not self._connection.can_promote_properties():
            self.promote_embedding = False

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

        self._retry_policy = RetryPolicy(attempts=retry_attempts)

        self._create_if_absent(sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
            sql.Identifier(self._graph_name)
        ))
        self.database_query(sql.SQL("SET graph_path = {}").format(
            sql.Literal(self._graph_name)
        ))

        self.verify_vector_support()
        # A filtered search visits about hnsw.ef_search candidates and filters
        # those, so without this it answers with however many happened to pass.
        # Whatever the caller asked for wins.
        self.search_options = known_search_options(
            self._connection,
            {**COMPLETE_FILTERED_SEARCH, **dict(self.search_options)},
        )
        self._connection.commit()
        apply_search_options(self._connection, self.search_options)

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
        self._create_if_absent(
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
            bounded_name(target_label, property_name, "idx"),
            "btree",
            None,
        )
        with self._acquire() as conn:
            try:
                with conn.transaction():
                    conn.ensure_indexes([desired], graph=self._graph_name)
            except (
                psycopg.errors.DuplicateTable,
                psycopg.errors.DuplicateObject,
            ):
                # Another construction built the same index a moment ago, which is
                # what this asked for.
                pass
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

    def _embedding_is_promoted(self) -> bool:
        """Whether this label's embedding has a column of its own."""
        if self._promoted:
            return True
        with self._acquire() as conn:
            try:
                declared = conn.declared_properties(
                    self.node_label, graph=self._graph_name
                )
            finally:
                try:
                    conn.commit()
                except psycopg.Error:
                    pass
        self._promoted = any(
            prop.name == self.embedding_node_property for prop in declared
        )
        return self._promoted

    def _promote_embedding_column(self) -> None:
        """Give the embedding a column of its own.

        Read out of the property map it is decimal text in a bag that has to come
        out of TOAST and be parsed before a distance can be taken, once per node
        the filter kept -- which is where a metadata-filtered search spends its
        time. The column is generated from the property, so writes are unchanged
        and a label that already holds nodes is filled from what it already has.
        Adding it to a populated label rewrites that label once.
        """
        if not self.promote_embedding or self._embedding_is_promoted():
            return
        self.database_query(
            sql.SQL(
                "ALTER VLABEL {} ADD COLUMN "
                + generated_column(
                    self.embedding_node_property, int(self.embedding_dimension)
                )
            ).format(sql.Identifier(self.node_label))
        )
        self._promoted = True
        # Whatever index was there was built over the property map, and the query
        # now asks about the column, so it cannot serve one.
        self.database_query(
            sql.SQL("DROP PROPERTY INDEX IF EXISTS {}").format(
                sql.Identifier(self.index_name)
            )
        )

    def create_new_index(self) -> None:
        """Build the HNSW index over the embeddings.

        The expression has to be what the query ends up asking for. Against a
        column the query says ``n.embedding`` and the cast falls away, so an index
        built over the cast cannot serve it -- measured, 568 ms against 1.8 ms,
        and nothing says the index went unused.
        """
        self.verify_label_existence()
        self._promote_embedding_column()
        operator_class = sql.SQL(DISTANCES[self.distance_strategy][1])
        expression = (
            sql.SQL("({} ") .format(sql.Identifier(self.embedding_node_property))
            + operator_class
            + sql.SQL(")")
            if self._embedding_is_promoted()
            else sql.SQL("(({}::vector({})) ").format(
                sql.Identifier(self.embedding_node_property),
                sql.SQL(str(int(self.embedding_dimension))),
            )
            + operator_class
            + sql.SQL(")")
        )
        # m and ef_construction decide how much of the graph an insertion walks and
        # how well the finished one connects; the defaults suit a small corpus and
        # are worth raising for a large one.
        options = sql.SQL("")
        if self.index_options:
            options = sql.SQL(" WITH (") + sql.SQL(", ").join(
                sql.SQL("{} = {}").format(sql.SQL(name), sql.Literal(value))
                for name, value in sorted(self.index_options.items())
            ) + sql.SQL(")")
        self._create_if_absent(
            sql.SQL(
                "CREATE PROPERTY INDEX IF NOT EXISTS {index_name} "
                "ON {node_label} USING hnsw "
            ).format(
                index_name=sql.Identifier(self.index_name),
                node_label=sql.Identifier(self.node_label),
            )
            + expression
            + options
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

    def _promoted_dimensions(
        self, label: str, prop: str, cache: Dict[str, int]
    ) -> Optional[int]:
        """The width of ``prop`` where it is a column of its own on ``label``.

        An index over a promoted column names the column and nothing else, so the
        width has to come from the column's declared type.
        """
        key = f"{label}.{prop}"
        if key in cache:
            return cache[key]
        with self._acquire() as conn:
            try:
                declared = conn.declared_properties(label, graph=self._graph_name)
            except psycopg.Error:
                return None
            finally:
                conn.rollback()
        for column in declared:
            if column.name != prop:
                continue
            width = VECTOR_TYPE_EXPR.search(column.type or "")
            if width is not None:
                cache[key] = int(width.group("dimensions"))
                return cache[key]
        return None

    def retrieve_existing_index(self) -> bool:
        """Adopt the vector index already on this label, if there is one.

        Sets this store's index name, label, embedded property and dimension from
        it, so a store opened against an existing index agrees with what is there
        rather than trying to build a second one beside it.
        """
        found = []
        promoted_dimensions: Dict[str, int] = {}
        for index in self._graph_indexes():
            match = VECTOR_INDEX_EXPR.search(index.definition)
            if match is not None:
                dimensions = int(match.group("dimensions"))
            else:
                match = PROMOTED_INDEX_EXPR.search(index.definition)
                if match is None:
                    continue
                dimensions = self._promoted_dimensions(
                    index.label, match.group("property"), promoted_dimensions
                )
                if dimensions is None:
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
                    "dimensions": dimensions,
                }
            )
        if not found:
            return False
        # The one named is the one meant; any other is a fallback.
        chosen = sort_by_index_name(found, self.index_name)[0]
        if (
            self.embedding_dimension
            and chosen["dimensions"] != self.embedding_dimension
        ):
            # Taking the width from the index instead would open the store and
            # refuse every embedding written to it afterwards, one at a time, with
            # nothing pointing back here. The index cannot be rebuilt to the asked
            # width either -- what is already stored is the other one.
            raise ValueError(
                f"index {chosen['name']!r} on {chosen['labelortype']!r} holds "
                f"{chosen['dimensions']}-dimensional embeddings and this store was "
                f"asked for {self.embedding_dimension}. Pass "
                f"embedding_dimension={chosen['dimensions']}, or use another "
                f"index_name or node_label."
            )
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

        fts_parts = [
            sql.SQL('(to_tsvector(\'english\', {}))').format(sql.Identifier(el))
            for el in node_props
        ]
        fts_index_query = """CREATE PROPERTY INDEX IF NOT EXISTS {index_name}
                             ON {node_label} USING gin ({expr})"""

        self._create_if_absent(
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
        self,
        query: VectorStoreQuery,
        mode: Optional[VectorStoreQueryMode] = None,
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build the formatted query SQL and parameters for a vector query.

        For MMR the candidates come back carrying their embeddings, and there are
        more of them than were asked for: the discounting happens afterwards and
        can only choose among what the search handed it.
        """
        # Filter only on IS NOT NULL. An `array_size(embedding) = dim` guard is
        # evaluated per row, which stops the planner from using the HNSW index
        # (forcing a sequential scan); the IS NOT NULL check plus the ::vector(N)
        # cast below already enforce a correctly-dimensioned embedding.
        base_index_query = (
                """MATCH (n:{label})
                WHERE n.{embedding_property} IS NOT NULL {filter_clause} """
        )

        base_cosine_query = """
            WITH n, n.{embedding_property}::vector({embedding_dimension})
                    {distance} {bound_embedding} AS inv_score
            ORDER BY inv_score
            LIMIT %(k)s
            WITH n, 1 - inv_score AS score 
            """

        filter_params: Dict[str, Any] = {}
        filter_clause: sql.Composed = sql.SQL("")
        if query.filters:
            snippet, filter_params = metadata_filters_to_cypher(
                query.filters, alias="n"
            )
            filter_clause = sql.SQL("AND (") + snippet + sql.SQL(")")
        # A hybrid store asked for a plain vector search gets one; the modes that
        # search the text are run from query()/aquery(), filter and all.
        index_query = base_index_query + base_cosine_query

        index_query = index_query + " WITH *, n as node "
        default_retrieval = """
            RETURN node.{text_property} AS text, score, 
            node.id AS id, 
            node || jsonb_build_object({text_property_literal}, Null, 
            {embedding_property_literal}, Null, 'id', Null) AS metadata
        """

        mmr_retrieval = """
            RETURN node.{text_property} AS text, score,
            node.id AS id,
            node.{embedding_property} AS mmr_embedding,
            node || jsonb_build_object({text_property_literal}, Null,
            {embedding_property_literal}, Null, 'id', Null) AS metadata
        """
        if mode is VectorStoreQueryMode.MMR:
            index_query += mmr_retrieval
            # Four times what was asked for, so there is something to choose
            # between; discounting a list of exactly k changes nothing.
            fetch_k = max(query.similarity_top_k * 4, query.similarity_top_k + 10)
        else:
            index_query += self.retrieval_query or default_retrieval
            fetch_k = query.similarity_top_k

        parameters = {
            "k": fetch_k,
            "embedding": self._bind_embedding(query.query_embedding),
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }

        formatted_query = sql.SQL(index_query).format(
            distance=sql.SQL(DISTANCES[self.distance_strategy][0].value),
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

    def _hybrid_modality_sql(
        self, modality: str, filter_clause: Optional[sql.Composed] = None
    ) -> sql.Composed:
        """Top-level Cypher for one hybrid modality, returning (id, text, metadata)
        ordered by relevance. Kept top-level (not nested in a SQL sub-query) so the
        HNSW / full-text index is used; the modalities are fused in _rrf_fuse.

        A metadata filter reaches both modalities. Asked for together they used to
        raise, saying filtering does not work with hybrid -- what did not work was
        that nothing passed the filter down to here."""
        tail = """
            WITH n AS node
            RETURN node.{text_property} AS text, node.id AS id,
                node || jsonb_build_object({text_property_literal}, Null,
                    {embedding_property_literal}, Null, 'id', Null) AS metadata
        """
        if modality == "semantic":
            head = """
                MATCH (n:{label})
                WHERE n.{embedding_property} IS NOT NULL {filter_clause}
                WITH n, n.{embedding_property}::vector({embedding_dimension})
                        {distance} {bound_embedding} AS d
                ORDER BY d LIMIT %(semantic_k)s
            """
        else:  # keyword
            head = """
                MATCH (n:{label})
                WHERE n.{text_property} IS NOT NULL AND
                      to_tsvector('english', n.{text_property})
                          @@ plainto_tsquery('english', %(query)s)
                      {filter_clause}
                WITH n, ts_rank_cd(
                        to_tsvector('english', n.{text_property}),
                        plainto_tsquery('english', %(query)s)) AS s
                ORDER BY s DESC LIMIT %(keyword_k)s
            """
        return sql.SQL(head + tail).format(
            distance=sql.SQL(DISTANCES[self.distance_strategy][0].value),
            filter_clause=filter_clause if filter_clause is not None else sql.SQL(""),
            label=sql.Identifier(self.node_label),
            embedding_property=sql.Identifier(self.embedding_node_property),
            embedding_dimension=self.embedding_dimension,
            bound_embedding=self._embedding_placeholder(),
            text_property=sql.Identifier(self.text_node_property),
            text_property_literal=sql.Literal(self.text_node_property),
            embedding_property_literal=sql.Literal(self.embedding_node_property),
        )

    def _rrf_fuse(
        self,
        modalities: List[List[Dict[str, Any]]],
        k: int,
        weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Reciprocal-rank-fusion of ranked per-modality result rows.

        ``weights`` says how much each modality counts, which is what ``alpha`` on
        the query asks for: 1.0 is the vector side alone, 0.0 the text side alone.
        Every modality counted equally before, so alpha did nothing."""
        rc = 60  # RRF rank constant in 1/(rc+rank); 60 is the de-facto default
        if weights is None:
            weights = [1.0] * len(modalities)
        scores: Dict[str, float] = {}
        data: Dict[str, Dict[str, Any]] = {}
        for rows, weight in zip(modalities, weights):
            for rank, r in enumerate(rows):
                scores[r["id"]] = scores.get(r["id"], 0.0) + weight / (rc + rank + 1)
                data.setdefault(r["id"], r)
        top = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        return [{**data[i], "score": scores[i]} for i in top]

    # The modes this store answers. The rest of the enum names a classifier that
    # ranks by fitting a model over the candidates, which is not something a graph
    # query does -- refused by name rather than quietly answered as a plain vector
    # search, which is what every mode used to get, because the field was never
    # read at all.
    _MODES = frozenset(
        {
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.SEMANTIC_HYBRID,
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.SPARSE,
            VectorStoreQueryMode.MMR,
        }
    )
    _TEXT_MODES = frozenset(
        {
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.SEMANTIC_HYBRID,
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.SPARSE,
        }
    )

    def _check_mode(self, query: VectorStoreQuery) -> VectorStoreQueryMode:
        """What the query asked for, and whether this store can answer it."""
        mode = query.mode or VectorStoreQueryMode.DEFAULT
        if mode not in self._MODES:
            raise ValueError(
                f"{mode.value} is not a mode this store answers; it can do "
                + ", ".join(sorted(m.value for m in self._MODES))
            )
        if mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID):
            if not self.hybrid_search:
                raise ValueError(
                    "asked for a hybrid search on a store built without one; pass "
                    "hybrid_search=True so the full-text index is there to search"
                )
        if mode in self._TEXT_MODES and not query.query_str:
            raise ValueError(f"{mode.value} searches the text, so it needs query_str")
        return mode

    def _text_plan(
        self, query: VectorStoreQuery, mode: VectorStoreQueryMode
    ) -> Tuple[List[sql.Composed], Dict[str, Any], List[float], int]:
        """The statements a text-involving search runs, and how to weigh them."""
        filter_clause: sql.Composed = sql.SQL("")
        filter_params: Dict[str, Any] = {}
        if query.filters:
            snippet, filter_params = metadata_filters_to_cypher(
                query.filters, alias="n"
            )
            filter_clause = sql.SQL("AND (") + snippet + sql.SQL(")")

        keyword_k = query.sparse_top_k or query.similarity_top_k
        params = {
            "semantic_k": query.similarity_top_k,
            "keyword_k": keyword_k,
            "embedding": self._bind_embedding(query.query_embedding),
            "query": remove_lucene_chars(query.query_str),
            **filter_params,
        }
        if mode in (VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.SPARSE):
            return (
                [self._hybrid_modality_sql("keyword", filter_clause)],
                params,
                [1.0],
                keyword_k,
            )
        # alpha weighs the vector side: 1.0 is that alone, 0.0 the text alone.
        alpha = 0.5 if query.alpha is None else float(query.alpha)
        return (
            [
                self._hybrid_modality_sql("semantic", filter_clause),
                self._hybrid_modality_sql("keyword", filter_clause),
            ],
            params,
            [alpha, 1.0 - alpha],
            query.hybrid_top_k or query.similarity_top_k,
        )

    def _mmr_rerank(
        self, query: VectorStoreQuery, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep the nearest that are not also near each other.

        The candidates come back carrying their embeddings, because the distance
        between two stored rows is not something the index answers -- the
        discounting is done here, over the candidates the search was given.
        """
        # Not named with a leading underscore: a row factory building named
        # tuples renames a field that cannot be a Python identifier, and the key
        # silently became "f_embedding".
        embeddings = [record.pop("mmr_embedding", None) for record in results]
        usable = [i for i, e in enumerate(embeddings) if e]
        if not usable:
            return results[: query.similarity_top_k]
        _, chosen = get_top_k_mmr_embeddings(
            query.query_embedding,
            [list(embeddings[i]) for i in usable],
            similarity_top_k=query.similarity_top_k,
            embedding_ids=usable,
            mmr_threshold=query.mmr_threshold,
        )
        return [results[i] for i in chosen]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        mode = self._check_mode(query)
        if mode in self._TEXT_MODES:
            statements, params, weights, fused_k = self._text_plan(query, mode)
            ranked = [self.database_query(st, params=params) for st in statements]
            return self._results_to_query_result(
                self._rrf_fuse(ranked, fused_k, weights)
            )
        formatted_query, parameters = self._build_query(query, mode=mode)
        results = self.database_query(formatted_query, params=parameters)
        if mode is VectorStoreQueryMode.MMR:
            results = self._mmr_rerank(query, results)
        return self._results_to_query_result(results)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """True-async counterpart of :meth:`query`."""
        mode = self._check_mode(query)
        if mode in self._TEXT_MODES:
            statements, params, weights, fused_k = self._text_plan(query, mode)
            # The modalities do not depend on each other and each borrows its own
            # connection, so they go at once rather than one after the other.
            ranked = await asyncio.gather(
                *[self.adatabase_query(st, params=params) for st in statements]
            )
            return self._results_to_query_result(
                self._rrf_fuse(list(ranked), fused_k, weights)
            )
        formatted_query, parameters = self._build_query(query, mode=mode)
        results = await self.adatabase_query(formatted_query, params=parameters)
        if mode is VectorStoreQueryMode.MMR:
            results = self._mmr_rerank(query, results)
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
                    "Vector extension not supported\n"
                    "Unable to install pg_vector extension"
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
        """Run ``query``, repeating it while the driver says the refusal was timing.

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters bound into the query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set

        Eight concurrent writers against one label refused 30% of their statements
        with a deadlock or a serialization failure, and nothing here tried again --
        each was a collision that would have succeeded a moment later. The driver
        owns which refusals are worth repeating and the wait between attempts, and a
        success is reported back to it because the allowance is spent by every
        refusal and paid back only by that report.
        """
        check_no_copy(query)
        number = 0
        while True:
            try:
                result = self._query_once(query, params)
                self._retry_policy.succeeded()
                return result
            except AgensQueryException as failure:
                cause = failure.__cause__
                number += 1
                if cause is None:
                    raise
                decision = self._retry_policy.decide(
                    cause, number=number, wrote=True, merging=True
                )
                if not decision.retry:
                    raise
                logger.debug("attempt %d: %s", number, decision.reason)
                time.sleep(decision.delay)

    def _create_if_absent(self, query: Any, params: dict = {}) -> None:
        """Create something if it is missing, allowing for another process racing us.

        ``IF NOT EXISTS`` checks and then creates, so two stores opening on the same
        graph both see it missing and the slower one fails -- on the graph, the
        label and each index in turn. Of eight opened at once, two survived.
        """
        try:
            self.database_query(query, params)
        except AgensQueryException as failure:
            if not lost_the_creation_race(failure):
                raise

    def _query_once(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """One attempt at ``query``, rolling back on a refusal."""
        with self._acquire() as conn:
            with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    curs.execute(query, params)
                    conn.commit()
                except psycopg.Error as e:
                    conn.rollback()
                    # Every vector-store query failure comes through here, and it
                    # used to arrive with the key the reader does not look at, the
                    # server's DETAIL line spliced in by str(e), and no cause -- so
                    # it read back as "unknown" and nothing downstream could ask
                    # the driver whether another attempt was worth making.
                    raise query_failed(query, e) from e
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
                apply_search_options(conn, self.search_options)
                yield conn
        else:
            apply_search_options(self._connection, self.search_options)
            yield self._connection

    async def _apool(self) -> "agensgraph.AsyncNullConnectionPool":
        """The connections the async methods borrow when no engine was supplied.

        One connection was held and handed to every caller, so two coroutines in
        flight at once ran their statements down the same wire and into each
        other's transactions. A borrower gets a connection to itself now.

        It connects per borrow rather than keeping a set warm. A pool that keeps
        them runs its workers on the loop's executor, and ``asyncio.run`` waits
        for that executor before it returns -- so a program that used one async
        method and then ended would hang there, silently. Keeping them is worth
        having, and is offered as ``AgensEngine``: a thing the caller holds and
        closes, rather than a default that hangs on the way out.
        """
        running = asyncio.get_running_loop()
        if self._apool_pool is not None and self._apool_loop is running:
            return self._apool_pool
        async with self._apool_lock:
            if self._apool_pool is not None and self._apool_loop is running:
                return self._apool_pool

            async def configure(conn: "agensgraph.AsyncConnection") -> None:
                if self._vectors_registered:
                    await conn.register_vectors()

            pool = agensgraph.AsyncNullConnectionPool(
                self._url,
                graph=self._graph_name,
                min_size=0,
                max_size=10,
                configure=configure,
                kwargs={"autocommit": True},
                check_connections=False,
            )
            await pool.open()
            self._apool_pool = pool
            self._apool_loop = running
        return self._apool_pool

    @asynccontextmanager
    async def _aacquire(self) -> "AsyncIterator[psycopg.AsyncConnection]":
        """Async sibling of :meth:`_acquire`."""
        if self._engine is not None:
            async with self._engine.aconnection(graph_path=self._graph_name) as conn:
                await aapply_search_options(conn, self.search_options)
                yield conn
        else:
            pool = await self._apool()
            async with pool.connection() as conn:
                await aapply_search_options(conn, self.search_options)
                yield conn

    async def __aenter__(self) -> "AgensgraphVectorStore":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def close(self) -> None:
        """Close the connection this store opened for itself.

        The async side has to be closed from the loop it was used on, so
        :meth:`aclose` is separate.
        """
        if self._connection is not None and not self._connection.closed:
            self._connection.close()

    def __enter__(self) -> "AgensgraphVectorStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def aclose(self) -> None:
        """Give back the connections the async methods borrowed."""
        if self._apool_pool is not None:
            await self._apool_pool.close()
            self._apool_pool = None
            self._apool_loop = None
        if self._aconn is not None and not self._aconn.closed:
            await self._aconn.close()
            self._aconn = None

    async def adatabase_query(
        self, query: str, params: dict = {}
    ) -> List[Dict[str, Any]]:
        """Async counterpart of :meth:`database_query` (true async I/O)."""
        check_no_copy(query)
        number = 0
        while True:
            try:
                result = await self._aquery_once(query, params)
                self._retry_policy.succeeded()
                return result
            except AgensQueryException as failure:
                cause = failure.__cause__
                number += 1
                if cause is None:
                    raise
                decision = self._retry_policy.decide(
                    cause, number=number, wrote=True, merging=True
                )
                if not decision.retry:
                    raise
                logger.debug("attempt %d: %s", number, decision.reason)
                await asyncio.sleep(decision.delay)

    async def _aquery_once(
        self, query: str, params: dict = {}
    ) -> List[Dict[str, Any]]:
        """One attempt at ``query``, rolling back on a refusal."""
        async with self._aacquire() as conn:
            async with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
                try:
                    await curs.execute(query, params)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise query_failed(query, e) from e
                try:
                    data = await curs.fetchall()
                except psycopg.ProgrammingError:
                    data = []

                if data is None:
                    return []
                return [self._record_to_dict(d) for d in data]