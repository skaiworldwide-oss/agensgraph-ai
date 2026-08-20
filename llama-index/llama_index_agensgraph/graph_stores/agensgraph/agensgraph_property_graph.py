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

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import re, json
import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager, contextmanager

from llama_index.core.graph_stores.types import (
    TRIPLET_SOURCE_KEY,
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.core.graph_stores.utils import value_sanitize
from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.filters import metadata_filters_to_cypher
from llama_index_agensgraph.graph_stores.agensgraph.utils import *
from llama_index_agensgraph.graph_stores.agensgraph.utils import query_failed
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
import agensgraph
from psycopg.conninfo import make_conninfo
from agensgraph import Edge, RetryPolicy, Vertex
from agensgraph.vector import Vector, generated_column
from agensgraph.cypher import check_single_statement
from agensgraph.errors import safe_message
from agensgraph.introspect import (
    MAX_IDENTIFIER,
    Check,
    DesiredIndex,
    DesiredLabel,
    Unique,
)
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
CHUNK_LABEL = "Chunk"
# Every element is written on the label naming what it is, and each of those inherits
# from BASE_NODE_LABEL. A match on the parent still reaches all of them, so a read that
# wants everything is unchanged, while `label(n)` answers what a thing is and a match on
# one type reads only that type's storage.
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
CHUNK_SIZE = 1000
# Max example values kept per property in the enhanced schema.
ENHANCED_MAX_EXAMPLES = 5
VECTOR_INDEX_NAME = "entity"
LONG_TEXT_THRESHOLD = 52




logger = logging.getLogger(__name__)

# Default Text2Cypher prompt for this store. LlamaIndex's generic
# DEFAULT_CYPHER_TEMPALTE does not describe what this store rejects: AgensGraph wants a
# label double-quoted and refuses the Neo4j-only constructs that template invites, so a
# model given it writes Cypher that does not run. This one says how an element is
# stored and what the dialect will not take.
AGENS_CYPHER_TEMPLATE_STR = """\
Task: generate a single read-only AgensGraph (openCypher) query to answer the question.

How this graph is stored (important):
- An entity's TYPE is its vertex label, written double-quoted: match a person as
  (n:"Person"). The types available are the ones in the schema below.
- Every type inherits from "__Node__", so (n:"__Node__") matches an element of any
  type, and label(n) returns the type of one.
- The human-readable name of an entity is the property n.name; other properties are
  as named in the schema.
- Relationship TYPES are edge labels, also double-quoted, e.g.
  (a)-[r:"WORKS_AT"]->(b), or use an untyped (a)-[r]->(b) and read type(r). Use only
  relationship types shown in the schema.

Hard rules:
- Read-only ONLY: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT.
  Never CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH.
- AgensGraph is NOT Neo4j. Do NOT use Neo4j-only constructs: no pattern expressions
  inside expressions such as size((n)--()) or [(n)-->(m) | m]; no COUNT { ... };
  no EXISTS { ... }; no apoc.*; no CALL { ... } subqueries. To count a node's
  degree, MATCH its relationships and use count().
- Use only the entity types, relationship types and properties shown in the schema.
- To count, use count(*) (or count(a small property like n.name)). Do NOT write
  count(n) on a node variable -- that materializes each node's full properties
  (including large embeddings) and is far slower.
- Always end with a LIMIT of at most 50. Return ONLY the Cypher query -- no prose,
  no markdown fences.

Examples:
  Q: Which authors have the most papers?
  MATCH (a:"Author")<-[:"AUTHORED_BY"]-(p)
  RETURN a.name AS author, count(*) AS papers ORDER BY papers DESC LIMIT 10

  Q: How many entities of each type?
  MATCH (n:"__Node__")
  RETURN label(n) AS type, count(*) AS n ORDER BY n DESC LIMIT 50

Schema:
{schema}

Question: {question}
Cypher query:"""

AGENS_CYPHER_TEMPLATE = PromptTemplate(AGENS_CYPHER_TEMPLATE_STR)


def _cypher_params(given: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Bind a caller's parameters in the form the server reads them.

    A Cypher comparison is against a jsonb property, so the value has to arrive as
    jsonb. Sent as itself, a string is parsed as JSON and rejected -- ``"Alice"`` is
    not a JSON document -- and a mapping cannot be adapted at all. Both are wrapped.

    A number is left alone, and deliberately: it is already valid JSON, so it compares
    correctly either way, and it is the one kind that also appears as a ``LIMIT``,
    where jsonb is refused outright ("argument of LIMIT must be type bigint"). Wrapping
    everything would have made every paged statement in this class fail.

    Anything already wrapped by the caller is passed through untouched.
    """
    if not given:
        return {}
    bound: Dict[str, Any] = {}
    for name, value in given.items():
        bound[name] = (
            Jsonb(value) if isinstance(value, (str, dict, list)) else value
        )
    return bound


def _property_equalities(
    alias: str, properties: Dict[str, Any], prefix: str
) -> Tuple[str, Dict[str, Any]]:
    """Render ``alias."key" = %(param)s`` for each property, safely.

    Both halves have to be built rather than interpolated. A property name is quoted,
    because a name is chosen by the caller and reaches the statement as an identifier:
    a key of ``secret" IS NOT NULL OR e."secret`` turned a filter matching nothing into
    one matching everything, and gave ``delete`` the same reach. And the parameter is
    named by position rather than after the property, because a name holding a bracket
    or a percent sign does not survive being written into a placeholder.
    """
    fragments, params = [], {}
    for index, key in enumerate(properties):
        name = f"{prefix}_{index}"
        fragments.append(f'{alias}.{sql.Identifier(key).as_string(None)} = %({name})s')
        params[name] = Jsonb(properties[key])
    return " AND ".join(fragments), params




def _bounded_name(*parts: str) -> str:
    """A name for a constraint or an index that stays distinct after truncation.

    An identifier is 63 bytes and a label can be longer -- these are whatever an LLM
    called an entity type. Two labels agreeing for the first 63 bytes would be given
    one name, and the second would be read as already there and skipped, leaving that
    label without the uniqueness that stops two writers making two of one node.
    """
    name = "_".join(parts)
    encoded = name.encode()
    if len(encoded) <= MAX_IDENTIFIER:
        return name
    digest = hashlib.blake2b(encoded, digest_size=8).hexdigest()
    head = encoded[: MAX_IDENTIFIER - len(digest) - 1].decode("utf-8", "ignore")
    return f"{head}_{digest}"


def _unique_id_name(label: str) -> str:
    """The name for a label's assertion that ``id`` is unique."""
    return _bounded_name(label, "unique_id")


def _merge_race(exc: BaseException) -> bool:
    """Whether a refusal is two writers reaching the same key at the same moment.

    A graph's ``ASSERT id IS UNIQUE`` is carried by an exclusion constraint and not by
    a unique index, so the writer arriving second is told ``23P01`` where the same race
    on a table would say ``23505``. The driver knows the second is worth another try and
    not the first, since an exclusion violation is normally a fact about the row rather
    than about the timing. Here it is about the timing: the writer that lost finds the
    node already there and merges onto it.
    """
    return isinstance(exc, psycopg.errors.ExclusionViolation)


class AgensPropertyGraphStore(PropertyGraphStore):
    """
    AgensGraph Property Graph Store.

    This class implements a AgensGraph property graph store.
    """


    supports_structured_queries: bool = True
    supports_vector_queries: bool = True
    text_to_cypher_template: PromptTemplate = AGENS_CYPHER_TEMPLATE

    @require_psycopg
    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        vector_dimension: int = None,
        sanitize_query_output: bool = True,
        enhanced_schema: bool = False,
        create_indexes: bool = True,
        create: bool = True,
        refresh_schema: bool = True,
        engine: Optional[AgensEngine] = None,
        retry_attempts: int = 3,
        schema_sample: int = 100,
        async_pool_size: int = 10,
        promote_embedding: bool = True,
        statement_timeout: Optional[float] = None,
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        # How many times a statement the server says to try again is tried again.
        self.retry_policy = RetryPolicy(attempts=retry_attempts)
        # How many rows of a label establish its shape. Reading every row of a
        # label full of embeddings to learn that one key holds an array is
        # minutes rather than milliseconds.
        self.schema_sample = schema_sample
        self.async_pool_size = async_pool_size
        self._label_counts: Dict[str, int] = {}
        # Properties a caller has asked to be indexed, kept so a label
        # declared later is indexed on the way in.
        self._indexed_properties: set = set()
        # Labels whose embedding has a column of its own, so the index built over
        # it is spelled to match.
        self._promote_embedding = promote_embedding
        self._is_promoted = False
        # Two coroutines reaching the first await would each build a pool
        # and the second assignment would drop the first still holding its
        # connections.
        self._apool_lock = asyncio.Lock()
        self._promotion_read = False
        # Depth, not a flag: a caller's read-only block can hold another.
        self._read_only_depth = 0
        # Inside a bulk block the vector indexes are deliberately absent, so
        # a label first written there must not build one on the way in.
        self._bulk_depth = 0
        self._allow_server_programs = False
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema
        self.create_indexes = create_indexes
        # The driver's connection: it refuses a server too old to speak to from the
        # startup packet, decodes an element into a Vertex or an Edge rather than
        # leaving it as text to be matched, and carries the vector types once
        # registered.
        # A limit on every statement, carried in the connection's own options so it is
        # in force before the first one and costs nothing per call. Set per statement it
        # is a round trip each time, and asked for as a per-caller deadline it is more:
        # measured elsewhere at 5 round trips a read against 9.
        if statement_timeout is not None:
            options = conf.get("options", "")
            conf = {
                **conf,
                "options": (
                    f"{options} -c statement_timeout={int(statement_timeout * 1000)}"
                ).strip(),
            }
        self._conf = conf
        self.connection = agensgraph.Connection.connect(**conf)
        # With the vector types registered, an embedding travels as itself rather
        # than as its decimal spelling for the server to parse back: measured on
        # 1536 numbers, 6 KB against 31 KB and a whole query 1.35x faster.
        self._vectors_registered = self.connection.has_vectors()
        if self._vectors_registered:
            self.connection.register_vectors()
        self.vector_dimension = vector_dimension
        # The engine (pool) is wired in only after setup completes: graph/index
        # creation must run on the dedicated connection before the graph exists
        # (a pooled checkout would try to `SET graph_path` to a missing graph).
        self._engine = None
        self._aconn = None
        self._apool_pool = None
        self._apool_loop = None

        with self._get_cursor() as curs:
            graphid = get_graph_id(curs, graph_name)
            if graphid is None:
                if create:
                    create_graph(curs, graph_name)
                    self.connection.commit()
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )
                graphid = get_graph_id(curs, graph_name)

            self.graphid = graphid
            # Element labels this store has already declared, so a repeated
            # write does not re-issue the DDL.
            self._declared_labels: set = {BASE_NODE_LABEL}
            set_graph_path(curs, graph_name)

            # One builder at a time. Every one of the statements below is a CREATE OR
            # REPLACE, so two processes constructing a store against the same graph
            # rewrite the same catalog rows and one of them loses with "tuple
            # concurrently updated" -- which is any multi-worker boot, and no
            # constructor argument turned it off. The lock is transaction-scoped, so
            # the commit at the end of this block releases it whatever happens.
            curs.execute(
                "SELECT pg_advisory_xact_lock(%s)", (int(self.graphid),)
            )

            # The label list this used to keep, and the catalog and trigger that
            # maintained it, are gone: an element is written on the label naming
            # what it is.
            self.connection.ensure_labels(
                [DesiredLabel(BASE_NODE_LABEL, "v", None)], graph=graph_name
            )

            self.connection.commit()

        # What the graph already holds, so a store reopened on it knows its labels
        # before writing one. Without this a label written by an earlier run is
        # invisible until something writes it again -- and a bulk block would
        # leave that label's vector index in place, which is the one thing it is
        # there to remove.
        self._declared_labels.update(self._established_labels())

        # Schema introspection scans every node's properties (O(N)). With
        # refresh_schema=False it is deferred and computed lazily on the first
        # get_schema()/get_schema_str() call.
        self._schema_refreshed = False
        self.structured_schema = {
            "node_props": {}, "rel_props": {}, "relationships": {}, "metadata": {},
        }
        if refresh_schema:
            self.refresh_schema()
        self.verify_vector_support()
        if create_indexes and self._supports_vector_store and not self.vector_dimension:
            logger.warning(
                "vector_dimension was not provided; the HNSW vector index will "
                "not be created. Pass vector_dimension=<N> to enable indexed "
                "vector search (search still works without it, but unindexed)."
            )
        if create_indexes:
            # Asked for by shape rather than by statement: the driver reads what is
            # there and issues only what is missing. The block this replaces ran the
            # DDL and swallowed every exception it raised, so a constraint that
            # failed for a reason other than already existing failed silently.
            self._ensure_constraints(
                [Unique(BASE_NODE_LABEL, "id", _unique_id_name(BASE_NODE_LABEL))]
            )
            # A chunk is written on its own label like anything else, and a constraint
            # on the base label does not reach a child -- so without this, reading a
            # chunk by id has no index to use and two writers can make two of one.
            self._ensure_element_labels([CHUNK_LABEL])
            # Nothing indexes the type: an element is written on the label naming it,
            # so asking for one type reads that label's storage and nothing else.
            # Measured on twenty thousand of each of two types, counting one of them:
            # 248 buffers by label against 495 through a btree over a scalar copy,
            # which also cost a property in every element and an index on every write.
            if self._supports_vector_index:
                self._create_vector_index()
                self._ensure_constraints(
                    [
                        Check(
                            BASE_NODE_LABEL,
                            "jsonb_typeof(embedding) = 'array' AND "
                            f"jsonb_array_length(embedding) = {self.vector_dimension}",
                            "embedding_length",
                        )
                    ]
                )

        # An element says what it is by the label it is written on, so there is no list
        # to assert the shape of. The assertion that stood here read
        # `properties->'labels'` against a map that is already the properties, so it
        # asked about `properties.properties.labels` and never held anything back.

        # Setup is done; runtime queries may now use the pool.
        self._engine = engine

    @require_psycopg
    def _get_cursor(self) -> psycopg.Cursor:
        cursor = self.connection.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cursor

    @contextmanager
    def _acquire(self) -> "Iterator[psycopg.Connection]":
        """Yield the connection ``structured_query`` should run on.

        Uses a pooled connection from the engine when one is configured; falls
        back to the dedicated connection otherwise (the pre-engine behavior).
        """
        if self._engine is not None:
            with self._engine.connection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            yield self.connection

    async def _apool(self) -> "agensgraph.AsyncConnectionPool":
        """The pool the async methods borrow from when no engine was supplied.

        One connection was held and handed to every caller, so two coroutines in
        flight at once ran their statements down the same wire and into each
        other's transactions -- twelve concurrent reads gave "transaction commit
        at the wrong nesting level" and "graph_path is NULL". A borrower gets a
        connection to itself now.

        It connects per borrow rather than keeping a set of connections warm. A
        pool that keeps them runs its workers on the loop's executor, and
        ``asyncio.run`` waits for that executor before it returns -- so a program
        that used one async method and then ended would hang there, silently, with
        nothing to suggest what it was waiting for. Keeping the connections is
        worth having: twelve concurrent reads take 21.8 ms through a warm pool
        against 104.3 ms connecting each time. It is offered as ``AgensEngine``,
        which is a thing the caller holds and closes, rather than as a default
        that hangs on the way out.

        Built on first use and keyed to the loop it was built on, since a pool
        left over from a loop that has closed cannot be handed out.
        """
        running = asyncio.get_running_loop()
        if self._apool_pool is not None and self._apool_loop is running:
            return self._apool_pool
        async with self._apool_lock:
            if self._apool_pool is not None and self._apool_loop is running:
                return self._apool_pool

            async def configure(conn: "agensgraph.AsyncConnection") -> None:
                if await conn.has_vectors():
                    await conn.register_vectors()

            pool = agensgraph.AsyncNullConnectionPool(
                make_conninfo(**self._conf),
                graph=self.graph_name,
                min_size=0,
                max_size=self.async_pool_size,
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
            async with self._engine.aconnection(graph_path=self.graph_name) as conn:
                yield conn
        else:
            pool = await self._apool()
            async with pool.connection() as conn:
                yield conn

    async def aclose(self) -> None:
        """Give back the connections the async methods borrowed.

        A pool keeps workers of its own, and a loop will not finish while they are
        running: without this, a program that used one async method and then
        returned from ``asyncio.run`` never got there. An engine's pool belongs to
        the engine and is left for it to close.
        """
        if self._apool_pool is not None:
            await self._apool_pool.close()
            self._apool_pool = None
            self._apool_loop = None
        if self._aconn is not None and not self._aconn.closed:
            await self._aconn.close()
            self._aconn = None

    def close(self) -> None:
        """Close the connection this store opened for itself.

        The async side has to be closed from the loop it was used on, so
        :meth:`aclose` is separate.
        """
        if self.connection is not None and not self.connection.closed:
            self.connection.close()

    @property
    def client(self) -> Any:
        return self.connection

    @require_psycopg
    def verify_vector_support(self) -> None:
        """
        Verify if the graph store supports vector operations
        """
        # check if the vector index is supported
        self._supports_vector_index = False
        self._supports_vector_store = False
        with self._get_cursor() as curs:
            try:
                curs.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.connection.commit()
                self._supports_vector_store = True
                if self.vector_dimension:
                    self._supports_vector_index = True
                    # A column needs its width when it is declared, so a store
                    # that was never told the dimension cannot have one.
                    self._promote_embedding = (
                        self._promote_embedding
                        and self.connection.can_promote_properties()
                    )
            except psycopg.Error:
                self.connection.rollback()
                logger.log(logging.WARNING, """Vector extension not supported\nUnable to install pg_vector extension""")
                pass

    def _established_labels(self) -> Set[str]:
        """The labels this graph already holds that carry their own uniqueness.

        A label without it is left out deliberately: it is treated as undeclared,
        so the next write to it goes through the path that adds the constraint.
        Two writers merging the same id onto a label that has none each create an
        element rather than finding one.
        """
        try:
            labels = self.connection.labels(graph=self.graph_name)
            constraints = self.connection.constraints(graph=self.graph_name)
        finally:
            self.connection.commit()
        unique_on_id = {
            constraint.label for constraint in constraints if constraint.unique
        }
        return {
            label.name
            for label in labels
            if label.kind == "v"
            and label.parent == BASE_NODE_LABEL
            and label.name in unique_on_id
        }

    def _promoted(self) -> bool:
        """Whether the base label's embedding has a column of its own.

        A column on a parent is a column on each label that inherits it -- an
        element written to the child fills it, and an index over it builds on the
        child -- so this is asked of the base once.
        """
        if self._promotion_read:
            return self._is_promoted
        declared = self.connection.declared_properties(
            BASE_NODE_LABEL, graph=self.graph_name
        )
        self.connection.commit()
        self._is_promoted = any(prop.name == "embedding" for prop in declared)
        self._promotion_read = True
        return self._is_promoted

    def _promoted_label(self, label: str) -> bool:
        """Whether *this* label's embedding is in a column.

        Inheriting the base is what carries the column down, so a label that
        stands outside it -- one the graph already held, written by something
        else -- keeps its embedding in the property map however the base is
        stored. Indexed as though it had the column, the server refuses it:
        "operator class vector_cosine_ops does not accept data type jsonb".
        """
        if not self._promoted():
            return False
        if label == BASE_NODE_LABEL:
            return True
        parents = {
            entry.name: entry.parent
            for entry in self.connection.labels(graph=self.graph_name)
        }
        self.connection.commit()
        seen: Set[str] = set()
        name: Optional[str] = label
        while name and name not in seen:
            seen.add(name)
            name = parents.get(name)
            if name == BASE_NODE_LABEL:
                return True
        return False

    def _ensure_promoted_column(self) -> None:
        """Give the embedding a column of its own, on the label every element is a.

        Read out of the property map, an embedding is decimal text in a bag that
        has to come out of TOAST and be parsed before a distance can be taken,
        once per element the filter kept. That is what a metadata-filtered search
        spends its time on: of the 475 ms a filter keeping 2,000 of 20,000
        elements took, the index found them in 0.8 ms and the rest was reading
        their bags. In a column the same search takes 1.5 ms.

        The column is generated from the property, so writes are unchanged and a
        graph that already holds elements is filled from what it already has.
        """
        if not (self._promote_embedding and self._supports_vector_index):
            return
        if self._promoted():
            return
        self.structured_query(
            sql.SQL(
                "ALTER VLABEL {} ADD COLUMN "
                + generated_column("embedding", int(self.vector_dimension))
            ).format(sql.Identifier(BASE_NODE_LABEL))
        )
        self._is_promoted = True
        # Every index that was there was built over the property map, and the
        # query now asks about the column, so none of them can serve one. Dropped
        # here and built again below in the spelling that matches -- left alone
        # they stay, are never used, and nothing says so.
        for label in sorted(self._declared_labels):
            self.structured_query(
                sql.SQL("DROP PROPERTY INDEX IF EXISTS {}").format(
                    sql.Identifier(self._vector_index_name(label))
                )
            )

    def _vector_index_name(self, label: str) -> str:
        """The name of a label's vector index."""
        if label == BASE_NODE_LABEL:
            # What it has always been called, so a graph written before this keeps
            # the index it has instead of gaining a second one beside it.
            return VECTOR_INDEX_NAME
        return _bounded_name(label, VECTOR_INDEX_NAME)

    def _create_vector_index(self, labels: Optional[Iterable[str]] = None) -> None:
        """Build the HNSW index over the embeddings, on every label that holds any.

        An index belongs to one label's storage and does not reach the labels that
        inherit from it. Since an element is written on the label naming what it
        is, indexing only the base left the search reading every embedded element:
        on two thousand of them the child branch of the plan was a sequential scan
        at 5.27 ms, against 0.38 ms once the label itself was indexed.

        The expression has to match what the vector query emits --
        ``n.embedding::vector(N)`` -- or the planner cannot use it, and the search
        goes back to reading everything without saying so.
        """
        if not self._supports_vector_index:
            return
        wanted = sorted(self._declared_labels if labels is None else labels)
        self._ensure_promoted_column()
        for label in wanted:
            # The expression has to be what the query ends up asking for. Against
            # a column the query says `n.embedding` and the cast falls away, so an
            # index built over the cast cannot serve it -- measured, 568 ms
            # against 1.8 ms, and nothing says the index went unused.
            expression = (
                sql.SQL("(embedding vector_cosine_ops)")
                if self._promoted_label(label)
                else sql.SQL(
                    "((embedding::vector("
                    + str(int(self.vector_dimension))
                    + ")) vector_cosine_ops)"
                )
            )
            self.structured_query(
                sql.SQL(
                    "CREATE PROPERTY INDEX IF NOT EXISTS {name} ON {label} "
                    "USING hnsw "
                ).format(
                    name=sql.Identifier(self._vector_index_name(label)),
                    label=sql.Identifier(label),
                )
                + expression
            )

    @contextmanager
    def bulk_ingest(self) -> Iterator[None]:
        """Drop the vector indexes for the writes in this block and build them after.

        Keeping them current costs more than the writing does. Measured on 4,000
        elements of 384 numbers, with the arms alternating in one process: 12.40 s
        with the index kept against 6.23 s with it built afterwards, and the gap
        widens as the label grows, because every insertion walks a graph that is
        getting bigger.

        **A search inside this block reads every element**, for everyone using the
        graph and not only this caller, because for its duration the indexes are
        not there. It is for loading a corpus, not for a store answering
        questions. If the process is killed inside the block they stay dropped;
        constructing the store builds them again.
        """
        if not self._supports_vector_index:
            yield
            return
        for label in sorted(self._declared_labels):
            self.structured_query(
                sql.SQL("DROP PROPERTY INDEX IF EXISTS {}").format(
                    sql.Identifier(self._vector_index_name(label))
                )
            )
        self._bulk_depth += 1
        try:
            yield
        finally:
            self._bulk_depth -= 1
            if not self._bulk_depth:
                # Every label, not only the ones dropped: a label first written
                # inside the block needs one too.
                self._create_vector_index()

    def create_property_index(
        self, property_name: str, label: Optional[str] = None
    ) -> None:
        """Index a property, on every label that holds elements.

        A metadata-filtered ``vector_query`` cannot use the HNSW index for the
        filter, so without a property index the filter reads every embedded
        element.

        An index belongs to one label's storage and does not reach the labels that
        inherit from it. Indexing only the base label left every element unindexed,
        because an element is written on the label naming what it is -- and the
        base holds nothing. The property is remembered, so a label declared later
        is indexed as it appears.
        """
        self._indexed_properties.add(property_name)
        labels = [label] if label else sorted(self._declared_labels)
        self._ensure_property_indexes(labels, [property_name])

    def _ensure_property_indexes(
        self, labels: Iterable[str], properties: Iterable[str]
    ) -> None:
        """Make a btree index exist for each label and property named."""
        desired = [
            DesiredIndex(
                label, (prop,), False, _bounded_name(label, prop, "idx"), "btree", None
            )
            for label in labels
            for prop in properties
        ]
        if not desired:
            return

        def once() -> None:
            with self._acquire() as conn:
                try:
                    with conn.transaction():
                        conn.execute(
                            "SELECT pg_advisory_xact_lock(%s)", (int(self.graphid),)
                        )
                        conn.ensure_indexes(desired, graph=self.graph_name)
                except psycopg.Error as e:
                    raise query_failed("declaring property indexes", e) from e

        self._run_with_retry(once)

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        self._schema_from(self._describe())

    async def arefresh_schema(self) -> None:
        """True-async counterpart of :meth:`refresh_schema`.

        ``enhanced_schema`` is the exception: its per-property statistics are a
        statement each and still run synchronously, so a store that asked for
        them holds the loop for that part.
        """
        self._schema_from(await self._adescribe())

    def _schema_from(self, description: Any) -> None:
        """Record what a description says the graph holds."""
        self.structured_schema = {
            "node_props": self._properties_of(description, "v"),
            "rel_props": self._properties_of(description, "e"),
            "relationships": [
                {"start": triple.start, "type": triple.edge, "end": triple.end}
                for triple in description.triples
            ],
            "metadata": {},
        }
        self._label_counts = dict(description.counts)

        if self.enhanced_schema:
            self._enhance_schema()

        self._schema_refreshed = True

    def get_schema(self, refresh: bool = False) -> Any:
        # Read on first access if it was deferred at construction.
        if refresh or not self._schema_refreshed:
            self.refresh_schema()

        return self.structured_schema

    async def aget_schema(self, refresh: bool = False) -> Any:
        """True-async counterpart of :meth:`get_schema`."""
        if refresh or not self._schema_refreshed:
            await self.arefresh_schema()

        return self.structured_schema

    async def aget_schema_str(
        self,
        refresh: bool = False,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
    ) -> str:
        """True-async counterpart of :meth:`get_schema_str`.

        Only the reading is async; the formatting below it is arithmetic on what
        was read.
        """
        await self.aget_schema(refresh=refresh)
        return self.get_schema_str(
            refresh=False, exclude_types=exclude_types, include_types=include_types
        )

    def get_schema_str(
        self,
        refresh: bool = False,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
    ) -> str:
        schema = self.get_schema(refresh=refresh)
        def filter_func(x: str) -> bool:
            return x in include_types if include_types else x not in exclude_types

        filtered_schema: Dict[str, Any] = {
            "node_props": {
                k: v for k, v in schema.get("node_props", {}).items() if filter_func(k)
            },
            "rel_props": {
                k: v for k, v in schema.get("rel_props", {}).items() if filter_func(k)
            },
            "relationships": [
                r
                for r in schema.get("relationships", [])
                if all(filter_func(r[t]) for t in ["start", "end", "type"])
            ],
        }

        formatted_node_props = []
        formatted_rel_props = []
        # Format node properties
        for label, props in filtered_schema["node_props"].items():
            prop_strs = []
            for prop in props:
                prop_str = f"{prop['property']}: {prop['type']}"
                # Statistics are only present when ``enhanced_schema`` is on.
                if "min" in prop and "max" in prop:
                    prop_str += f" (min: {prop['min']}, max: {prop['max']})"
                elif "min_size" in prop and "max_size" in prop:
                    prop_str += (
                        f" (list size min: {prop['min_size']}, max: {prop['max_size']})"
                    )
                elif prop.get("values"):
                    examples = ", ".join(str(v) for v in prop["values"])
                    prop_str += f" (e.g. {examples})"
                prop_strs.append(prop_str)
            props_str = ", ".join(prop_strs)
            formatted_node_props.append(f"{label} {{{props_str}}}")

        # Format relationship properties using structured_schema
        for type, props in filtered_schema["rel_props"].items():
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in props]
            )
            formatted_rel_props.append(f"{type} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
            for el in filtered_schema["relationships"]
        ]

        return "\n".join(
            [
                "Node properties:",
                "\n".join(formatted_node_props),
                "Relationship properties:",
                "\n".join(formatted_rel_props),
                "The relationships:",
                "\n".join(formatted_rels),
            ]
        )

    def _ensure_element_labels(self, labels: Iterable[str]) -> None:
        """Declare the labels about to be written, once per store.

        A label is an identifier and cannot be a parameter, so a write is grouped by it
        and the label has to exist first. Each inherits from the base label: a read that
        matches the parent still sees every element, which is what leaves the rest of
        this class's statements alone.

        Each also carries its own uniqueness on ``id``. A constraint on the parent does
        not reach a child -- measured, the same id written to two child labels gave two
        elements -- so without one per label two writers merging the same id would each
        create an element rather than find one.
        """
        # Deduplicated: a batch of nodes usually shares a label, and asking for the
        # same one twice is asking for two constraints of the same name.
        wanted = list(
            dict.fromkeys(
                label
                for label in labels
                if label and label not in self._declared_labels
            )
        )
        if not wanted:
            return
        self._run_with_retry(lambda: self._declare_labels(wanted))
        self._declared_labels.update(wanted)

    def _declare_labels(self, wanted: List[str]) -> None:
        """Create the labels in ``wanted``, one writer at a time."""
        with self._acquire() as conn:
            try:
                # Everything here in one transaction, so a refusal takes the whole
                # group back rather than leaving the connection unable to run
                # anything else.
                with conn.transaction():
                    # Reconciling reads what is there and then creates what is not,
                    # which is two steps: eight writers all found Person missing and
                    # the losers were told it already exists. The lock lasts as long
                    # as the transaction, so it is released whichever way this ends.
                    conn.execute(
                        "SELECT pg_advisory_xact_lock(%s)", (int(self.graphid),)
                    )
                    conn.ensure_labels(
                        [
                            DesiredLabel(label, "v", BASE_NODE_LABEL)
                            for label in wanted
                        ],
                        graph=self.graph_name,
                    )
                    # A constraint on the parent does not reach a child -- measured,
                    # the same id written to two child labels gave two elements -- so
                    # each label carries its own.
                    conn.ensure_constraints(
                        [
                            Unique(label, "id", _unique_id_name(label))
                            for label in wanted
                        ],
                        graph=self.graph_name,
                    )
            except psycopg.Error as e:
                raise query_failed(f"declaring labels {wanted}", e) from e
        # A property asked for before this label existed is indexed on it now.
        if self._indexed_properties:
            self._ensure_property_indexes(wanted, sorted(self._indexed_properties))
        # And so is the embedding, for the same reason: the index belongs to this
        # label's storage and the base label's does not reach it. Not inside a bulk
        # block -- building it there is the cost that block exists to avoid, and
        # the block builds every label's on the way out.
        if not self._bulk_depth:
            self._create_vector_index(wanted)

    def _ensure_constraints(self, desired: List[Any]) -> None:
        """Make the constraints named in ``desired`` exist, under the graph's lock."""

        def once() -> None:
            with self._acquire() as conn:
                try:
                    with conn.transaction():
                        conn.execute(
                            "SELECT pg_advisory_xact_lock(%s)", (int(self.graphid),)
                        )
                        conn.ensure_constraints(desired, graph=self.graph_name)
                except psycopg.Error as e:
                    raise query_failed(f"declaring constraints {desired}", e) from e

        self._run_with_retry(once)

    def _build_upsert_nodes_ops(
        self, nodes: List[LabelledNode]
    ) -> List[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the ordered (query, params) operations for ``upsert_nodes``."""
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.model_dump(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.model_dump(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        ops: List[Tuple[sql.Composed, Dict[str, Any]]] = []

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MERGE (c:{CHUNK_LABEL} {{id: row.id}})
                    SET c.text = row.text
                    WITH c, row
                    SET c += row.properties, c.embedding = row.embedding
                    RETURN count(*)
                    """).format(
                        CHUNK_LABEL=sql.Identifier(CHUNK_LABEL)
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))

        # Grouped by label, because a label is an identifier and cannot be bound. The
        # element carries its type by being written on it, so nothing appends to a list
        # and nothing keeps a scalar copy of the label beside it.
        by_label: Dict[str, List[dict]] = {}
        for entity in entity_dicts:
            by_label.setdefault(entity["label"], []).append(entity)

        for label, entities in by_label.items():
            for index in range(0, len(entities), CHUNK_SIZE):
                chunked_params = entities[index : index + CHUNK_SIZE]
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MERGE (e:{label} {{id: row.id}})
                    SET e += CASE WHEN row.properties IS NOT NULL THEN row.properties ELSE properties(e) END
                    SET e.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE e.name END
                    """).format(label=sql.Identifier(label)),
                    {"chunked_params": Jsonb(chunked_params)}
                ))
                # Write embeddings in a SEPARATE statement from the MENTIONS link
                # below: AgensGraph does not persist an earlier SET when a later
                # `WITH ... WHERE` filters out every row ahead of a MERGE, so
                # combining them drops the embedding for entities with no chunk.
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    MATCH (e:{label} {{id: row.id}})
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    """).format(label=sql.Identifier(label)),
                    {"chunked_params": Jsonb(chunked_params)}
                ))
                # Link each entity to its source chunk via MENTIONS, for the rows
                # that carry a triplet_source_id.
                ops.append((
                    sql.SQL("""
                    UNWIND %(chunked_params)s AS row
                    WITH row WHERE row.properties.triplet_source_id IS NOT NULL
                    MATCH (e:{label} {{id: row.id}})
                    MERGE (c:{CHUNK_LABEL} {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:"MENTIONS"]-(c)
                    """).format(
                        label=sql.Identifier(label),
                        CHUNK_LABEL=sql.Identifier(CHUNK_LABEL),
                    ), {"chunked_params": Jsonb(chunked_params)}
                ))

        return ops

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        self._ensure_element_labels(
            item.label for item in nodes if isinstance(item, EntityNode)
        )
        for query, params in self._build_upsert_nodes_ops(nodes):
            self.structured_query(query, params)

    async def aupsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """True-async counterpart of :meth:`upsert_nodes`."""
        self._ensure_element_labels(
            item.label for item in nodes if isinstance(item, EntityNode)
        )
        for query, params in self._build_upsert_nodes_ops(nodes):
            await self.astructured_query(query, params)

    def _build_upsert_relations_ops(
        self, relations: List[Relation]
    ) -> List[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the ordered (query, params) operations for ``upsert_relations``.

        Relations are grouped by label and each group is UNWIND-batched in
        CHUNK_SIZE rows (the relationship type must be a literal in MERGE, so a
        batch can only span one label). The batched ``MERGE (n {id: row.id})`` is
        index-backed.

        The endpoints are merged on the base label, which reaches an element of any
        type because every element label inherits from it -- so a relation finds the
        entity a caller wrote whatever type it was given. An endpoint that is not there
        is created on the base label itself, carrying no type, since a relation names
        only an id. Nodes are written before relations by the ingest that calls this,
        so that is the out-of-order case rather than the ordinary one.
        """
        by_label: Dict[str, List[dict]] = {}
        for r in relations:
            d = r.model_dump()
            by_label.setdefault(d["label"], []).append(d)

        ops: List[Tuple[sql.Composed, Dict[str, Any]]] = []
        for label, rels in by_label.items():
            for index in range(0, len(rels), CHUNK_SIZE):
                chunk = rels[index : index + CHUNK_SIZE]
                rows = [
                    {
                        "source_id": p["source_id"],
                        "target_id": p["target_id"],
                        "properties": p["properties"],
                    }
                    for p in chunk
                ]
                ops.append((
                    sql.SQL("""
                    UNWIND %(rows)s AS row
                    MERGE (source: {BASE_NODE_LABEL} {{id: row.source_id}})
                    MERGE (target: {BASE_NODE_LABEL} {{id: row.target_id}})
                    WITH source, target, row
                    MERGE (source)-[r:{label}]->(target)
                    SET r += row.properties
                    RETURN count(*)
                    """).format(
                        BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                        label=sql.Identifier(label),
                    ), {"rows": Jsonb(rows)}
                ))
        return ops

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        for query, params in self._build_upsert_relations_ops(relations):
            self.structured_query(query, params)

    async def aupsert_relations(self, relations: List[Relation]) -> None:
        """True-async counterpart of :meth:`upsert_relations`."""
        for query, params in self._build_upsert_relations_ops(relations):
            await self.astructured_query(query, params)

    @staticmethod
    def _or_equalities(
        field: str, values: List[str], prefix: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Build an index-friendly ``(field = v0 OR field = v1 ...)`` fragment.

        The ``<@`` / ``IN`` containment forms never use the btree index; an
        OR-of-equalities is served via a BitmapOr index scan when ``field`` is
        indexed (``id`` always is; other properties when explicitly indexed).
        """
        terms = []
        params: Dict[str, Any] = {}
        for i, value in enumerate(values):
            pname = f"{prefix}_{i}"
            params[pname] = Jsonb(value)
            terms.append(f"{field} = %({pname})s")
        return "(" + " OR ".join(terms) + ")", params

    @staticmethod
    def _keyed_equalities(
        field: str, values: List[str], name: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """A prelude and a predicate that ask the index once per value.

        An OR of equalities reaches the index too, and executes at the same speed
        -- but the planner has to work through every term of it, and that grows
        with the list: on 5,000 ids, 151.7 ms of planning against 0.2 ms, roughly
        half the statement. One bound list costs the same to plan whatever its
        length.

        The form matters. ``UNWIND range(...) AS i ... WHERE e.id = ids[i]``,
        subscripting the list inside the predicate, is not something the index can
        serve and reads the whole label; unwinding the values themselves and
        comparing against the unwound one is.
        """
        key = f"{name}_key"
        return (
            f"UNWIND %({name})s AS {key} ",
            f"{field} = {key}",
            {name: Jsonb(list(values))},
        )

    def _build_get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build the (query, params) for :meth:`get`."""
        query = """SELECT t.name,
                            t.type,
                            t.properties - 'embedding' - 'id' AS properties
                     FROM ("""
        params: Dict[str, Any] = {}
        predicate = ""
        if ids:
            prelude, predicate, id_params = self._keyed_equalities(
                "e.id", ids, "get_ids"
            )
            query += prelude
            params.update(id_params)
        query += 'MATCH (e:{BASE_NODE_LABEL}) '
        query += "WHERE e.id IS NOT NULL "
        if ids is not None and len(ids) == 0:
            # An explicit empty id list means "no nodes". Without this guard the
            # `if ids:` below skips the filter and the query matches every node.
            query += "AND false "
        elif ids:
            query += "AND " + predicate + " "

        if properties:
            frag, prop_params = _property_equalities("e", properties, "get_prop")
            query += "AND " + frag
            params.update(prop_params)

        query += """
            RETURN
            e.id AS name,
            label(e) AS type,
            properties(e) AS properties
        """
        query += ")t"
        return (
            sql.SQL(query).format(
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL)
            ),
            params,
        )

    @staticmethod
    def _get_response_to_nodes(
        response: Optional[List[Dict[str, Any]]]
    ) -> List[LabelledNode]:
        response = response if response else []
        nodes: List[LabelledNode] = []
        for record in response:
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=record["properties"],
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=record["properties"],
                    )
                )

        return nodes

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        query, params = self._build_get(properties, ids)
        response = self.structured_query(query, params=params)
        return self._get_response_to_nodes(response)

    async def aget(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """True-async counterpart of :meth:`get`."""
        query, params = self._build_get(properties, ids)
        response = await self.astructured_query(query, params=params)
        return self._get_response_to_nodes(response)

    async def aget_llama_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """True-async counterpart of :meth:`get_llama_nodes`."""
        converted: List[BaseNode] = []
        for node in await self.aget(ids=node_ids):
            try:
                converted.append(metadata_dict_to_node(node.properties))
                converted[-1].set_content(node.text)  # type: ignore[attr-defined]
            except Exception:
                continue
        return converted

    async def adelete_llama_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        ref_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """True-async counterpart of :meth:`delete_llama_nodes`.

        The base class reads each id with a statement of its own and then calls
        the synchronous delete, holding the event loop for all of it.
        """
        nodes: List[LabelledNode] = []
        node_ids = node_ids or []
        for id_ in node_ids:
            nodes.extend(await self.aget(properties={TRIPLET_SOURCE_KEY: id_}))
        if node_ids:
            nodes.extend(await self.aget(ids=node_ids))

        ref_doc_ids = ref_doc_ids or []
        for id_ in ref_doc_ids:
            nodes.extend(await self.aget(properties={"ref_doc_id": id_}))
        if ref_doc_ids:
            nodes.extend(await self.aget(ids=ref_doc_ids))

        await self.adelete(ids=[node.id for node in nodes])

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Triplet]:
        """The triplets matching every argument given."""
        query, params = self._build_get_triplets(
            entity_names, relation_names, properties, ids, limit
        )
        return self._triplets_from(self.structured_query(query, params=params))

    async def aget_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Triplet]:
        """True-async counterpart of :meth:`get_triplets`.

        The base class answers this by calling the synchronous one, which holds
        the event loop for the whole round trip. An async retriever would do a
        real async vector query and then block on this.
        """
        query, params = self._build_get_triplets(
            entity_names, relation_names, properties, ids, limit
        )
        return self._triplets_from(await self.astructured_query(query, params=params))

    def _build_get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> Tuple[sql.Composed, Dict[str, Any]]:
        """Build the (query, params) for :meth:`get_triplets`.

        ``limit`` bounds what comes back. It was a literal 100 written into the
        statement, so a caller asking a broad question was given a hundred rows and
        no way to know the answer had been cut.
        """
        params: Dict[str, Any] = {"triplet_limit": limit}

        query = """
                SELECT t.type,
                        t.rel_prop,
                        t.source_id,
                        t.source_type,
                        t.source_properties - 'embedding' - 'name' AS source_properties,
                        t.target_id,
                        t.target_type,
                        t.target_properties - 'embedding' - 'name' AS target_properties
                FROM ("""
        if ids:
            prelude, keyed, id_params = self._keyed_equalities("e.id", ids, "gt_ids")
            query += prelude
            params.update(id_params)
        query += "MATCH (e)-[r]->(t) "

        # Collected and joined once. Each argument used to add its own separator, so a
        # combination nobody had tried emitted `AND AND` or two fragments with nothing
        # between them -- six of the fifteen combinations of these four arguments were
        # a syntax error, and the one test covering the method passes only entity_names.
        predicates = [f"label(e) <> '{CHUNK_LABEL}'"]

        if entity_names:
            frag, p = self._or_equalities("e.name", entity_names, "etn")
            predicates.append(frag)
            params.update(p)

        if relation_names:
            # type(r) is the edge label, not a property, so it can't use a
            # property index; the containment form is fine here.
            predicates.append("type(r) <@ %(relation_names)s")
            params["relation_names"] = Jsonb(relation_names)

        if ids:
            predicates.append(keyed)

        if properties:
            frag, prop_params = _property_equalities("e", properties, "get_prop")
            predicates.append(frag)
            params.update(prop_params)

        query += "WHERE " + " AND ".join(predicates) + " "

        query += """
            RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
            label(e) AS source_type,
            properties(e) AS source_properties,
            t.id as target_id,
            label(t) AS target_type, properties(t) AS target_properties
            LIMIT %(triplet_limit)s
        """

        query += ")t"
        return (
            sql.SQL(query).format(
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
            ),
            params,
        )

    @staticmethod
    def _triplets_from(data: Optional[List[Dict[str, Any]]]) -> List[Triplet]:
        """Shape what the triplet read returned."""
        triplets = []
        for record in data or []:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=record["source_properties"],
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=record["target_properties"],
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=record["rel_prop"],
            )
            triplets.append([source, rel, target])
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        built = self._build_get_rel_map(graph_nodes, depth, limit)
        if built is None:
            return []
        query, params = built
        return self._rel_map_from(
            self.structured_query(query, params=params), ignore_rels
        )

    async def aget_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """True-async counterpart of :meth:`get_rel_map`.

        The base class answers this by calling the synchronous one, which holds
        the event loop for the whole round trip -- and this is the expensive half
        of what an async retriever does after its vector query.
        """
        built = self._build_get_rel_map(graph_nodes, depth, limit)
        if built is None:
            return []
        query, params = built
        return self._rel_map_from(
            await self.astructured_query(query, params=params), ignore_rels
        )

    def _build_get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
    ) -> Optional[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the (query, params) for :meth:`get_rel_map`, or None for no seeds."""
        ids = [node.id for node in graph_nodes]
        if not ids:
            return None
        query = """SELECT t.source_id,
                            t.source_type,
                            t.source_properties - 'embedding' - 'id' AS source_properties,
                            t.type,
                            t.rel_properties,
                            t.target_id,
                            t.target_type,
                            t.target_properties - 'embedding' - 'id' AS target_properties
                      FROM (
                """
        # The seeds are unwound and compared against the unwound value, which the
        # id index serves. Subscripting the list inside the predicate --
        # `UNWIND range(...) AS i ... WHERE e.id = ids[i]`, which stood here once
        # -- is not something it can serve, and read the whole label per call.
        seed_prelude, seed_frag, seed_params = self._keyed_equalities(
            "e.id", ids, "rm_ids"
        )
        # AgensGraph's variable-length-edge engine is far slower than an equivalent
        # fixed pattern even at depth 1, so use a plain 1-hop match for the common
        # depth<=1 case and the *1..depth path only for multi-hop maps.
        if depth <= 1:
            traversal = (
                seed_prelude + """
            MATCH (e:{BASE_NODE_LABEL})
            WHERE """ + seed_frag + """
            MATCH (e)-[rel]-(other)
            WHERE type(rel) <> 'MENTIONS'
                """
            )
        else:
            traversal = (
                seed_prelude + """
            MATCH (e:{BASE_NODE_LABEL})
            WHERE """ + seed_frag + """
            MATCH p=(e)-[r*1..{depth}]-(other)
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, collect(type(rel)) AS types
            WHERE all(x IN types WHERE x <> 'MENTIONS')
                """
            )
        query += traversal + """
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel AS rel_properties,
                endNode(rel) AS endNode
            RETURN source.id AS source_id,
                label(source) AS source_type,
                properties(source) AS source_properties,
                type,
                properties(rel_properties) as rel_properties,
                endNode.id AS target_id,
                label(endNode) AS target_type,
                properties(endNode) AS target_properties
            LIMIT %(limit)s
            """
        query += ")t"
        return (
            sql.SQL(query).format(
                BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                depth=depth,
            ),
            {**seed_params, "limit": limit},
        )

    @staticmethod
    def _rel_map_from(
        response: Optional[List[Dict[str, Any]]],
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Shape what the rel-map read returned."""
        triples: List[Triplet] = []
        ignore_rels = ignore_rels or []
        for record in response or []:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=record["source_properties"],
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=record["target_properties"],
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=record["rel_properties"],
            )
            triples.append([source, rel, target])

        return triples
    
    def _delete_ops(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """The statements :meth:`delete` runs, in order."""
        ops: List[Tuple[Any, Dict[str, Any]]] = []
        if entity_names:
            frag, params = self._or_equalities("n.name", entity_names, "etn")
            ops.append(
                ('MATCH (n:"__Node__") WHERE ' + frag + " DETACH DELETE n", params)
            )

        if ids:
            prelude, keyed, params = self._keyed_equalities("n.id", ids, "del_ids")
            ops.append(
                (
                    prelude
                    + 'MATCH (n:"__Node__") WHERE '
                    + keyed
                    + " DETACH DELETE n",
                    params,
                )
            )

        if relation_names:
            for rel in relation_names:
                ops.append(
                    (
                        sql.SQL("MATCH ()-[r:{rel}]->() DELETE r").format(
                            rel=sql.Identifier(rel)
                        ),
                        {},
                    )
                )

        if properties:
            frag, params = _property_equalities("e", properties, "del_prop")
            ops.append(("MATCH (e) WHERE " + frag + " DETACH DELETE e", params))
        return ops

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        for query, params in self._delete_ops(
            entity_names, relation_names, properties, ids
        ):
            self.structured_query(query, params=params)

    async def adelete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """True-async counterpart of :meth:`delete`.

        The base class answers this by calling the synchronous one, which holds
        the event loop for every statement it runs.
        """
        for query, params in self._delete_ops(
            entity_names, relation_names, properties, ids
        ):
            await self.astructured_query(query, params=params)

    def _build_vector_query(
        self, query: VectorStoreQuery
    ) -> Optional[Tuple[sql.Composed, Dict[str, Any]]]:
        """Build the (query, params) for :meth:`vector_query`, or None if vector
        operations are unsupported."""
        # Translate metadata filters into a parameterized WHERE fragment so the
        # ANN search can be scoped (mirrors the filtered-vector-search feature of
        # other graph integrations, but injection-safe).
        filter_clause: sql.Composed = sql.SQL("")
        filter_params: Dict[str, Any] = {}
        if query.filters:
            snippet, filter_params = metadata_filters_to_cypher(
                query.filters, alias="n"
            )
            filter_clause = sql.SQL("AND (") + snippet + sql.SQL(")")

        if self._supports_vector_index:
            # The nearest-neighbour ORDER BY + LIMIT live INSIDE the Cypher
            # sub-query against the actual query embedding, so AgensGraph can
            # use the HNSW index; the outer SQL only reshapes the properties.
            # Dimension is the store's configured dimension (a pgvector typmod
            # must be a literal, so it is interpolated, not bound).
            vector_query = (
                """
                SELECT
                    t.name,
                    t.type,
                    t.properties - 'embedding' - 'name' - 'id' AS properties,
                    t.similarity
                FROM (
                    MATCH (n: {BASE_NODE_LABEL})
                    WHERE n.embedding IS NOT NULL {filter_clause}
                    WITH n,
                         (n.embedding::vector({dim}) <=> {bound_embedding}) AS dist
                    RETURN n.id as name,
                           properties(n) AS properties,
                           (1 - dist) AS similarity,
                           label(n) AS type
                    ORDER BY dist
                    LIMIT %(top_k)s
                )t;
                """
            )
            return (
                sql.SQL(vector_query).format(
                    BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                    BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                    dim=sql.SQL(str(int(self.vector_dimension))),
                    bound_embedding=self._embedding_placeholder(
                        "query_embedding", self.vector_dimension
                    ),
                    filter_clause=filter_clause,
                ),
                {
                    "query_embedding": self._bind_embedding(query.query_embedding),
                    "top_k": query.similarity_top_k,
                    **filter_params,
                },
            )
        elif self._supports_vector_store:
            vector_query = """SELECT t.name,
                                t.type,
                                t.similarity,
                                t.properties - 'embedding' - 'name' - 'id' AS properties
                            FROM (
                            """
            vector_query += """
                            MATCH (n: {BASE_NODE_LABEL})
                            WHERE n.embedding IS NOT NULL {filter_clause}
                            WITH n,
                                {bound_embedding} <=> n.embedding::vector AS cos_d
                            RETURN n.id as name,
                                properties(n) AS properties,
                                1-cos_d as similarity,
                                label(n) AS type
                            ORDER BY cos_d
                            LIMIT %(top_k)s
                            """
            vector_query += ")t"
            return (
                sql.SQL(vector_query).format(
                    BASE_NODE_LABEL=sql.Identifier(BASE_NODE_LABEL),
                    BASE_ENTITY_LABEL=sql.Literal(BASE_ENTITY_LABEL),
                    bound_embedding=self._embedding_placeholder("query_embedding"),
                    filter_clause=filter_clause,
                ),
                {
                    "query_embedding": self._bind_embedding(query.query_embedding),
                    "top_k": query.similarity_top_k,
                    **filter_params,
                },
            )
        else:
            return None

    def _bind_embedding(self, embedding: Sequence[float]) -> Any:
        """Bind an embedding as itself where the server can read one.

        A list sent as jsonb is written out as decimal text and parsed back: on
        1536 numbers that is 31 KB on the wire against 6 KB, and the whole query
        measured 1.35x slower. Where the vector types are not registered -- no
        vector extension -- it stays jsonb with a cast, which still works.
        """
        if self._vectors_registered:
            return Vector(embedding)
        return Jsonb(list(embedding))

    def _embedding_placeholder(
        self, name: str, dimension: Optional[int] = None
    ) -> sql.SQL:
        """How a bound embedding is spelled in the statement.

        A vector needs no cast; a list of numbers in jsonb does, and a typmod has
        to be a literal, so it is written in rather than bound.
        """
        if self._vectors_registered:
            return sql.SQL("%({})s").format(sql.SQL(name))
        if dimension:
            return sql.SQL("%({})s::vector({})").format(
                sql.SQL(name), sql.SQL(str(int(dimension)))
            )
        return sql.SQL("%({})s::vector").format(sql.SQL(name))

    @staticmethod
    def _vector_data_to_result(
        data: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[LabelledNode], List[float]]:
        data = data if data else []
        nodes: List[LabelledNode] = []
        scores: List[float] = []
        for record in data:
            node = EntityNode(
                name=record["name"],
                label=record["type"],
                properties=record["properties"],
            )
            nodes.append(node)
            scores.append(record["similarity"])

        return (nodes, scores)

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        built = self._build_vector_query(query)
        data = self.structured_query(*built) if built is not None else []
        return self._vector_data_to_result(data)

    async def avector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """True-async counterpart of :meth:`vector_query`."""
        built = self._build_vector_query(query)
        data = await self.astructured_query(*built) if built is not None else []
        return self._vector_data_to_result(data)

    @staticmethod
    def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
        """Turn a row into plain Python.

        The driver decodes an element before it gets here, so a vertex arrives as a
        Vertex and a relationship as an Edge carrying its own properties and the ids of
        both its endpoints. Reading them out of the text they print as lost all of
        that: a relationship came back as an empty pair with a type between them, a
        path was never recognised, a label with a space in it stayed a raw string, and
        a stored string shaped like a vertex was read back as one -- so stored text
        could stand in for a real element, and which element you got depended on the
        order the columns were returned in.

        A relationship keeps the shape callers read, ``(start, type, end)``. The
        endpoints are filled from the vertices the same row returned, matched on their
        ids rather than on their spelling, so a value that merely looks like a vertex
        cannot be mistaken for one.
        """
        vertices: Dict[Any, Dict[str, Any]] = {}
        for name in record._fields:
            value = getattr(record, name)
            if isinstance(value, Vertex):
                vertices[value.id] = dict(value.properties)

        row: Dict[str, Any] = {}
        for name in record._fields:
            value = getattr(record, name)
            if isinstance(value, Edge):
                row[name] = (
                    vertices.get(value.start, {}),
                    value.label,
                    vertices.get(value.end, {}),
                )
            elif isinstance(value, Vertex):
                row[name] = dict(value.properties)
            else:
                row[name] = value
        return row

    @require_psycopg
    def structured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a Cypher statement and return its rows.

        Args:
            query: the statement to run.
            param_map: its parameters. This is the name the ``PropertyGraphStore``
                contract uses, and ``CypherTemplateRetriever`` passes it by keyword.
            params: the same thing under this class's older name, kept working.

        Values are bound rather than written into the text: a string or a mapping is
        wrapped so the server reads it as the jsonb a Cypher comparison expects, and a
        number is passed as itself so it can still be a ``LIMIT``.
        """

        bound = _cypher_params(param_map if param_map is not None else params)
        if self._read_only_depth:
            check_single_statement(str(query))
        return self._run_with_retry(lambda: self._run_once(query, bound))

    @contextmanager
    def read_only(self, *, allow_server_programs: bool = False) -> Iterator[None]:
        """Run the statements in this block in a transaction that cannot write.

        For statements this store did not write -- an LLM's, most often. The
        refusal is the server's, so a write is refused however it is spelled, and
        nothing here has to recognise what writing looks like. That is the whole
        argument for it: it is PostgreSQL underneath, so ``INSERT``, ``TRUNCATE``,
        ``GRANT`` and ``COPY`` are all available and none of them is Cypher, and
        the keyword list this replaces let every one of them through while
        refusing a read whose text merely contained the word DELETE.

        Text arriving this way is also held to one statement. A statement sent
        with no parameters goes over the protocol that runs the whole string and
        reports only the first result, so a read with a write after a semicolon
        runs the write and looks like the read.
        """
        self._read_only_depth += 1
        if self._read_only_depth == 1:
            self._allow_server_programs = allow_server_programs
        try:
            yield
        finally:
            self._read_only_depth -= 1

    def _run_once(
        self, query: str, bound: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """One attempt at a statement, rolling back if the server refuses it."""
        with self._acquire() as conn:
            if self._read_only_depth:
                with conn.read_only_transaction(
                    allow_server_programs=self._allow_server_programs
                ):
                    return self._rows(conn, query, bound)
            return self._rows(conn, query, bound)

    def _rows(
        self,
        conn: "psycopg.Connection",
        query: str,
        bound: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Send one statement and decode what comes back."""
        with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
            try:
                curs.execute(query, bound)
                if not self._read_only_depth:
                    conn.commit()
            except psycopg.Error as e:
                if not self._read_only_depth:
                    try:
                        conn.rollback()
                    except psycopg.Error:
                        # A connection the server has already dropped cannot
                        # be rolled back, and saying so here would replace the
                        # real reason the statement failed.
                        pass
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

            if self.sanitize_query_output:
                result = [value_sanitize(el) for el in result]

            return result

    def _retry_decision(self, cause: BaseException, number: int):
        """Whether to try again after ``cause``, and how long to wait first.

        The driver owns both answers -- which refusals are worth repeating, and the
        backoff between attempts -- so a race it reads as final is put to it a second
        time under the spelling it recognises rather than being decided here.
        """
        decision = self.retry_policy.decide(
            cause, number=number, wrote=True, merging=True
        )
        if not decision.retry and _merge_race(cause):
            decision = self.retry_policy.decide(
                psycopg.errors.UniqueViolation(),
                number=number,
                wrote=True,
                merging=True,
            )
        return decision

    def _run_with_retry(self, attempt: Callable[[], Any]) -> Any:
        """Run ``attempt``, repeating it while the driver says the refusal was timing.

        Eight writers merging onto a shared set of keys refused 95 statements in 200
        without this, because every one of those refusals was another writer holding the
        key for the moment it took to commit.
        """
        number = 0
        while True:
            try:
                return attempt()
            except AgensQueryException as failure:
                cause = failure.__cause__
                number += 1
                if cause is None:
                    raise
                decision = self._retry_decision(cause, number)
                if not decision.retry:
                    raise
                logger.debug(
                    "attempt %d: %s", number, decision.reason
                )
                time.sleep(decision.delay)

    async def astructured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Async counterpart of :meth:`structured_query` (true async I/O)."""
        bound = _cypher_params(param_map if param_map is not None else params)
        if self._read_only_depth:
            check_single_statement(str(query))

        number = 0
        while True:
            try:
                return await self._arun_once(query, bound)
            except AgensQueryException as failure:
                cause = failure.__cause__
                number += 1
                if cause is None:
                    raise
                decision = self._retry_decision(cause, number)
                if not decision.retry:
                    raise
                logger.debug("attempt %d: %s", number, decision.reason)
                await asyncio.sleep(decision.delay)

    async def _arun_once(
        self, query: str, bound: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`_run_once`."""
        async with self._aacquire() as conn:
            if self._read_only_depth:
                async with conn.read_only_transaction(
                    allow_server_programs=self._allow_server_programs
                ):
                    return await self._arows(conn, query, bound)
            return await self._arows(conn, query, bound)

    async def _arows(
        self,
        conn: "psycopg.AsyncConnection",
        query: str,
        bound: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`_rows`."""
        async with conn.cursor(row_factory=psycopg.rows.namedtuple_row) as curs:
            try:
                await curs.execute(query, bound)
                if not self._read_only_depth:
                    await conn.commit()
            except psycopg.Error as e:
                if not self._read_only_depth:
                    try:
                        await conn.rollback()
                    except psycopg.Error:
                        pass
                raise query_failed(query, e) from e
            try:
                data = await curs.fetchall()
            except psycopg.ProgrammingError:
                data = []

            if data is None:
                result = []
            else:
                result = [self._record_to_dict(d) for d in data]

            if self.sanitize_query_output:
                result = [value_sanitize(el) for el in result]

            return result

    def _describe(self) -> Any:
        """What the graph holds, read from the catalogs rather than from the graph.

        This replaces three statements that walked every element's properties and
        every edge, and a plpgsql function the package used to install in the
        caller's database to name a JSON type -- ``jsonb_typeof`` is built in, and
        the driver does the one thing it does not, telling a whole number from a
        fractional one.
        """
        with self._acquire() as conn:
            try:
                # The triple catalog is the server's own and only a gather fills it,
                # so a graph nobody has gathered has no relationships to report.
                # Gathering is a write; doing it when the catalog already describes
                # the graph would be a write for nothing.
                stale = not conn.meta_is_current(graph=self.graph_name)
                try:
                    return conn.describe(
                        graph=self.graph_name, sample=self.schema_sample, refresh=stale
                    )
                except psycopg.Error as e:
                    if not stale:
                        raise query_failed("describing the graph", e) from e
                    # A reader who may not gather still deserves the labels and
                    # their properties; only the relationships are lost.
                    logger.debug(
                        "could not gather the triple catalog: %s", safe_message(e)
                    )
                    conn.rollback()
                    return conn.describe(
                        graph=self.graph_name, sample=self.schema_sample
                    )
            finally:
                # Reads, but on a connection that is not in autocommit they leave a
                # transaction open holding a share lock on the label tables -- and
                # the next store's CREATE CONSTRAINT wants the table to itself, so
                # it waits for as long as this store lives.
                try:
                    conn.commit()
                except psycopg.Error:
                    pass

    # The schema dict names a JSON type the way a Cypher writer says it, and the
    # driver names it the way JSON does. Everything else is the same word.
    _SCHEMA_TYPES = {"array": "LIST", "object": "MAP"}

    async def _adescribe(self) -> Any:
        """Async sibling of :meth:`_describe`."""
        async with self._aacquire() as conn:
            try:
                stale = not await conn.meta_is_current(graph=self.graph_name)
                try:
                    return await conn.describe(
                        graph=self.graph_name, sample=self.schema_sample, refresh=stale
                    )
                except psycopg.Error as e:
                    if not stale:
                        raise query_failed("describing the graph", e) from e
                    logger.debug(
                        "could not gather the triple catalog: %s", safe_message(e)
                    )
                    await conn.rollback()
                    return await conn.describe(
                        graph=self.graph_name, sample=self.schema_sample
                    )
            finally:
                try:
                    await conn.commit()
                except psycopg.Error:
                    pass

    def _properties_of(self, description: Any, kind: str) -> Dict[str, Any]:
        """The properties of every label of ``kind``, as the schema dict spells them.

        The base label is left out: it inherits every element, so reporting it would
        repeat each child's properties under a name nothing is written on.
        """
        kinds = {label.name: label.kind for label in description.labels}
        skip = {"ag_vertex", "ag_edge", BASE_NODE_LABEL}
        return {
            name: [
                {
                    "property": shape.name,
                    "type": self._SCHEMA_TYPES.get(shape.kind, shape.kind.upper()),
                }
                for shape in shapes
            ]
            for name, shapes in description.properties.items()
            if name not in skip and kinds.get(name) == kind
        }

    def _get_triples(self) -> List[Dict[str, str]]:
        """The distinct relationships in the graph, for an LLM's context.

        Returns:
            List[Dict[str, str]]: relationships as a list of dicts in the format
                "{'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}"
        """
        if not self._schema_refreshed:
            self.refresh_schema()
        return self.structured_schema.get("relationships", [])

    def _get_triples_str(self) -> List[str]:
        """
        Get a set of distinct relationship types (as a list of strings) in the graph
        to be used as context by an llm.

        Returns:
            List[str]: relationships as a list of strings in the format
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """

        triples = self._get_triples()
        return format_triples(triples)

    # Property type groups for enhanced-schema statistics.
    _NUMERIC_TYPES = {"INTEGER", "FLOAT", "NUMBER"}

    def _label_count(self, label: str) -> int:
        """How many elements a label holds, as the catalog already recorded it.

        Counting these by statement was one round trip and one scan per label.
        """
        return int(self._label_counts.get(label, 0))

    def _enhance_schema(self) -> None:
        """
        Enrich each node-label property in ``structured_schema`` with concrete
        statistics for text-to-Cypher prompting:

        - numeric properties get ``min`` / ``max`` / ``distinct_count``,
        - list properties get ``min_size`` / ``max_size``,
        - everything else (strings, etc.) gets example ``values`` +
          ``distinct_count``.

        Stats are computed exhaustively when the label has at most
        ``EXHAUSTIVE_SEARCH_LIMIT`` nodes, otherwise over a bounded sample of
        that many nodes. The embedding/labels properties are always skipped, so
        this never materializes vectors or grows unbounded with the graph.
        """
        node_props = self.structured_schema.get("node_props", {})
        for label, props in node_props.items():
            if label == BASE_ENTITY_LABEL:
                continue
            try:
                count = self._label_count(label)
            except Exception as exc:  # pragma: no cover - best-effort enrichment
                logger.warning("Enhanced schema count failed for %s: %s", label, exc)
                continue
            if count == 0:
                continue
            # None => exhaustive (no LIMIT); else cap the scan to a sample.
            sample = None if count <= EXHAUSTIVE_SEARCH_LIMIT else EXHAUSTIVE_SEARCH_LIMIT

            for prop in props:
                name = prop["property"]
                if name in ("embedding", "labels"):
                    continue
                try:
                    if prop.get("type") in self._NUMERIC_TYPES:
                        prop.update(self._numeric_stats(label, name, sample))
                    elif prop.get("type") == "LIST":
                        prop.update(self._list_stats(label, name, sample))
                    else:
                        prop.update(self._value_stats(label, name, sample))
                except Exception as exc:  # pragma: no cover - best-effort enrichment
                    logger.warning(
                        "Enhanced schema sampling failed for %s.%s: %s",
                        label,
                        name,
                        exc,
                    )

    def _stat_subquery(
        self, prop: str, sample: Optional[int]
    ) -> Tuple[sql.Composed, sql.Composed]:
        """Build the shared Cypher subquery returning a property's values and a
        LIMIT clause (empty when exhaustive)."""
        limit = (
            sql.SQL("LIMIT {n}").format(n=sql.SQL(str(int(sample))))
            if sample is not None
            else sql.SQL("")
        )
        subquery = sql.SQL(
            "MATCH (a:{base_label}) WHERE label(a) = %(label)s AND a.{prop} IS NOT NULL "
            "RETURN a.{prop} AS v {limit}"
        ).format(
            base_label=sql.Identifier(BASE_NODE_LABEL),
            prop=sql.Identifier(prop),
            limit=limit,
        )
        return subquery, limit

    def _numeric_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT min((t.v #>> '{{}}')::numeric) AS min, "
                "max((t.v #>> '{{}}')::numeric) AS max, "
                "count(DISTINCT t.v) AS distinct_count FROM ({sub})t"
            ).format(sub=subquery),
            {"label": Jsonb(label)},
        )
        if not rows or rows[0]["min"] is None:
            return {}
        row = rows[0]
        return {
            "min": float(row["min"]),
            "max": float(row["max"]),
            "distinct_count": row["distinct_count"],
        }

    def _list_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT min(jsonb_array_length(t.v)) AS min_size, "
                "max(jsonb_array_length(t.v)) AS max_size FROM ({sub})t"
            ).format(sub=subquery),
            {"label": Jsonb(label)},
        )
        if not rows or rows[0]["min_size"] is None:
            return {}
        return {"min_size": rows[0]["min_size"], "max_size": rows[0]["max_size"]}

    def _value_stats(
        self, label: str, prop: str, sample: Optional[int]
    ) -> Dict[str, Any]:
        subquery, _ = self._stat_subquery(prop, sample)
        rows = self.structured_query(
            sql.SQL(
                "SELECT (array_agg(DISTINCT t.v))[1:{max_examples}] AS examples, "
                "count(DISTINCT t.v) AS distinct_count FROM ({sub})t"
            ).format(
                max_examples=sql.SQL(str(ENHANCED_MAX_EXAMPLES)),
                sub=subquery,
            ),
            {"label": Jsonb(label)},
        )
        if not rows or not rows[0].get("examples"):
            return {}
        return {
            "values": rows[0]["examples"],
            "distinct_count": rows[0]["distinct_count"],
        }
