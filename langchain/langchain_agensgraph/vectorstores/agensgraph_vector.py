"""A LangChain vector store whose chunks are vertices of an AgensGraph graph.

Each chunk is a vertex of the store's label, so the knowledge graph can link to it with ordinary
edges and Cypher can read it. Its embedding lives in a table of the companion schema
``"<graph>_store"``, keyed to the vertex and deleted with it, which is what lets a metadata filter
and an approximate index serve one search, on every release from 2.14.

::

    from langchain_agensgraph import AgensEngine, AgensgraphVector

    engine = AgensEngine("postgresql://app_rw:secret@localhost/db")
    store = AgensgraphVector(embeddings, engine=engine, graph_name="kg", create_graph=True)
    store.add_texts(["first chunk", "second chunk"], metadatas=[{"year": 2024}, {"year": 2025}])
    store.similarity_search("chunk", k=1, filter={"year": {"$gte": 2025}})

Nothing is sent when the store is made. The first call creates whatever is missing -- the label,
its indexes, the vector table -- and later calls cost one catalog read the first time and nothing
after. The approximate index is built once, after the first write, which is much faster than
maintaining it while a large first load arrives; until it exists, a search reads every row.

``*_with_score`` returns the distance, smaller being nearer, except in a hybrid search, where it
returns the fused score, larger being better; ``similarity_search_with_relevance_scores`` turns
either into a relevance between 0 and 1. ``relevance_score_fn`` says what a distance means as a
relevance, in place of the distance strategy's own, for a vector search and for the vector half of
a max-fused hybrid search.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import psycopg
from agensgraph import InTable, Jsonb, StorageState, Vector, VectorStorage
from agensgraph.adapters import dump_jsonb
from agensgraph.settings import check_settings, settings_statement
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from .._core.component import Component
from .._core.engine import AgensEngine, bind
from .._core.names import checked, companion_schema, companion_table, printed
from .._core.program import Call, InTransaction, Pipeline, Program, Statement, mappings, scalar
from .._core.settings import EF_SEARCH_MAX, NESTED_LOOPS
from .._core.setup import Setup, made_graph, present
from ..errors import ConfigurationError, SchemaMismatchError
from ._sql import ID_KEY, Layout, keyword_text
from .config import (
    DistanceStrategy,
    IndexConfig,
    KeywordConfig,
    SearchType,
    distance_of,
    relevance_function,
)
from .filters import names_a_field, translate

logger = logging.getLogger(__name__)

MAX_INDEXED_VECTOR = 2000
MAX_INDEXED_HALFVEC = 4000
DEFAULT_BATCH = 2000
DELETE_BATCH = 1000


@dataclass(frozen=True, slots=True)
class HybridHit:
    """One result of a hybrid search, with what each half of the search made of it.

    ``fused_score`` is what the results are ordered by, larger being better. A chunk found by one
    half only has ``None`` for the other's distance or score and rank.
    """

    document: Document
    fused_score: float
    vector_distance: float | None
    keyword_score: float | None
    vector_rank: int | None
    keyword_rank: int | None


class AgensgraphVector(Component, VectorStore):
    """Chunks of text as vertices of a graph, searched by embedding and, optionally, by keyword.

    ``engine`` is an :class:`AgensEngine` or a conninfo string (sharing one engine per string);
    ``graph`` an :class:`AgensGraph`, whose engine and graph name are used. ``graph_name`` names
    the graph otherwise. ``dims`` is read from an existing store or, failing that, from embedding
    one text. ``filter_keys`` are metadata keys searched often enough to deserve an index of their
    own; every key is searchable through the label's catch-all index.
    """

    _what = "vector store"

    def __init__(
        self,
        embedding: Embeddings,
        *,
        engine: AgensEngine | str | None = None,
        graph: Any = None,
        graph_name: str | None = None,
        label: str = "Chunk",
        text_property: str = "text",
        embedding_property: str = "embedding",
        dims: int | None = None,
        distance_strategy: DistanceStrategy | str = DistanceStrategy.COSINE,
        vector_type: Literal["vector", "halfvec"] | None = None,
        index: IndexConfig | None = IndexConfig(),
        index_build: Literal["auto", "manual"] = "auto",
        search_type: SearchType | str = SearchType.VECTOR,
        keyword: KeywordConfig | None = None,
        fusion: Literal["rrf", "max"] = "rrf",
        fusion_weights: tuple[float, float] = (1.0, 1.0),
        rank_constant: int = 60,
        filter_keys: Sequence[str] = (),
        search_mode: Literal["approximate", "exact"] = "approximate",
        ef_search: int | None = None,
        iterative_scan: Literal["strict_order", "relaxed_order", "off"] = "strict_order",
        max_scan_tuples: int | None = None,
        probes: int | None = None,
        retrieval_query: str | None = None,
        relevance_score_fn: Callable[[float], float] | None = None,
        create_graph: bool = False,
        setup: bool | Literal["verify"] = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self._embedding = embedding
        self._engine, self._owns_engine, self.graph_name = bind(engine, graph, graph_name)
        self.label = checked(label, "label")
        self.text_property = checked(text_property, "text property")
        self.embedding_property = checked(embedding_property, "embedding property")
        self.distance_strategy = DistanceStrategy(distance_strategy)
        self._distance = distance_of(self.distance_strategy)
        self.search_type = SearchType(search_type)
        if self.search_type is SearchType.HYBRID and keyword is None:
            keyword = KeywordConfig()
        self.keyword = keyword
        if fusion not in ("rrf", "max"):
            raise ValueError(f"fusion is 'rrf' or 'max', not {fusion!r}")
        self.fusion = fusion
        self.fusion_weights = (float(fusion_weights[0]), float(fusion_weights[1]))
        self.rank_constant = int(rank_constant)
        self.filter_keys = tuple(dict.fromkeys(filter_keys))
        if search_mode not in ("approximate", "exact"):
            raise ValueError(f"search_mode is 'approximate' or 'exact', not {search_mode!r}")
        self.search_mode = search_mode
        self.index = index
        self.index_build = index_build
        self.ef_search = ef_search
        self.iterative_scan = iterative_scan
        self.max_scan_tuples = max_scan_tuples
        self.probes = probes
        self.retrieval_query = retrieval_query
        self.override_relevance_score_fn = relevance_score_fn
        self.create_graph = create_graph
        self._setup_mode = setup
        self.batch_size = int(batch_size)
        self._requested_type = vector_type
        self.dims = dims
        self._layout: Layout | None = None
        self._setup: Setup | None = None
        self._indexed = False
        self._index_pending = False
        self._index_refused = False
        if dims is not None:
            self._configure(dims)

    # -- what the store is ---------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def layout(self) -> Layout:
        """Where the chunks, vectors and text live, once the dimension is known."""
        if self._layout is None:
            raise RuntimeError("the store's dimension is not known until its first call")
        return self._layout

    def _configure(self, dims: int) -> None:
        """Fix the dimension and everything that follows from it."""
        if dims < 1:
            raise ValueError(f"dims must be positive, not {dims}")
        vector_type = self._requested_type
        index = self.index
        if vector_type is None:
            vector_type = "vector" if dims <= MAX_INDEXED_VECTOR or index is None else "halfvec"
            if vector_type == "halfvec":
                logger.info(
                    "%d dimensions is more than an index over vector holds; keeping halfvec", dims
                )
        limit = MAX_INDEXED_VECTOR if vector_type == "vector" else MAX_INDEXED_HALFVEC
        if index is not None and dims > limit:
            warnings.warn(
                f"{dims} dimensions is more than an index over {vector_type} holds ({limit}); "
                "searching exactly instead",
                stacklevel=3,
            )
            index = None
        self.dims = dims
        self.vector_type = vector_type
        self.index = index
        storage = VectorStorage(
            self.graph_name,
            self.label,
            dimensions=dims,
            property=self.embedding_property,
            type=vector_type,
            distance=self._distance,
            index=index.driver() if index is not None else None,
            storage=InTable(schema=companion_schema(self.graph_name), name=self._vector_table),
        )
        self._layout = Layout(
            graph=self.graph_name,
            label=self.label,
            text_property=self.text_property,
            storage=storage,
            keyword=self.keyword,
            filter_keys=self.filter_keys,
        )
        self._setup = Setup(
            self._engine, self.graph_name, self._check, self._create, forgotten=self._forgotten
        )

    @property
    def _vector_table(self) -> str:
        """The vector table's name in the companion schema."""
        return companion_table(self.label, self.embedding_property, "vec")

    def _forgotten(self) -> None:
        self._indexed = False
        self._index_pending = False

    # -- setup ---------------------------------------------------------------------------------

    def setup(self) -> None:
        """Create whatever the store needs and is missing, now rather than on the first call."""
        self._ensure()

    async def asetup(self) -> None:
        await self._aensure()

    def _ensure(self) -> None:
        self._open()
        if self._setup is None:
            self._configure(self._engine.run(self._existing_dims, replay=True) or self._probe())
        assert self._setup is not None
        if self._setup_mode == "verify":
            self._setup.verify(self._missing)
        elif self._setup_mode:
            self._setup.ensure()

    async def _aensure(self) -> None:
        self._open()
        if self._setup is None:
            existing = await self._engine.arun(self._existing_dims, replay=True)
            self._configure(existing or await self._aprobe())
        assert self._setup is not None
        if self._setup_mode == "verify":
            await self._setup.averify(self._missing)
        elif self._setup_mode:
            await self._setup.aensure()

    def _missing(self) -> SchemaMismatchError:
        return SchemaMismatchError(
            f"the store {self.label!r} on graph {self.graph_name!r} is not set up as configured; "
            "make it with setup=True"
        )

    def _probe(self) -> int:
        return len(self._embedding.embed_query("dimension"))

    async def _aprobe(self) -> int:
        return len(await self._embedding.aembed_query("dimension"))

    def _existing_dims(self) -> Program[int | None]:
        """The dimension of a store already made, read from its vector table's column."""
        name = printed(companion_schema(self.graph_name), self._vector_table)
        found = yield Statement(Layout.VECTOR_DIMENSIONS, (name, self.embedding_property))
        declared = scalar(found)
        if not declared or "(" not in declared:
            return None
        if self._requested_type is None:
            self._requested_type = declared.split("(", 1)[0]
        return int(declared.split("(", 1)[1].rstrip(")"))

    def _check(self) -> Program[bool]:
        layout = self.layout
        if not (yield from present(layout.required())):
            return False
        state = yield Call("vector_storage_state", (layout.storage,))
        if state in (StorageState.STALE, StorageState.MISMATCHED):
            return False
        self._indexed = state is StorageState.READY
        return True

    def _create(self) -> Program[None]:
        layout = self.layout
        yield from made_graph(self.graph_name, create=self.create_graph)
        if not (yield Call("has_vectors")):
            raise ConfigurationError(
                "the pgvector extension is not created in this database: run CREATE EXTENSION "
                "vector as its owner. AgensGraph does not bundle pgvector; build it against the "
                "server's own PostgreSQL with `make PG_CONFIG=<agensgraph>/bin/pg_config install`"
            )
        yield Call("graph", (self.graph_name,))
        yield Statement(layout.create_label)
        yield Statement(layout.create_id_index)
        yield Statement(layout.create_catch_all_index)
        for key in self.filter_keys:
            yield Statement(layout.create_filter_index(key))
        yield Statement(layout.create_schema)
        state = yield Call("ensure_vectors", (layout.storage,), {"index": False})
        self._indexed = state is StorageState.READY
        if self.keyword is not None:
            yield Statement(layout.create_text_table)
            yield Statement(layout.create_text_index)

    # -- writing -------------------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: Sequence[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and store texts; the ids they were stored under, in the order given.

        ``ids`` wins over a document's own id, and a text with neither gets a new uuid4. A text
        given twice in one call is stored once, as the last one given. A chunk already stored
        under an id is replaced: its text, metadata and embedding.
        """
        texts = list(texts)
        keys = self._ids(texts, ids)
        prepared = self._metadatas(texts, metadatas)
        size = int(kwargs.get("batch_size", self.batch_size))
        self._ensure()
        for start in range(0, len(texts), size):
            chunk = slice(start, start + size)
            vectors = self._embedding.embed_documents(texts[chunk])
            self._write(texts[chunk], vectors, prepared[chunk], keys[chunk])
        self._after_write()
        return keys

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: Sequence[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        keys = self._ids(texts, ids)
        prepared = self._metadatas(texts, metadatas)
        size = int(kwargs.get("batch_size", self.batch_size))
        await self._aensure()
        for start in range(0, len(texts), size):
            chunk = slice(start, start + size)
            vectors = await self._embedding.aembed_documents(texts[chunk])
            await self._awrite(texts[chunk], vectors, prepared[chunk], keys[chunk])
        await self._aafter_write()
        return keys

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        texts, metadatas, ids = self._from_documents(documents, kwargs.pop("ids", None))
        return self.add_texts(texts, metadatas, ids=ids, **kwargs)

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        texts, metadatas, ids = self._from_documents(documents, kwargs.pop("ids", None))
        return await self.aadd_texts(texts, metadatas, ids=ids, **kwargs)

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store texts with embeddings computed elsewhere."""
        texts = list(texts)
        vectors = list(embeddings)
        if len(vectors) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(vectors)} embeddings")
        keys = self._ids(texts, ids)
        prepared = self._metadatas(texts, metadatas)
        if self._setup is None:
            # A store already made says its own dimension and type; the vectors only a new one's.
            existing = self._engine.run(self._existing_dims, replay=True)
            self._configure(existing or (len(vectors[0]) if vectors else self._probe()))
        self._ensure()
        size = int(kwargs.get("batch_size", self.batch_size))
        for start in range(0, len(texts), size):
            chunk = slice(start, start + size)
            self._write(texts[chunk], vectors[chunk], prepared[chunk], keys[chunk])
        self._after_write()
        return keys

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        vectors = list(embeddings)
        if len(vectors) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(vectors)} embeddings")
        keys = self._ids(texts, ids)
        prepared = self._metadatas(texts, metadatas)
        if self._setup is None:
            existing = await self._engine.arun(self._existing_dims, replay=True)
            self._configure(existing or (len(vectors[0]) if vectors else await self._aprobe()))
        await self._aensure()
        size = int(kwargs.get("batch_size", self.batch_size))
        for start in range(0, len(texts), size):
            chunk = slice(start, start + size)
            await self._awrite(texts[chunk], vectors[chunk], prepared[chunk], keys[chunk])
        await self._aafter_write()
        return keys

    def _write(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
    ) -> None:
        batch = self._batch(texts, vectors, metadatas, keys)
        # Another writer may create one of these keys after this batch looked for it. The batch
        # is an upsert, so running it again updates what that writer made.
        self._engine.run(
            lambda: self._write_batch(batch), graph=self.graph_name, replay=True, merging=True
        )

    async def _awrite(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
    ) -> None:
        batch = self._batch(texts, vectors, metadatas, keys)
        await self._engine.arun(
            lambda: self._write_batch(batch), graph=self.graph_name, replay=True, merging=True
        )

    def _batch(
        self,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        keys: Sequence[str],
    ) -> list[tuple[dict[str, Any], Vector]]:
        """Rows ready to send: one per key, the last given winning, in key order so concurrent
        writers lock vertices in the same order."""
        rows: dict[str, tuple[dict[str, Any], Vector]] = {}
        for text, vector, metadata, key in zip(texts, vectors, metadatas, keys, strict=True):
            if not isinstance(text, str):
                raise TypeError(f"a text must be a string, not {type(text).__name__}")
            properties = {k: v for k, v in metadata.items() if v is not None}
            properties[ID_KEY] = key
            properties[self.text_property] = text
            rows[key] = (properties, self._vector(vector))
        return [rows[key] for key in sorted(rows)]

    def _vector(self, values: Sequence[float]) -> Vector:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != self.dims:
            raise ValueError(f"an embedding of {array.shape} where {self.dims} numbers belong")
        if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float32).max):
            raise ValueError("an embedding holds a value that is not a finite 32-bit number")
        if self._distance.name == "COSINE" and not np.any(array):
            raise ValueError("a zero embedding has no direction, so no cosine distance")
        return Vector(array.astype(np.float32))

    def _write_batch(self, batch: list[tuple[dict[str, Any], Vector]]) -> Program[None]:
        layout = self.layout
        rows = [properties for properties, _ in batch]
        keys = [properties[ID_KEY] for properties in rows]
        # The update and the create read the same rows, encoded once for both.
        encoded = dump_jsonb(rows)
        payload = Jsonb(rows, dumps=lambda _: encoded)
        statements = [
            Statement(layout.update_vertices, (payload,), settings=NESTED_LOOPS),
            Statement(layout.create_vertices, (payload,)),
            Statement(layout.upsert_vectors, (Jsonb(keys), keys, [v for _, v in batch])),
        ]
        if self.keyword is not None:
            fields = self.keyword.properties or (self.text_property,)
            bodies = [keyword_text(properties, fields) for properties in rows]
            statements.append(
                Statement(layout.upsert_text, (Jsonb(keys), self.keyword.regconfig, keys, bodies))
            )
        yield Pipeline(tuple(statements))

    def _ids(self, texts: Sequence[str], ids: Sequence[str | None] | None) -> list[str]:
        if ids is None:
            return [str(uuid.uuid4()) for _ in texts]
        if len(ids) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(ids)} ids")
        return [str(uuid.uuid4()) if key is None else str(key) for key in ids]

    def _metadatas(
        self, texts: Sequence[str], metadatas: Sequence[Mapping[str, Any] | None] | None
    ) -> list[Mapping[str, Any]]:
        if metadatas is None:
            return [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(metadatas)} metadatas")
        reserved = {ID_KEY, self.text_property}
        out = []
        for metadata in metadatas:
            metadata = metadata or {}
            clash = reserved & set(metadata)
            if clash:
                raise ValueError(
                    f"metadata may not hold {sorted(clash)}: the store keeps the chunk's id and "
                    "text under those keys"
                )
            out.append(metadata)
        return out

    @staticmethod
    def _from_documents(
        documents: Sequence[Document], ids: Sequence[str | None] | None
    ) -> tuple[list[str], list[dict[str, Any]], list[str | None]]:
        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata) for document in documents]
        if ids is None:
            ids = [document.id for document in documents]
        return texts, metadatas, list(ids)

    # -- the index -----------------------------------------------------------------------------

    def _after_write(self) -> None:
        if not self._wants_index():
            return
        if self._engine.in_transaction():
            self._index_pending = True
            return
        try:
            built = self._engine.run(self._build, kind="maintenance")
        except psycopg.errors.InsufficientPrivilege:
            self._refused_index()
            return
        self._built(built)

    async def _aafter_write(self) -> None:
        if not self._wants_index():
            return
        if self._engine.in_transaction():
            self._index_pending = True
            return
        try:
            built = await self._engine.arun(self._build, kind="maintenance")
        except psycopg.errors.InsufficientPrivilege:
            self._refused_index()
            return
        self._built(built)

    def _wants_index(self) -> bool:
        return (
            self.index is not None
            and self.index_build == "auto"
            and not self._indexed
            and not self._index_refused
        )

    def _built(self, built: bool) -> None:
        """Remember whether there is an index, once what made it so is committed; one there is
        leaves no build waiting."""

        def remembered() -> None:
            self._indexed = built
            if built:
                self._index_pending = False

        self._engine.remember(remembered)

    def _refused_index(self) -> None:
        self._index_refused = True
        logger.warning(
            "the role cannot build the index on %s, which its owner must do with build_index(); "
            "searches read every row until then",
            self.layout.vector_table,
        )

    def build_index(self, *, concurrently: bool = False) -> bool:
        """Build the approximate index now; whether there is one afterwards.

        ``concurrently`` keeps the table writable while it builds, which takes longer, and cannot
        run inside a transaction. Nothing is built over an empty table.
        """
        self._concurrent_outside_transaction(concurrently)
        self._ensure()
        if self.index is None:
            return False
        built = self._engine.run(lambda: self._build(concurrently=concurrently), kind="maintenance")
        self._built(built)
        return built

    async def abuild_index(self, *, concurrently: bool = False) -> bool:
        self._concurrent_outside_transaction(concurrently)
        await self._aensure()
        if self.index is None:
            return False
        built = await self._engine.arun(
            lambda: self._build(concurrently=concurrently), kind="maintenance"
        )
        self._built(built)
        return built

    def _concurrent_outside_transaction(self, concurrently: bool) -> None:
        if concurrently and self._engine.in_transaction():
            raise ValueError(
                "a concurrent index build cannot run inside a transaction; build it after the "
                "transaction ends, or without concurrently"
            )

    def index_status(self) -> dict[str, Any]:
        """Whether the approximate index exists, and whether a build is waiting on something."""
        self._ensure()
        state = self._engine.run(lambda: _call("vector_storage_state", self.layout.storage))
        self._built(state is StorageState.READY)
        return {
            "indexed": state is StorageState.READY,
            "pending": self._index_pending,
            "refused": self._index_refused,
            "state": state.value,
        }

    def _build(self, *, concurrently: bool = False) -> Program[bool]:
        """Build with the server's parallel maintenance workers, or with none when theirs fails.

        Parallel workers share the graph being built through a segment of ``maintenance_work_mem``
        in ``/dev/shm``, which a small one (Docker's 64 MB) refuses with 53100; a serial build
        keeps the graph in the backend's own memory instead.
        """
        rows = int(scalar((yield Statement(self.layout.count_vectors))) or 0)
        if rows == 0:
            return False
        memory = f"{self._build_memory(rows)}MB"
        try:
            yield from self._build_with(memory, concurrently=concurrently, serial=False)
        except psycopg.errors.DiskFull:
            if concurrently:
                # The failed concurrent build left its index behind, invalid, which the retry's
                # "if not exists" would otherwise take for a finished one.
                failed = yield Statement(
                    self.layout.FAILED_INDEX_BUILDS, (self.layout.vector_table_regclass,)
                )
                for (name,) in failed.records:
                    yield Statement(f"DROP INDEX CONCURRENTLY IF EXISTS {name.replace('%', '%%')}")
            yield from self._build_with(memory, concurrently=concurrently, serial=True)
        return True

    def _build_with(self, memory: str, *, concurrently: bool, serial: bool) -> Program[None]:
        settings: dict[str, object] = {"maintenance_work_mem": memory}
        if serial:
            settings["max_parallel_maintenance_workers"] = 0
        if not concurrently:
            yield InTransaction(_call("build_vector_index", self.layout.storage), settings=settings)
            return
        # A concurrent build runs outside any transaction, so its settings are the session's
        # until put back.
        yield Statement(*settings_statement(check_settings(settings), local=False))
        try:
            yield Call("build_vector_index", (self.layout.storage,), {"concurrently": True})
        finally:
            for name in settings:
                yield Statement(f"RESET {name}")

    def _build_memory(self, rows: int) -> int:
        """Megabytes enough to build an HNSW graph in memory, which is far faster than on disk."""
        width = (4 if self.vector_type == "vector" else 2) * (self.dims or 0)
        links = 2 * (self.index.m if self.index is not None else 16) * 8
        need = rows * (width + links + 64) * 1.5 / 2**20 + 64
        return int(min(max(need, 64), 2048))

    # -- reading by id -------------------------------------------------------------------------

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """The chunks stored under these ids, in the order asked; a missing id is left out."""
        keys = [str(key) for key in ids]
        if not keys:
            return []
        self._ensure()
        found = self._engine.run(lambda: self._get(keys), graph=self.graph_name, replay=True)
        return self._in_order(keys, found)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        keys = [str(key) for key in ids]
        if not keys:
            return []
        await self._aensure()
        found = await self._engine.arun(lambda: self._get(keys), graph=self.graph_name, replay=True)
        return self._in_order(keys, found)

    def _get(self, keys: list[str]) -> Program[list[Document]]:
        result = yield Statement(self.layout.get_by_key, (Jsonb(keys),), settings=NESTED_LOOPS)
        return [self._document(row[0]) for row in result.records]

    @staticmethod
    def _in_order(keys: list[str], found: list[Document]) -> list[Document]:
        by_key = {document.id: document for document in found}
        return [by_key[key] for key in dict.fromkeys(keys) if key in by_key]

    def _document(self, properties: Mapping[str, Any]) -> Document:
        metadata = dict(properties)
        key = metadata.pop(ID_KEY, None)
        text = metadata.pop(self.text_property, "")
        return Document(
            id=None if key is None else str(key),
            page_content=text if isinstance(text, str) else str(text),
            metadata=metadata,
        )

    # -- searching -----------------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        hits = await self.asimilarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in hits]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """The ``k`` chunks nearest the query, with their distance; in a hybrid search, the fused
        score instead, larger being better.

        ``retrieval_query=`` (or the store's own) reads each hit again through Cypher: see
        :meth:`_bridge`.
        """
        return [(doc, score) for doc, score, _ in self._search_hits(query, k, filter, **kwargs)]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        hits = await self._asearch_hits(query, k, filter, **kwargs)
        return [(doc, score) for doc, score, _ in hits]

    def _search_hits(
        self, query: str, k: int, filter: dict[str, Any] | None, **kwargs: Any
    ) -> list[tuple[Document, float, Any]]:
        """Each hit's document, score and vertex id, by the store's search type."""
        _check_k(k)
        retrieval_query = kwargs.pop("retrieval_query", self.retrieval_query)
        if self.search_type is SearchType.HYBRID:
            fused = self._hybrid_hits(query, k, filter, kwargs.get("fetch_k"))
            hits = [(hit.document, hit.fused_score, gid) for hit, gid in fused]
        else:
            vector = self._embedding.embed_query(query)
            hits = self._vector_hits(vector, k, filter)
        if retrieval_query:
            return self._engine.run(
                lambda: self._bridge(hits, retrieval_query), graph=self.graph_name, replay=True
            )
        return hits

    async def _asearch_hits(
        self, query: str, k: int, filter: dict[str, Any] | None, **kwargs: Any
    ) -> list[tuple[Document, float, Any]]:
        _check_k(k)
        retrieval_query = kwargs.pop("retrieval_query", self.retrieval_query)
        if self.search_type is SearchType.HYBRID:
            fused = await self._ahybrid_hits(query, k, filter, kwargs.get("fetch_k"))
            hits = [(hit.document, hit.fused_score, gid) for hit, gid in fused]
        else:
            vector = await self._embedding.aembed_query(query)
            hits = await self._avector_hits(vector, k, filter)
        if retrieval_query:
            return await self._engine.arun(
                lambda: self._bridge(hits, retrieval_query), graph=self.graph_name, replay=True
            )
        return hits

    def _bridge(
        self, hits: list[tuple[Document, float, Any]], retrieval_query: str
    ) -> Program[list[tuple[Document, float, Any]]]:
        """The hits read again by the caller's Cypher: ``node`` and ``score`` in, ``text``,
        ``score``, ``metadata`` and optionally ``doc_id`` out, one document per row returned.

        The query sees the hits best first, but one that aggregates hands its rows back in no
        order, so the rows are put back in the hits' order by their score: a row returning its
        hit's score takes that hit's place. A query that returns a score of its own orders its
        rows itself.
        """
        if not hits:
            return []
        found = [
            {"gid": str(gid), "score": score, "rank": rank}
            for rank, (_, score, gid) in enumerate(hits)
        ]
        result = yield Statement(self.layout.bridge(retrieval_query), (Jsonb(found),))
        out = []
        for values in mappings(result):
            metadata = values.get("metadata") or {}
            doc_id = values.get("doc_id")
            document = Document(
                id=None if doc_id is None else str(doc_id),
                page_content=str(values.get("text") or ""),
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {"value": metadata},
            )
            out.append((document, float(values.get("score") or 0.0), None))
        place: dict[float, int] = {}
        for rank, (_, score, _) in enumerate(hits):
            place.setdefault(float(score), rank)
        return sorted(out, key=lambda row: place.get(row[1], len(hits)))

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        hits = self.similarity_search_with_score_by_vector(embedding, k, filter, **kwargs)
        return [doc for doc, _ in hits]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        hits = await self.asimilarity_search_with_score_by_vector(embedding, k, filter, **kwargs)
        return [doc for doc, _ in hits]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """The ``k`` chunks nearest an embedding, with their distance."""
        _check_k(k)
        if self.search_type is SearchType.HYBRID:
            raise ValueError("a hybrid search needs the query's text, not only its embedding")
        hits = self._vector_hits(embedding, k, filter)
        retrieval_query = kwargs.get("retrieval_query", self.retrieval_query)
        if retrieval_query:
            hits = self._engine.run(
                lambda: self._bridge(hits, retrieval_query), graph=self.graph_name, replay=True
            )
        return [(doc, score) for doc, score, _ in hits]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        _check_k(k)
        if self.search_type is SearchType.HYBRID:
            raise ValueError("a hybrid search needs the query's text, not only its embedding")
        hits = await self._avector_hits(embedding, k, filter)
        retrieval_query = kwargs.get("retrieval_query", self.retrieval_query)
        if retrieval_query:
            hits = await self._engine.arun(
                lambda: self._bridge(hits, retrieval_query), graph=self.graph_name, replay=True
            )
        return [(doc, score) for doc, score, _ in hits]

    def _vector_hits(
        self, vector: Sequence[float], k: int, where: Mapping[str, Any] | None
    ) -> list[tuple[Document, float, Any]]:
        self._ensure()
        rows = self._engine.run(
            lambda: self._nearest(vector, k, where), graph=self.graph_name, replay=True
        )
        return [(self._document(row[0]), float(row[1]), row[-1]) for row in rows]

    async def _avector_hits(
        self, vector: Sequence[float], k: int, where: Mapping[str, Any] | None
    ) -> list[tuple[Document, float, Any]]:
        await self._aensure()
        rows = await self._engine.arun(
            lambda: self._nearest(vector, k, where), graph=self.graph_name, replay=True
        )
        return [(self._document(row[0]), float(row[1]), row[-1]) for row in rows]

    def search_with_embeddings(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float, list[float]]]:
        """The nearest chunks with their distance and stored embedding."""
        _check_k(k)
        vector = self._embedding.embed_query(query)
        self._ensure()
        rows = self._engine.run(
            lambda: self._nearest(vector, k, filter, embeddings=True),
            graph=self.graph_name,
            replay=True,
        )
        return [(self._document(r[0]), float(r[1]), _floats(r[2]).tolist()) for r in rows]

    async def asearch_with_embeddings(
        self, query: str, k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float, list[float]]]:
        _check_k(k)
        vector = await self._embedding.aembed_query(query)
        await self._aensure()
        rows = await self._engine.arun(
            lambda: self._nearest(vector, k, filter, embeddings=True),
            graph=self.graph_name,
            replay=True,
        )
        return [(self._document(r[0]), float(r[1]), _floats(r[2]).tolist()) for r in rows]

    def _nearest(
        self,
        vector: Sequence[float],
        k: int,
        where: Mapping[str, Any] | None,
        *,
        embeddings: bool = False,
    ) -> Program[list[tuple[Any, ...]]]:
        predicate, params = translate(where)
        text = self.layout.nearest(predicate, embeddings=embeddings, exact=self._exact)
        result = yield Statement(
            text,
            (self._vector(vector), *params, k),
            settings=self._search_settings(k, filtered=bool(where)),
        )
        return list(result.records)

    @property
    def _exact(self) -> bool:
        return self.search_mode == "exact" or self.index is None

    def _search_settings(self, k: int, *, filtered: bool) -> dict[str, object] | None:
        if self._exact or self.index is None:
            return None
        settings: dict[str, object] = {}
        if self.index.method == "hnsw":
            wanted = max(self.ef_search or 40, k)
            settings["hnsw.ef_search"] = min(wanted, EF_SEARCH_MAX)
            # The index hands back ef_search rows at most, so a filter that drops some, or a k
            # above pgvector's ceiling, takes an iterative scan to read on until k rows are found.
            beyond = wanted > EF_SEARCH_MAX
            if beyond and self.iterative_scan == "off":
                raise ValueError(
                    f"{k} rows is more than one index scan returns ({EF_SEARCH_MAX}); it takes "
                    "an iterative scan, which iterative_scan='off' turns off"
                )
            if (filtered or beyond) and self.iterative_scan != "off":
                settings["hnsw.iterative_scan"] = self.iterative_scan
                if self.max_scan_tuples is not None:
                    settings["hnsw.max_scan_tuples"] = self.max_scan_tuples
        else:
            if self.probes is not None:
                settings["ivfflat.probes"] = self.probes
            if filtered and self.iterative_scan != "off":
                settings["ivfflat.iterative_scan"] = "relaxed_order"
        return settings or None

    # -- maximal marginal relevance ------------------------------------------------------------

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Chunks near the query and unlike each other, from the ``fetch_k`` nearest."""
        vector = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            vector, k, fetch_k, lambda_mult, filter, **kwargs
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        vector = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            vector, k, fetch_k, lambda_mult, filter, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        _check_k(k)
        self._ensure()
        rows = self._engine.run(
            lambda: self._nearest(embedding, max(fetch_k, k), filter, embeddings=True),
            graph=self.graph_name,
            replay=True,
        )
        return self._marginal(embedding, rows, k, lambda_mult)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        _check_k(k)
        await self._aensure()
        rows = await self._engine.arun(
            lambda: self._nearest(embedding, max(fetch_k, k), filter, embeddings=True),
            graph=self.graph_name,
            replay=True,
        )
        return self._marginal(embedding, rows, k, lambda_mult)

    def _marginal(
        self,
        query: Sequence[float],
        rows: list[tuple[Any, ...]],
        k: int,
        lambda_mult: float,
    ) -> list[Document]:
        if not rows:
            return []
        candidates = np.vstack([_floats(row[2]) for row in rows])
        chosen = maximal_marginal_relevance(
            # Annotated as a list of lists; an array is what it computes with either way.
            np.asarray(query, dtype=np.float32),
            cast(Any, candidates),
            lambda_mult=lambda_mult,
            k=k,
        )
        return [self._document(rows[i][0]) for i in chosen]

    # -- hybrid --------------------------------------------------------------------------------

    def hybrid_search_with_details(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        fetch_k: int | None = None,
        **kwargs: Any,
    ) -> list[HybridHit]:
        """A search by embedding and by keyword at once, the two rankings fused.

        Each half reads ``fetch_k`` candidates (by default four times ``k``, at least twenty).
        """
        return [hit for hit, _ in self._hybrid_hits(query, k, filter, fetch_k)]

    async def ahybrid_search_with_details(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        *,
        fetch_k: int | None = None,
        **kwargs: Any,
    ) -> list[HybridHit]:
        return [hit for hit, _ in await self._ahybrid_hits(query, k, filter, fetch_k)]

    def _hybrid_hits(
        self, query: str, k: int, where: dict[str, Any] | None, fetch_k: int | None
    ) -> list[tuple[HybridHit, Any]]:
        _check_k(k)
        self._require_keyword()
        vector = self._embedding.embed_query(query)
        self._ensure()
        candidates = fetch_k or max(4 * k, 20)
        rows = self._engine.run(
            lambda: self._hybrid(vector, query, candidates, where),
            graph=self.graph_name,
            replay=True,
        )
        return self._fuse(rows, k)

    async def _ahybrid_hits(
        self, query: str, k: int, where: dict[str, Any] | None, fetch_k: int | None
    ) -> list[tuple[HybridHit, Any]]:
        _check_k(k)
        self._require_keyword()
        vector = await self._embedding.aembed_query(query)
        await self._aensure()
        candidates = fetch_k or max(4 * k, 20)
        rows = await self._engine.arun(
            lambda: self._hybrid(vector, query, candidates, where),
            graph=self.graph_name,
            replay=True,
        )
        return self._fuse(rows, k)

    def _require_keyword(self) -> None:
        if self.keyword is None:
            raise ValueError("a hybrid search needs keyword=KeywordConfig(...) on the store")

    def _hybrid(
        self,
        vector: Sequence[float],
        query: str,
        candidates: int,
        where: Mapping[str, Any] | None,
    ) -> Program[list[tuple[Any, ...]]]:
        assert self.keyword is not None
        predicate, params = translate(where)
        text = self.layout.hybrid(predicate, exact=self._exact)
        result = yield Statement(
            text,
            (
                self._vector(vector),
                *params,
                candidates,
                self.keyword.regconfig,
                query,
                *params,
                candidates,
            ),
            settings=self._search_settings(candidates, filtered=bool(where)),
        )
        return list(result.records)

    def _fuse(self, rows: list[tuple[Any, ...]], k: int) -> list[tuple[HybridHit, Any]]:
        vector_weight, keyword_weight = self.fusion_weights
        if self.fusion == "rrf":

            def fused(distance: Any, vrank: Any, score: Any, krank: Any) -> float:
                total = 0.0
                if vrank is not None:
                    total += vector_weight / (self.rank_constant + vrank)
                if krank is not None:
                    total += keyword_weight / (self.rank_constant + krank)
                return total

        else:
            relevance = self._relevance()
            top_vector = max((relevance(r[1]) for r in rows if r[1] is not None), default=0.0)
            top_keyword = max((float(r[3]) for r in rows if r[3] is not None), default=0.0)

            def fused(distance: Any, vrank: Any, score: Any, krank: Any) -> float:
                total = 0.0
                if distance is not None and top_vector > 0:
                    total += vector_weight * relevance(distance) / top_vector
                if score is not None and top_keyword > 0:
                    total += keyword_weight * float(score) / top_keyword
                return total

        hits = [
            (
                HybridHit(
                    document=self._document(properties),
                    fused_score=fused(distance, vrank, score, krank),
                    vector_distance=None if distance is None else float(distance),
                    keyword_score=None if score is None else float(score),
                    vector_rank=vrank,
                    keyword_rank=krank,
                ),
                gid,
            )
            for properties, distance, vrank, score, krank, gid in rows
        ]
        hits.sort(key=lambda pair: (-pair[0].fused_score, pair[0].document.id or ""))
        return hits[:k]

    # -- relevance -----------------------------------------------------------------------------

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.search_type is SearchType.HYBRID:
            if self.fusion == "max":
                top = sum(self.fusion_weights)
                return lambda score: score / top if top else score
            best = sum(self.fusion_weights) / (self.rank_constant + 1)
            return lambda score: score / best if best else score
        return self._relevance()

    def _relevance(self) -> Callable[[float], float]:
        """What a distance means as a relevance: ``relevance_score_fn`` when given, for a vector
        search and for the vector half of a max-fused one alike."""
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn
        return relevance_function(self.distance_strategy)

    # -- deleting ------------------------------------------------------------------------------

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Delete chunks by id, by metadata filter, or all of them with ``delete_all=True``.

        Deleting an id that is not stored is not an error. With no ids and no filter, or with a
        filter that tests no field and so matches every chunk, nothing is deleted unless
        ``delete_all`` says that everything is meant.
        """
        if _deletes_everything(ids, filter, delete_all):
            self.clear()
            return True
        self._ensure()
        if filter is not None:
            keys = self._engine.run(lambda: self._keys(filter), graph=self.graph_name)
            ids = list(ids or []) + keys
        keys = [str(key) for key in ids or []]
        for start in range(0, len(keys), DELETE_BATCH):
            part = keys[start : start + DELETE_BATCH]
            self._engine.run(lambda: self._delete(part), graph=self.graph_name, replay=True)
        return True

    async def adelete(
        self,
        ids: list[str] | None = None,
        *,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
        **kwargs: Any,
    ) -> bool:
        if _deletes_everything(ids, filter, delete_all):
            await self.aclear()
            return True
        await self._aensure()
        if filter is not None:
            keys = await self._engine.arun(lambda: self._keys(filter), graph=self.graph_name)
            ids = list(ids or []) + keys
        keys = [str(key) for key in ids or []]
        for start in range(0, len(keys), DELETE_BATCH):
            part = keys[start : start + DELETE_BATCH]
            await self._engine.arun(lambda: self._delete(part), graph=self.graph_name, replay=True)
        return True

    def _keys(self, where: Mapping[str, Any]) -> Program[list[str]]:
        predicate, params = translate(where)
        result = yield Statement(self.layout.keys_where(predicate), params)
        return [str(row[0]) for row in result.records if row[0] is not None]

    def _delete(self, keys: list[str]) -> Program[None]:
        yield Statement(self.layout.delete_by_key, (Jsonb(keys),), settings=NESTED_LOOPS)

    def clear(self) -> None:
        """Delete every chunk, keeping the store's tables and indexes for what comes next."""
        self._ensure()
        self._engine.run(self._clear, graph=self.graph_name, replay=True)

    async def aclear(self) -> None:
        await self._aensure()
        await self._engine.arun(self._clear, graph=self.graph_name, replay=True)

    def _clear(self) -> Program[None]:
        layout = self.layout

        def both() -> Program[None]:
            yield Statement(layout.clear_tables)
            yield Statement(layout.clear_vertices)

        yield InTransaction(both())

    def drop(self) -> None:
        """Delete every chunk and drop the store's vector and text tables.

        The label and its indexes stay, since other things in the graph may use the label.
        """
        self._ensure()
        assert self._setup is not None
        self._engine.run(self._drop, graph=self.graph_name, kind="maintenance")
        self._setup.forget()

    async def adrop(self) -> None:
        await self._aensure()
        assert self._setup is not None
        await self._engine.arun(self._drop, graph=self.graph_name, kind="maintenance")
        self._setup.forget()

    def _drop(self) -> Program[None]:
        layout = self.layout

        def everything() -> Program[None]:
            yield Statement(layout.clear_vertices)
            yield Call("drop_vectors", (layout.storage,))
            if self.keyword is not None:
                yield Statement(f"DROP TABLE IF EXISTS {layout.text_table}")

        yield InTransaction(everything(), lock=self.graph_name)

    # -- making a store ------------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        store = cls(embedding, **kwargs)
        await store.aadd_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: list[tuple[str, list[float]]],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        """A store holding texts with embeddings computed elsewhere."""
        store = cls(embedding, **kwargs)
        texts = [text for text, _ in text_embeddings]
        vectors = [vector for _, vector in text_embeddings]
        store.add_embeddings(texts, vectors, metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: list[tuple[str, list[float]]],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AgensgraphVector:
        store = cls(embedding, **kwargs)
        texts = [text for text, _ in text_embeddings]
        vectors = [vector for _, vector in text_embeddings]
        await store.aadd_embeddings(texts, vectors, metadatas, ids=ids)
        return store

    @classmethod
    def from_existing_index(cls, embedding: Embeddings, **kwargs: Any) -> AgensgraphVector:
        """A store over one already made; refused if what it needs is not there."""
        store = cls(embedding, setup="verify", **kwargs)
        store._ensure()
        return store

    @classmethod
    async def afrom_existing_index(cls, embedding: Embeddings, **kwargs: Any) -> AgensgraphVector:
        store = cls(embedding, setup="verify", **kwargs)
        await store._aensure()
        return store


def _call(method: str, *args: Any) -> Program[Any]:
    return (yield Call(method, args))


def _floats(vector: Any) -> np.ndarray:
    values = getattr(vector, "values", vector)
    return (
        np.frombuffer(values, dtype=np.float32)
        if isinstance(values, bytes | memoryview)
        else (np.asarray(values, dtype=np.float32))
    )


def _check_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, not {k!r}")


def _deletes_everything(
    ids: list[str] | None, filter: Mapping[str, Any] | None, delete_all: bool
) -> bool:
    """Whether a delete is of every chunk: no ids and no filter, or a filter testing no field,
    which matches them all. Refused unless ``delete_all`` says so."""
    if filter is not None:
        everything = not names_a_field(filter)
    else:
        everything = ids is None
    if everything and not delete_all:
        raise ValueError(
            "delete() needs ids, a filter that tests a field, or delete_all=True to delete every "
            "chunk"
        )
    return everything


__all__ = ["AgensgraphVector", "HybridHit"]
