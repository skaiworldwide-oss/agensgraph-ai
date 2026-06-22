"""Construction helpers so every demo shares ONE AgensEngine (connection pool).

Centralizing this means demos never duplicate connection wiring and they all
exercise pooled connections against the live database. The property-graph store
and the vector store both accept ``engine=`` and can safely share one pool: the
engine re-applies ``SET graph_path`` on every checkout, so each store stays bound
to its own graph (different ``graph_name``s never collide).
"""

from __future__ import annotations

from typing import Optional

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

from . import config
from .models import EMBED_DIM

_engine: Optional[AgensEngine] = None


def get_engine(*, min_size: int = 2, max_size: int = 20) -> AgensEngine:
    """Return a process-wide shared connection pool."""
    global _engine
    if _engine is None:
        _engine = AgensEngine.from_url(config.url(), min_size=min_size, max_size=max_size)
    return _engine


def make_pg_store(
    graph_name: str,
    *,
    vector_dimension: Optional[int] = EMBED_DIM,
    enhanced_schema: bool = False,
    create: bool = True,
    create_indexes: bool = True,
    refresh_schema: bool = False,
) -> AgensPropertyGraphStore:
    """Build an ``AgensPropertyGraphStore`` on the shared engine.

    ``vector_dimension`` must be set (1536 for text-embedding-3-small) for the
    HNSW index on entity embeddings to be created — otherwise ``vector_query``
    falls back to a sequential scan.

    ``refresh_schema`` defaults to False: the schema scan is O(N) over the whole
    graph, so we defer it (it is computed lazily when Text2Cypher actually needs
    the schema) instead of paying it on every construction.
    """
    return AgensPropertyGraphStore(
        graph_name,
        conf=config.conf(),
        vector_dimension=vector_dimension,
        enhanced_schema=enhanced_schema,
        create=create,
        create_indexes=create_indexes,
        refresh_schema=refresh_schema,
        engine=get_engine(),
    )


def make_vector_store(
    *,
    graph_name: str = "vector_store",
    node_label: str = "Chunk",
    hybrid_search: bool = False,
    embedding_dimension: int = EMBED_DIM,
    **kwargs,
) -> AgensgraphVectorStore:
    """Build an ``AgensgraphVectorStore`` bound to the shared engine.

    ``hybrid_search=True`` enables RRF keyword+vector fusion but is incompatible
    with metadata filters — use a separate instance for filtered queries.
    """
    return AgensgraphVectorStore(
        url=config.url(),
        embedding_dimension=embedding_dimension,
        graph_name=graph_name,
        node_label=node_label,
        hybrid_search=hybrid_search,
        engine=get_engine(),
        **kwargs,
    )


def close() -> None:
    """Close the shared (sync) pool — call from a finally block at the end of a demo."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


async def aclose() -> None:
    """Close both pools from inside the event loop that used async.

    The async pool's worker tasks are bound to the loop they were created in, so
    a demo that did async work should ``await agens.aclose()`` at the end of that
    same loop — otherwise the workers are orphaned ("Task was destroyed") at exit.
    """
    global _engine
    if _engine is not None:
        await _engine.aclose()
        _engine = None
