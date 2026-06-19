"""Construction helpers so every demo shares ONE AgensEngine (connection pool).

Centralizing this means demos never duplicate connection wiring and they all
exercise pooled connections against the live database.
"""

from __future__ import annotations

from typing import Optional

from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector

from . import config

_engine: Optional[AgensEngine] = None


def get_engine(*, min_size: int = 1, max_size: int = 10) -> AgensEngine:
    """Return a process-wide shared connection pool."""
    global _engine
    if _engine is None:
        _engine = AgensEngine.from_url(config.url(), min_size=min_size, max_size=max_size)
    return _engine


def make_graph(
    graph_name: str,
    *,
    create: bool = True,
    enhanced_schema: bool = False,
    sanitize: bool = False,
    timeout: Optional[float] = None,
    refresh_schema: bool = True,
) -> AgensGraph:
    return AgensGraph(
        graph_name,
        config.conf(),
        create=create,
        enhanced_schema=enhanced_schema,
        sanitize=sanitize,
        timeout=timeout,
        refresh_schema=refresh_schema,
        engine=get_engine(),
    )


def make_vector(embedding, *, graph_name: str = "vector_store", **kwargs) -> AgensgraphVector:
    """Build an AgensgraphVector bound to the shared engine."""
    return AgensgraphVector(embedding, graph_name=graph_name, engine=get_engine(), **kwargs)


def close() -> None:
    """Close the shared pool (call from a finally block at the end of a demo)."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None
