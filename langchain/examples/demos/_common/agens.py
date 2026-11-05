"""One ``AgensEngine`` for the whole process, and the components the demos build on it.

The engine connects as the role that owns the demos' graphs (``config.url()``) and runs
model-written Cypher as the role ``AGENSGRAPH_UNTRUSTED_URL`` names. :func:`grant` gives the roles
of ``langchain_agensgraph.roles`` their privileges on a graph, which the untrusted role needs
before it can read it.
"""

from __future__ import annotations

from typing import Any, Optional

import psycopg
from langchain_core.embeddings import Embeddings

from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector
from langchain_agensgraph.roles import render_grants

from . import config

_engine: Optional[AgensEngine] = None


def get_engine(*, min_size: int = 1, max_size: int = 10) -> AgensEngine:
    """The process's engine, made on first use."""
    global _engine
    if _engine is None or _engine.closed:
        _engine = AgensEngine(
            config.url(),
            untrusted_conninfo=config.untrusted_url(),
            min_size=min_size,
            max_size=max_size,
        )
    return _engine


def make_graph(
    graph_name: str,
    *,
    create: bool = False,
    enhanced_schema: bool = False,
    timeout: Optional[float] = None,
) -> AgensGraph:
    """A graph on the shared engine; ``create`` makes it if it does not exist."""
    return AgensGraph(
        get_engine(),
        graph_name=graph_name,
        create_graph=create,
        enhanced_schema=enhanced_schema,
        timeout=timeout,
    )


def make_vector(embedding: Embeddings, *, graph_name: str, **kwargs: Any) -> AgensgraphVector:
    """A vector store on the shared engine."""
    return AgensgraphVector(embedding, engine=get_engine(), graph_name=graph_name, **kwargs)


def grant(graph_name: str) -> None:
    """Run ``render_grants`` for a graph as its owner, so the untrusted role can read it.

    The graph has to exist. Labels the owner makes later are covered by the default privileges
    the grants set.
    """
    with psycopg.connect(config.url(), autocommit=True) as conn:
        conn.execute(render_grants(graph_name))


def close() -> None:
    """Close the shared engine's blocking pools (call from a finally block)."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None


async def aclose() -> None:
    """Close the shared engine, the awaiting pools of the running loop included."""
    global _engine
    if _engine is not None:
        await _engine.aclose()
        _engine = None
