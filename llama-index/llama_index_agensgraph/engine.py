"""Connection pooling for AgensGraph.

``AgensEngine`` wraps a ``psycopg_pool.ConnectionPool`` (and a lazily-created
``AsyncConnectionPool``) so a single pool can be shared between an
``AgensPropertyGraphStore`` and one or more ``AgensgraphVectorStore`` instances.
This is the production answer to the single-connection bottleneck: concurrent
requests each check out their own connection instead of serializing on one.

Usage::

    from llama_index_agensgraph.engine import AgensEngine
    from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
    from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

    engine = AgensEngine.from_url(
        "postgresql://user:pwd@host:5432/db", min_size=2, max_size=20
    )
    pg = AgensPropertyGraphStore("my_graph", conf={...}, engine=engine)
    vs = AgensgraphVectorStore(url="...", embedding_dimension=1536, engine=engine)
    ...
    engine.close()

When no engine is supplied, both stores behave exactly as before (a single
dedicated ``psycopg`` connection).
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool, ConnectionPool


def _conf_to_conninfo(conf: Dict[str, Any]) -> str:
    """Build a libpq conninfo string from a psycopg-style conf dict."""
    return make_conninfo(**conf)


class AgensEngine:
    """A shareable connection pool for AgensGraph.

    Construct via :meth:`from_url` or :meth:`from_conf`. The sync pool is opened
    eagerly; the async pool is created on first async use (so the engine can be
    built outside an event loop).
    """

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        application_name: str = "llama-index-agensgraph",
        **pool_kwargs: Any,
    ) -> None:
        # Tag pooled connections for pg_stat_activity unless already set.
        if "application_name=" not in conninfo:
            conninfo = make_conninfo(conninfo, application_name=application_name)
        self.conninfo = conninfo
        self._min_size = min_size
        self._max_size = max_size
        self._pool_kwargs = pool_kwargs
        self._pool: ConnectionPool = ConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            open=True,
            **pool_kwargs,
        )
        self._apool: Optional[AsyncConnectionPool] = None

    # ---- constructors ----

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "AgensEngine":
        return cls(url, **kwargs)

    @classmethod
    def from_conf(cls, conf: Dict[str, Any], **kwargs: Any) -> "AgensEngine":
        return cls(_conf_to_conninfo(conf), **kwargs)

    # ---- sync ----

    @contextmanager
    def connection(
        self, graph_path: Optional[str] = None
    ) -> Iterator[psycopg.Connection]:
        """Check out a pooled connection, optionally binding ``graph_path``.

        ``graph_path`` is (re)applied on every checkout because pooled
        connections are reused and the session variable would otherwise carry
        over from a previous borrower.
        """
        with self._pool.connection() as conn:
            if graph_path is not None:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SET graph_path = {n}").format(
                            n=sql.Identifier(graph_path)
                        )
                    )
                conn.commit()
            yield conn

    def open_connection(self, graph_path: Optional[str] = None) -> psycopg.Connection:
        """Open a standalone (non-pooled) connection from this engine's conninfo.

        Used by stores for one-off setup/introspection work that wants a
        dedicated connection rather than borrowing from the pool.
        """
        conn = psycopg.connect(self.conninfo)
        if graph_path is not None:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SET graph_path = {n}").format(
                        n=sql.Identifier(graph_path)
                    )
                )
            conn.commit()
        return conn

    def close(self) -> None:
        self._pool.close()

    # ---- async ----

    async def _aget_pool(self) -> AsyncConnectionPool:
        if self._apool is None:
            self._apool = AsyncConnectionPool(
                self.conninfo,
                min_size=self._min_size,
                max_size=self._max_size,
                open=False,
                **self._pool_kwargs,
            )
            await self._apool.open()
        return self._apool

    @asynccontextmanager
    async def aconnection(
        self, graph_path: Optional[str] = None
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        pool = await self._aget_pool()
        async with pool.connection() as conn:
            if graph_path is not None:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("SET graph_path = {n}").format(
                            n=sql.Identifier(graph_path)
                        )
                    )
                await conn.commit()
            yield conn

    async def aclose(self) -> None:
        if self._apool is not None:
            await self._apool.close()
            self._apool = None
        self._pool.close()


__all__ = ["AgensEngine"]
