"""Connections shared between the stores.

``AgensEngine`` holds one pool of connections that an ``AgensPropertyGraphStore`` and
any number of ``AgensgraphVectorStore`` instances borrow from, so concurrent callers
each get their own connection rather than serialising on one.

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

Without an engine both stores open a connection of their own.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import agensgraph
from psycopg.conninfo import make_conninfo


def _conf_to_conninfo(conf: Dict[str, Any]) -> str:
    """Build a libpq conninfo string from a psycopg-style conf dict."""
    return make_conninfo(**conf)


class AgensEngine:
    """A shareable pool of connections to one graph.

    Construct via :meth:`from_url` or :meth:`from_conf`. The sync pool opens with the
    engine; the async one is built on first use, so an engine can be made outside a
    running loop.
    """

    def __init__(
        self,
        conninfo: str,
        *,
        graph: Optional[str] = None,
        min_size: int = 1,
        max_size: int = 10,
        application_name: str = "llama-index-agensgraph",
        **pool_kwargs: Any,
    ) -> None:
        # Tag pooled connections for pg_stat_activity unless already set.
        if "application_name=" not in conninfo:
            conninfo = make_conninfo(conninfo, application_name=application_name)
        self.conninfo = conninfo
        self._graph = graph
        self._min_size = min_size
        self._max_size = max_size

        # Autocommit, so a pooled connection holds no snapshot and no locks between
        # statements: a read costs the statement rather than the statement plus the
        # commit closing the transaction it opened. Anything wanting a transaction opens
        # one explicitly.
        pool_kwargs.setdefault("kwargs", {}).setdefault("autocommit", True)
        # The pool's own liveness check is a round trip on every borrow, and a borrower
        # that finds a dead connection is told so by the statement it was going to send
        # anyway.
        pool_kwargs.setdefault("check_connections", False)
        self._pool_kwargs = pool_kwargs
        self._vectors: Optional[bool] = None

        self._pool = agensgraph.ConnectionPool(
            conninfo,
            graph=graph,
            min_size=min_size,
            max_size=max_size,
            configure=self._configure,
            **pool_kwargs,
        )
        self._pool.open()
        self._apool: Optional[agensgraph.AsyncConnectionPool] = None
        # The loop an async pool's workers belong to, so one left over from a loop that
        # has since closed is not handed out.
        self._apool_loop: Optional[asyncio.AbstractEventLoop] = None
        # Two coroutines reaching the first await would each build a pool and the second
        # assignment would drop the first still holding its connections. Bound to a loop
        # on first use rather than here.
        self._apool_lock = asyncio.Lock()

    # ---- constructors ----

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "AgensEngine":
        return cls(url, **kwargs)

    @classmethod
    def from_conf(cls, conf: Dict[str, Any], **kwargs: Any) -> "AgensEngine":
        return cls(_conf_to_conninfo(conf), **kwargs)

    # ---- connection preparation ----

    def _configure(self, conn: agensgraph.Connection) -> None:
        """Prepare a connection the pool has just made, once rather than per borrow.

        The graph is selected by the pool itself when the connection is made, which is
        the round trip per borrow this used to spend on ``SET graph_path`` and the
        commit that followed it.

        Registering the vector types lets an embedding travel as itself rather than as
        its decimal spelling for the server to parse back.
        """
        if self._vectors is None:
            self._vectors = conn.has_vectors()
        if self._vectors:
            conn.register_vectors()

    async def _aconfigure(self, conn: agensgraph.AsyncConnection) -> None:
        """Async sibling of :meth:`_configure`."""
        if self._vectors is None:
            self._vectors = await conn.has_vectors()
        if self._vectors:
            await conn.register_vectors()

    # ---- sync ----

    @contextmanager
    def connection(
        self, graph_path: Optional[str] = None
    ) -> Iterator[agensgraph.Connection]:
        """Borrow a pooled connection.

        ``graph_path`` is accepted for callers that still name it, and selected only
        when it differs from the graph this pool was built for -- the pool has already
        put every connection on that graph.
        """
        with self._pool.connection() as conn:
            if graph_path is not None and graph_path != self._graph:
                conn.graph(graph_path)
            yield conn

    def open_connection(
        self, graph_path: Optional[str] = None
    ) -> agensgraph.Connection:
        """Open a connection of its own, outside the pool.

        Used for the one-off work a store does at construction, which wants a
        connection it can keep rather than one it has to give back.
        """
        conn = agensgraph.Connection.connect(self.conninfo, autocommit=True)
        if self._vectors is None:
            self._vectors = conn.has_vectors()
        if self._vectors:
            conn.register_vectors()
        if graph_path is not None:
            conn.graph(graph_path)
        return conn

    def close(self) -> None:
        self._pool.close()

    # ---- async ----

    async def _aget_pool(self) -> agensgraph.AsyncConnectionPool:
        running = asyncio.get_running_loop()
        if self._apool is not None and self._apool_loop is running:
            return self._apool
        async with self._apool_lock:
            if self._apool is not None and self._apool_loop is running:
                return self._apool
            self._apool = agensgraph.AsyncConnectionPool(
                self.conninfo,
                graph=self._graph,
                min_size=self._min_size,
                max_size=self._max_size,
                configure=self._aconfigure,
                **self._pool_kwargs,
            )
            await self._apool.open()
            self._apool_loop = running
        return self._apool

    @asynccontextmanager
    async def aconnection(
        self, graph_path: Optional[str] = None
    ) -> AsyncIterator[agensgraph.AsyncConnection]:
        pool = await self._aget_pool()
        async with pool.connection() as conn:
            if graph_path is not None and graph_path != self._graph:
                await conn.graph(graph_path)
            yield conn

    async def aclose(self) -> None:
        if self._apool is not None:
            await self._apool.close()
            self._apool = None
            self._apool_loop = None
        self._pool.close()


__all__ = ["AgensEngine"]
