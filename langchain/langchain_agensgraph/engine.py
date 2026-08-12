"""Connection pooling for AgensGraph.

``AgensEngine`` wraps the driver's own pool so a single pool can be shared between an
``AgensGraph`` and one or more ``AgensgraphVector`` stores. This is the production
answer to the single-connection bottleneck: concurrent requests each check out their own
connection instead of serializing on one.

Usage::

    from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector

    engine = AgensEngine.from_url(
        "postgresql://user:pwd@host:5432/db", min_size=2, max_size=20
    ) graph = AgensGraph("my_graph", conf={...}, engine=engine, create=True) store =
    AgensgraphVector(embeddings, graph_name="my_graph", engine=engine)
    ...
    engine.close()

When no engine is supplied, ``AgensGraph``/``AgensgraphVector`` behave exactly as before
(a single dedicated connection).

**The graph is bound once per connection, not once per checkout.** The previous
implementation issued ``SET graph_path`` *and a commit* on every borrow, so one logical
query cost three round trips before it ran. The driver's pool distinguishes a hook that
runs when a connection is made from one that runs when it is lent, and selecting a graph
belongs to the first.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import agensgraph
from agensgraph import Deadline
from psycopg.conninfo import make_conninfo


def _conf_to_conninfo(conf: Dict[str, Any]) -> str:
    """Build a libpq conninfo string from a psycopg-style conf dict."""
    return make_conninfo(**conf)


def _budget(deadline: Optional[float]) -> Optional[Deadline]:
    """A caller's time budget, as the pool takes it.

    The pool spends the wait for a connection out of the same budget as the statement
    that will run on it, and sets the statement timeout once per borrow rather than once
    per statement.
    """
    return None if deadline is None else Deadline(deadline)


def _database_has_vectors(conninfo: str) -> bool:
    """Whether this database can read vectors, asked once for the pool.

    Asked here rather than on each connection: it is a property of the database, and the
    answer decides what every connection the pool makes has to be told.
    """
    try:
        with agensgraph.connect(conninfo, autocommit=True) as conn:
            return conn.has_vectors()
    except Exception:  # pragma: no cover - the pool reports a bad conninfo itself
        return False


class AgensEngine:
    """A shareable connection pool for AgensGraph.

    Construct via :meth:`from_url` or :meth:`from_conf`. The sync pool is opened
    eagerly; the async pool is created on first async use (so the engine can be built
    outside an event loop).
    """

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        application_name: str = "langchain-agensgraph",
        graph: Optional[str] = None,
        **pool_kwargs: Any,
    ) -> None:
        # Tag pooled connections for pg_stat_activity unless already set. Read from the
        # parsed connection string rather than by searching its text, so the spaced
        # keyword form is recognised too.
        from psycopg.conninfo import conninfo_to_dict

        if not conninfo_to_dict(conninfo).get("application_name"):
            conninfo = make_conninfo(conninfo, application_name=application_name)
        self.conninfo = conninfo
        self._min_size = min_size
        self._max_size = max_size
        self._graph = graph
        # Autocommit, so a pooled connection holds no snapshot and no locks between
        # statements and a read costs the statement rather than the statement and the
        # commit that closes the transaction it opened. A block that wants a transaction
        # opens one explicitly, which is what `read_only` and `add_graph_documents` do.
        pool_kwargs.setdefault("kwargs", {}).setdefault("autocommit", True)
        self._pool_kwargs = pool_kwargs
        self._vectors = _database_has_vectors(conninfo)
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
        # The loop the async pool's workers belong to, so a pool left over from a loop
        # that has since closed is not handed out.
        self._apool_loop: Optional[asyncio.AbstractEventLoop] = None
        # Two coroutines reaching the first `await` here would each build a pool, and the
        # second assignment would drop the first on the floor still holding its
        # connections. The lock is bound to a loop on first use, not here, so an engine
        # built outside a running loop is fine.
        self._apool_lock = asyncio.Lock()

    def _configure(self, conn: agensgraph.Connection) -> None:
        """Prepare a connection the pool has just made, once rather than per borrow.

        Registering vector types lets an embedding be sent as itself rather than as its
        decimal spelling, which the server would have to parse.
        """
        if self._vectors:
            conn.register_vectors()

    async def _aconfigure(self, conn: agensgraph.AsyncConnection) -> None:
        """Async sibling of :meth:`_configure`."""
        if self._vectors:
            await conn.register_vectors()

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
        self,
        graph_path: Optional[str] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Iterator[agensgraph.Connection]:
        """Check out a pooled connection, on the graph the caller asked for.

        Selecting a graph is skipped when the connection is already on it, and which
        graph a connection is on is something the driver knows without asking the server
        -- it holds the label table for it. So a pool serving one graph pays for the
        selection once per connection, and a pool shared between callers on different
        graphs pays only when the graph actually changes. The previous implementation
        issued ``SET graph_path`` *and a commit* on every single checkout.
        """
        with self._pool.connection(deadline=_budget(deadline)) as conn:
            wanted = graph_path if graph_path is not None else self._graph
            if wanted is not None and conn.label_table.graph != wanted:
                conn.graph(wanted)
            yield conn

    def open_connection(
        self, graph_path: Optional[str] = None
    ) -> agensgraph.Connection:
        """Open a standalone (non-pooled) connection from this engine's conninfo.

        Used by stores for one-off setup work that wants a dedicated connection rather
        than borrowing from the pool.
        """
        conn = agensgraph.connect(self.conninfo, autocommit=True)
        if graph_path is not None:
            conn.graph(graph_path)
        return conn

    def close(self) -> None:
        """Close the sync pool. The async pool, if any, is closed by :meth:`aclose`."""
        self._pool.close()

    # ---- async ----

    async def _aget_pool(self) -> agensgraph.AsyncConnectionPool:
        # A pool's workers are tasks of the loop that opened it. Handed to a second loop
        # -- which is what a script calling `asyncio.run` twice has -- nothing services a
        # request for a connection, and the caller waits out the pool's timeout before
        # being told it could not get one. The loop is recorded so that pool is dropped
        # and rebuilt instead. Its connections go with the loop that owned them; close the
        # engine with `aclose()` inside the loop that used it to return them properly.
        running = asyncio.get_running_loop()
        if self._apool is not None and self._apool_loop is not running:
            self._apool = None
            self._apool_loop = None
            self._apool_lock = asyncio.Lock()
        if self._apool is not None:
            return self._apool
        async with self._apool_lock:
            # Checked again inside the lock: the coroutine that waited here arrives after
            # the one that held it has already published a pool.
            if self._apool is None:
                pool = agensgraph.AsyncConnectionPool(
                    self.conninfo,
                    graph=self._graph,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    configure=self._aconfigure,
                    **self._pool_kwargs,
                )
                await pool.open()
                # Published only once it is open, so another task cannot take a pool that
                # is not yet accepting connections.
                self._apool = pool
                self._apool_loop = running
        return self._apool

    @asynccontextmanager
    async def aconnection(
        self,
        graph_path: Optional[str] = None,
        *,
        deadline: Optional[float] = None,
    ) -> AsyncIterator[agensgraph.AsyncConnection]:
        pool = await self._aget_pool()
        async with pool.connection(deadline=_budget(deadline)) as conn:
            wanted = graph_path if graph_path is not None else self._graph
            if wanted is not None and conn.label_table.graph != wanted:
                await conn.graph(wanted)
            yield conn

    async def aclose(self) -> None:
        """Close both pools.

        The sync one is closed here as well, because an async teardown is a teardown;
        the reverse is not true, and :meth:`close` leaves the async pool alone rather
        than closing it from outside its event loop.
        """
        if self._apool is not None:
            await self._apool.close()
            self._apool = None
        self._pool.close()

    def __enter__(self) -> "AgensEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AgensEngine":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


__all__ = ["AgensEngine"]
