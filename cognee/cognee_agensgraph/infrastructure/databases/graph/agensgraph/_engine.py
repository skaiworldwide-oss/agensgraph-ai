"""
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
"""

"""
Shared connection pool for AgensGraph.

The graph adapter and the vector adapter of one cognee run point at the same
database, so they share one ``AgensEngine`` per connection string. The engine
wraps the driver's ``AsyncConnectionPool``: the graph is selected once per
connection, not once per checkout, and connections run in autocommit mode so a
statement costs one round trip.

The pool belongs to the event loop that opened it. cognee keeps one adapter for
the life of the process, and a pool handed to a second ``asyncio.run()`` never
answers, so the engine rebuilds the pool when it sees a different loop.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Set

import agensgraph
from psycopg import errors, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

logger = logging.getLogger(__name__)

_ENGINES: Dict[str, "AgensEngine"] = {}
_ENGINES_LOCK = threading.Lock()

APPLICATION_NAME = "cognee-agensgraph"


class AgensEngine:
    """A connection pool shared by every adapter that uses the same connection string."""

    def __init__(
        self,
        conninfo: str,
        *,
        graph_name: str = "cognee",
        min_size: int = 0,
        max_size: int = 16,
    ) -> None:
        if not conninfo_to_dict(conninfo).get("application_name"):
            conninfo = make_conninfo(conninfo, application_name=APPLICATION_NAME)
        self.conninfo = conninfo
        self.graph_name = graph_name
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[agensgraph.AsyncConnectionPool] = None
        self._pool_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pool_open = False
        # A plain lock. An asyncio.Lock binds to the first loop that waits on it and
        # then fails from every other loop. Nothing held under this lock awaits.
        self._lock = threading.Lock()
        self._graph_ready = False
        self.graph_id: Optional[int] = None
        self.vectors = False
        self._done: Set[str] = set()

    @classmethod
    def get(cls, conninfo: str, **kwargs: Any) -> "AgensEngine":
        """The engine for ``conninfo``, created on first use."""
        with _ENGINES_LOCK:
            engine = _ENGINES.get(conninfo)
            if engine is None:
                engine = cls(conninfo, **kwargs)
                _ENGINES[conninfo] = engine
            return engine

    # ---- the pool ----

    async def _prepare(self) -> None:
        """Create the graph if it is missing and learn what the database has.

        Runs on a connection of its own, before the pool opens: the pool selects the
        graph on every connection it makes, so the graph must exist first.
        """
        if self._graph_ready:
            return
        async with await agensgraph.AsyncConnection.connect(
            self.conninfo, autocommit=True
        ) as conn:
            try:
                await conn.execute(
                    sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
                        sql.Identifier(self.graph_name)
                    )
                )
            except (errors.DuplicateSchema, errors.DuplicateObject):
                pass  # another process created it between the check and the create
            cur = await conn.execute(
                "SELECT oid FROM ag_graph WHERE graphname = %s", (self.graph_name,)
            )
            row = await cur.fetchone()
            self.graph_id = row[0] if row else None
            self.vectors = await conn.has_vectors()
        self._graph_ready = True

    async def _configure(self, conn: agensgraph.AsyncConnection) -> None:
        """Prepare a new pooled connection. Runs once per connection, not per checkout."""
        if self.vectors:
            await conn.register_vectors()

    async def pool(self) -> agensgraph.AsyncConnectionPool:
        """The pool for the running event loop, opened on first use."""
        running = asyncio.get_running_loop()
        with self._lock:
            if self._pool is None or self._pool_loop is not running:
                if self._pool is not None:
                    logger.warning(
                        "the AgensGraph pool belongs to an event loop that has finished; "
                        "its connections cannot be returned from another loop. Call "
                        "finalize() before leaving a loop."
                    )
                self._pool = agensgraph.AsyncConnectionPool(
                    self.conninfo,
                    graph=self.graph_name,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    configure=self._configure,
                    kwargs={"autocommit": True},
                    check_connections=False,
                )
                self._pool_loop = running
                self._pool_open = False
            pool = self._pool
            opened = self._pool_open
        if not opened:
            await self._prepare()
            # Opening an open pool is not free: the driver checks the server with a
            # connection of its own each time, about 3 ms. So the pool is opened once.
            # Two callers arriving together both open it, which the pool itself allows.
            await pool.open()
            with self._lock:
                if self._pool is pool:
                    self._pool_open = True
        return pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[agensgraph.AsyncConnection]:
        """Borrow a pooled connection. The graph is already selected on it."""
        pool = await self.pool()
        async with pool.connection() as conn:
            yield conn

    # ---- one-time work ----

    async def setup_once(
        self, tag: str, work: Callable[[agensgraph.AsyncConnection], Awaitable[None]]
    ) -> None:
        """Run ``work`` on a pooled connection once per engine.

        Two callers arriving together may both run it; the work is DDL written with
        IF NOT EXISTS, so that is harmless.
        """
        if tag in self._done:
            return
        async with self.connection() as conn:
            await work(conn)
        self._done.add(tag)

    async def enable_vectors(self) -> None:
        """Install pgvector if it is missing and make every pooled connection use it."""
        if self.vectors:
            return
        async with await agensgraph.AsyncConnection.connect(
            self.conninfo, autocommit=True
        ) as conn:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except errors.DuplicateObject:
                pass
            self.vectors = await conn.has_vectors()
        with self._lock:
            pool = self._pool if self._pool_loop is asyncio.get_running_loop() else None
        if pool is not None and not pool.closed:
            # Connections register the vector type when they are made; remake them.
            await pool.drain()

    async def aclose(self) -> None:
        """Close the running loop's pool. The next use opens a new one."""
        running = asyncio.get_running_loop()
        with self._lock:
            pool = self._pool if self._pool_loop is running else None
            if pool is not None:
                self._pool = None
                self._pool_loop = None
                self._pool_open = False
        if pool is not None:
            await pool.close()

    def forget_graph(self) -> None:
        """Called after the graph is dropped: the next use creates it again."""
        self._graph_ready = False
        self.graph_id = None
        self._done.clear()


__all__ = ["AgensEngine"]
