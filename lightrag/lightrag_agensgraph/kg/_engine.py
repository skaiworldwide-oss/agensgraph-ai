# Copyright (c) 2025, SKAI Worldwide Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared connection pool for AgensGraph.

One LightRAG run holds up to twelve storage objects (one graph store, three
vector stores, seven key-value stores and the document status) that all point
at the same database and the same graph. They share one ``AgensEngine`` per
(connection string, graph): a driver pool that selects the graph once per
connection, not once per checkout, with connections in autocommit mode so a
statement costs one round trip and no commit statement.

The engine is refcounted so ``finalize_storages()`` closes the pool when the
last store lets go. The pool belongs to the event loop that opened it; a
script that runs a second ``asyncio.run()`` gets a fresh pool.
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, Set, Tuple

import agensgraph
import psycopg
from agensgraph import RetryPolicy
from agensgraph.errors import safe_message
from psycopg import errors, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

logger = logging.getLogger(__name__)

APPLICATION_NAME = "lightrag-agensgraph"


def conninfo_from_env() -> str:
    """The connection string LightRAG's environment describes.

    ``AGENSGRAPH_URI`` is taken whole when set. Otherwise ``AGENSGRAPH_DB``,
    ``AGENSGRAPH_USER`` and ``AGENSGRAPH_PASSWORD`` are required, ``AGENSGRAPH_HOST``
    and ``AGENSGRAPH_PORT`` default to localhost:5432. The values go through the
    driver's conninfo builder, so a password with a quote or a space is fine.
    """
    uri = os.environ.get("AGENSGRAPH_URI")
    if uri:
        return make_conninfo(uri)
    parts = {
        "dbname": os.environ["AGENSGRAPH_DB"],
        "user": os.environ["AGENSGRAPH_USER"],
        "host": os.environ.get("AGENSGRAPH_HOST", "localhost"),
        "port": os.environ.get("AGENSGRAPH_PORT", "5432"),
    }
    password = os.environ["AGENSGRAPH_PASSWORD"]
    if password:
        parts["password"] = password
    return make_conninfo(**parts)


async def run_with_retry(policy: RetryPolicy, attempt: Callable[[], Awaitable[Any]], *, wrote: bool) -> Any:
    """Run ``attempt`` again while the driver says the failure was timing.

    Up to two dozen coroutines merge entities and edges at once during an insert,
    serialised only by LightRAG's per-name locks, so two of them can meet on the
    same key. A merge that loses such a race under a uniqueness constraint is
    reported as an exclusion violation (23P01), so it is judged as the unique
    violation it is.
    """
    number = 0
    while True:
        try:
            result = await attempt()
        except psycopg.Error as exc:
            number += 1
            decision = policy.decide(exc, number=number, wrote=wrote, merging=wrote)
            if not decision.retry and isinstance(exc, errors.ExclusionViolation):
                decision = policy.decide(
                    errors.UniqueViolation(), number=number, wrote=wrote, merging=True
                )
            if not decision.retry:
                logger.error("AgensGraph statement failed: %s", safe_message(exc))
                raise
            await asyncio.sleep(decision.delay)
        else:
            policy.succeeded()
            return result


_ENGINES: Dict[Tuple[str, str], "AgensEngine"] = {}
_ENGINES_LOCK = threading.Lock()


class AgensEngine:
    """A connection pool shared by every store on the same database and graph."""

    def __init__(
        self,
        conninfo: str,
        graph_name: str,
        *,
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
        self._refcount = 0
        self._graph_ready = False
        self.vectors = False
        self._done: Set[str] = set()

    @classmethod
    async def acquire(cls, conninfo: Optional[str] = None, *, graph: str, **kwargs: Any) -> "AgensEngine":
        """The engine for this connection string and graph, created on first use, with one more holder."""
        conninfo = conninfo or conninfo_from_env()
        with _ENGINES_LOCK:
            engine = _ENGINES.get((conninfo, graph))
            if engine is None:
                engine = cls(conninfo, graph, **kwargs)
                _ENGINES[(conninfo, graph)] = engine
            engine._refcount += 1
        return engine

    async def release(self) -> None:
        """One holder fewer; the pool closes with the last one."""
        with _ENGINES_LOCK:
            self._refcount -= 1
            last = self._refcount <= 0
            if last:
                self._refcount = 0
                _ENGINES.pop((self.conninfo, self.graph_name), None)
        if last:
            await self.aclose()
            self._done.clear()
            self._graph_ready = False

    # ---- the pool ----

    async def _prepare(self) -> None:
        """Create the graph if it is missing and learn whether pgvector is installed.

        Runs on a connection of its own, before the pool opens: the pool selects the
        graph on every connection it makes, so the graph must exist first.
        """
        if self._graph_ready:
            return
        async with await agensgraph.AsyncConnection.connect(self.conninfo, autocommit=True) as conn:
            try:
                await conn.execute(
                    sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(sql.Identifier(self.graph_name))
                )
            except (errors.DuplicateSchema, errors.DuplicateObject):
                pass  # another process created it between the check and the create
            self.vectors = await conn.has_vectors()
        self._graph_ready = True

    async def _configure(self, conn: agensgraph.AsyncConnection) -> None:
        """Prepare a new pooled connection. Runs once per connection, not per checkout."""
        # Every statement here is a fixed shape with bound parameters, and a prepared
        # statement is planned again for each of its first five runs before the server
        # settles on a generic plan. The generic plan is asked for from the first run.
        await conn.execute("SET plan_cache_mode = force_generic_plan")
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
                        "finalize_storages() before leaving a loop."
                    )
                # psycopg prepares a statement after its fifth run on each connection, and
                # a pool rotates connections, so most runs would pay for planning. The
                # stores run a few fixed statement shapes: prepare on first use.
                self._pool = agensgraph.AsyncConnectionPool(
                    self.conninfo,
                    graph=self.graph_name,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    configure=self._configure,
                    kwargs={"autocommit": True, "prepare_threshold": 0},
                    check_connections=False,
                )
                self._pool_loop = running
                self._pool_open = False
            pool = self._pool
            opened = self._pool_open
        if not opened:
            await self._prepare()
            # Opening an open pool is not free: the driver checks the server with a
            # connection of its own each time. So the pool is opened once. Two callers
            # arriving together both open it, which the pool itself allows.
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

        Two callers arriving together may both run it; the work is DDL written to be
        run twice, so that is harmless.
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
        async with await agensgraph.AsyncConnection.connect(self.conninfo, autocommit=True) as conn:
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


__all__ = ["AgensEngine", "APPLICATION_NAME", "conninfo_from_env", "run_with_retry"]
