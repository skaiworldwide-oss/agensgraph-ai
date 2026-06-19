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
Shared async connection pool for AgensGraph.

A Cognee run can configure both an AgensGraph graph adapter and an AgensGraph
vector adapter against the same database. Rather than each opening its own
``AsyncConnectionPool``, they share one ``AgensEngine`` per process — keyed by
connection string in a refcounted registry: the pool opens once on first use
and closes only when the last adapter releases it. ``graph_path`` is reapplied
on every checkout because pooled connections are reused; relational/vector
queries check out with ``graph_path=None``.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool, PoolTimeout

_ENGINES: Dict[str, "AgensEngine"] = {}
_ENGINES_LOCK = asyncio.Lock()


class AgensEngine:
    """A shareable async connection pool for AgensGraph."""

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 4,
        max_size: int = 16,
        **pool_kwargs: Any,
    ) -> None:
        self.conninfo = conninfo
        self._min_size = min_size
        self._max_size = max_size
        self._pool_kwargs = pool_kwargs
        self._pool: Optional[AsyncConnectionPool] = None
        self._opened = False
        self._refcount = 0
        self._lock = asyncio.Lock()
        self._bootstrapped_graphs: set[str] = set()
        self._bootstrapped_relational: set[str] = set()

    # ---- lifecycle ----

    @classmethod
    async def acquire(cls, conninfo: str, **kwargs: Any) -> "AgensEngine":
        """Get-or-create the shared engine for ``conninfo`` and ref it."""
        async with _ENGINES_LOCK:
            engine = _ENGINES.get(conninfo)
            if engine is None:
                engine = cls(conninfo, **kwargs)
                _ENGINES[conninfo] = engine
            engine._refcount += 1
        await engine._open()
        return engine

    async def _open(self) -> None:
        async with self._lock:
            if not self._opened:
                self._pool = AsyncConnectionPool(
                    self.conninfo,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    open=False,
                    **self._pool_kwargs,
                )
                await self._pool.open()
                self._opened = True

    async def release(self) -> None:
        """Drop a reference; close the pool when the last holder releases."""
        async with _ENGINES_LOCK:
            self._refcount -= 1
            if self._refcount <= 0:
                self._refcount = 0
                pool, self._pool, self._opened = self._pool, None, False
                self._bootstrapped_graphs.clear()
                self._bootstrapped_relational.clear()
                _ENGINES.pop(self.conninfo, None)
            else:
                pool = None
        if pool is not None:
            await pool.close()

    # ---- connections ----

    @asynccontextmanager
    async def _checkout(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """Check out a pooled connection (workaround for a psycopg_pool bug)."""
        try:
            conn = await self._pool.getconn()
        except PoolTimeout:
            await self._pool._add_connection(None)  # pragma: no cover - pool workaround
            conn = await self._pool.getconn()
        try:
            async with conn:
                yield conn
        finally:
            await self._pool.putconn(conn)

    @asynccontextmanager
    async def aconnection(
        self, *, graph_path: Optional[str] = None
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        async with self._checkout() as conn:
            if graph_path is not None:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("SET graph_path = {}").format(
                            sql.Identifier(graph_path)
                        )
                    )
            yield conn

    # ---- one-time bootstrap ----

    async def ensure_graph(self, graph_name: str, ddl) -> None:
        """Run graph-creation DDL once per (engine, graph_name).

        ``ddl`` is an async callable receiving a cursor with ``graph_path`` set.
        """
        async with self._lock:
            if graph_name in self._bootstrapped_graphs:
                return
        async with self.aconnection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("CREATE GRAPH IF NOT EXISTS {}").format(
                        sql.Identifier(graph_name)
                    )
                )
                await cur.execute(
                    sql.SQL("SET graph_path = {}").format(sql.Identifier(graph_name))
                )
                await ddl(cur)
            await conn.commit()
        async with self._lock:
            self._bootstrapped_graphs.add(graph_name)

    async def ensure_relational(self, tag: str, ddl) -> None:
        """Run a named relational (table/index) bootstrap once per engine."""
        async with self._lock:
            if tag in self._bootstrapped_relational:
                return
        async with self.aconnection() as conn:
            async with conn.cursor() as cur:
                await ddl(cur)
            await conn.commit()
        async with self._lock:
            self._bootstrapped_relational.add(tag)


__all__ = ["AgensEngine"]
