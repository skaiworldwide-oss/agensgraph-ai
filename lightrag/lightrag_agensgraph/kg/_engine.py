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

A single LightRAG run instantiates up to four AgensGraph storage classes
(graph + vector + KV + doc-status), all pointing at the same database. Rather
than each opening its own ``AsyncConnectionPool``, they share one
``AgensEngine`` per process (keyed by connection string) via a refcounted
registry: the pool is opened once on first use and closed only when the last
storage releases it. ``graph_path`` is (re)applied per checkout because pooled
connections are reused.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool, PoolTimeout

# Registry of live engines, keyed by connection string, so all storages in a
# process that share a DB share one pool.
_ENGINES: Dict[str, "AgensEngine"] = {}
_ENGINES_LOCK = asyncio.Lock()


def conninfo_from_env() -> str:
    """Build a libpq connection string from the AGENSGRAPH_* environment."""
    db = os.environ["AGENSGRAPH_DB"]
    user = os.environ["AGENSGRAPH_USER"]
    password = os.environ["AGENSGRAPH_PASSWORD"]
    host = os.environ.get("AGENSGRAPH_HOST", "localhost")
    port = os.environ.get("AGENSGRAPH_PORT", "5432")
    return f"dbname='{db}' user='{user}' password='{password}' host='{host}' port={port}"


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
        # Track which graphs / relational schema we have already bootstrapped so
        # the DDL runs once per engine, not on every query or store init.
        self._bootstrapped_graphs: set[str] = set()
        self._bootstrapped_relational: set[str] = set()

    # ---- lifecycle ----

    @classmethod
    async def acquire(
        cls, conninfo: Optional[str] = None, **kwargs: Any
    ) -> "AgensEngine":
        """Get-or-create the shared engine for ``conninfo`` and ref it."""
        conninfo = conninfo or conninfo_from_env()
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
        """Check out a connection, binding ``graph_path`` when given.

        ``graph_path`` is reapplied on every checkout because pooled connections
        are reused. Relational stores pass ``graph_path=None``: a stale
        ``graph_path`` from a previous Cypher borrower does not affect plain SQL
        table resolution (which uses ``search_path``), so no reset is needed.
        """
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

        ``ddl`` is an async callable receiving an open cursor with
        ``graph_path`` already set to ``graph_name``.
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
        """Run a named relational (table/index) bootstrap once per engine.

        ``tag`` distinguishes the KV / vector / doc-status schemas so each runs
        exactly once even though several storage instances share the engine.
        """
        async with self._lock:
            if tag in self._bootstrapped_relational:
                return
        async with self.aconnection() as conn:
            async with conn.cursor() as cur:
                await ddl(cur)
            await conn.commit()
        async with self._lock:
            self._bootstrapped_relational.add(tag)


__all__ = ["AgensEngine", "conninfo_from_env"]
