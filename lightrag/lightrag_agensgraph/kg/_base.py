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

"""Shared base for the AgensGraph LightRAG storage backends."""

import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import agensgraph
from agensgraph import GraphId, RetryPolicy, TokenBucket
from agensgraph.types import Edge, Path, Vertex
from lightrag.namespace import NameSpace
from psycopg.rows import dict_row, tuple_row

from ._engine import AgensEngine, conninfo_from_env, run_with_retry

DEFAULT_GRAPH = NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION


def resolve_workspace(workspace: str, global_config: dict) -> str:
    """The tenant workspace: the environment wins, then what the caller passed, then the run's."""
    return os.environ.get("AGENSGRAPH_WORKSPACE") or workspace or (global_config or {}).get("workspace") or ""


def graph_name_for(base_name: str, workspace: str) -> str:
    """The graph a workspace lives in: a valid identifier, unchanged for no workspace.

    LightRAG names its one graph ``chunk_entity_relation``; a tenant gets a graph of
    its own so two tenants' knowledge graphs never mix.
    """
    if not workspace:
        return base_name
    safe = re.sub(r"[^a-z0-9]+", "_", workspace.lower()).strip("_")
    if not safe or safe[0].isdigit() or safe.startswith("pg_"):
        safe = "w_" + safe
    return f"{safe}_{base_name}"[:63]


def plain(value: Any) -> Any:
    """The driver's graph values as the plain values LightRAG reads.

    A vertex and an edge become their property maps, a graph id its text, a path
    its elements. Everything else is already a plain value.
    """
    if isinstance(value, (Vertex, Edge)):
        return value.properties
    if isinstance(value, GraphId):
        return str(value)
    if isinstance(value, Path):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


class _AgensStorageBase:
    """The shared engine, connection and statement helpers of every store.

    A subclass sets ``self.workspace`` before ``initialize()``; the graph the store
    uses follows from it. Every statement runs on a pooled autocommit connection:
    one round trip, no commit statement.
    """

    _engine: Optional[AgensEngine] = None
    # The driver's default retry allowance is shared by the whole process and sized for a
    # few conflicts a minute. LightRAG merges entities from two dozen coroutines at once,
    # and two writers meeting on one hub is the normal case, so the stores keep an
    # allowance of their own that a burst of conflicts does not spend for everyone.
    _retry: RetryPolicy = RetryPolicy(attempts=6, bucket=TokenBucket(capacity=4000, refill=60.0))

    def _graph_name(self) -> str:
        """The graph this store's workspace lives in.

        LightRAG keeps one graph per run, named after the graph store's namespace
        (``chunk_entity_relation``); ``AGENSGRAPH_GRAPHNAME`` renames it. The key-value,
        vector and status stores compute the same name so that every store of a
        workspace shares one pool.
        """
        base = os.environ.get("AGENSGRAPH_GRAPHNAME") or DEFAULT_GRAPH
        return graph_name_for(base, getattr(self, "workspace", "") or "")

    async def _acquire_engine(self) -> None:
        # A second initialize() is the same initialize: it holds the engine once.
        if self._engine is None:
            self._engine = await AgensEngine.acquire(conninfo_from_env(), graph=self._graph_name())

    async def _release_engine(self) -> None:
        if self._engine is not None:
            await self._engine.release()
            self._engine = None

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[agensgraph.AsyncConnection]:
        async with self._engine.connection() as conn:
            yield conn

    @staticmethod
    async def _rows(conn, query: Any, params: Any, row_factory: Any, by_index: bool) -> List[Any]:
        """Run one statement on ``conn`` and read its rows.

        ``by_index`` is for a statement that matches nodes from a bound list of names.
        The server plans such a statement once, for a list of a hundred names, and on a
        graph of a few thousand nodes that plan hashes the whole node table instead of
        probing the unique index once per name. Turning sequential scans off for the
        statement's own transaction keeps the probes, and the four statements go out in
        one flush.
        """
        if not by_index:
            async with conn.cursor(row_factory=row_factory) as cur:
                await cur.execute(query, params)
                return await cur.fetchall() if cur.description is not None else []
        async with conn.transaction():
            async with conn.pipeline():
                await conn.execute("SET LOCAL enable_seqscan = off")
                cur = conn.cursor(row_factory=row_factory)
                await cur.execute(query, params)
            rows = await cur.fetchall() if cur.description is not None else []
            await cur.close()
            return rows

    async def _fetch(
        self, query: Any, params: Any = None, *, wrote: bool = False, by_index: bool = False
    ) -> List[Dict[str, Any]]:
        """Run one statement and return its rows as dicts of plain values."""

        async def attempt():
            async with self._connection() as conn:
                rows = await self._rows(conn, query, params, dict_row, by_index)
            return [{key: plain(value) for key, value in row.items()} for row in rows]

        return await run_with_retry(self._retry, attempt, wrote=wrote)

    async def _fetch_tuples(self, query: Any, params: Any = None, *, by_index: bool = False) -> List[tuple]:
        """Rows as tuples, with no conversion. For reads of plain SQL values only."""

        async def attempt():
            async with self._connection() as conn:
                return await self._rows(conn, query, params, tuple_row, by_index)

        return await run_with_retry(self._retry, attempt, wrote=False)

    async def _run(self, query: Any, params: Any = None, *, wrote: bool = True) -> int:
        """Run one statement whose rows are not wanted; the number of rows it touched."""

        async def attempt():
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    return cur.rowcount

        return await run_with_retry(self._retry, attempt, wrote=wrote)

    async def _run_many(self, query: Any, seq_params: Any, *, wrote: bool = True) -> None:
        """Run one statement once per row of ``seq_params``."""
        seq = list(seq_params)
        if not seq:
            return

        async def attempt():
            async with self._connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(query, seq)

        await run_with_retry(self._retry, attempt, wrote=wrote)


__all__ = [
    "DEFAULT_GRAPH",
    "graph_name_for",
    "plain",
    "resolve_workspace",
    "_AgensStorageBase",
]
