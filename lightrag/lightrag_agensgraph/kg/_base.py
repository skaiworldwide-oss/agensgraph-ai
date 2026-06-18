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

"""Shared base for the AgensGraph LightRAG storage backends."""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, NamedTuple, Optional, Pattern

import psycopg
from psycopg.rows import dict_row, namedtuple_row

from ._engine import AgensEngine


class AgensgraphQueryException(Exception):
    """Exception for AgensGraph queries."""

    def __init__(self, exception) -> None:
        if isinstance(exception, dict):
            self.message = exception.get("message", "unknown")
            self.details = exception.get("detail", exception.get("details", "unknown"))
        else:
            self.message = exception
            self.details = "unknown"
        super().__init__(self.message)

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


# AgensGraph returns vertices/edges as agtype *strings* ``label[id]{json}``;
# these regexes pull the JSON properties out. Scalars/maps/lists already arrive
# as native Python types, so they must NOT be re-parsed.
VERTEX_REGEX: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
EDGE_REGEX: Pattern = re.compile(
    r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})"
)


def record_to_dict(record: NamedTuple) -> Dict[str, Any]:
    """Decode an AgensGraph result row into a dict.

    Vertex columns become their property dict; edge columns become a
    ``(start_props, label, end_props)`` tuple; everything else is passed through
    as the native Python value psycopg already produced (no blanket json.loads,
    which would corrupt numeric-looking string ids).
    """
    vertices: Dict[str, Any] = {}
    for k in record._fields:
        v = getattr(record, k)
        if isinstance(v, str):
            vertex = VERTEX_REGEX.match(v)
            if vertex:
                _, vertex_id, properties = vertex.groups()
                vertices[str(vertex_id)] = json.loads(properties)

    d: Dict[str, Any] = {}
    for k in record._fields:
        v = getattr(record, k)
        if isinstance(v, str):
            vertex = VERTEX_REGEX.match(v)
            edge = EDGE_REGEX.match(v)
            if vertex:
                d[k] = json.loads(vertex.group(3))
            elif edge:
                elabel, _, start_id, end_id, _props = edge.groups()
                d[k] = (vertices.get(start_id, {}), elabel, vertices.get(end_id, {}))
            else:
                d[k] = v
        else:
            d[k] = v
    return d


def resolve_workspace(namespace: str, global_config: dict) -> str:
    """Resolve the tenant workspace (env > global_config > '')."""
    return (
        os.environ.get("AGENSGRAPH_WORKSPACE")
        or (global_config or {}).get("workspace")
        or ""
    )


class _AgensStorageBase:
    """Mixin providing the shared engine, connection, and query helpers.

    Subclasses set ``self._graph_path`` (the graph name for the graph store, or
    ``None`` for the relational stores) and ``self.workspace``.
    """

    _graph_path: Optional[str] = None
    _engine: Optional[AgensEngine] = None

    async def _acquire_engine(self) -> None:
        self._engine = await AgensEngine.acquire()

    async def _release_engine(self) -> None:
        if self._engine is not None:
            await self._engine.release()
            self._engine = None

    @asynccontextmanager
    async def _connection(
        self, *, graph_path: Optional[str] = ...
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        gp = self._graph_path if graph_path is ... else graph_path
        async with self._engine.aconnection(graph_path=gp) as conn:
            yield conn

    async def _query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query (graph_path bound) and decode vertex/edge rows."""
        async with self._connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as cur:
                try:
                    await cur.execute(query, params or {})
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        {"message": f"Error executing graph query: {query}", "detail": str(e)}
                    ) from e
                try:
                    data = await cur.fetchall()
                except psycopg.ProgrammingError:
                    data = []
                return [record_to_dict(d) for d in (data or [])]

    async def _execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        fetch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a plain SQL statement (no graph_path, no decode); dict rows."""
        async with self._connection(graph_path=None) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                try:
                    await cur.execute(query, params or {})
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        {"message": f"Error executing SQL: {query}", "detail": str(e)}
                    ) from e
                if not fetch:
                    return []
                try:
                    return await cur.fetchall()
                except psycopg.ProgrammingError:
                    return []

    async def _executemany(self, query: str, seq_params) -> None:
        """Run a plain SQL statement once per row of ``seq_params`` (no decode)."""
        seq = list(seq_params)
        if not seq:
            return
        async with self._connection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.executemany(query, seq)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AgensgraphQueryException(
                        {"message": f"Error executing SQL: {query}", "detail": str(e)}
                    ) from e


__all__ = [
    "AgensgraphQueryException",
    "VERTEX_REGEX",
    "EDGE_REGEX",
    "record_to_dict",
    "resolve_workspace",
    "_AgensStorageBase",
]
