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

import os
from dataclasses import dataclass
from typing import Any, Dict, List, final

from psycopg.types.json import Jsonb

from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from lightrag_agensgraph.kg._base import _AgensStorageBase, resolve_workspace
from lightrag_agensgraph.kg._sql_templates import KV_TABLE_DDL


@final
@dataclass
class AgensgraphKVStorage(_AgensStorageBase, BaseKVStorage):
    """Key-value storage backed by a generic JSONB table in AgensGraph.

    KV records are opaque dicts addressed by id, so all KV namespaces share one
    ``LIGHTRAG_KV`` table partitioned by ``(workspace, namespace)``. Matches the
    reference ``JsonKVStorage`` contract: round-trips the stored dict and injects
    ``_id`` / ``create_time`` / ``update_time`` on read.
    """

    def __post_init__(self):
        self.workspace = os.environ.get("AGENSGRAPH_WORKSPACE") or self.workspace or ""
        self._graph_path = None
        self._engine = None

    async def initialize(self):
        await self._acquire_engine()

        async def _ddl(cur):
            await cur.execute(KV_TABLE_DDL)

        await self._engine.ensure_relational("kv", _ddl)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        # AgensGraph persists synchronously.
        pass

    def _scope(self) -> Dict[str, Any]:
        return {"ws": self.workspace, "ns": self.namespace}

    @staticmethod
    def _row_to_value(row: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(row.get("value") or {})
        value["_id"] = row["id"]
        value["create_time"] = row.get("create_time") or 0
        value["update_time"] = row.get("update_time") or 0
        return value

    async def get_by_id(self, id: str) -> Dict[str, Any] | None:
        rows = await self._execute(
            """
            SELECT id, value,
                   EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
                   EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
            FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s AND id = %(id)s
            """,
            {**self._scope(), "id": id},
        )
        return self._row_to_value(rows[0]) if rows else None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._execute(
            """
            SELECT id, value,
                   EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
                   EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
            FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s AND id = ANY(%(ids)s)
            """,
            {**self._scope(), "ids": list(ids)},
        )
        by_id = {r["id"]: self._row_to_value(r) for r in rows}
        # Preserve order and None-pad missing ids (JsonKVStorage contract).
        return [by_id.get(i) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        keys = set(keys)
        if not keys:
            return set()
        rows = await self._execute(
            """
            SELECT id FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s AND id = ANY(%(ids)s)
            """,
            {**self._scope(), "ids": list(keys)},
        )
        return keys - {r["id"] for r in rows}

    async def get_all(self) -> Dict[str, Dict[str, Any]]:
        rows = await self._execute(
            """
            SELECT id, value,
                   EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
                   EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
            FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s
            """,
            self._scope(),
        )
        return {r["id"]: self._row_to_value(r) for r in rows}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        params = []
        for id_, payload in data.items():
            value = {
                k: v
                for k, v in (payload or {}).items()
                if k not in ("_id", "create_time", "update_time")
            }
            params.append((self.workspace, self.namespace, id_, Jsonb(value)))
        await self._executemany(
            """
            INSERT INTO LIGHTRAG_KV (workspace, namespace, id, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (workspace, namespace, id)
            DO UPDATE SET value = EXCLUDED.value, update_time = CURRENT_TIMESTAMP
            """,
            params,
        )

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._execute(
            """
            DELETE FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s AND id = ANY(%(ids)s)
            """,
            {**self._scope(), "ids": list(ids)},
            fetch=False,
        )

    async def is_empty(self) -> bool:
        rows = await self._execute(
            """
            SELECT 1 FROM LIGHTRAG_KV
            WHERE workspace = %(ws)s AND namespace = %(ns)s LIMIT 1
            """,
            self._scope(),
        )
        return not rows

    async def drop(self) -> Dict[str, str]:
        try:
            await self._execute(
                "DELETE FROM LIGHTRAG_KV WHERE workspace = %(ws)s AND namespace = %(ns)s",
                self._scope(),
                fetch=False,
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error dropping KV namespace {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
