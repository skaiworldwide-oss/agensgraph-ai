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

from dataclasses import dataclass
from typing import Any, Dict, List, final

from agensgraph import Jsonb
from lightrag.base import BaseKVStorage
from lightrag.utils import logger

from lightrag_agensgraph.kg._base import _AgensStorageBase, resolve_workspace
from lightrag_agensgraph.kg._sql_templates import KV_TABLE_DDL

_INJECTED = ("_id", "create_time", "update_time")

_READ = """
    SELECT id, value,
           EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
           EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
    FROM LIGHTRAG_KV
    WHERE workspace = %(ws)s AND namespace = %(ns)s
"""


@final
@dataclass
class AgensgraphKVStorage(_AgensStorageBase, BaseKVStorage):
    """Key-value storage backed by one JSONB table.

    Every KV namespace (documents, chunks, the LLM cache, the entity and relation
    chunk lists) is an opaque dict addressed by id, so one table partitioned by
    ``(workspace, namespace)`` serves them all. A record is returned as it was
    stored plus ``_id``, ``create_time`` and ``update_time``.
    """

    # A miss here is a confirmed miss: a failed statement raises, it is never read as absent.
    supports_strict_point_reads = True

    def __post_init__(self):
        self.workspace = resolve_workspace(self.workspace, self.global_config)
        self._engine = None

    async def initialize(self):
        await self._acquire_engine()

        async def ddl(conn):
            await conn.execute(KV_TABLE_DDL)

        await self._engine.setup_once("kv", ddl)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        pass  # every write is already committed

    def _scope(self) -> Dict[str, Any]:
        return {"ws": self.workspace, "ns": self.namespace}

    @staticmethod
    def _record(row: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(row["value"] or {})
        value["_id"] = row["id"]
        value["create_time"] = row.get("create_time") or 0
        value["update_time"] = row.get("update_time") or 0
        return value

    async def get_by_id(self, id: str) -> Dict[str, Any] | None:
        rows = await self._fetch(_READ + " AND id = %(id)s", {**self._scope(), "id": id})
        return self._record(rows[0]) if rows else None

    async def get_by_id_strict(self, id: str) -> Dict[str, Any] | None:
        return await self.get_by_id(id)

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._fetch(_READ + " AND id = ANY(%(ids)s)", {**self._scope(), "ids": list(ids)})
        by_id = {r["id"]: self._record(r) for r in rows}
        # In the order asked for, None where nothing was found.
        return [by_id.get(i) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        keys = set(keys)
        if not keys:
            return set()
        rows = await self._fetch_tuples(
            "SELECT id FROM LIGHTRAG_KV WHERE workspace = %(ws)s AND namespace = %(ns)s "
            "AND id = ANY(%(ids)s)",
            {**self._scope(), "ids": list(keys)},
        )
        return keys - {r[0] for r in rows}

    async def get_all(self) -> Dict[str, Dict[str, Any]]:
        rows = await self._fetch(_READ, self._scope())
        return {r["id"]: self._record(r) for r in rows}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        ids, values = [], []
        for id_, payload in data.items():
            ids.append(id_)
            stored = {k: v for k, v in (payload or {}).items() if k not in _INJECTED}
            values.append(Jsonb(stored))
        # One statement for the whole batch: the rows arrive as two arrays.
        await self._run(
            """
            INSERT INTO LIGHTRAG_KV (workspace, namespace, id, value)
            SELECT %(ws)s, %(ns)s, u.id, u.value
            FROM unnest(%(ids)s::text[], %(values)s::jsonb[]) AS u(id, value)
            ON CONFLICT (workspace, namespace, id)
            DO UPDATE SET value = EXCLUDED.value, update_time = CURRENT_TIMESTAMP
            """,
            {**self._scope(), "ids": ids, "values": values},
        )

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._run(
            "DELETE FROM LIGHTRAG_KV WHERE workspace = %(ws)s AND namespace = %(ns)s "
            "AND id = ANY(%(ids)s)",
            {**self._scope(), "ids": list(ids)},
        )

    async def is_empty(self) -> bool:
        rows = await self._fetch_tuples(
            "SELECT 1 FROM LIGHTRAG_KV WHERE workspace = %(ws)s AND namespace = %(ns)s LIMIT 1", self._scope()
        )
        return not rows

    async def drop(self) -> Dict[str, str]:
        try:
            await self._run(
                "DELETE FROM LIGHTRAG_KV WHERE workspace = %(ws)s AND namespace = %(ns)s", self._scope()
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("Error dropping KV namespace %s: %s", self.namespace, e)
            return {"status": "error", "message": str(e)}
