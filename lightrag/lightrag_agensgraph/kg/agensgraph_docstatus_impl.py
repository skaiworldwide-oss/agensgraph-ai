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

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, final

from psycopg.types.json import Jsonb

from lightrag.base import DocProcessingStatus, DocStatus, DocStatusStorage
from lightrag.utils import logger

from lightrag_agensgraph.kg._base import _AgensStorageBase
from lightrag_agensgraph.kg._sql_templates import (
    DOC_STATUS_INDEX_DDL,
    DOC_STATUS_TABLE_DDL,
)

_DPS_FIELDS = {f.name for f in dataclasses.fields(DocProcessingStatus)}
_SORT_FIELDS = {"created_at", "updated_at", "id", "file_path"}


@final
@dataclass
class AgensgraphDocStatusStorage(_AgensStorageBase, DocStatusStorage):
    """Document-status storage backed by a typed table in AgensGraph.

    The full DocProcessingStatus record is kept in a ``value`` JSONB column for
    faithful round-tripping; status / file_path / content_hash / track_id are
    promoted to indexed columns for filtering, counting, and pagination.
    """

    def __post_init__(self):
        self.workspace = os.environ.get("AGENSGRAPH_WORKSPACE") or self.workspace or ""
        self._graph_path = None
        self._engine = None

    async def initialize(self):
        await self._acquire_engine()

        async def _ddl(cur):
            await cur.execute(DOC_STATUS_TABLE_DDL)
            for ix in DOC_STATUS_INDEX_DDL:
                await cur.execute(ix)

        await self._engine.ensure_relational("doc_status", _ddl)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        pass

    def _scope(self) -> Dict[str, Any]:
        return {"ws": self.workspace}

    @staticmethod
    def _to_status(stored: dict) -> DocProcessingStatus:
        data = {k: v for k, v in (stored or {}).items() if k in _DPS_FIELDS}
        data.pop("content", None)
        if not data.get("file_path"):
            data["file_path"] = "no-file-path"
        data.setdefault("metadata", {})
        data.setdefault("error_msg", None)
        return DocProcessingStatus(**data)

    # ---- KV-style accessors ----

    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        rows = await self._execute(
            "SELECT value FROM LIGHTRAG_DOC_STATUS WHERE workspace = %(ws)s AND id = %(id)s",
            {**self._scope(), "id": id},
        )
        return rows[0]["value"] if rows else None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._execute(
            "SELECT id, value FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._scope(), "ids": list(ids)},
        )
        by_id = {r["id"]: r["value"] for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        keys = set(keys)
        if not keys:
            return set()
        rows = await self._execute(
            "SELECT id FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._scope(), "ids": list(keys)},
        )
        return keys - {r["id"] for r in rows}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        params = []
        for id_, payload in data.items():
            p = dict(payload or {})
            p.setdefault("chunks_list", [])
            status = p.get("status")
            status = status.value if isinstance(status, DocStatus) else status
            params.append(
                (
                    self.workspace,
                    id_,
                    status,
                    p.get("file_path"),
                    p.get("content_hash"),
                    p.get("track_id"),
                    Jsonb(p),
                )
            )
        await self._executemany(
            """
            INSERT INTO LIGHTRAG_DOC_STATUS
                (workspace, id, status, file_path, content_hash, track_id, value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (workspace, id) DO UPDATE SET
                status = EXCLUDED.status,
                file_path = EXCLUDED.file_path,
                content_hash = EXCLUDED.content_hash,
                track_id = EXCLUDED.track_id,
                value = EXCLUDED.value,
                update_time = CURRENT_TIMESTAMP
            """,
            params,
        )

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._execute(
            "DELETE FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._scope(), "ids": list(ids)},
            fetch=False,
        )

    async def is_empty(self) -> bool:
        rows = await self._execute(
            "SELECT 1 FROM LIGHTRAG_DOC_STATUS WHERE workspace = %(ws)s LIMIT 1",
            self._scope(),
        )
        return not rows

    async def drop(self) -> Dict[str, str]:
        try:
            await self._execute(
                "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = %(ws)s",
                self._scope(),
                fetch=False,
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error dropping doc-status: {e}")
            return {"status": "error", "message": str(e)}

    # ---- status queries ----

    async def get_status_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in DocStatus}
        rows = await self._execute(
            "SELECT status, count(*) AS c FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s GROUP BY status",
            self._scope(),
        )
        for r in rows:
            if r["status"] in counts:
                counts[r["status"]] = int(r["c"])
        return counts

    async def get_all_status_counts(self) -> Dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_docs_by_statuses(
        self, statuses: List[DocStatus]
    ) -> Dict[str, DocProcessingStatus]:
        values = [s.value if isinstance(s, DocStatus) else s for s in statuses]
        rows = await self._execute(
            "SELECT id, value FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND status = ANY(%(st)s)",
            {**self._scope(), "st": values},
        )
        return self._rows_to_status_map(rows)

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> Dict[str, DocProcessingStatus]:
        rows = await self._execute(
            "SELECT id, value FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND track_id = %(tid)s",
            {**self._scope(), "tid": track_id},
        )
        return self._rows_to_status_map(rows)

    def _rows_to_status_map(self, rows) -> Dict[str, DocProcessingStatus]:
        out: Dict[str, DocProcessingStatus] = {}
        for r in rows:
            try:
                out[r["id"]] = self._to_status(r["value"])
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Failed to build DocProcessingStatus for {r['id']}: {e}")
        return out

    async def get_docs_paginated(
        self,
        status_filter: Optional[DocStatus] = None,
        status_filters: Optional[List[DocStatus]] = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> Tuple[List[Tuple[str, DocProcessingStatus]], int]:
        statuses = self.resolve_status_filter_values(
            status_filter=status_filter, status_filters=status_filters
        )
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        if sort_field not in _SORT_FIELDS:
            sort_field = "updated_at"
        direction = "ASC" if str(sort_direction).lower() == "asc" else "DESC"

        where = "WHERE workspace = %(ws)s"
        params: Dict[str, Any] = {**self._scope()}
        if statuses is not None:
            where += " AND status = ANY(%(st)s)"
            params["st"] = list(statuses)

        total = (
            await self._execute(
                f"SELECT count(*) AS c FROM LIGHTRAG_DOC_STATUS {where}", params
            )
        )[0]["c"]

        # id/file_path are columns; created_at/updated_at live in the JSONB value
        # (ISO strings sort chronologically).
        if sort_field in ("id", "file_path"):
            order_expr = sort_field
        else:
            order_expr = f"value->>'{sort_field}'"

        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        rows = await self._execute(
            f"""
            SELECT id, value FROM LIGHTRAG_DOC_STATUS {where}
            ORDER BY {order_expr} {direction} NULLS LAST, id ASC
            LIMIT %(lim)s OFFSET %(off)s
            """,
            params,
        )
        result: List[Tuple[str, DocProcessingStatus]] = []
        for r in rows:
            try:
                result.append((r["id"], self._to_status(r["value"])))
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Failed to build DocProcessingStatus for {r['id']}: {e}")
        return result, int(total)

    # ---- single-document lookups ----

    async def get_doc_by_file_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        rows = await self._execute(
            "SELECT value FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND file_path = %(fp)s ORDER BY id ASC LIMIT 1",
            {**self._scope(), "fp": file_path},
        )
        return rows[0]["value"] if rows else None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        rows = await self._execute(
            """
            SELECT id, value FROM LIGHTRAG_DOC_STATUS
            WHERE workspace = %(ws)s
              AND regexp_replace(file_path, '^.*/', '') = %(b)s
            ORDER BY id ASC LIMIT 1
            """,
            {**self._scope(), "b": basename},
        )
        return (rows[0]["id"], rows[0]["value"]) if rows else None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        rows = await self._execute(
            "SELECT id, value FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace = %(ws)s AND content_hash = %(h)s ORDER BY id ASC LIMIT 1",
            {**self._scope(), "h": content_hash},
        )
        return (rows[0]["id"], rows[0]["value"]) if rows else None
