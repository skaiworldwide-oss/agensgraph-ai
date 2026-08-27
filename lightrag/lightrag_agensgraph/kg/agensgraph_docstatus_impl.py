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

"""Document status storage: the table LightRAG's ingestion pipeline schedules from."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, final

from agensgraph import Jsonb
from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceConflict,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
    SourceResolution,
    SourceUnique,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.utils import logger
from psycopg import sql

from lightrag_agensgraph.kg._base import _AgensStorageBase, resolve_workspace
from lightrag_agensgraph.kg._sql_templates import (
    DOC_STATUS_INDEX_DDL,
    DOC_STATUS_TABLE_DDL,
    DOC_STATUS_UPGRADE_DDL,
)

TABLE = "LIGHTRAG_DOC_STATUS"

# The columns, in the order the upsert binds them.
COLUMNS = (
    "id",
    "status",
    "content_summary",
    "content_length",
    "chunks_count",
    "chunks_list",
    "file_path",
    "track_id",
    "content_hash",
    "error_msg",
    "metadata",
    "created_at",
    "updated_at",
)
# What a targeted update may touch. created_at is the sort key of a sweep in progress.
UPDATABLE = frozenset(COLUMNS) - {"id", "created_at"}
JSON_COLUMNS = frozenset({"chunks_list", "metadata"})
TIME_COLUMNS = frozenset({"created_at", "updated_at"})
ARRAY_TYPES = {
    "chunks_list": "jsonb[]",
    "metadata": "jsonb[]",
    "created_at": "timestamptz[]",
    "updated_at": "timestamptz[]",
    "content_length": "integer[]",
    "chunks_count": "integer[]",
}
SORT_FIELDS = frozenset({"created_at", "updated_at", "id", "file_path"})

# The projection a scheduling sweep reads: enough to order, bound and route a page.
SCHEDULING_COLUMNS = "id, status, created_at, updated_at, file_path, track_id, metadata"
ORDER = "ORDER BY created_at ASC NULLS FIRST, id ASC"
# A row that another document's record points at is not a holder of that content.
PRIMARY = "COALESCE((metadata->>'is_duplicate')::boolean, false) = false"
# Sources that name no file are never resolved or listed as conflicts.
PLACEHOLDER_SOURCES = ("", "unknown_source", "no-file-path")
CONFLICT_SAMPLE = 32


def parse_time(value: Any) -> Optional[datetime]:
    """A timestamp from what LightRAG writes: an ISO string, or already a datetime.

    A string without a zone is read as UTC, which is what LightRAG writes.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def render_time(value: Any) -> Optional[str]:
    """A stored timestamp as the ISO string LightRAG compares and displays."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def fingerprint(doc_ids: Sequence[str]) -> str:
    """A digest of a candidate set, the token a repair is committed against."""
    return hashlib.sha256(b"\x00".join(d.encode() for d in sorted(doc_ids))).hexdigest()


@final
@dataclass
class AgensgraphDocStatusStorage(_AgensStorageBase, DocStatusStorage):
    """Document status in a typed table.

    Every field of a status record is a column, so the pipeline's sweeps, counts
    and lookups are index reads, and a page of the sweep is one statement.
    """

    # A miss here is a confirmed miss: a failed statement raises, it is never read as absent.
    supports_strict_point_reads = True

    def __post_init__(self):
        self.workspace = resolve_workspace(self.workspace, self.global_config)
        self._engine = None

    async def initialize(self):
        await self._acquire_engine()

        async def ddl(conn):
            await conn.execute(DOC_STATUS_TABLE_DDL)
            cur = await conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'lightrag_doc_status' AND column_name = 'value'"
            )
            if await cur.fetchone():
                logger.info("bringing the document status table to its column layout")
                async with conn.transaction():
                    for statement in DOC_STATUS_UPGRADE_DDL:
                        await conn.execute(statement)
            for statement in DOC_STATUS_INDEX_DDL:
                await conn.execute(statement)

        await self._engine.setup_once("doc_status", ddl)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        pass  # every write is already committed

    # ---- rows ----

    def _ws(self) -> Dict[str, Any]:
        return {"ws": self.workspace}

    @staticmethod
    def _as_dict(row: Dict[str, Any]) -> Dict[str, Any]:
        """A row as the plain record LightRAG reads with ``get_by_id``."""
        out = {k: row.get(k) for k in COLUMNS if k != "id"}
        out["created_at"] = render_time(row.get("created_at"))
        out["updated_at"] = render_time(row.get("updated_at"))
        out["chunks_list"] = list(row.get("chunks_list") or [])
        out["metadata"] = dict(row.get("metadata") or {})
        return out

    @staticmethod
    def _as_status(row: Dict[str, Any]) -> DocProcessingStatus:
        """A row as the full record; raises on a row that cannot be one."""
        metadata = row.get("metadata")
        return DocProcessingStatus(
            content_summary=row.get("content_summary") or "",
            content_length=int(row.get("content_length") or 0),
            file_path=row.get("file_path") or "no-file-path",
            status=DocStatus(row["status"]),
            created_at=render_time(row.get("created_at")) or "",
            updated_at=render_time(row.get("updated_at")) or "",
            track_id=row.get("track_id"),
            chunks_count=row.get("chunks_count"),
            chunks_list=list(row.get("chunks_list") or []),
            error_msg=row.get("error_msg"),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            content_hash=row.get("content_hash"),
        )

    @staticmethod
    def _as_scheduling(row: Dict[str, Any]) -> DocSchedulingRecord:
        """A row as the light record a sweep routes; raises on a row without a timestamp."""
        created = row.get("created_at")
        if not isinstance(created, datetime):
            raise TypeError(f"document {row.get('id')!r} has no created_at, so it cannot be scheduled")
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        return DocSchedulingRecord(
            id=row["id"],
            status=DocStatus(row["status"]),
            created_at=render_time(created),
            updated_at=render_time(row.get("updated_at") or created),
            file_path=row.get("file_path") or "no-file-path",
            track_id=row.get("track_id"),
            has_custom_chunk_journal=isinstance(metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict),
        )

    def _statuses(self, rows: List[Dict[str, Any]], *, strict: bool) -> Dict[str, DocProcessingStatus]:
        out: Dict[str, DocProcessingStatus] = {}
        for row in rows:
            try:
                out[row["id"]] = self._as_status(row)
            except (KeyError, TypeError, ValueError) as exc:
                if strict:
                    raise
                logger.error("document status row %s cannot be read: %s", row.get("id"), exc)
        return out

    # ---- key-value reads and writes ----

    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND id = %(id)s", {**self._ws(), "id": id}
        )
        return self._as_dict(rows[0]) if rows else None

    async def get_by_id_strict(self, id: str) -> Optional[Dict[str, Any]]:
        return await self.get_by_id(id)

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._ws(), "ids": list(ids)},
        )
        by_id = {r["id"]: self._as_dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        keys = set(keys)
        if not keys:
            return set()
        rows = await self._fetch_tuples(
            f"SELECT id FROM {TABLE} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._ws(), "ids": list(keys)},
        )
        return keys - {r[0] for r in rows}

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        columns: Dict[str, list] = {c: [] for c in COLUMNS}
        for id_, payload in data.items():
            p = dict(payload or {})
            status = p.get("status")
            columns["id"].append(id_)
            columns["status"].append(status.value if isinstance(status, DocStatus) else status)
            columns["content_summary"].append(p.get("content_summary"))
            columns["content_length"].append(p.get("content_length"))
            columns["chunks_count"].append(p.get("chunks_count"))
            columns["chunks_list"].append(Jsonb(list(p.get("chunks_list") or [])))
            columns["file_path"].append(p.get("file_path"))
            columns["track_id"].append(p.get("track_id"))
            columns["content_hash"].append(p.get("content_hash"))
            columns["error_msg"].append(p.get("error_msg"))
            columns["metadata"].append(Jsonb(dict(p.get("metadata") or {})))
            columns["created_at"].append(parse_time(p.get("created_at")))
            columns["updated_at"].append(parse_time(p.get("updated_at")))
        names = ", ".join(COLUMNS)
        arrays = ", ".join(f"%({c})s::{ARRAY_TYPES.get(c, 'text[]')}" for c in COLUMNS)
        # created_at keeps its first value: a document written again is the same document,
        # and a sweep may be ordering on it.
        updates = ", ".join(
            f"created_at = COALESCE({TABLE}.created_at, EXCLUDED.created_at)"
            if c == "created_at"
            else f"{c} = EXCLUDED.{c}"
            for c in COLUMNS
            if c != "id"
        )
        # One statement for the batch: every column arrives as an array.
        await self._run(
            f"INSERT INTO {TABLE} (workspace, {names}) SELECT %(ws)s, * FROM unnest({arrays}) "
            f"ON CONFLICT (workspace, id) DO UPDATE SET {updates}",
            {**self._ws(), **columns},
        )

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._run(
            f"DELETE FROM {TABLE} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._ws(), "ids": list(ids)},
        )

    async def is_empty(self) -> bool:
        rows = await self._fetch_tuples(f"SELECT 1 FROM {TABLE} WHERE workspace = %(ws)s LIMIT 1", self._ws())
        return not rows

    async def drop(self) -> Dict[str, str]:
        try:
            await self._run(f"DELETE FROM {TABLE} WHERE workspace = %(ws)s", self._ws())
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("Error dropping document status: %s", e)
            return {"status": "error", "message": str(e)}

    async def update_doc_status_fields(
        self, doc_id: str, fields: Dict[str, Any], *, missing_ok: bool = False
    ) -> None:
        if "created_at" in fields:
            raise ValueError("created_at is the sort key of a scheduling sweep and cannot be changed")
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"not columns of a document status record: {sorted(unknown)}")
        if not fields:
            if missing_ok or await self.get_by_id(doc_id) is not None:
                return
            raise StorageRecordNotFoundError(doc_id)
        params: Dict[str, Any] = {**self._ws(), "id": doc_id}
        assignments = []
        for name, value in fields.items():
            if name in JSON_COLUMNS:
                value = Jsonb(value if value is not None else ([] if name == "chunks_list" else {}))
            elif name in TIME_COLUMNS:
                value = parse_time(value)
            elif name == "status" and isinstance(value, DocStatus):
                value = value.value
            params["f_" + name] = value
            assignments.append(f"{name} = %(f_{name})s")
        touched = await self._run(
            f"UPDATE {TABLE} SET {', '.join(assignments)} WHERE workspace = %(ws)s AND id = %(id)s", params
        )
        if touched == 0 and not missing_ok:
            raise StorageRecordNotFoundError(doc_id)

    # ---- counts and listings ----

    async def get_status_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in DocStatus}
        rows = await self._fetch_tuples(
            f"SELECT status, count(*) FROM {TABLE} WHERE workspace = %(ws)s GROUP BY status", self._ws()
        )
        for status, count in rows:
            counts[status] = int(count)
        return counts

    async def get_all_status_counts(self) -> Dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def count_docs_by_statuses(self, statuses: List[DocStatus], *, strict: bool = True) -> int:
        values = [s.value if isinstance(s, DocStatus) else s for s in statuses]
        if not values:
            return 0
        rows = await self._fetch_tuples(
            f"SELECT count(*) FROM {TABLE} WHERE workspace = %(ws)s AND status = ANY(%(st)s)",
            {**self._ws(), "st": values},
        )
        if not rows:
            raise StorageControlPlaneError("the document count came back without a row")
        return int(rows[0][0])

    async def get_docs_by_statuses(
        self, statuses: List[DocStatus], strict: bool = False
    ) -> Dict[str, DocProcessingStatus]:
        values = [s.value if isinstance(s, DocStatus) else s for s in statuses]
        if not values:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND status = ANY(%(st)s)",
            {**self._ws(), "st": values},
        )
        return self._statuses(rows, strict=strict)

    async def get_docs_by_status(self, status: DocStatus) -> Dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_track_id(self, track_id: str) -> Dict[str, DocProcessingStatus]:
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND track_id = %(tid)s",
            {**self._ws(), "tid": track_id},
        )
        return self._statuses(rows, strict=False)

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
        if sort_field not in SORT_FIELDS:
            sort_field = "updated_at"
        direction = sql.SQL("ASC" if str(sort_direction).lower() == "asc" else "DESC")
        where = sql.SQL("WHERE workspace = %(ws)s")
        params: Dict[str, Any] = {**self._ws(), "lim": page_size, "off": (page - 1) * page_size}
        if statuses is not None:
            where += sql.SQL(" AND status = ANY(%(st)s)")
            params["st"] = sorted(statuses)
        total = (
            await self._fetch_tuples(
                sql.SQL("SELECT count(*) FROM {} {}").format(sql.Identifier(TABLE.lower()), where), params
            )
        )[0][0]
        rows = await self._fetch(
            sql.SQL(
                "SELECT * FROM {} {} ORDER BY {} {} NULLS LAST, id ASC LIMIT %(lim)s OFFSET %(off)s"
            ).format(sql.Identifier(TABLE.lower()), where, sql.Identifier(sort_field), direction),
            params,
        )
        result = [
            (r["id"], status)
            for r in rows
            for status in [self._statuses([r], strict=False).get(r["id"])]
            if status
        ]
        return result, int(total)

    # ---- single-document lookups ----

    async def get_doc_by_file_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND file_path = %(fp)s {ORDER} LIMIT 1",
            {**self._ws(), "fp": file_path},
        )
        return self._as_dict(rows[0]) if rows else None

    async def get_doc_by_file_basename(self, basename: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s "
            f"AND (file_path = %(b)s OR regexp_replace(file_path, '^.*/', '') = %(b)s) {ORDER} LIMIT 1",
            {**self._ws(), "b": basename},
        )
        return (rows[0]["id"], self._as_dict(rows[0])) if rows else None

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: Optional[str] = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """The earliest document holding this content, or None when there is none.

        With ``exclude_doc_id`` two kinds of row do not count: the document itself,
        and a record that merely says its content belongs to that document.
        """
        if not content_hash:
            return None
        params: Dict[str, Any] = {**self._ws(), "h": content_hash}
        exclusion = ""
        if exclude_doc_id:
            params["x"] = exclude_doc_id
            exclusion = (
                " AND id <> %(x)s AND NOT (COALESCE((metadata->>'is_duplicate')::boolean, false) "
                "AND COALESCE(metadata->>'original_doc_id', '') = %(x)s)"
            )
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND content_hash = %(h)s{exclusion} "
            f"{ORDER} LIMIT 1",
            params,
        )
        return (rows[0]["id"], self._as_dict(rows[0])) if rows else None

    # ---- the scheduling sweep ----

    @staticmethod
    def _cursor(row: Dict[str, Any]) -> CursorAfter:
        created = row.get("created_at")
        return CursorAfter(
            json.dumps([created.astimezone(timezone.utc).isoformat() if created else None, row["id"]])
        )

    @staticmethod
    def _decode_cursor(position: CursorPosition) -> Tuple[Optional[datetime], str]:
        if not isinstance(position, CursorAfter):
            raise StorageControlPlaneError(f"not a cursor of this store: {position!r}")
        try:
            created, doc_id = json.loads(position.opaque)
            return (parse_time(created) if created else None), str(doc_id)
        except (ValueError, TypeError) as exc:
            raise StorageControlPlaneError(f"the scheduling cursor cannot be read: {exc}") from exc

    async def get_docs_by_statuses_page(
        self,
        statuses: List[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """One page of documents in the given statuses, in (created_at, id) order.

        One ordered, limited branch per status, merged: the planner answers each
        from the sweep index, where a single ``status = ANY`` read sorted the whole
        set. The cursor is the last row read, so a page whose rows could not all be
        turned into records still moves the sweep forward.
        """
        if limit <= 0:
            raise ValueError(f"a page holds at least one document, got limit={limit}")
        values = [s.value if isinstance(s, DocStatus) else s for s in statuses]
        if not values or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        params: Dict[str, Any] = {**self._ws(), "lim": limit}
        after = ""
        if position is not CURSOR_START:
            created, doc_id = self._decode_cursor(position)
            params["cid"] = doc_id
            if created is None:
                # Still inside the rows that have no timestamp, which sort first.
                after = " AND ((created_at IS NULL AND id > %(cid)s) OR created_at IS NOT NULL)"
            else:
                params["cts"] = created
                after = " AND created_at IS NOT NULL AND (created_at, id) > (%(cts)s, %(cid)s)"
        branches = []
        for i, value in enumerate(values):
            params[f"s{i}"] = value
            branches.append(
                f"(SELECT {SCHEDULING_COLUMNS} FROM {TABLE} "
                f"WHERE workspace = %(ws)s AND status = %(s{i})s{after} {ORDER} LIMIT %(lim)s)"
            )
        statement = " UNION ALL ".join(branches)
        if len(branches) > 1:
            statement = f"SELECT * FROM ({statement}) AS u {ORDER} LIMIT %(lim)s"
        rows = await self._fetch(statement, params)
        docs: Dict[str, DocSchedulingRecord] = {}
        for row in rows:
            try:
                docs[row["id"]] = self._as_scheduling(row)
            except (KeyError, TypeError, ValueError) as exc:
                if strict:
                    raise
                logger.error("document %s is skipped by the sweep: %s", row.get("id"), exc)
        next_position = self._cursor(rows[-1]) if len(rows) == limit else CURSOR_END
        return DocStatusPage(docs=docs, next_position=next_position)

    async def get_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> Dict[str, DocSchedulingRecord]:
        ids = list(doc_ids)
        if not ids:
            return {}
        rows = await self._fetch(
            f"SELECT {SCHEDULING_COLUMNS} FROM {TABLE} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._ws(), "ids": ids},
        )
        out: Dict[str, DocSchedulingRecord] = {}
        for row in rows:
            try:
                out[row["id"]] = self._as_scheduling(row)
            except (KeyError, TypeError, ValueError):
                if strict:
                    raise
        return out

    async def get_full_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> Dict[str, DocProcessingStatus]:
        ids = list(doc_ids)
        if not ids:
            return {}
        rows = await self._fetch(
            f"SELECT * FROM {TABLE} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {**self._ws(), "ids": ids},
        )
        return self._statuses(rows, strict=strict)

    # ---- sources ----

    async def resolve_doc_source_strict(self, canonical_source_key: str) -> SourceResolution:
        if canonical_source_key in PLACEHOLDER_SOURCES:
            return SourceAbsent()
        params = {**self._ws(), "fp": canonical_source_key}
        rows = await self._fetch(
            f"SELECT {SCHEDULING_COLUMNS} FROM {TABLE} "
            f"WHERE workspace = %(ws)s AND file_path = %(fp)s AND {PRIMARY} {ORDER} LIMIT 2",
            params,
        )
        if not rows:
            return SourceAbsent()
        if len(rows) == 1:
            return SourceUnique(doc_id=rows[0]["id"], doc=self._as_scheduling(rows[0]))
        count = (
            await self._fetch_tuples(
                f"SELECT count(*) FROM {TABLE} WHERE workspace = %(ws)s AND file_path = %(fp)s AND {PRIMARY}",
                params,
            )
        )[0][0]
        return SourceConflict(candidate_count=int(count), sample_doc_ids=tuple(sorted(r["id"] for r in rows)))

    async def _candidates(self, conn, canonical_source_key: str, *, lock: bool) -> List[str]:
        cur = await conn.execute(
            f"SELECT id FROM {TABLE} WHERE workspace = %(ws)s AND file_path = %(fp)s AND {PRIMARY} "
            "ORDER BY id ASC" + (" FOR UPDATE" if lock else ""),
            {**self._ws(), "fp": canonical_source_key},
        )
        return [r[0] for r in await cur.fetchall()]

    async def list_source_conflicts_page(
        self, *, limit: int, position: CursorPosition = CURSOR_START
    ) -> SourceConflictPage:
        if limit <= 0:
            raise ValueError(f"a page holds at least one conflict, got limit={limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)
        params: Dict[str, Any] = {**self._ws(), "lim": limit, "skip": list(PLACEHOLDER_SOURCES)}
        after = ""
        if position is not CURSOR_START:
            if not isinstance(position, CursorAfter):
                raise StorageControlPlaneError(f"not a cursor of this store: {position!r}")
            params["last"] = json.loads(position.opaque)
            after = " AND file_path > %(last)s"
        rows = await self._fetch_tuples(
            f"SELECT file_path, count(*) FROM {TABLE} WHERE workspace = %(ws)s AND {PRIMARY} "
            f"AND file_path IS NOT NULL AND file_path <> ALL(%(skip)s){after} "
            f"GROUP BY file_path HAVING count(*) >= 2 ORDER BY file_path ASC LIMIT %(lim)s",
            params,
        )
        conflicts = []
        for file_path, count in rows:
            sample = await self._fetch_tuples(
                f"SELECT id FROM {TABLE} WHERE workspace = %(ws)s AND file_path = %(fp)s AND {PRIMARY} "
                f"ORDER BY id ASC LIMIT %(n)s",
                {**self._ws(), "fp": file_path, "n": CONFLICT_SAMPLE},
            )
            conflicts.append(
                SourceConflictSummary(
                    canonical_source_key=file_path,
                    candidate_count=int(count),
                    sample_doc_ids=tuple(r[0] for r in sample),
                )
            )
        next_position = CursorAfter(json.dumps(rows[-1][0])) if len(rows) == limit else CURSOR_END
        return SourceConflictPage(conflicts=tuple(conflicts), next_position=next_position)

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        """Keep one document for a source and mark the others as its duplicates.

        Nothing is deleted. A commit re-reads the candidates under a row lock and
        refuses when they are not the set the caller was shown.
        """
        async with self._connection() as conn:
            async with conn.transaction():
                ids = await self._candidates(conn, canonical_source_key, lock=not dry_run)
                if primary_doc_id not in ids:
                    raise ValueError(f"{primary_doc_id!r} is not a candidate for {canonical_source_key!r}")
                digest = fingerprint(ids)
                losers = [i for i in ids if i != primary_doc_id]
                if dry_run:
                    return SourceConflictRepairResult(
                        canonical_source_key=canonical_source_key,
                        primary_doc_id=primary_doc_id,
                        candidate_count=len(ids),
                        fingerprint=digest,
                        demoted_sample_doc_ids=tuple(losers[:CONFLICT_SAMPLE]),
                        committed=False,
                    )
                if len(ids) != expected_candidate_count or digest != expected_candidate_fingerprint:
                    raise SourceConflictRepairCASError(
                        f"the candidates for {canonical_source_key!r} changed since the dry run; run it again"
                    )
                if losers:
                    await conn.execute(
                        f"UPDATE {TABLE} SET metadata = metadata || %(mark)s, updated_at = now() "
                        f"WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
                        {
                            **self._ws(),
                            "ids": losers,
                            "mark": Jsonb({"is_duplicate": True, "original_doc_id": primary_doc_id}),
                        },
                    )
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=len(ids),
            fingerprint=digest,
            demoted_sample_doc_ids=tuple(losers[:CONFLICT_SAMPLE]),
            committed=True,
        )
