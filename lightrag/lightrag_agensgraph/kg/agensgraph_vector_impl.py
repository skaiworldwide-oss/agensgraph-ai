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

"""Vector storage: LightRAG's entity, relation and chunk embeddings in pgvector tables.

LightRAG hands a record over one at a time while it merges a document and asks
for the embeddings to be made by the store. Records are kept here until the
document is done and ``index_done_callback`` runs, then embedded in batches and
written with one statement per batch. A search runs on the HNSW index with the
planner told so, because it cannot see that the vectors live out of line and
would rather read the whole table.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, final

from agensgraph import Vector
from lightrag.base import BaseVectorStorage
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.namespace import NameSpace, is_namespace
from lightrag.utils import compute_mdhash_id, logger
from psycopg import sql

from lightrag_agensgraph.kg._base import _AgensStorageBase, resolve_workspace
from lightrag_agensgraph.kg._sql_templates import (
    VECTOR_CHUNK_TABLE,
    VECTOR_ENTITY_TABLE,
    VECTOR_LOOKUP_INDEX_DDL,
    VECTOR_RELATION_TABLE,
    VECTOR_TABLE_DDL,
    vector_index_ddl,
    vector_index_name,
)

# The columns each kind of record carries besides its id and vector, and how a payload
# key maps onto them. chunk_ids holds what LightRAG calls source_id: the chunk ids joined.
KINDS = {
    "entities": (
        VECTOR_ENTITY_TABLE,
        {
            "entity_name": "entity_name",
            "content": "content",
            "chunk_ids": "source_id",
            "file_path": "file_path",
        },
    ),
    "relationships": (
        VECTOR_RELATION_TABLE,
        {
            "src_id": "src_id",
            "tgt_id": "tgt_id",
            "content": "content",
            "chunk_ids": "source_id",
            "file_path": "file_path",
        },
    ),
    "chunks": (
        VECTOR_CHUNK_TABLE,
        {
            "full_doc_id": "full_doc_id",
            "chunk_order_index": "chunk_order_index",
            "tokens": "tokens",
            "content": "content",
            "file_path": "file_path",
        },
    ),
}
INTEGER_COLUMNS = frozenset({"chunk_order_index", "tokens"})
ROWS_PER_STATEMENT = 500
DEFAULT_EF_SEARCH = 40
CREATED = "EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at"


def as_vector(values: Any) -> Vector:
    if isinstance(values, Vector):
        return values
    return Vector(values.tolist() if hasattr(values, "tolist") else values)


@final
@dataclass
class AgensgraphVectorStorage(_AgensStorageBase, BaseVectorStorage):
    """Embeddings in a pgvector table with an HNSW index, written in batches."""

    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    maintenance_work_mem: str = "1GB"
    _pending: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _bulk: bool = field(default=False, init=False, repr=False)
    _index_dropped: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self._validate_embedding_func()
        self.workspace = resolve_workspace(self.workspace, self.global_config)
        self._engine = None
        for kind, (table, columns) in KINDS.items():
            if is_namespace(self.namespace, getattr(NameSpace, f"VECTOR_STORE_{kind.upper()}")):
                self._kind, self.table, self._columns = kind, table, columns
                break
        else:
            raise ValueError(f"Unsupported vector namespace: {self.namespace}")
        config = (self.global_config or {}).get("vector_db_storage_cls_kwargs", {})
        if config.get("cosine_better_than_threshold") is not None:
            self.cosine_better_than_threshold = config["cosine_better_than_threshold"]
        self.hnsw_m = int(config.get("hnsw_m", self.hnsw_m))
        self.hnsw_ef_construction = int(config.get("hnsw_ef_construction", self.hnsw_ef_construction))
        self._batch = int((self.global_config or {}).get("embedding_batch_num", 32))
        self._dim = int(self.embedding_func.embedding_dim)

    async def initialize(self):
        await self._acquire_engine()
        await self._engine.enable_vectors()
        dim = self._dim

        async def ddl(conn):
            for statement in VECTOR_TABLE_DDL.values():
                await conn.execute(statement.format(dim=dim))
            await self._upgrade(conn)
            for table in VECTOR_TABLE_DDL:
                await conn.execute(
                    vector_index_ddl(table, m=self.hnsw_m, ef_construction=self.hnsw_ef_construction)
                )
            for statement in VECTOR_LOOKUP_INDEX_DDL:
                await conn.execute(statement)

        await self._engine.setup_once("vector", ddl)
        # The tables were made for one embedding width; a different model needs its own.
        rows = await self._fetch_tuples(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'content_vector'",
            (self.table.lower(),),
        )
        if rows and rows[0][0] != f"vector({dim})":
            raise ValueError(
                f"{self.table} holds {rows[0][0]} embeddings and this embedding function produces "
                f"vector({dim}); drop the LIGHTRAG_VDB_* tables to switch embedding models"
            )

    @staticmethod
    async def _upgrade(conn) -> None:
        """Bring tables in the previous layout to this one, keeping their rows.

        The relation table named its endpoints source_id and target_id, and both the
        entity and the relation table kept the chunk ids as an array.
        """
        cur = await conn.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_name IN ('lightrag_vdb_entity', 'lightrag_vdb_relation') "
            "AND column_name IN ('source_id', 'target_id', 'chunk_ids')"
        )
        columns = {(t, c): kind for t, c, kind in await cur.fetchall()}
        async with conn.transaction():
            if ("lightrag_vdb_relation", "source_id") in columns:
                logger.info("renaming the relation table's endpoint columns")
                await conn.execute("ALTER TABLE LIGHTRAG_VDB_RELATION RENAME COLUMN source_id TO src_id")
                await conn.execute("ALTER TABLE LIGHTRAG_VDB_RELATION RENAME COLUMN target_id TO tgt_id")
            for table in ("lightrag_vdb_entity", "lightrag_vdb_relation"):
                if columns.get((table, "chunk_ids")) == "ARRAY":
                    logger.info("storing %s chunk ids as one string", table)
                    # A utility statement takes no bound parameter; the separator is a literal.
                    await conn.execute(
                        sql.SQL(
                            "ALTER TABLE {} ALTER COLUMN chunk_ids TYPE TEXT "
                            "USING array_to_string(chunk_ids, {})"
                        ).format(sql.Identifier(table), sql.Literal(GRAPH_FIELD_SEP))
                    )

    async def finalize(self):
        await self._release_engine()

    # ---- writes ----

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Keep the records; they are embedded and written when the document is done."""
        for id_, item in data.items():
            if "content" not in item:
                raise ValueError(f"vector record {id_!r} has no content to embed")
            self._pending[id_] = dict(item)

    async def index_done_callback(self) -> None:
        pending, self._pending = self._pending, {}
        if not pending:
            return
        try:
            if self._bulk and not self._index_dropped:
                await self._run(f"DROP INDEX IF EXISTS {vector_index_name(self.table)}")
                self._index_dropped = True
            ids = list(pending)
            vectors = await self._embed([pending[i]["content"] for i in ids])
            for start in range(0, len(ids), ROWS_PER_STATEMENT):
                chunk = ids[start : start + ROWS_PER_STATEMENT]
                await self._write(
                    chunk, [pending[i] for i in chunk], vectors[start : start + ROWS_PER_STATEMENT]
                )
        except BaseException:
            # What was not written is still owed; a record written since keeps its newer value.
            for id_, item in pending.items():
                self._pending.setdefault(id_, item)
            raise

    async def drop_pending_index_ops(self) -> None:
        self._pending.clear()

    async def _embed(self, texts: List[str]) -> List[Vector]:
        vectors: List[Vector] = []
        for start in range(0, len(texts), self._batch):
            for row in await self.embedding_func(texts[start : start + self._batch]):
                vectors.append(as_vector(row))
        return vectors

    async def _write(self, ids: List[str], items: List[Dict[str, Any]], vectors: List[Vector]) -> None:
        columns = list(self._columns)
        arrays: Dict[str, list] = {"ids": ids, "vectors": vectors}
        for column in columns:
            arrays[column] = [item.get(self._columns[column]) for item in items]
        typed = ", ".join(f"%({c})s::{'integer[]' if c in INTEGER_COLUMNS else 'text[]'}" for c in columns)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
        # One statement per batch: every column arrives as an array, the vectors in binary.
        await self._run(
            f"INSERT INTO {self.table} (workspace, id, content_vector, {', '.join(columns)}) "
            f"SELECT %(ws)s, * FROM unnest(%(ids)s::text[], %(vectors)b::vector[], {typed}) "
            f"ON CONFLICT (workspace, id) DO UPDATE SET content_vector = EXCLUDED.content_vector, "
            f"{updates}, update_time = now()",
            {"ws": self.workspace, **arrays},
        )

    @asynccontextmanager
    async def bulk_ingest(self) -> AsyncIterator[None]:
        """Load many records: the HNSW index is dropped for the duration and rebuilt at the end.

        Inserting into an HNSW index costs far more than building it afterwards, and the
        build runs with ``maintenance_work_mem`` raised so the graph fits in memory.
        """
        self._bulk = True
        try:
            yield
            await self.index_done_callback()
        finally:
            self._bulk = False
            if self._index_dropped:
                self._index_dropped = False
                async with self._connection() as conn:
                    async with conn.transaction():
                        await conn.execute(f"SET LOCAL maintenance_work_mem = '{self.maintenance_work_mem}'")
                        await conn.execute(
                            vector_index_ddl(
                                self.table, m=self.hnsw_m, ef_construction=self.hnsw_ef_construction
                            )
                        )

    # ---- search ----

    async def query(
        self, query: str, top_k: int, query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        if self._pending:
            await self.index_done_callback()
        embedding = (
            query_embedding if query_embedding is not None else (await self.embedding_func([query]))[0]
        )
        columns = ", ".join(self._columns)
        statement = (
            f"SELECT id, {columns}, content_vector <=> %(v)b AS distance, {CREATED} FROM {self.table} "
            "WHERE workspace = %(ws)s AND content_vector <=> %(v)b < %(threshold)s "
            "ORDER BY content_vector <=> %(v)b LIMIT %(k)s"
        )
        params = {
            "ws": self.workspace,
            "v": as_vector(embedding),
            "threshold": 1 - self.cosine_better_than_threshold,
            "k": top_k,
        }

        async def attempt():
            async with self._connection() as conn:
                async with conn.transaction():
                    async with conn.pipeline():
                        # The planner costs the HNSW index above reading the table, because the
                        # vectors are stored out of line where it cannot see them; the workspace
                        # key then offers it a bitmap or a primary-key scan followed by a sort.
                        # With those three off for this transaction the index walk is the only
                        # plan left that needs no sort. The walk goes further when the threshold
                        # drops candidates, and as far as the limit asks.
                        await conn.execute("SET LOCAL enable_seqscan = off")
                        await conn.execute("SET LOCAL enable_bitmapscan = off")
                        await conn.execute("SET LOCAL enable_sort = off")
                        await conn.vector_search_options(
                            {
                                "hnsw.iterative_scan": "relaxed_order",
                                "hnsw.ef_search": max(DEFAULT_EF_SEARCH, int(top_k)),
                            }
                        )
                        cur = conn.cursor()
                        await cur.execute(statement, params)
                    rows = await cur.fetchall()
                    keys = [c.name for c in cur.description]
                    await cur.close()
            return [self._record(dict(zip(keys, row))) for row in rows]

        from lightrag_agensgraph.kg._engine import run_with_retry

        return await run_with_retry(self._retry, attempt, wrote=False)

    def _record(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row.pop("content_vector", None)
        if "chunk_ids" in row:
            joined = row.pop("chunk_ids")
            row["source_id"] = joined
            row["chunk_ids"] = joined.split(GRAPH_FIELD_SEP) if joined else []
        return row

    def _pending_record(self, id_: str) -> Dict[str, Any]:
        item = self._pending[id_]
        row = {"id": id_}
        for column, key in self._columns.items():
            row[column] = item.get(key)
        return self._record(row)

    # ---- reads ----

    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        if id in self._pending:
            return self._pending_record(id)
        rows = await self._fetch(
            f"SELECT id, {', '.join(self._columns)}, {CREATED} FROM {self.table} "
            "WHERE workspace = %(ws)s AND id = %(id)s",
            {"ws": self.workspace, "id": id},
        )
        return self._record(rows[0]) if rows else None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._fetch(
            f"SELECT id, {', '.join(self._columns)}, {CREATED} FROM {self.table} "
            "WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {"ws": self.workspace, "ids": list(ids)},
        )
        found = {r["id"]: self._record(r) for r in rows}
        for id_ in ids:
            if id_ in self._pending:
                found[id_] = self._pending_record(id_)
        return [found[i] for i in ids if i in found]

    async def get_vectors_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """The embedding of every id asked for, including ones not yet written."""
        if not ids:
            return {}

        async def attempt():
            async with self._connection() as conn:
                async with conn.cursor(binary=True) as cur:
                    await cur.execute(
                        f"SELECT id, content_vector FROM {self.table} "
                        "WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
                        {"ws": self.workspace, "ids": list(ids)},
                    )
                    return await cur.fetchall()

        from lightrag_agensgraph.kg._engine import run_with_retry

        result = {
            id_: vector.tolist() for id_, vector in await run_with_retry(self._retry, attempt, wrote=False)
        }
        waiting = [i for i in ids if i in self._pending and i not in result]
        if waiting:
            for id_, vector in zip(
                waiting, await self._embed([self._pending[i]["content"] for i in waiting])
            ):
                result[id_] = vector.tolist()
        return result

    # ---- deletes ----

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        for id_ in ids:
            self._pending.pop(id_, None)
        await self._run(
            f"DELETE FROM {self.table} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {"ws": self.workspace, "ids": list(ids)},
        )

    async def delete_entity(self, entity_name: str) -> None:
        for id_ in [i for i, item in self._pending.items() if item.get("entity_name") == entity_name]:
            self._pending.pop(id_)
        await self._run(
            f"DELETE FROM {VECTOR_ENTITY_TABLE} "
            "WHERE workspace = %(ws)s AND (id = %(id)s OR entity_name = %(name)s)",
            {"ws": self.workspace, "id": compute_mdhash_id(entity_name, prefix="ent-"), "name": entity_name},
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        for id_ in [
            i for i, item in self._pending.items() if entity_name in (item.get("src_id"), item.get("tgt_id"))
        ]:
            self._pending.pop(id_)
        await self._run(
            f"DELETE FROM {VECTOR_RELATION_TABLE} "
            "WHERE workspace = %(ws)s AND (src_id = %(name)s OR tgt_id = %(name)s)",
            {"ws": self.workspace, "name": entity_name},
        )

    async def drop(self) -> Dict[str, str]:
        try:
            self._pending.clear()
            await self._run(f"DELETE FROM {self.table} WHERE workspace = %(ws)s", {"ws": self.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error("Error dropping vector namespace %s: %s", self.namespace, e)
            return {"status": "error", "message": str(e)}


__all__ = ["AgensgraphVectorStorage"]
