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
from typing import Any, Dict, List, Optional, final

from lightrag.base import BaseVectorStorage
from lightrag.namespace import NameSpace, is_namespace
from lightrag.utils import compute_mdhash_id, logger

try:
    from lightrag.constants import GRAPH_FIELD_SEP
except ImportError:  # pragma: no cover
    from lightrag.prompt import GRAPH_FIELD_SEP

from lightrag_agensgraph.kg._base import _AgensStorageBase
from lightrag_agensgraph.kg._sql_templates import (
    VECTOR_CHUNK_TABLE,
    VECTOR_ENTITY_TABLE,
    VECTOR_INDEX_DDL,
    VECTOR_RELATION_TABLE,
    VECTOR_TABLE_DDL,
)

# Per-kind cosine search returning exactly the columns LightRAG expects.
_QUERY_SQL = {
    "entities": """
        SELECT entity_name, EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
        FROM {table}
        WHERE workspace = %(ws)s AND content_vector <=> %(emb)s::vector < %(thr)s
        ORDER BY content_vector <=> %(emb)s::vector
        LIMIT %(k)s
    """,
    "relationships": """
        SELECT source_id AS src_id, target_id AS tgt_id,
               EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
        FROM {table}
        WHERE workspace = %(ws)s AND content_vector <=> %(emb)s::vector < %(thr)s
        ORDER BY content_vector <=> %(emb)s::vector
        LIMIT %(k)s
    """,
    "chunks": """
        SELECT id, content, file_path,
               EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
        FROM {table}
        WHERE workspace = %(ws)s AND content_vector <=> %(emb)s::vector < %(thr)s
        ORDER BY content_vector <=> %(emb)s::vector
        LIMIT %(k)s
    """,
}

_UPSERT_SQL = {
    "entities": """
        INSERT INTO {table}
            (workspace, id, entity_name, content, content_vector, chunk_ids, file_path)
        VALUES (%s, %s, %s, %s, %s::vector, %s::varchar[], %s)
        ON CONFLICT (workspace, id) DO UPDATE SET
            entity_name = EXCLUDED.entity_name, content = EXCLUDED.content,
            content_vector = EXCLUDED.content_vector, chunk_ids = EXCLUDED.chunk_ids,
            file_path = EXCLUDED.file_path, update_time = CURRENT_TIMESTAMP
    """,
    "relationships": """
        INSERT INTO {table}
            (workspace, id, source_id, target_id, content, content_vector, chunk_ids, file_path)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s::varchar[], %s)
        ON CONFLICT (workspace, id) DO UPDATE SET
            source_id = EXCLUDED.source_id, target_id = EXCLUDED.target_id,
            content = EXCLUDED.content, content_vector = EXCLUDED.content_vector,
            chunk_ids = EXCLUDED.chunk_ids, file_path = EXCLUDED.file_path,
            update_time = CURRENT_TIMESTAMP
    """,
    "chunks": """
        INSERT INTO {table}
            (workspace, id, tokens, chunk_order_index, full_doc_id, content,
             content_vector, file_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
        ON CONFLICT (workspace, id) DO UPDATE SET
            tokens = EXCLUDED.tokens, chunk_order_index = EXCLUDED.chunk_order_index,
            full_doc_id = EXCLUDED.full_doc_id, content = EXCLUDED.content,
            content_vector = EXCLUDED.content_vector, file_path = EXCLUDED.file_path,
            update_time = CURRENT_TIMESTAMP
    """,
}


def _vec_literal(vector) -> str:
    """Render an embedding as a pgvector text literal ``[v0,v1,...]``."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


@final
@dataclass
class AgensgraphVectorStorage(_AgensStorageBase, BaseVectorStorage):
    """Vector storage backed by pgvector tables (HNSW cosine) in AgensGraph."""

    def __post_init__(self):
        self.workspace = os.environ.get("AGENSGRAPH_WORKSPACE") or self.workspace or ""
        self._graph_path = None
        self._engine = None
        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            self._kind, self.table = "entities", VECTOR_ENTITY_TABLE
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            self._kind, self.table = "relationships", VECTOR_RELATION_TABLE
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            self._kind, self.table = "chunks", VECTOR_CHUNK_TABLE
        else:
            raise ValueError(f"Unsupported vector namespace: {self.namespace}")
        cfg = (self.global_config or {}).get("vector_db_storage_cls_kwargs", {})
        threshold = cfg.get("cosine_better_than_threshold")
        if threshold is not None:
            self.cosine_better_than_threshold = threshold
        self._max_batch = int((self.global_config or {}).get("embedding_batch_num", 32))

    async def initialize(self):
        await self._acquire_engine()
        dim = int(self.embedding_func.embedding_dim)

        async def _ddl(cur):
            for ddl in VECTOR_TABLE_DDL:
                await cur.execute(ddl.format(dim=dim))
            for ix in VECTOR_INDEX_DDL:
                await cur.execute(ix)

        await self._engine.ensure_relational("vector", _ddl)

    async def finalize(self):
        await self._release_engine()

    async def index_done_callback(self) -> None:
        pass

    def _chunk_ids(self, item: dict) -> List[str]:
        source_id = item.get("source_id")
        if isinstance(source_id, str) and GRAPH_FIELD_SEP in source_id:
            return source_id.split(GRAPH_FIELD_SEP)
        return [source_id] if source_id is not None else []

    async def _embed(self, texts: List[str]) -> List[Any]:
        vectors: List[Any] = []
        for start in range(0, len(texts), self._max_batch):
            batch = texts[start : start + self._max_batch]
            vectors.extend(await self.embedding_func(batch))
        return vectors

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        if not data:
            return
        ids = list(data.keys())
        contents = [data[i]["content"] for i in ids]
        vectors = await self._embed(contents)

        params = []
        for i, id_ in enumerate(ids):
            item = data[id_]
            vec = _vec_literal(vectors[i])
            if self._kind == "entities":
                params.append((
                    self.workspace, id_, item["entity_name"], item["content"],
                    vec, self._chunk_ids(item), item.get("file_path"),
                ))
            elif self._kind == "relationships":
                params.append((
                    self.workspace, id_, item["src_id"], item["tgt_id"],
                    item["content"], vec, self._chunk_ids(item), item.get("file_path"),
                ))
            else:  # chunks
                params.append((
                    self.workspace, id_, item.get("tokens"),
                    item.get("chunk_order_index"), item.get("full_doc_id"),
                    item["content"], vec, item.get("file_path"),
                ))
        await self._executemany(
            _UPSERT_SQL[self._kind].format(table=self.table), params
        )

    async def query(
        self, query: str, top_k: int, query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = (await self.embedding_func([query]))[0]
        return await self._execute(
            _QUERY_SQL[self._kind].format(table=self.table),
            {
                "ws": self.workspace,
                "thr": 1 - self.cosine_better_than_threshold,
                "k": top_k,
                "emb": _vec_literal(embedding),
            },
        )

    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        rows = await self._execute(
            f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at "
            f"FROM {self.table} WHERE workspace = %(ws)s AND id = %(id)s",
            {"ws": self.workspace, "id": id},
        )
        if not rows:
            return None
        row = dict(rows[0])
        row.pop("content_vector", None)  # not JSON-serializable; use get_vectors_by_ids
        return row

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = await self._execute(
            f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at "
            f"FROM {self.table} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {"ws": self.workspace, "ids": list(ids)},
        )
        out = []
        for r in rows:
            r = dict(r)
            r.pop("content_vector", None)
            out.append(r)
        return out

    async def get_vectors_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        if not ids:
            return {}
        rows = await self._execute(
            f"SELECT id, content_vector::text AS vec FROM {self.table} "
            f"WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {"ws": self.workspace, "ids": list(ids)},
        )
        result: Dict[str, List[float]] = {}
        for r in rows:
            raw = r.get("vec")
            if raw:
                result[r["id"]] = [float(x) for x in raw.strip("[]").split(",") if x]
        return result

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._execute(
            f"DELETE FROM {self.table} WHERE workspace = %(ws)s AND id = ANY(%(ids)s)",
            {"ws": self.workspace, "ids": list(ids)},
            fetch=False,
        )

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self._execute(
            f"DELETE FROM {VECTOR_ENTITY_TABLE} "
            f"WHERE workspace = %(ws)s AND (id = %(id)s OR entity_name = %(name)s)",
            {"ws": self.workspace, "id": entity_id, "name": entity_name},
            fetch=False,
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._execute(
            f"DELETE FROM {VECTOR_RELATION_TABLE} "
            f"WHERE workspace = %(ws)s AND (source_id = %(name)s OR target_id = %(name)s)",
            {"ws": self.workspace, "name": entity_name},
            fetch=False,
        )

    async def drop(self) -> Dict[str, str]:
        try:
            await self._execute(
                f"DELETE FROM {self.table} WHERE workspace = %(ws)s",
                {"ws": self.workspace},
                fetch=False,
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error dropping vector namespace {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
