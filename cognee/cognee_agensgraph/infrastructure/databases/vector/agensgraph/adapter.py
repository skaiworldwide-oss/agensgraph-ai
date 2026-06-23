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

"""AgensGraph (pgvector) vector adapter for Cognee."""

import asyncio
from typing import List, Optional
from uuid import UUID

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint

from ...graph.agensgraph._engine import AgensEngine

CHUNK_SIZE = 1000


def _to_uuid(value):
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return value


def _vec_literal(vector) -> str:
    """Render an embedding as a pgvector text literal ``[v0,v1,...]``."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


class IndexSchema(DataPoint):
    """A minimal embeddable data point (id + text) used for field-level indexing."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


class AgensgraphVectorAdapter(VectorDBInterface):
    """
    Cognee vector storage backed by pgvector tables in AgensGraph.

    Each collection is a relational table ``(id, payload, vector)`` with an HNSW
    (``vector_cosine_ops``) index; ``vector`` is typed ``VECTOR(dim)`` so the
    query's ``<=>`` cast matches the index expression and the index is used at
    scale. Shares the same async engine/pool as the graph adapter.
    """

    def __init__(self, url: str, api_key: Optional[str] = None, embedding_engine=None):
        self.conninfo = url
        self.embedding_engine = embedding_engine
        self._engine: Optional[AgensEngine] = None

    async def _ensure_engine(self) -> AgensEngine:
        if self._engine is None:
            self._engine = await AgensEngine.acquire(self.conninfo)
        return self._engine

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        engine = await self._ensure_engine()
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                # Exact-case match: collection names are created as quoted
                # (case-sensitive) identifiers, which to_regclass would fold.
                await cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = %s LIMIT 1",
                    (collection_name,),
                )
                return await cur.fetchone() is not None

    async def create_collection(self, collection_name: str, payload_schema=None):
        engine = await self._ensure_engine()
        dim = int(self.embedding_engine.get_vector_size())
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {t} "
                        "(id TEXT PRIMARY KEY, payload JSONB, vector VECTOR({d}))"
                    ).format(t=sql.Identifier(collection_name), d=sql.SQL(str(dim)))
                )
                await cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {ix} ON {t} "
                        "USING hnsw (vector vector_cosine_ops)"
                    ).format(
                        ix=sql.Identifier(f"{collection_name}_hnsw"),
                        t=sql.Identifier(collection_name),
                    )
                )
            await conn.commit()

    async def create_data_points(
        self, collection_name: str, data_points: List[DataPoint]
    ):
        if not data_points:
            return
        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, type(data_points[0]))

        vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(dp) for dp in data_points]
        )
        params = [
            (str(dp.id), Jsonb(dp.model_dump(mode="json")), _vec_literal(vectors[i]))
            for i, dp in enumerate(data_points)
        ]
        engine = await self._ensure_engine()
        query = sql.SQL(
            "INSERT INTO {t} (id, payload, vector) VALUES (%s, %s, %s::vector) "
            "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, vector = EXCLUDED.vector"
        ).format(t=sql.Identifier(collection_name))
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                for start in range(0, len(params), CHUNK_SIZE):
                    await cur.executemany(query, params[start : start + CHUNK_SIZE])
            await conn.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the collection cognee indexes a DataPoint field into."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        """Embed + store the indexable field of each data point in its collection."""
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(id=dp.id, text=DataPoint.get_embeddable_data(dp))
                for dp in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        if not data_point_ids or not await self.has_collection(collection_name):
            return []
        engine = await self._ensure_engine()
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    sql.SQL("SELECT id, payload FROM {t} WHERE id = ANY(%(ids)s)").format(
                        t=sql.Identifier(collection_name)
                    ),
                    {"ids": [str(i) for i in data_point_ids]},
                )
                rows = await cur.fetchall()
        return [
            ScoredResult(id=_to_uuid(r["id"]), payload=r["payload"], score=0)
            for r in rows
        ]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        if not await self.has_collection(collection_name):
            return []
        if query_text is not None and query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]
        if query_vector is None:
            return []

        engine = await self._ensure_engine()
        limit_clause = sql.SQL("LIMIT {n}").format(n=sql.SQL(str(int(limit)))) if limit and limit > 0 else sql.SQL("")
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    sql.SQL(
                        "SELECT id, payload, vector <=> %(q)s::vector AS distance {vec} "
                        "FROM {t} ORDER BY vector <=> %(q)s::vector {lim}"
                    ).format(
                        t=sql.Identifier(collection_name),
                        vec=sql.SQL(", vector::text AS vector") if with_vector else sql.SQL(""),
                        lim=limit_clause,
                    ),
                    {"q": _vec_literal(query_vector)},
                )
                rows = await cur.fetchall()
        results = []
        for r in rows:
            payload = r["payload"]
            if with_vector and r.get("vector"):
                payload = {**(payload or {}), "vector": r["vector"]}
            results.append(
                ScoredResult(id=_to_uuid(r["id"]), score=float(r["distance"]), payload=payload)
            )
        return results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        query_vectors = await self.embed_data(query_texts)
        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=qv,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for qv in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: List[str]):
        if not data_point_ids or not await self.has_collection(collection_name):
            return
        engine = await self._ensure_engine()
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("DELETE FROM {t} WHERE id = ANY(%(ids)s)").format(
                        t=sql.Identifier(collection_name)
                    ),
                    {"ids": [str(i) for i in data_point_ids]},
                )
            await conn.commit()

    async def prune(self):
        # Drop every collection this adapter created. They are uniquely
        # identified by their (payload jsonb + vector) column signature, so this
        # never touches the graph tables or unrelated user tables.
        engine = await self._ensure_engine()
        async with engine.aconnection(graph_path=None) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT c_vec.table_name
                    FROM information_schema.columns c_vec
                    JOIN information_schema.columns c_pl
                      ON c_vec.table_schema = c_pl.table_schema
                     AND c_vec.table_name = c_pl.table_name
                    WHERE c_vec.table_schema = 'public'
                      AND c_vec.column_name = 'vector' AND c_vec.udt_name = 'vector'
                      AND c_pl.column_name = 'payload'
                    """
                )
                tables = [r[0] for r in await cur.fetchall()]
                for table in tables:
                    await cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {t} CASCADE").format(
                            t=sql.Identifier(table)
                        )
                    )
            await conn.commit()
