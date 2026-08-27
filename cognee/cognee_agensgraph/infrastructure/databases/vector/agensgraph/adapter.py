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
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from agensgraph import RetryPolicy, Vector
from cognee.infrastructure.databases.vector.exceptions.exceptions import (
    CollectionNotFoundError,
)
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint
from psycopg import errors, sql
from psycopg.rows import tuple_row
from psycopg.types.json import Jsonb

from ...graph.agensgraph._engine import AgensEngine, run_with_retry

# Rows per insert statement.
CHUNK_SIZE = 1000

# The HNSW index returns at most hnsw.ef_search candidates, and this is its default. A
# search for more rows than that raises the setting for its transaction, or the index
# would silently return fewer rows than asked for.
DEFAULT_EF_SEARCH = 40

# Memory for building an index in one go. Building the index for 13,264 rows of 1,536
# dimensions took 70.8 s with the server default of 64 MB and 27.9 s with this.
BUILD_WORK_MEM = "1GB"


def _to_uuid(value):
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return value


def _vector(values) -> Vector:
    return Vector(values.tolist() if hasattr(values, "tolist") else values)


class IndexSchema(DataPoint):
    """A minimal embeddable data point (id + text) used for field-level indexing."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


class AgensgraphVectorAdapter(VectorDBInterface):
    """
    Cognee vector storage backed by pgvector tables in AgensGraph.

    Each collection is a table ``(id TEXT PRIMARY KEY, payload JSONB, vector VECTOR(dim))``
    with an HNSW cosine index. Embeddings travel in binary. A search for the nearest
    rows runs with sequential scans off for its transaction: the planner cannot see that
    a 1536-dimension vector is stored out of line, so it prices the scan below the index.
    A search for every row (``limit=0``) keeps the scan, which is right for it. Shares
    the engine, and so the pool, with the graph adapter.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        embedding_engine=None,
        *,
        retry_attempts: int = 6,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ):
        self.conninfo = url
        self.embedding_engine = embedding_engine
        self._engine: Optional[AgensEngine] = None
        # Collections known to exist. Only positive answers are kept: another process
        # may create a collection at any time.
        self._known: Set[str] = set()
        self.retry_policy = RetryPolicy(attempts=retry_attempts)
        # HNSW build parameters. Inserting into the index is the cost of a vector write:
        # 7 ms per 1,536-dimension row against 0.07 ms without the index, and 4.9 ms with
        # ef_construction = 32, which finds fewer of the true neighbours.
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self._bulk = False
        self._bulk_touched: Set[str] = set()

    async def _ensure_engine(self) -> AgensEngine:
        if self._engine is None:
            self._engine = AgensEngine.get(self.conninfo)
        return self._engine

    async def _ensure_vectors(self) -> AgensEngine:
        """The engine, with pgvector installed and registered on every connection."""
        engine = await self._ensure_engine()
        if not engine.vectors:
            await engine.enable_vectors()
        return engine

    async def _run(self, work, *, wrote: bool = False):
        engine = await self._ensure_vectors()

        async def attempt():
            async with engine.connection() as conn:
                return await work(conn)

        return await run_with_retry(self.retry_policy, attempt, wrote=wrote)

    async def _on_collection(self, collection_name: str, work, *, wrote: bool = False):
        """Run ``work``; a missing table is reported as a missing collection."""
        try:
            return await self._run(work, wrote=wrote)
        except errors.UndefinedTable:
            self._known.discard(collection_name)
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found") from None

    @staticmethod
    def _table(collection_name: str) -> sql.Identifier:
        return sql.Identifier(collection_name)

    # ---- collections ----

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        if collection_name in self._known:
            return True

        async def work(conn):
            cur = await conn.execute(
                "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = 'public' AND c.relname = %s AND c.relkind = 'r'",
                (collection_name,),
            )
            return await cur.fetchone() is not None

        found = await self._run(work)
        if found:
            self._known.add(collection_name)
        return found

    async def create_collection(self, collection_name: str, payload_schema=None):
        dim = int(self.embedding_engine.get_vector_size())
        table = self._table(collection_name)

        async def work(conn):
            try:
                await conn.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS {t} "
                        "(id TEXT PRIMARY KEY, payload JSONB, vector VECTOR({d}))"
                    ).format(t=table, d=sql.SQL(str(dim)))
                )
            except (errors.DuplicateTable, errors.DuplicateObject, errors.UniqueViolation):
                pass  # another process created it first
            if not self._bulk:
                await self._create_index(conn, collection_name)

        await self._run(work, wrote=True)
        self._known.add(collection_name)
        if self._bulk:
            self._bulk_touched.add(collection_name)

    async def _create_index(self, conn, collection_name: str, *, build_work_mem: Optional[str] = None) -> None:
        statement = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {ix} ON {t} USING hnsw (vector vector_cosine_ops) "
            "WITH (m = {m}, ef_construction = {ef})"
        ).format(
            ix=sql.Identifier(f"{collection_name}_hnsw"),
            t=self._table(collection_name),
            m=sql.Literal(self.hnsw_m),
            ef=sql.Literal(self.hnsw_ef_construction),
        )
        try:
            if build_work_mem:
                async with conn.transaction():
                    await conn.execute(
                        sql.SQL("SET LOCAL maintenance_work_mem = {}").format(sql.Literal(build_work_mem))
                    )
                    await conn.execute(statement)
            else:
                await conn.execute(statement)
        except (errors.DuplicateTable, errors.DuplicateObject):
            pass

    @asynccontextmanager
    async def bulk_ingest(self):
        """Write many rows, then build the indexes once.

        Inserting into an HNSW index costs about 7 ms per 1,536-dimension row; inserting
        without one costs 0.07 ms, and building the index afterwards for 13,264 rows took
        27.9 s. Inside this block the indexes of every collection written are dropped
        first and built again on the way out, with more memory for the build. Searches
        inside the block on those collections read every row.

        Use around a load, for example ``async with vector_engine.bulk_ingest(): await
        cognee.cognify(...)``.
        """
        self._bulk = True
        self._bulk_touched = set()
        try:
            yield self
        finally:
            self._bulk = False
            touched = sorted(self._bulk_touched)
            self._bulk_touched = set()
            if touched:
                async def work(conn):
                    for name in touched:
                        await self._create_index(conn, name, build_work_mem=BUILD_WORK_MEM)

                await self._run(work, wrote=True)

    async def _drop_index_for_bulk(self, conn, collection_name: str) -> None:
        await conn.execute(
            sql.SQL("DROP INDEX IF EXISTS {ix}").format(ix=sql.Identifier(f"{collection_name}_hnsw"))
        )

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the collection cognee indexes a DataPoint field into."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    # ---- writes ----

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        if not data_points:
            return
        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, type(data_points[0]))

        vectors = await self.embed_data([DataPoint.get_embeddable_data(dp) for dp in data_points])
        # One row per id; a later point with the same id wins, as it would in sequence.
        rows: Dict[str, Any] = {}
        for dp, vector in zip(data_points, vectors):
            rows[str(dp.id)] = (Jsonb(dp.model_dump(mode="json")), _vector(vector))
        ids = list(rows)
        # %b for the vectors: the driver sends a Vector in binary only, and for a list the
        # automatic placeholder chooses text, which has no dumper.
        statement = sql.SQL(
            "INSERT INTO {t} (id, payload, vector) "
            "SELECT * FROM unnest(%(ids)s::text[], %(payloads)s::jsonb[], %(vectors)b::vector[]) "
            "ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload, vector = EXCLUDED.vector"
        ).format(t=self._table(collection_name))

        async def work(conn):
            if self._bulk and collection_name not in self._bulk_touched:
                await self._drop_index_for_bulk(conn, collection_name)
                self._bulk_touched.add(collection_name)
            for start in range(0, len(ids), CHUNK_SIZE):
                chunk = ids[start : start + CHUNK_SIZE]
                await conn.execute(
                    statement,
                    {
                        "ids": chunk,
                        "payloads": [rows[i][0] for i in chunk],
                        "vectors": [rows[i][1] for i in chunk],
                    },
                )

        await self._on_collection(collection_name, work, wrote=True)

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        """Embed + store the indexable field of each data point in its collection."""
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [IndexSchema(id=dp.id, text=DataPoint.get_embeddable_data(dp)) for dp in data_points],
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: List[str]):
        if not data_point_ids:
            return

        async def work(conn):
            await conn.execute(
                sql.SQL("DELETE FROM {t} WHERE id = ANY(%(ids)s)").format(t=self._table(collection_name)),
                {"ids": [str(i) for i in data_point_ids]},
            )

        try:
            await self._on_collection(collection_name, work, wrote=True)
        except CollectionNotFoundError:
            return

    async def prune(self):
        """Drop every collection this adapter created.

        They are the tables with a ``payload`` column and a ``vector`` column of type
        vector, so the graph's tables and unrelated tables are never touched.
        """

        async def work(conn):
            cur = await conn.execute(
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
                await conn.execute(sql.SQL("DROP TABLE IF EXISTS {t} CASCADE").format(t=sql.Identifier(table)))

        await self._run(work, wrote=True)
        self._known.clear()

    # ---- reads ----

    async def _rows(self, collection_name: str, statement, params, *, top_k: int = 0):
        """Rows of one statement on a collection.

        ``top_k > 0`` marks a search for the nearest rows, which is run with sequential
        scans off and, when more rows are asked for than the index returns by default,
        with the index told to return that many.
        """

        async def work(conn):
            async with conn.cursor(row_factory=tuple_row) as cur:
                if top_k <= 0:
                    await cur.execute(statement, params)
                    return await cur.fetchall()
                async with conn.transaction():
                    async with conn.pipeline():
                        await conn.execute("SET LOCAL enable_seqscan = off")
                        if top_k > DEFAULT_EF_SEARCH:
                            await conn.vector_search_options({"hnsw.ef_search": top_k})
                        await cur.execute(statement, params)
                    return await cur.fetchall()

        return await self._on_collection(collection_name, work)

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        if not data_point_ids:
            return []
        rows = await self._rows(
            collection_name,
            sql.SQL("SELECT id, payload FROM {t} WHERE id = ANY(%(ids)s)").format(
                t=self._table(collection_name)
            ),
            {"ids": [str(i) for i in data_point_ids]},
        )
        return [ScoredResult(id=_to_uuid(i), payload=p, score=0) for i, p in rows]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        """The rows nearest to the query, closest first.

        ``score`` is the cosine distance, 0 for an identical vector. ``limit=0`` (or
        None) returns every row, which is how cognee reads a whole collection.
        """
        if query_text is not None and query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]
        if query_vector is None:
            return []
        top_k = int(limit) if limit and limit > 0 else 0
        columns = sql.SQL("id, payload, vector <=> %(q)s AS distance")
        if with_vector:
            columns = columns + sql.SQL(", vector")
        statement = sql.SQL("SELECT {cols} FROM {t} ORDER BY vector <=> %(q)s {lim}").format(
            cols=columns,
            t=self._table(collection_name),
            lim=sql.SQL("LIMIT {}").format(sql.Literal(top_k)) if top_k else sql.SQL(""),
        )
        rows = await self._rows(collection_name, statement, {"q": _vector(query_vector)}, top_k=top_k)
        results = []
        for row in rows:
            payload = row[1]
            if with_vector:
                payload = {**(payload or {}), "vector": list(row[3])}
            results.append(ScoredResult(id=_to_uuid(row[0]), score=float(row[2]), payload=payload))
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
