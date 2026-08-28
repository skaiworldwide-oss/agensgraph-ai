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

import hashlib
import math
import os
import re
from typing import List

import agensgraph
import numpy as np
import pytest
import pytest_asyncio

import cognee_agensgraph  # noqa: F401  (registers the adapters)

TEST_URL = os.environ.get("AGENSGRAPH_TEST_URL")

if not TEST_URL:
    # Every test here talks to a server. Skipping them all would report a green run
    # that tested nothing, so the run stops instead.
    pytest.exit(
        "Set AGENSGRAPH_TEST_URL to a libpq connection string, for example "
        "postgresql://user@127.0.0.1:55432/cognee_test",
        returncode=4,
    )

# Kept for tests that still mark themselves with it; it never skips now.
requires_agens = pytest.mark.skipif(not TEST_URL, reason="AGENSGRAPH_TEST_URL is not set")

EMBED_DIM = 8


class FakeEmbeddingEngine:
    """Deterministic embeddings, no network.

    ``dim=8`` is a bag of characters, which keeps identical texts identical and lets a
    test reason about nearest neighbours. A larger ``dim`` (1536, the size of a real
    embedding) gives pseudo-random unit vectors keyed by the text, so the rows are as
    wide as production rows and are stored out of line the way real ones are.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def get_vector_size(self) -> int:
        return self.dim

    async def embed_text(self, texts: List[str]):
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            if self.dim == EMBED_DIM:
                for ch in t or "":
                    out[i, ord(ch) % self.dim] += 1.0
            else:
                seed = int.from_bytes(
                    hashlib.blake2b((t or "").encode(), digest_size=8).digest(), "little"
                )
                out[i] = np.random.default_rng(seed).standard_normal(self.dim)
            norm = math.sqrt(float((out[i] * out[i]).sum())) or 1.0
            out[i] /= norm
        return out


class StatementCounter:
    """Counts statements the driver sends, through its query logger."""

    def __init__(self):
        self.statements: List[str] = []

    def __call__(self, record: agensgraph.QueryRecord) -> None:
        self.statements.append(record.statement)

    def __len__(self) -> int:
        return len(self.statements)

    def reset(self) -> None:
        self.statements.clear()


@pytest.fixture
def conn_url():
    return TEST_URL


@pytest.fixture
def embedding_engine():
    return FakeEmbeddingEngine()


@pytest.fixture
def wide_embedding_engine():
    return FakeEmbeddingEngine(dim=1536)


@pytest.fixture
def statements():
    counter = StatementCounter()
    agensgraph.add_query_logger(counter)
    try:
        yield counter
    finally:
        agensgraph.remove_query_logger(counter)


_SEQ_SCAN = re.compile(r"Seq Scan on (\S+).*?\(actual rows=(\d+)")


async def explain(conn, statement: str, params=None, *, by_index: bool = False) -> str:
    """The plan the server picks and what each step really did.

    Read with the planner's default settings, unless ``by_index`` asks for the form the
    adapter uses for a statement that matches nodes from a bound list. Runs inside a
    transaction that is rolled back, so a write can be explained too.
    """
    async with conn.transaction(force_rollback=True):
        if by_index:
            await conn.execute("SET LOCAL enable_seqscan = off")
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN (ANALYZE, TIMING OFF) " + statement, params)
            rows = await cur.fetchall()
    return "\n".join(r[0] for r in rows)


def scanned_tables(plan: str) -> List[str]:
    """The relations a plan read sequentially and found rows in.

    An empty label is scanned sequentially because it has no pages to speak of; that is
    not the scan a test is looking for.
    """
    return [table for table, rows in _SEQ_SCAN.findall(plan) if int(rows) > 0]


@pytest_asyncio.fixture(autouse=True)
async def _close_engine_pools():
    """Close each engine's pool for this test's event loop before the loop is torn down.

    pytest-asyncio gives every test its own event loop. The engine keeps one pool per
    loop, and a loop closed while its pool is open hangs in asyncio's task cancellation,
    waiting on a pool worker's network read that never returns. Production runs one loop
    for the whole process, so this is a test-lifecycle concern, not a runtime one.
    """
    yield
    from cognee_agensgraph.infrastructure.databases.graph.agensgraph import _engine

    for engine in list(_engine._ENGINES.values()):
        try:
            await engine.aclose()
        except Exception:
            pass
