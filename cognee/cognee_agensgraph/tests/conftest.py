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
from typing import List

import agensgraph
import numpy as np
import pytest

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


async def explain(conn, statement: str, params=None) -> str:
    """The plan the server picks with its default settings.

    Not ``enable_seqscan = off``: that only proves an index exists, not that the
    planner chooses it.
    """
    async with conn.cursor() as cur:
        await cur.execute("EXPLAIN " + statement, params)
        rows = await cur.fetchall()
    return "\n".join(r[0] for r in rows)
