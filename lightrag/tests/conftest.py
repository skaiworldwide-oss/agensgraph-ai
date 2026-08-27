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

"""Test fixtures. Every test needs a server: a run without one refuses to start.

    AGENSGRAPH_DB=... AGENSGRAPH_USER=... AGENSGRAPH_PASSWORD=... [AGENSGRAPH_HOST/PORT] pytest
"""

import hashlib
import os
from typing import List

import numpy as np
import pytest
from agensgraph import QueryRecord, add_query_logger, remove_query_logger
from lightrag.utils import EmbeddingFunc

import lightrag_agensgraph  # noqa: F401  (registers the storages)

if not (os.environ.get("AGENSGRAPH_DB") and os.environ.get("AGENSGRAPH_USER")):
    pytest.exit(
        "the suite needs a server: set AGENSGRAPH_DB, AGENSGRAPH_USER and AGENSGRAPH_PASSWORD "
        "(AGENSGRAPH_HOST/AGENSGRAPH_PORT default to localhost:5432)",
        returncode=4,
    )
os.environ.setdefault("AGENSGRAPH_PASSWORD", "")

# The width of a real embedding, so a row's vector is stored out of line as it would be.
EMBED_DIM = 1536


def embed_one(text: str) -> List[float]:
    """A deterministic unit vector for a text in which texts sharing words are near.

    Each word and each character trigram lands in a hashed dimension, so a query
    that mentions an entity's name is close to that entity's record and far from
    an unrelated one, which is what the searches in these tests rely on.
    """
    v = np.zeros(EMBED_DIM)
    text = (text or "").lower()
    pieces = text.split() + [text[i : i + 3] for i in range(max(len(text) - 2, 0))]
    for piece in pieces:
        digest = hashlib.md5(piece.encode()).digest()
        v[int.from_bytes(digest[:4], "big") % EMBED_DIM] += 1.0
    norm = np.linalg.norm(v) or 1.0
    return (v / norm).tolist()


@pytest.fixture
def embedding_func():
    async def _embed(texts, **kwargs):
        return np.array([embed_one(t) for t in texts], dtype=float)

    return EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=_embed)


class StatementCounter:
    """Every statement the driver sends while the fixture is active."""

    def __init__(self) -> None:
        self.records: List[QueryRecord] = []

    def __call__(self, record: QueryRecord) -> None:
        self.records.append(record)

    def reset(self) -> None:
        self.records.clear()

    @property
    def statements(self) -> List[str]:
        return [r.statement for r in self.records]

    def __len__(self) -> int:
        return len(self.records)


@pytest.fixture
def statements():
    counter = StatementCounter()
    add_query_logger(counter)
    try:
        yield counter
    finally:
        remove_query_logger(counter)


async def explain(conn, statement: str, params=None) -> str:
    """The plan a statement gets under the settings the connection has, rolled back."""
    async with conn.transaction(force_rollback=True):
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN (ANALYZE, COSTS OFF) " + statement, params)
            return "\n".join(row[0] for row in await cur.fetchall())
