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

import math
import os

import numpy as np
import pytest

import cognee_agensgraph  # noqa: F401  (registers the adapters)

TEST_URL = os.environ.get("AGENSGRAPH_TEST_URL")

requires_agens = pytest.mark.skipif(
    not TEST_URL,
    reason="Set AGENSGRAPH_TEST_URL to a libpq connection string to run.",
)

EMBED_DIM = 8


class FakeEmbeddingEngine:
    """Deterministic bag-of-chars embedding, no network."""

    def get_vector_size(self) -> int:
        return EMBED_DIM

    async def embed_text(self, texts):
        out = []
        for t in texts:
            v = [0.0] * EMBED_DIM
            for ch in t or "":
                v[ord(ch) % EMBED_DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return np.array(out, dtype=float)


@pytest.fixture
def conn_url():
    return TEST_URL


@pytest.fixture
def embedding_engine():
    return FakeEmbeddingEngine()
