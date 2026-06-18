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

import lightrag_agensgraph  # noqa: F401  (registers the storages)
from lightrag.utils import EmbeddingFunc

AGENS_AVAILABLE = bool(
    os.environ.get("AGENSGRAPH_DB")
    and os.environ.get("AGENSGRAPH_USER")
    and os.environ.get("AGENSGRAPH_PASSWORD")
)

requires_agens = pytest.mark.skipif(
    not AGENS_AVAILABLE,
    reason="Requires AGENSGRAPH_DB / AGENSGRAPH_USER / AGENSGRAPH_PASSWORD env vars.",
)

EMBED_DIM = 8


def _embed_one(text: str):
    """Deterministic bag-of-chars embedding, L2-normalized."""
    v = [0.0] * EMBED_DIM
    for ch in text or "":
        v[ord(ch) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def embedding_func():
    async def _embed(texts, **kwargs):
        return np.array([_embed_one(t) for t in texts], dtype=float)

    return EmbeddingFunc(embedding_dim=EMBED_DIM, max_token_size=8192, func=_embed)
