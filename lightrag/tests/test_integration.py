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

import numpy as np
import pytest
import pytest_asyncio

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from conftest import EMBED_DIM, requires_agens, _embed_one

pytestmark = [requires_agens, pytest.mark.asyncio]

CUSTOM_KG = {
    "entities": [
        {"entity_name": "CompanyA", "entity_type": "Organization",
         "description": "A technology company", "source_id": "Source1"},
        {"entity_name": "ProductX", "entity_type": "Product",
         "description": "A product by CompanyA", "source_id": "Source1"},
        {"entity_name": "PersonA", "entity_type": "Person",
         "description": "An AI researcher", "source_id": "Source2"},
    ],
    "relationships": [
        {"src_id": "CompanyA", "tgt_id": "ProductX",
         "description": "CompanyA develops ProductX", "keywords": "develop",
         "weight": 1.0, "source_id": "Source1"},
    ],
    "chunks": [
        {"content": "ProductX, developed by CompanyA, revolutionized the market.",
         "source_id": "Source1", "source_chunk_index": 0},
        {"content": "PersonA is a prominent AI researcher.",
         "source_id": "Source2", "source_chunk_index": 0},
    ],
}


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    return "ok"


async def _embed(texts, **kwargs):
    return np.array([_embed_one(t) for t in texts], dtype=float)


@pytest_asyncio.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM, max_token_size=8192, func=_embed
        ),
        graph_storage="AgensgraphStorage",
        vector_storage="AgensgraphVectorStorage",
        kv_storage="AgensgraphKVStorage",
        doc_status_storage="AgensgraphDocStatusStorage",
        workspace="itest",
    )
    await instance.initialize_storages()
    await initialize_pipeline_status()
    try:
        yield instance
    finally:
        for store in (
            instance.chunk_entity_relation_graph,
            instance.entities_vdb,
            instance.relationships_vdb,
            instance.chunks_vdb,
            instance.full_docs,
            instance.text_chunks,
            instance.doc_status,
            instance.llm_response_cache,
        ):
            try:
                await store.drop()
            except Exception:
                pass
        await instance.finalize_storages()


async def test_all_four_stores_are_agensgraph(rag):
    assert type(rag.chunk_entity_relation_graph).__name__ == "AgensgraphStorage"
    assert type(rag.entities_vdb).__name__ == "AgensgraphVectorStorage"
    assert type(rag.full_docs).__name__ == "AgensgraphKVStorage"
    assert type(rag.doc_status).__name__ == "AgensgraphDocStatusStorage"


async def test_custom_kg_lands_in_all_stores(rag):
    await rag.ainsert_custom_kg(CUSTOM_KG)

    # graph
    assert await rag.chunk_entity_relation_graph.has_node("CompanyA") is True
    assert await rag.chunk_entity_relation_graph.has_edge("CompanyA", "ProductX") is True

    # entity vector store
    hits = await rag.entities_vdb.query("CompanyA", top_k=5)
    assert any(h.get("entity_name") == "CompanyA" for h in hits)

    # chunk vector store has the ingested chunks
    chunk_hits = await rag.chunks_vdb.query("ProductX market", top_k=5)
    assert len(chunk_hits) >= 1
