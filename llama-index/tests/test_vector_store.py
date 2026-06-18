'''
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
'''

import os

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

agens_db = os.environ.get("AGENS_DB")
agens_user = os.environ.get("AGENS_USER")
agens_password = os.environ.get("AGENS_PASSWORD")
agens_host = os.environ.get("AGENS_HOST") or "localhost"
agens_port = os.environ.get("AGENS_PORT") or 5432

agens_available = bool(agens_db and agens_user and agens_password)

requires_agens = pytest.mark.skipif(
    not agens_available,
    reason="Requires AGENS_DB, AGENS_USER and AGENS_PASSWORD environment variables.",
)


def _url() -> str:
    return (
        f"postgresql://{agens_user}:{agens_password}"
        f"@{agens_host}:{agens_port}/{agens_db}"
    )


def test_class():
    names_of_base_classes = [b.__name__ for b in AgensgraphVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
    assert "client" not in AgensgraphVectorStore.__abstractmethods__


@pytest.fixture()
def vector_store() -> AgensgraphVectorStore:
    if not agens_available:
        pytest.skip("No agensgraph credentials provided")
    store = AgensgraphVectorStore(
        url=_url(),
        embedding_dimension=4,
        graph_name="test_vec_store",
        node_label="Chunk",
    )
    store.database_query('MATCH (n) DETACH DELETE n')
    return store


def _nodes():
    return [
        TextNode(
            text="alpha document", embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"topic": "a"},
        ),
        TextNode(
            text="beta document", embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={"topic": "b"},
        ),
        TextNode(
            text="gamma document", embedding=[0.0, 0.0, 1.0, 0.0],
            metadata={"topic": "a"},
        ),
    ]


@requires_agens
def test_add_and_query(vector_store: AgensgraphVectorStore):
    ids = vector_store.add(_nodes())
    assert len(ids) == 3

    res = vector_store.query(
        VectorStoreQuery(query_embedding=[0.95, 0.0, 0.0, 0.0], similarity_top_k=1)
    )
    assert len(res.nodes) == 1
    assert "alpha" in res.nodes[0].get_content()


@requires_agens
def test_query_metadata_filter(vector_store: AgensgraphVectorStore):
    vector_store.add(_nodes())
    filters = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="a", operator=FilterOperator.EQ)]
    )
    res = vector_store.query(
        VectorStoreQuery(
            query_embedding=[0.0, 0.0, 1.0, 0.0],
            similarity_top_k=5,
            filters=filters,
        )
    )
    # Only the two topic="a" nodes should be returned.
    assert len(res.nodes) == 2
    assert all("document" in n.get_content() for n in res.nodes)


@requires_agens
def test_delete(vector_store: AgensgraphVectorStore):
    nodes = _nodes()
    # Give them a shared ref_doc_id so delete() can target them.
    for n in nodes:
        n.relationships = {}
    ids = vector_store.add(nodes)
    assert len(ids) == 3
    # Deleting a non-existent ref doc id is a no-op and must not error.
    vector_store.delete("does-not-exist")
    remaining = vector_store.database_query('MATCH (n:"Chunk") RETURN count(n) AS c')
    assert remaining[0]["c"] == 3


@requires_agens
def test_id_index_created(vector_store: AgensgraphVectorStore):
    """The btree index on the MERGE key (`id`) must exist after construction."""
    rows = vector_store.database_query(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'test_vec_store' AND tablename = 'Chunk'"
    )
    names = {r["indexname"] for r in rows}
    assert "Chunk_id_idx" in names


@requires_agens
@pytest.mark.asyncio
async def test_async_add_and_query(vector_store: AgensgraphVectorStore):
    ids = await vector_store.async_add(_nodes())
    assert len(ids) == 3
    res = await vector_store.aquery(
        VectorStoreQuery(query_embedding=[0.0, 0.95, 0.0, 0.0], similarity_top_k=1)
    )
    assert len(res.nodes) == 1
    assert "beta" in res.nodes[0].get_content()


@requires_agens
def test_engine_pooling(vector_store: AgensgraphVectorStore):
    engine = AgensEngine.from_url(_url(), min_size=1, max_size=4)
    try:
        store = AgensgraphVectorStore(
            url=_url(),
            embedding_dimension=4,
            graph_name="test_vec_pool",
            node_label="Chunk",
            engine=engine,
        )
        store.database_query('MATCH (n) DETACH DELETE n')
        store.add(_nodes())
        res = store.query(
            VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=1)
        )
        assert "alpha" in res.nodes[0].get_content()
    finally:
        engine.close()
