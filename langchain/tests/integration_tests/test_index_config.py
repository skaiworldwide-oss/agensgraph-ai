"""Integration tests for typed IndexConfig + HybridSearchConfig."""

from __future__ import annotations

import os

import pytest

from langchain_agensgraph.vectorstores.agensgraph_vector import (
    AgensgraphVector,
    HybridSearchConfig,
    IndexConfig,
    SearchType,
    VectorIndexAM,
)
from tests.integration_tests.fake_embeddings import FakeEmbeddings

URL = os.environ.get("AGENSGRAPH_URL")


def _drop_vector_indexes(store: AgensgraphVector) -> None:
    for idx in store.query("SELECT name FROM ag_list_vector_indexes()"):
        store.query(f'''DROP PROPERTY INDEX "{idx['name']}" CASCADE''')
    store.query("MATCH (n) DETACH DELETE n")


def test_create_hnsw_index_with_build_params():
    store = AgensgraphVector.from_texts(
        ["a", "b", "c", "d"],
        embedding=FakeEmbeddings(),
        url=URL,
        graph_name="cfgtest",
        node_label="HnswCfg",
        index_name="hnsw_cfg_idx",
        pre_delete_collection=True,
    )
    # Replace the auto-created index with one carrying explicit HNSW params.
    _drop_vector_indexes(store)
    # re-seed since drop cleared data
    store.add_texts(["a", "b", "c", "d"])
    store.create_new_index(
        index_config=IndexConfig(am=VectorIndexAM.HNSW, m=8, ef_construction=32)
    )
    # Index is present and usable.
    names = [i["name"] for i in store.query("SELECT name FROM ag_list_vector_indexes()")]
    assert "hnsw_cfg_idx" in names
    assert store.similarity_search("a", k=2)
    store.close()


def test_create_ivfflat_index_with_lists():
    store = AgensgraphVector.from_texts(
        ["a", "b", "c", "d", "e", "f"],
        embedding=FakeEmbeddings(),
        url=URL,
        graph_name="cfgtest",
        node_label="IvfCfg",
        index_name="ivf_cfg_idx",
        pre_delete_collection=True,
    )
    _drop_vector_indexes(store)
    store.add_texts(["a", "b", "c", "d", "e", "f"])
    store.create_new_index(
        index_config=IndexConfig(am=VectorIndexAM.IVFFLAT, lists=1)
    )
    assert store.similarity_search("a", k=2)
    store.close()


def test_hybrid_custom_rank_constant_runs():
    store = AgensgraphVector.from_texts(
        ["the quick brown fox", "lazy dog", "fox and hound", "cat nap"],
        embedding=FakeEmbeddings(),
        url=URL,
        graph_name="cfgtest",
        node_label="HybridCfg",
        index_name="hybrid_cfg_idx",
        keyword_index_name="hybrid_cfg_kw",
        search_type=SearchType.HYBRID,
        pre_delete_collection=True,
    )
    default_hits = store.similarity_search("fox", k=3)
    tuned_hits = store.similarity_search(
        "fox", k=3, hybrid_config=HybridSearchConfig(rank_constant=1, keyword_weight=5.0)
    )
    assert default_hits and tuned_hits
    assert len(tuned_hits) <= 3
    store.close()
