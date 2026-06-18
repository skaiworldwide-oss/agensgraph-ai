"""Unit tests for IndexConfig / HybridSearchConfig (no DB)."""

from __future__ import annotations

from langchain_agensgraph.vectorstores.agensgraph_vector import (
    HybridSearchConfig,
    IndexConfig,
    VectorIndexAM,
)


def test_hnsw_with_options_clause():
    cfg = IndexConfig(am=VectorIndexAM.HNSW, m=16, ef_construction=64)
    assert cfg.with_options_clause() == " WITH (m = 16, ef_construction = 64)"


def test_hnsw_partial_options():
    cfg = IndexConfig(am=VectorIndexAM.HNSW, m=8)
    assert cfg.with_options_clause() == " WITH (m = 8)"


def test_hnsw_no_options_is_empty():
    assert IndexConfig(am=VectorIndexAM.HNSW).with_options_clause() == ""


def test_ivfflat_with_lists():
    cfg = IndexConfig(am=VectorIndexAM.IVFFLAT, lists=100)
    assert cfg.with_options_clause() == " WITH (lists = 100)"


def test_ivfflat_ignores_hnsw_opts():
    # m/ef_construction don't apply to ivfflat and must be ignored.
    cfg = IndexConfig(am=VectorIndexAM.IVFFLAT, m=16, ef_construction=64, lists=10)
    assert cfg.with_options_clause() == " WITH (lists = 10)"


def test_hybrid_defaults():
    h = HybridSearchConfig()
    assert h.fusion == "rrf"
    assert h.rank_constant == 60
    assert h.vector_weight == 1.0 and h.keyword_weight == 1.0
