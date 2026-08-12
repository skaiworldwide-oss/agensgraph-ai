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


def test_a_build_parameter_for_the_other_access_method_is_refused():
    """Dropping it builds an index with the defaults while the caller believes they
    tuned it, and the difference only shows up as recall they cannot explain."""
    import pytest

    cfg = IndexConfig(am=VectorIndexAM.IVFFLAT, m=16, ef_construction=64, lists=10)
    with pytest.raises(ValueError, match="IVFFlat takes `lists`"):
        cfg.with_options_clause()

    cfg = IndexConfig(am=VectorIndexAM.HNSW, m=16, lists=10)
    with pytest.raises(ValueError, match="HNSW takes `m` and `ef_construction`"):
        cfg.with_options_clause()


def test_each_method_still_takes_its_own():
    assert IndexConfig(
        am=VectorIndexAM.HNSW, m=16, ef_construction=64
    ).with_options_clause() == " WITH (m = 16, ef_construction = 64)"
    assert (
        IndexConfig(am=VectorIndexAM.IVFFLAT, lists=10).with_options_clause()
        == " WITH (lists = 10)"
    )


def test_hybrid_defaults():
    h = HybridSearchConfig()
    # `fusion` is gone: it was declared, defaulted to "rrf", and read nowhere -- the
    # hybrid query does reciprocal rank fusion unconditionally.
    assert not hasattr(h, "fusion")
    assert h.rank_constant == 60
    assert h.vector_weight == 1.0 and h.keyword_weight == 1.0
