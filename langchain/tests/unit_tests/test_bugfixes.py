"""Regression tests for v0.2.0 Phase 2 fixes."""

from __future__ import annotations

import logging

import pytest

from langchain_agensgraph.graphs.agensgraph import AgensGraph
from langchain_agensgraph.vectorstores.agensgraph_vector import (
    DEFAULT_VECTOR_INDEX_AM,
    VectorIndexAM,
)


def test_ivfflat_enum_value_is_pgvector_name() -> None:
    # Old enum value was "IVFLLAT" (extra L) which pgvector rejects in DDL.
    assert VectorIndexAM.IVFFLAT.value == "ivfflat"
    assert VectorIndexAM.HNSW.value == "HNSW"
    # Default remains HNSW; the bugfix is just that the IVFFLAT option now works.
    assert DEFAULT_VECTOR_INDEX_AM is VectorIndexAM.HNSW


def test_format_properties_escapes_single_quotes() -> None:
    # Bug: f"'{v}'" with no escape — values containing apostrophes broke the
    # Cypher map literal. Fix uses backslash-escape.
    out = AgensGraph._format_properties({"name": "O'Brien"})
    assert out == "{\"name\": 'O\\'Brien'}"


def test_format_properties_escapes_backslash() -> None:
    out = AgensGraph._format_properties({"path": "c:\\users\\x"})
    assert out == "{\"path\": 'c:\\\\users\\\\x'}"


def test_format_properties_keeps_non_string_unchanged() -> None:
    # Behavior parity with prior test_format_properties test.
    out = AgensGraph._format_properties({"a": "b", "c": 1, "d": True})
    assert out == "{\"a\": 'b', \"c\": 1, \"d\": True}"


def test_get_triples_str_removed() -> None:
    # Dead method that called _get_triples with mismatched signature.
    assert not hasattr(AgensGraph, "_get_triples_str")


def test_no_debug_prints_in_vectorstore_module(caplog: pytest.LogCaptureFixture) -> None:
    # Bug: `print("DEBUG: ...")` at two callsites — replaced by self.logger.debug.
    import langchain_agensgraph.vectorstores.agensgraph_vector as mod
    src = open(mod.__file__).read()
    assert 'print("DEBUG:' not in src
    assert "print('DEBUG:" not in src


def test_refresh_schema_accepts_force_and_ttl() -> None:
    import inspect

    sig = inspect.signature(AgensGraph.refresh_schema)
    assert "force" in sig.parameters
    init_sig = inspect.signature(AgensGraph.__init__)
    assert "schema_cache_ttl" in init_sig.parameters


def test_triple_query_unwinds_all_labels() -> None:
    # The pre-0.2.0 query took start[0] (only the first label of a multi-label
    # node). The fix uses UNWIND on both start and end label lists.
    from langchain_agensgraph.graphs import agensgraph as mod

    src = open(mod.__file__).read()
    assert "UNWIND startlbls" in src
    assert "UNWIND endlbls" in src
    assert "start[0]" not in src


def test_property_type_detection_filters_nulls() -> None:
    from langchain_agensgraph.graphs import agensgraph as mod

    src = open(mod.__file__).read()
    # Both property-aggregation queries WHERE-filter NULL values before typeof.
    # Count >= 2 (node + edge queries).
    assert src.count("WHERE value IS NOT NULL") >= 2


def test_verify_vector_support_error_mentions_pg_config() -> None:
    # The message should point the operator at the pgvector source build step
    # because AgensGraph does not bundle the pgvector control file.
    from langchain_agensgraph.vectorstores import agensgraph_vector as mod
    src = open(mod.__file__).read()
    # Two anchors in the new message
    assert "pgvector" in src
    assert "pg_config" in src
