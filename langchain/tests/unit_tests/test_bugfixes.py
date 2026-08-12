"""Regression tests for behaviour that needs no server to check.

The assertions here are about what the code does, not about the text of the statements it
sends: the schema is read from the catalogs and the wire format is decoded by the driver,
so there is no query text to match against.
"""

from __future__ import annotations

import inspect

import pytest

from langchain_agensgraph.graphs.agensgraph import (
    AgensGraph,
    AgensQueryException,
    _sanitize_value,
)
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


def test_query_exception_keeps_the_servers_own_account() -> None:
    # Every raise site writes "detail", so that is the spelling the class has to read for
    # the server's own account of a failure to survive.
    exc = AgensQueryException({"message": "boom", "detail": "the real cause"})
    assert exc.get_details() == "the real cause"
    assert exc.get_message() == "boom"
    # And the message reaches `str()`, which needs __init__ to call super().
    assert str(exc) == "boom"


def test_query_exception_still_reads_the_other_spelling() -> None:
    exc = AgensQueryException({"message": "boom", "details": "the real cause"})
    assert exc.get_details() == "the real cause"


def test_one_query_exception_class_for_the_package() -> None:
    # Two byte-identical copies would mean `except` on one does not catch the other, so
    # both modules must name the same object.
    from langchain_agensgraph.vectorstores import agensgraph_vector as vec

    assert vec.AgensQueryException is AgensQueryException


def test_refresh_schema_accepts_force_and_ttl() -> None:
    sig = inspect.signature(AgensGraph.refresh_schema)
    assert "force" in sig.parameters
    init_sig = inspect.signature(AgensGraph.__init__)
    assert "schema_cache_ttl" in init_sig.parameters


def test_no_debug_prints_in_vectorstore_module() -> None:
    # Bug: `print("DEBUG: ...")` at two callsites — replaced by self.logger.debug.
    import langchain_agensgraph.vectorstores.agensgraph_vector as mod

    src = open(mod.__file__).read()
    assert 'print("DEBUG:' not in src
    assert "print('DEBUG:" not in src


def test_nothing_installs_a_plpgsql_typeof() -> None:
    # Three of the integrations this package sits beside create a plpgsql `typeof(jsonb)`
    # in the caller's database to name a JSON type. `jsonb_typeof` is built in, and the
    # driver's describe() uses it, so nothing here should carry that DDL any more.
    from langchain_agensgraph.graphs import agensgraph as mod

    src = open(mod.__file__).read()
    assert "CREATE OR REPLACE FUNCTION" not in src
    assert "LANGUAGE plpgsql" not in src


def test_the_regex_wire_decoder_is_gone() -> None:
    # The decoder dropped the vertex label, the vertex id and the edge's whole property
    # map, and could not match a label containing a space. The driver decodes the wire now.
    from langchain_agensgraph.graphs import agensgraph as graphs
    from langchain_agensgraph.vectorstores import agensgraph_vector as vec

    for mod in (graphs, vec):
        src = open(mod.__file__).read()
        assert "vertex_regex" not in src, mod.__name__
        assert "edge_regex" not in src, mod.__name__
        assert "_record_to_dict" not in src, mod.__name__


def test_verify_vector_support_error_mentions_pg_config() -> None:
    # The message should point the operator at the pgvector source build step
    # because AgensGraph does not bundle the pgvector control file.
    from langchain_agensgraph.vectorstores import agensgraph_vector as mod

    src = open(mod.__file__).read()
    assert "pgvector" in src
    assert "pg_config" in src


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}),
        # An oversized list takes its key with it, rather than leaving a null behind.
        ({"a": list(range(200))}, {}),
        ({"a": 1, "b": "x", "c": None}, {"a": 1, "b": "x", "c": None}),
        ({"a": {"b": list(range(200))}}, {"a": {}}),
    ],
)
def test_sanitize_drops_only_oversized_lists(value: dict, expected: dict) -> None:
    assert _sanitize_value(value) == expected


def test_sanitize_describes_a_graph_element_rather_than_walking_past_it() -> None:
    # A vertex is not a dict, so the previous implementation returned it untouched and its
    # embedding reached the prompt anyway. It is described first, then sanitized.
    from agensgraph import Vertex
    from agensgraph._protocol.graphid import GraphId

    vertex = Vertex(GraphId(3, 1), "Chunk", {"text": "hi", "embedding": list(range(200))})
    out = _sanitize_value(vertex)
    assert out["label"] == "Chunk"
    assert out["id"] == "3.1"
    assert out["properties"]["text"] == "hi"
    assert "embedding" not in out["properties"]


class TestSanitizeSeesAVector:
    """The value most likely to flood a prompt is the one the length rule cannot see.

    A vector is not a ``list`` -- ``isinstance`` says so -- though it compares equal to
    one, so a guard written against ``list`` let 1,536 numbers through untouched.
    """

    def test_a_wide_vector_is_dropped(self):
        from agensgraph import Vector

        assert _sanitize_value({"embedding": Vector([1.0] * 1536)}) == {}

    def test_a_narrow_one_is_kept_as_its_numbers(self):
        from agensgraph import Vector

        assert _sanitize_value({"embedding": Vector([1.0, 2.0])}) == {
            "embedding": [1.0, 2.0]
        }

    def test_a_sparse_vector_is_measured_by_what_it_holds(self):
        from agensgraph import SparseVector

        held = SparseVector({1: 1.0, 7: 2.0}, 4096)
        assert _sanitize_value({"embedding": held}) == {"embedding": {1: 1.0, 7: 2.0}}

        crowded = SparseVector({i: 1.0 for i in range(200)}, 4096)
        assert _sanitize_value({"embedding": crowded}) == {}

    def test_a_vector_is_not_a_list(self):
        """The premise, so the test above cannot pass for the wrong reason."""
        from agensgraph import Vector

        held = Vector([1.0] * 1536)
        assert not isinstance(held, list)
        assert held == [1.0] * 1536


class TestSeeingWhatWasSent:
    """The usual question about a slow chain is which part of it was slow."""

    def test_a_block_collects_the_statements_it_ran(self):
        import os

        from langchain_agensgraph import AgensGraph, log_queries

        conf = {
            "dbname": os.getenv("AGENSGRAPH_DB"),
            "user": os.getenv("AGENSGRAPH_USER"),
            "password": os.getenv("AGENSGRAPH_PASSWORD"),
            "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
            "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
        }
        graph = AgensGraph("observe_it", conf, create=True, refresh_schema=False)
        try:
            with log_queries() as statements:
                graph.query("RETURN 1 AS x")
                graph.query("RETURN 2 AS x")
            assert len(statements) == 2
            assert all(one.elapsed is not None for one in statements)

            # and it stops collecting once the block is done
            before = len(statements)
            graph.query("RETURN 3 AS x")
            assert len(statements) == before
        finally:
            graph.close()

    def test_each_one_is_handed_over_as_it_finishes(self):
        import os

        from langchain_agensgraph import AgensGraph, log_queries

        conf = {
            "dbname": os.getenv("AGENSGRAPH_DB"),
            "user": os.getenv("AGENSGRAPH_USER"),
            "password": os.getenv("AGENSGRAPH_PASSWORD"),
            "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
            "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
        }
        graph = AgensGraph("observe_it", conf, create=True, refresh_schema=False)
        seen: list = []
        try:
            with log_queries(seen.append):
                graph.query("RETURN 1 AS x")
            assert len(seen) == 1
        finally:
            graph.close()


class TestANameThatWouldNotArriveIntactIsRefused:
    """The server's lexer stops at a null byte.

    So a label holding one is composed into a statement that ends somewhere other than
    where it appears to: ``Ses\0sion`` is written as ``"Ses"``, and the server is asked
    about a different label than the caller named.
    """

    def test_a_null_byte_in_a_label_is_refused(self):
        import os

        import pytest

        from langchain_agensgraph import AgensChatMessageHistory, AgensGraph

        conf = {
            "dbname": os.getenv("AGENSGRAPH_DB"),
            "user": os.getenv("AGENSGRAPH_USER"),
            "password": os.getenv("AGENSGRAPH_PASSWORD"),
            "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
            "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
        }
        graph = AgensGraph("nullname_it", conf, create=True, refresh_schema=False)
        try:
            with pytest.raises(ValueError, match="null byte"):
                AgensChatMessageHistory(
                    session_id="s", graph=graph, session_node_label="Ses\x00sion"
                )
        finally:
            graph.close()

    def test_an_ordinary_name_is_accepted(self):
        from langchain_agensgraph.graphs.agensgraph import checked_names

        checked_names(label="Session", key="firstName", other="has space")
