"""Integration tests for 0.3.0 connection lifecycle + ergonomics:
close(), context managers, timeout, sanitize, application_name.
"""

from __future__ import annotations

import os

import pytest

from langchain_agensgraph.graphs.agensgraph import AgensGraph, AgensQueryException
from langchain_agensgraph.vectorstores.agensgraph_vector import AgensgraphVector
from tests.integration_tests.fake_embeddings import FakeEmbeddings


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


def test_close_is_idempotent():
    g = AgensGraph("lifecycle", _conf(), create=True)
    g.close()
    assert g.connection.closed
    g.close()  # second call must not raise


def test_sync_context_manager_closes():
    with AgensGraph("lifecycle", _conf(), create=True) as g:
        assert g.query("MATCH (n) RETURN count(n) AS c")[0]["c"] >= 0
    assert g.connection.closed


def test_timeout_raises_on_slow_query():
    g = AgensGraph("lifecycle", _conf(), create=True, timeout=0.001)
    with pytest.raises(AgensQueryException):
        # pg_sleep(2s) under a 1ms statement_timeout must abort
        g.query("MATCH (n) WHERE pg_sleep(2) IS NULL RETURN n")
    g.close()


def test_per_call_timeout_overrides_instance():
    g = AgensGraph("lifecycle", _conf(), create=True)  # no instance timeout
    with pytest.raises(AgensQueryException):
        g.query("MATCH (n) WHERE pg_sleep(2) IS NULL RETURN n", timeout=0.001)
    g.close()


def test_application_name_is_set():
    g = AgensGraph("lifecycle", _conf(), create=True)
    rows = g.query(
        "SELECT current_setting('application_name') AS app"
    )
    assert rows[0]["app"].startswith("langchain-agensgraph/")
    g.close()


def test_application_name_not_overridden():
    conf = _conf()
    conf["application_name"] = "my-app"
    g = AgensGraph("lifecycle", conf, create=True)
    rows = g.query("SELECT current_setting('application_name') AS app")
    assert rows[0]["app"] == "my-app"
    g.close()


def test_sanitize_strips_large_lists():
    g = AgensGraph("lifecycle", _conf(), create=True, sanitize=True)
    g.query("MATCH (n) DETACH DELETE n")
    g.query(
        "CREATE (n:SanTest {small: %(small)s, big: %(big)s, name: 'x'})",
        params={"small": list(range(3)), "big": list(range(200))},
    )
    rows = g.query("MATCH (n:SanTest) RETURN properties(n) AS p")
    p = rows[0]["p"]
    assert p.get("small") == [0, 1, 2]
    assert "big" not in p or p["big"] is None
    assert p.get("name") == "x"
    g.query("MATCH (n) DETACH DELETE n")
    g.close()


def test_sanitize_off_keeps_large_lists():
    g = AgensGraph("lifecycle", _conf(), create=True)  # sanitize defaults off
    g.query("MATCH (n) DETACH DELETE n")
    g.query(
        "CREATE (n:SanTest2 {big: %(big)s})",
        params={"big": list(range(200))},
    )
    rows = g.query("MATCH (n:SanTest2) RETURN properties(n) AS p")
    assert len(rows[0]["p"]["big"]) == 200
    g.query("MATCH (n) DETACH DELETE n")
    g.close()


def test_vector_store_context_manager():
    url = os.environ.get("AGENSGRAPH_URL")
    with AgensgraphVector.from_texts(
        ["a", "b"],
        embedding=FakeEmbeddings(),
        graph_name="lifecycle",
        url=url,
        node_label="LifeChunk",
        pre_delete_collection=True,
    ) as store:
        assert store.similarity_search("a", k=1)
    assert store.connection.closed
