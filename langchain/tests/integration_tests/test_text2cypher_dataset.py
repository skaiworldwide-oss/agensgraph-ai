"""Every text2cypher dataset entry, proven against the server. No model.

A gold query runs as a model's statement does, through ``query_untrusted`` as ``llm_ro``, and
returns what the entry says it returns. A recorded habit errors, is refused or disagrees with
gold, or answers by reading the whole label, as the entry declares. An entry a release cannot
hold names the capability it needs in ``requires``.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.errors import AgensQueryError, UnsafeCypherError
from langchain_agensgraph.graphs import AgensGraph
from langchain_agensgraph.roles import render_grants

EVAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "evals" / "text2cypher"
sys.path.insert(0, str(EVAL_DIR))

import t2c  # noqa: E402

pytestmark = pytest.mark.server

ENTRIES = [e for e in t2c.load_dataset() if e["graph"] in t2c.FIXTURE_GRAPHS]
REFUSED = (UnsafeCypherError, AgensQueryError)


@pytest.fixture(scope="module")
def graphs(dsn: str) -> Iterator[dict[str, AgensGraph]]:
    try:
        with AgensEngine(dsn) as owner:
            for name in t2c.FIXTURE_GRAPHS:
                t2c.BUILDERS[name](AgensGraph(owner, graph_name=name, create_graph=True))
        with psycopg.connect(dsn, autocommit=True) as conn:
            for name in t2c.FIXTURE_GRAPHS:
                conn.execute(render_grants(name))
        untrusted = make_conninfo(dsn, user="llm_ro", password="agens")
        with AgensEngine(dsn, untrusted_conninfo=untrusted) as engine:
            yield {name: AgensGraph(engine, graph_name=name) for name in t2c.FIXTURE_GRAPHS}
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            for name in t2c.FIXTURE_GRAPHS:
                conn.execute(
                    sql.SQL("DROP GRAPH IF EXISTS {} CASCADE").format(sql.Identifier(name))
                )
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(f"{name}_store")
                    )
                )


def run(graph: AgensGraph, cypher: str) -> list[dict[str, Any]]:
    return graph.query_untrusted(cypher, max_rows=11, timeout=30)


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["id"] for e in ENTRIES])
def test_the_entry_holds(entry: dict[str, Any], graphs: dict[str, AgensGraph], dsn: str) -> None:
    graph = graphs[entry["graph"]]
    unmet = t2c.requires_met(entry, graph.capabilities)
    if unmet:
        pytest.skip(f"the release lacks {unmet}")
    gold = run(graph, entry["gold"])
    if entry.get("expect_empty"):
        assert gold == [], "gold was declared empty but returned rows"
    else:
        assert gold, "gold returned no rows"
    assert len(gold) <= 10, "answers are designed to fit within a LIMIT 10"
    if entry.get("expect_index"):
        assert t2c.plan_has_index_path(dsn, entry["graph"], entry["gold"]), (
            "gold was declared index-served but the planner finds no index path"
        )
    habit = entry.get("habit")
    if habit is None:
        return
    ordered = entry.get("ordered", False)
    if entry["habit_outcome"] == "error":
        with pytest.raises(REFUSED):
            run(graph, habit)
    elif entry["habit_outcome"] == "unindexed":
        assert t2c.results_match(run(graph, habit), gold, ordered), (
            "an unindexed habit must return the same rows as gold"
        )
        assert not t2c.plan_has_index_path(dsn, entry["graph"], habit), (
            "the habit was declared unindexed but an index path exists"
        )
    else:
        try:
            rows = run(graph, habit)
        except REFUSED:
            return
        assert not t2c.results_match(rows, gold, ordered), (
            "the habit query agrees with gold; this trap is not real"
        )
