"""Every text2cypher dataset entry, proven against a live server. No model.

The dataset's value is that its gold queries are dialect-correct and its traps
are real. Both claims are executable, so this test executes them: every gold
runs read-only and returns what the entry says it returns, and every recorded
habit either errors or disagrees with gold — whichever the entry declares. An
engine change that invalidates an entry fails here, naming the entry.
"""

import os
import pathlib
import sys

import pytest

EVAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "evals" / "text2cypher"
sys.path.insert(0, str(EVAL_DIR))

import t2c  # noqa: E402

from langchain_agensgraph.graphs.agensgraph import AgensGraph  # noqa: E402

conf = {
    "dbname": os.getenv("AGENSGRAPH_DB"),
    "user": os.getenv("AGENSGRAPH_USER"),
    "password": os.getenv("AGENSGRAPH_PASSWORD"),
    "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
    "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
}

ENTRIES = [e for e in t2c.load_dataset() if e["graph"] in t2c.FIXTURE_GRAPHS]


@pytest.fixture(scope="module")
def graphs():
    built = {}
    for name in t2c.FIXTURE_GRAPHS:
        graph = AgensGraph(name, conf, create=True, refresh_schema=False)
        t2c.BUILDERS[name](graph)
        built[name] = graph
    yield built
    for graph in built.values():
        graph.close()


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["id"] for e in ENTRIES])
def test_the_entry_holds(entry, graphs) -> None:
    graph = graphs[entry["graph"]]
    unmet = t2c.requires_met(entry, graph.capabilities)
    if unmet:
        pytest.skip(f"server lacks {unmet}")

    with graph.read_only(allow_server_programs=True):
        gold_rows = graph.query(entry["gold"], timeout=30)
    if entry.get("expect_empty"):
        assert gold_rows == [], "gold was declared empty but returned rows"
    else:
        assert gold_rows, "gold returned no rows"
    assert len(gold_rows) <= 10, "answers are designed to fit within a LIMIT 10"

    habit = entry.get("habit")
    if habit is None:
        return
    if entry["habit_outcome"] == "error":
        with pytest.raises(Exception):
            with graph.read_only(allow_server_programs=True):
                graph.query(habit, timeout=30)
    else:
        with graph.read_only(allow_server_programs=True):
            habit_rows = graph.query(habit, timeout=30)
        assert not t2c.results_match(
            habit_rows, gold_rows, entry.get("ordered", False)
        ), "the habit query agrees with gold; this trap is not real"
