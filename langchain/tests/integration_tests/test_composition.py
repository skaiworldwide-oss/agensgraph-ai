"""What happens when a caller puts these pieces together.

Each of these is about two things used *together* -- two threads, a component inside a
caller's transaction, a setting and the statement it is meant to govern. A suite that
exercises one thing at a time cannot see any of them.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from langchain_core.messages import HumanMessage

from langchain_agensgraph.chat_message_histories.agensgraph import (
    AgensChatMessageHistory,
)
from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.graphs.agensgraph import AgensGraph

GRAPH = "composition_it"


def _conf() -> dict:
    return {
        "host": os.environ.get("AGENSGRAPH_HOST", "127.0.0.1"),
        "port": int(os.environ.get("AGENSGRAPH_PORT", "5830")),
        "user": os.environ.get("AGENSGRAPH_USER", "agens"),
        "dbname": os.environ.get("AGENSGRAPH_DB", "postgres"),
    }


def _clear(g: AgensGraph) -> None:
    """Everything these tests write. Sessions included: a run that leaves duplicates
    behind makes the next run's unique index impossible, and the failure lands somewhere
    unrelated."""
    g.query("MATCH (n:T) DETACH DELETE n")
    for label in ("Message", "Session", "Ghost"):
        g.query(f'MATCH (n:"{label}") DETACH DELETE n')


@pytest.fixture
def graph():
    g = AgensGraph(GRAPH, _conf(), create=True, refresh_schema=False)
    g.query("CREATE VLABEL IF NOT EXISTS T")
    for label in ("Message", "Session", "Ghost"):
        g.query(f'CREATE VLABEL IF NOT EXISTS "{label}"')
    _clear(g)
    yield g
    _clear(g)
    g.close()


@pytest.fixture
def pooled():
    engine = AgensEngine.from_conf(_conf(), min_size=1, max_size=3)
    g = AgensGraph(GRAPH, _conf(), create=True, refresh_schema=False, engine=engine)
    g.query("CREATE VLABEL IF NOT EXISTS T")
    g.query("MATCH (n:T) DETACH DELETE n")
    yield g
    g.query("MATCH (n:T) DETACH DELETE n")
    g.close()
    engine.close()


class TestConcurrentWritesOnOneHandle:
    """One connection does one thing at a time.

    Without a pool every caller shares the graph's own connection. Unless it is held, two
    of them opening a transaction on it interleave their BEGIN and COMMIT, and a caller is
    told its write failed while the row lands anyway.
    """

    THREADS, PER = 6, 20

    def test_every_write_lands_and_is_reported_as_landing(self, graph: AgensGraph):
        failures: list = []
        lock = threading.Lock()

        def work(t: int) -> None:
            for i in range(self.PER):
                try:
                    with graph.transaction():
                        graph.query("CREATE (:T {k: %(k)s})", {"k": f"{t}-{i}"})
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        failures.append(exc)

        threads = [
            threading.Thread(target=work, args=(t,)) for t in range(self.THREADS)
        ]
        for one in threads:
            one.start()
        for one in threads:
            one.join()

        assert failures == []
        landed = graph.query("MATCH (n:T) RETURN count(n) AS c")[0]["c"]
        assert int(landed) == self.THREADS * self.PER


class TestATransactionInsideATransaction:
    """A component writing inside a caller's transaction joins it.

    Taking a second connection instead is what lets an inner write outlive the rollback
    that was supposed to undo it -- and with a pool of one, deadlock against itself.
    """

    def test_a_nested_write_commits_with_the_outer_one(self, graph: AgensGraph):
        with graph.transaction():
            with graph.transaction():
                graph.query("CREATE (:T {k: 'nested'})")
        assert int(graph.query("MATCH (n:T {k: 'nested'}) RETURN count(n) AS c")[0]["c"]) == 1

    def test_a_nested_write_is_undone_by_the_outer_rollback(self, graph: AgensGraph):
        with pytest.raises(RuntimeError):
            with graph.transaction():
                with graph.transaction():
                    graph.query("CREATE (:T {k: 'rolled'})")
                raise RuntimeError("undo")
        assert int(graph.query("MATCH (n:T {k: 'rolled'}) RETURN count(n) AS c")[0]["c"]) == 0

    def test_the_same_holds_through_a_pool(self, pooled: AgensGraph):
        with pytest.raises(RuntimeError):
            with pooled.transaction():
                pooled.query("CREATE (:T {k: 'pooled'})")
                raise RuntimeError("undo")
        assert int(pooled.query("MATCH (n:T {k: 'pooled'}) RETURN count(n) AS c")[0]["c"]) == 0

    def test_a_nested_write_through_a_pool_is_undone_too(self, pooled: AgensGraph):
        """The pool is where a second connection is there to be taken.

        An inner block that borrows one of its own commits somewhere the enclosing
        rollback cannot reach, so the write survives being undone -- and with a pool of one
        there is no second connection to take, so it waits for itself until the pool gives
        up.
        """
        with pytest.raises(RuntimeError):
            with pooled.transaction():
                with pooled.transaction():
                    pooled.query("CREATE (:T {k: 'pooled-nested'})")
                raise RuntimeError("undo")
        found = pooled.query(
            "MATCH (n:T {k: 'pooled-nested'}) RETURN count(n) AS c"
        )[0]["c"]
        assert int(found) == 0

    def test_a_component_write_inside_a_pooled_transaction(self, pooled: AgensGraph):
        """The shape the package documents: a shared graph, a component, a transaction."""
        with pytest.raises(RuntimeError):
            with pooled.transaction():
                AgensChatMessageHistory(
                    session_id="in-a-transaction", graph=pooled
                ).add_messages([HumanMessage(content="undone")])
                raise RuntimeError("undo")
        found = pooled.query(
            'MATCH (m:"Message") WHERE m.session = %(s)s RETURN count(m) AS c',
            {"s": "in-a-transaction"},
        )[0]["c"]
        assert int(found) == 0

    @staticmethod
    def _linked_document(tag: str):
        from langchain_core.documents import Document

        from langchain_agensgraph.graphs.graph_document import (
            GraphDocument,
            Node,
            Relationship,
        )

        one = Node(id=f"{tag}-a", type="Ghost")
        two = Node(id=f"{tag}-b", type="Ghost")
        return GraphDocument(
            nodes=[one, two],
            relationships=[Relationship(source=one, target=two, type="HAUNTS")],
            source=Document(page_content="x"),
        )

    def test_an_ingest_joins_the_transaction_around_it(self, graph: AgensGraph):
        with pytest.raises(RuntimeError):
            with graph.transaction():
                graph.add_graph_documents([self._linked_document("undone")])
                raise RuntimeError("undo")
        found = graph.query('MATCH (n:"Ghost") RETURN count(n) AS c')[0]["c"]
        assert int(found) == 0, "an ingest inside a transaction is undone with it"

    @pytest.mark.parametrize("shape", ["shared", "pooled"])
    def test_an_ingest_writes_its_edges(self, request, shape: str):
        """The endpoints an edge merges on are the batch's own uncommitted vertices.

        Sent on any other connection the match finds nothing, and a MERGE that matches
        nothing creates nothing and reports success -- so an ingest wrote every vertex and
        not one edge, silently. Both shapes, because only one of them was wrong.
        """
        g: AgensGraph = request.getfixturevalue(
            "graph" if shape == "shared" else "pooled"
        )
        g.query('MATCH (n:"Ghost") DETACH DELETE n')
        g.add_graph_documents([self._linked_document(shape)], include_source=True)

        vertices = g.query('MATCH (n:"Ghost") RETURN count(n) AS c')[0]["c"]
        edges = g.query('MATCH (:"Ghost")-[r:"HAUNTS"]->(:"Ghost") RETURN count(r) AS c')[0]["c"]
        mentions = g.query('MATCH ()-[r:"MENTIONS"]->(:"Ghost") RETURN count(r) AS c')[0]["c"]
        g.query('MATCH (n:"Ghost") DETACH DELETE n')
        assert int(vertices) == 2
        assert int(edges) == 1, "the edge between the batch's own vertices"
        assert int(mentions) >= 1, "the edges to the source document"

    def test_read_only_inside_a_transaction_still_refuses_a_write(
        self, graph: AgensGraph
    ):
        with graph.transaction():
            with pytest.raises(Exception):
                with graph.read_only(allow_server_programs=True):
                    graph.query("CREATE (:T {k: 'refused'})")
            # and the transaction around it is still usable
            graph.query("CREATE (:T {k: 'after'})")
        assert int(graph.query("MATCH (n:T {k: 'after'}) RETURN count(n) AS c")[0]["c"]) == 1


class TestTheTimeoutHoldsInsideABlock:
    """A statement timeout is about the statement, not about which connection it is on.

    Applied only to the graph's own connection, a statement inside a transaction on a
    pooled one got none at all -- and generated Cypher runs in exactly such a block.
    """

    SLOW = "SELECT pg_sleep(2)"

    @pytest.mark.parametrize("shape", ["plain", "transaction", "read_only"])
    def test_a_slow_statement_is_refused(self, pooled: AgensGraph, shape: str):
        started = time.perf_counter()
        with pytest.raises(Exception):
            if shape == "plain":
                pooled.query(self.SLOW, timeout=0.3)
            elif shape == "transaction":
                with pooled.transaction():
                    pooled.query(self.SLOW, timeout=0.3)
            else:
                with pooled.read_only(allow_server_programs=True):
                    pooled.query(self.SLOW, timeout=0.3)
        assert time.perf_counter() - started < 1.5, "it waited the whole two seconds"

    def test_the_transaction_does_not_leave_its_timeout_behind(
        self, pooled: AgensGraph
    ):
        """Stated for the transaction, so the connection goes back as it was found."""
        with pooled.transaction():
            pooled.query("MATCH (n:T) RETURN count(n)", timeout=0.3)
        held = pooled.query("SHOW statement_timeout")[0]
        assert str(next(iter(held.values()))) in ("0", "0ms"), held


class TestASearchOptionReachesTheSearch:
    """An option set on one connection governs nothing that runs on another.

    The store set `hnsw.ef_search` on the graph's own connection and then searched through
    the pool, so the search that was meant to look further looked no further at all. The
    rows still come back, so only the backend running each statement shows it.
    """

    def test_the_option_and_the_search_are_on_one_connection(self, pooled: AgensGraph):
        from agensgraph.observability import add_query_logger, remove_query_logger

        seen: list = []

        def watch(record) -> None:
            text = str(record.statement or "")
            if "ef_search" in text or "pg_sleep" in text:
                seen.append(record.connection)

        add_query_logger(watch)
        try:
            with pooled.transaction() as conn:
                conn.vector_search_options({"hnsw.ef_search": 80})
                pooled.query("SELECT pg_sleep(0)")
        finally:
            remove_query_logger(watch)

        assert len(seen) == 2, seen
        assert seen[0] == seen[1], "the option landed on a different connection"


class TestAnIndexDroppedBehindTheAnswer:
    """Remembering an index was made is not the same as it being there."""

    def test_it_is_made_again(self, graph: AgensGraph):
        AgensChatMessageHistory(session_id="a", graph=graph)
        graph.query('DROP PROPERTY INDEX IF EXISTS "Session_id_idx"')
        assert not [i for i in graph.connection.indexes("Session") if i.unique]
        AgensChatMessageHistory(session_id="b", graph=graph)
        assert [i for i in graph.connection.indexes("Session") if i.unique]

    def test_many_at_once_do_not_refuse_each_other(self, graph: AgensGraph):
        """Each looks, each finds it missing, and only one can create it."""
        graph.query('DROP PROPERTY INDEX IF EXISTS "Session_id_idx"')
        failures: list = []
        lock = threading.Lock()

        def build(i: int) -> None:
            try:
                own = AgensGraph(GRAPH, _conf(), create=True, refresh_schema=False)
                AgensChatMessageHistory(session_id=f"race{i}", graph=own)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failures.append(exc)

        threads = [threading.Thread(target=build, args=(i,)) for i in range(8)]
        for one in threads:
            one.start()
        for one in threads:
            one.join()
        assert failures == []
        assert [i for i in graph.connection.indexes("Session") if i.unique]


class TestSharedComponentsUnderConcurrency:
    """A chain or a history handed to more than one caller at a time."""

    def test_two_questions_do_not_share_one_budget(self, graph: AgensGraph):
        from langchain_agensgraph.chains.cypher_qa import AgensCypherQAChain

        class Unused:
            def invoke(self, *a, **k):  # pragma: no cover
                raise AssertionError

        chain = AgensCypherQAChain(
            graph=graph, cypher_llm=Unused(), qa_llm=Unused(), timeout=30.0
        )
        seen: dict = {}

        # The second question has to start while the first is still going, and the first
        # has to finish before the second reads. Held on the chain, the second overwrites
        # the first's deadline on the way in and the first clears it on the way out,
        # leaving the second to find nothing and believe it has its whole budget.
        def ask(name: str, wait: float) -> None:
            with chain._budget():
                time.sleep(wait)
                seen[name] = chain._remaining

        first = threading.Thread(target=ask, args=("early", 0.5))
        second = threading.Thread(target=ask, args=("late", 1.0))
        first.start()
        time.sleep(0.1)
        second.start()
        first.join()
        second.join()

        # Each has spent what it has actually been running, and no more.
        assert 29.3 < seen["early"] < 29.7, seen
        assert 28.7 < seen["late"] < 29.1, seen

    def test_writers_sharing_keys_neither_lose_nor_duplicate(self, graph: AgensGraph):
        graph.query('MATCH (n:"Message") DETACH DELETE n')
        graph.query('MATCH (n:"Session") DETACH DELETE n')
        writers, per, keys = 6, 15, 10
        refused: list = []
        lock = threading.Lock()

        def work(w: int) -> None:
            own = AgensGraph(GRAPH, _conf(), create=True, refresh_schema=False)
            for i in range(per):
                history = AgensChatMessageHistory(
                    session_id=f"k{(w * per + i) % keys}", graph=own
                )
                try:
                    history.add_messages([HumanMessage(content=f"{w}-{i}")])
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        refused.append(exc)

        threads = [threading.Thread(target=work, args=(w,)) for w in range(writers)]
        for one in threads:
            one.start()
        for one in threads:
            one.join()

        sessions = int(graph.query('MATCH (s:"Session") RETURN count(s) AS c')[0]["c"])
        messages = int(graph.query('MATCH (m:"Message") RETURN count(m) AS c')[0]["c"])
        assert refused == []
        assert sessions == keys, "one session per key, not one per writer that raced"
        assert messages == writers * per
