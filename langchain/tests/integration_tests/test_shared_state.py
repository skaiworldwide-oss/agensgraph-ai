"""What one caller's transaction and timeout must not do to another's.

An ``AgensGraph`` is shared -- by threads serving requests, by tasks in one loop, by a
vector store built from it -- while a transaction and a statement timeout are both
properties of a single connection. These are the tests that the two are kept apart.
"""

from __future__ import annotations

import asyncio
import os
import threading

import pytest
from agensgraph.introspect import DesiredIndex

from langchain_agensgraph.engine import AgensEngine
from langchain_agensgraph.graphs.agensgraph import AgensGraph
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


def _url():
    c = _conf()
    auth = c["user"] + (f":{c['password']}" if c.get("password") else "")
    return f"postgresql://{auth}@{c['host']}:{c['port']}/{c['dbname']}"


@pytest.fixture
def graph():
    g = AgensGraph("shared_state_it", _conf(), create=True, refresh_schema=False)
    g.query("MATCH (n) DETACH DELETE n")
    yield g
    g.query("MATCH (n) DETACH DELETE n")
    g.close()


@pytest.fixture
def pooled():
    engine = AgensEngine.from_url(_url(), min_size=2, max_size=5)
    g = AgensGraph(
        "shared_state_it", _conf(), create=True, engine=engine, refresh_schema=False
    )
    g.query("MATCH (n) DETACH DELETE n")
    yield g
    g.query("MATCH (n) DETACH DELETE n")
    g.close()
    engine.close()


def _statement_timeout(graph: AgensGraph) -> str:
    """What the blocking connection carries, asked of the connection.

    Not through ``query``, which applies a timeout of its own before it runs and so
    would answer for itself rather than for the connection.
    """
    return graph.connection.execute("SHOW statement_timeout").fetchone()[0]


async def _astatement_timeout(graph: AgensGraph) -> str:
    conn = await graph._aconn_get()
    cur = await conn.execute("SHOW statement_timeout")
    return (await cur.fetchone())[0]


class TestATransactionBelongsToItsCaller:
    def test_another_thread_is_not_swept_into_it(self, pooled: AgensGraph):
        """A write from a second thread lands while the first holds a read-only block.

        The second thread has nothing to do with the first one's transaction, so it
        borrows a connection of its own and its write is a write.
        """
        holding, release = threading.Event(), threading.Event()
        outcome: dict = {}

        def holder():
            # The role these tests run as is privileged, so the boundary is accepted
            # explicitly; what is under test is which connection the *other* thread
            # lands on.
            with pooled.read_only(allow_server_programs=True):
                pooled.query("MATCH (n) RETURN count(*) AS c")
                holding.set()
                release.wait(10)

        def other():
            holding.wait(10)
            try:
                pooled.query('CREATE (:"Mark" {who: \'other\'})')
                outcome["other"] = "wrote"
            except Exception as exc:  # noqa: BLE001 - recorded, then asserted on
                outcome["other"] = f"refused: {exc}"
            release.set()

        threads = [threading.Thread(target=holder), threading.Thread(target=other)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)

        assert outcome["other"] == "wrote"
        assert pooled.query('MATCH (n:"Mark") RETURN count(*) AS c')[0]["c"] == 1

    def test_the_holder_is_still_inside_its_own(self, pooled: AgensGraph):
        with pooled.read_only(allow_server_programs=True):
            with pytest.raises(Exception):
                pooled.query('CREATE (:"Mark" {who: \'holder\'})')
        assert pooled.query('MATCH (n:"Mark") RETURN count(*) AS c')[0]["c"] == 0

    @pytest.mark.asyncio
    async def test_a_sibling_task_is_not_swept_into_it(self, pooled: AgensGraph):
        holding, release = asyncio.Event(), asyncio.Event()

        async def holder():
            async with pooled.aread_only(allow_server_programs=True):
                await pooled.aquery("MATCH (n) RETURN count(*) AS c")
                holding.set()
                await asyncio.wait_for(release.wait(), 10)

        async def other():
            await asyncio.wait_for(holding.wait(), 10)
            try:
                await pooled.aquery('CREATE (:"Mark" {who: \'other\'})')
                return "wrote"
            finally:
                release.set()

        _, outcome = await asyncio.gather(holder(), other())
        assert outcome == "wrote"
        assert pooled.query('MATCH (n:"Mark") RETURN count(*) AS c')[0]["c"] == 1


class TestAStoreSharesItsGraphsTransaction:
    """A store built from a graph runs on that graph's connection.

    So a block the graph opened is a block the store's statements are inside, and the
    store has to ask the graph rather than decide for itself.
    """

    def test_a_search_runs_on_the_transactions_connection(self, pooled: AgensGraph):
        store = AgensgraphVector(
            FakeEmbeddings(), graph=pooled, node_label="SharedChunk"
        )
        held = pooled.connection.execute("SELECT pg_backend_pid()").fetchone()[0]

        loose = store.query("RETURN pg_backend_pid() AS pid")[0]["pid"]
        assert loose != held, "outside a transaction a search belongs on the pool"

        with pooled.read_only(allow_server_programs=True):
            inside = store.query("RETURN pg_backend_pid() AS pid")[0]["pid"]
        assert inside == held

    @pytest.mark.asyncio
    async def test_the_async_search_does_too(self, pooled: AgensGraph):
        store = AgensgraphVector(
            FakeEmbeddings(), graph=pooled, node_label="SharedChunk"
        )
        conn = await pooled._aconn_get()
        held = (await (await conn.execute("SELECT pg_backend_pid()")).fetchone())[0]

        loose = (await store.aquery("RETURN pg_backend_pid() AS pid"))[0]["pid"]
        assert loose != held

        async with pooled.aread_only(allow_server_programs=True):
            inside = (await store.aquery("RETURN pg_backend_pid() AS pid"))[0]["pid"]
        assert inside == held
        await pooled.aclose()


class TestATimeoutBelongsToItsConnection:
    @pytest.mark.asyncio
    async def test_each_connection_is_given_its_own(self, graph: AgensGraph):
        """The async connection carries what an awaiting caller asked for.

        A statement timeout is session state, so what one connection carries says
        nothing about the other, however alike the two requests look.
        """
        await graph.aquery("MATCH (n) RETURN 1")
        graph.query("MATCH (n) RETURN 1", timeout=5)
        await graph.aquery("MATCH (n) RETURN 1", timeout=5)
        assert await _astatement_timeout(graph) == "5s"
        assert _statement_timeout(graph) == "5s"
        await graph.aclose()

    def test_a_rolled_back_block_does_not_leave_a_stale_one(self, graph: AgensGraph):
        """``SET`` is transactional, so a rollback takes the timeout back with it."""
        graph.query("MATCH (n) RETURN 1", timeout=5)
        with pytest.raises(RuntimeError):
            with graph.read_only(allow_server_programs=True):
                graph.query("MATCH (n) RETURN 1", timeout=7)
                raise RuntimeError("the block fails after asking for 7 seconds")
        assert _statement_timeout(graph) == "5s"
        graph.query("MATCH (n) RETURN 1", timeout=7)
        assert _statement_timeout(graph) == "7s"

    def test_the_same_timeout_twice_is_applied_once(self, graph: AgensGraph):
        """The reason there is a record of it at all."""
        graph.query("MATCH (n) RETURN 1", timeout=5)
        applied = graph._applied_timeout
        graph.query("MATCH (n) RETURN 1", timeout=5)
        assert applied == graph._applied_timeout == 5
        assert _statement_timeout(graph) == "5s"


class TestTwoWritersMergingTheSameKey:
    """A merge looks, finds nothing, and creates. Two of them can both look first.

    Only a unique index over the key the merge matches on can refuse the second, and
    only making the write again turns that refusal into the match it should have been.
    """

    WRITERS, KEYS = 6, 10

    def _race(self, build, write):
        """Each writer on its own connection, all starting together."""
        graphs = [
            AgensGraph(
                "shared_state_it", _conf(), create=True, refresh_schema=False
            )
            for _ in range(self.WRITERS)
        ]
        targets = [build(g) for g in graphs]
        start = threading.Barrier(self.WRITERS)
        refused: list = []

        def worker(w):
            start.wait()
            for k in range(self.KEYS):
                try:
                    write(targets[w], k, w)
                except Exception as exc:  # noqa: BLE001 - counted, then asserted on
                    refused.append(exc)

        threads = [
            threading.Thread(target=worker, args=(w,)) for w in range(self.WRITERS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        for g in graphs:
            g.close()
        return refused

    def test_a_checkpoint_per_thread_and_no_more(self, graph: AgensGraph):
        from langchain_agensgraph.checkpoint.agensgraph import AgensSaver

        AgensSaver(graph=graph)

        def write(saver, k, w):
            saver.put(
                {"configurable": {"thread_id": f"t{k}", "checkpoint_ns": ""}},
                {
                    "v": 1,
                    "id": f"c{k}",
                    "ts": "2024-01-01T00:00:00+00:00",
                    "channel_values": {},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
                {"w": w},
                {},
            )

        refused = self._race(lambda g: AgensSaver(graph=g), write)
        held = graph.query('MATCH (n:"Checkpoint") RETURN count(*) AS c')[0]["c"]
        assert held == self.KEYS, f"{refused[:1]}"

    def test_an_item_per_key_and_no_more(self, graph: AgensGraph):
        from langchain_agensgraph.store.agensgraph import AgensStore

        AgensStore(graph=graph)
        refused = self._race(
            lambda g: AgensStore(graph=g),
            lambda store, k, w: store.put(("ns",), f"k{k}", {"w": w}),
        )
        held = graph.query('MATCH (n:"StoreItem") RETURN count(*) AS c')[0]["c"]
        assert held == self.KEYS, f"{refused[:1]}"

    def test_a_session_per_id_and_no_more(self, graph: AgensGraph):
        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="warmup", graph=graph)
        refused = self._race(
            lambda g: g,
            lambda g, k, w: AgensChatMessageHistory(
                session_id=f"s{k}", graph=g
            ).add_user_message(f"m{w}"),
        )
        held = graph.query(
            "MATCH (n:\"Session\") WHERE n.id <> 'warmup' RETURN count(*) AS c"
        )[0]["c"]
        assert held == self.KEYS, f"{refused[:1]}"


class TestTheIndexesAreTheOnesAskedFor:
    """Reconciled against what is there, rather than created blind."""

    def test_a_non_unique_index_of_the_same_name_is_rebuilt(self, graph: AgensGraph):
        from agensgraph.introspect import DesiredIndex

        graph.query('CREATE VLABEL IF NOT EXISTS "Recon"')
        graph.query('DROP PROPERTY INDEX IF EXISTS "Recon_pk"')
        graph.query('CREATE PROPERTY INDEX "Recon_pk" ON "Recon" (a, b)')
        assert [i.unique for i in graph.connection.indexes("Recon")] == [False]

        wanted = [
            DesiredIndex(
                label="Recon", properties=("a", "b"), unique=True, name="Recon_pk"
            )
        ]
        graph.ensure_indexes(wanted)
        assert [i.unique for i in graph.connection.indexes("Recon")] == [True]
        assert graph.ensure_indexes(wanted) == [], "a second call changes nothing"

    def test_an_index_of_the_same_name_over_other_properties_gives_up_the_name(
        self, graph: AgensGraph
    ):
        """Reconciling is by what an index covers, so the name has to be reclaimed."""
        from agensgraph.introspect import DesiredIndex

        graph.query('CREATE VLABEL IF NOT EXISTS "Renamed"')
        graph.query('DROP PROPERTY INDEX IF EXISTS "Renamed_idx"')
        graph.query('CREATE PROPERTY INDEX "Renamed_idx" ON "Renamed" (a)')
        graph.ensure_indexes(
            [
                DesiredIndex(
                    label="Renamed",
                    properties=("a", "b", "c"),
                    unique=True,
                    name="Renamed_idx",
                )
            ]
        )
        found = graph.connection.indexes("Renamed")
        assert [i.name for i in found] == ["Renamed_idx"]
        assert found[0].unique

    def test_duplicates_already_there_are_reported_as_such(self, graph: AgensGraph):
        from agensgraph.introspect import DesiredIndex

        graph.query('CREATE VLABEL IF NOT EXISTS "Dupes"')
        graph.query('CREATE (:"Dupes" {a: 1})')
        graph.query('CREATE (:"Dupes" {a: 1})')
        with pytest.raises(Exception, match="already holds elements"):
            graph.ensure_indexes(
                [
                    DesiredIndex(
                        label="Dupes",
                        properties=("a",),
                        unique=True,
                        name="Dupes_pk",
                    )
                ]
            )


class TestAWriteThatLostARaceIsMadeAgain:
    def test_a_unique_violation_is_retried(self, graph: AgensGraph):
        from psycopg.errors import UniqueViolation

        calls = []

        def write():
            calls.append(1)
            if len(calls) == 1:
                raise UniqueViolation("someone else created it first")
            return "written"

        assert graph.merging(write) == "written"
        assert len(calls) == 2

    def test_a_failure_that_is_not_a_race_is_raised_at_once(self, graph: AgensGraph):
        calls = []

        def write():
            calls.append(1)
            raise ValueError("the caller asked for something impossible")

        with pytest.raises(ValueError):
            graph.merging(write)
        assert len(calls) == 1

    def test_giving_up_does_not_point_an_exception_at_itself(self, graph: AgensGraph):
        """Anything walking ``__cause__`` to the root would walk forever."""
        from psycopg.errors import UniqueViolation

        def write():
            raise UniqueViolation("always taken")

        with pytest.raises(UniqueViolation) as caught:
            graph.merging(write, attempts=2)
        seen, exc = set(), caught.value
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            exc = exc.__cause__
        assert exc is None, "the chain of causes loops back on itself"


class TestAPooledConnectionCarriesVectors:
    """A pool makes its own connections, and each has to be told about vector types.

    Told once when the connection is made rather than on every borrow, which is what the
    pool's configure hook is for.
    """

    def test_a_vector_can_be_sent_on_a_borrowed_connection(self):
        from agensgraph import Vector

        engine = AgensEngine.from_url(_url(), min_size=2, max_size=4)
        try:
            with engine.connection() as conn:
                conn.execute("DROP TABLE IF EXISTS pooled_vec_probe")
                conn.execute("CREATE TABLE pooled_vec_probe (v vector(3))")
                conn.execute(
                    "INSERT INTO pooled_vec_probe VALUES (%s)",
                    (Vector([1.0, 2.0, 3.0]),),
                )
                held = conn.execute("SELECT v FROM pooled_vec_probe").fetchone()[0]
                assert list(held) == [1.0, 2.0, 3.0]
                conn.execute("DROP TABLE pooled_vec_probe")
        finally:
            engine.close()


class TestAPooledReadCostsOneRoundTrip:
    """A pooled connection is autocommit, as the dedicated one is and for the reason.

    Without it every read opens a transaction the read then has to close, so a statement
    costs a statement and a commit, and the connection sits holding a snapshot and its
    locks between calls.
    """

    def test_a_borrowed_connection_is_autocommit(self):
        engine = AgensEngine.from_url(_url(), min_size=1, max_size=2)
        try:
            with engine.connection() as conn:
                assert conn.autocommit
        finally:
            engine.close()

    def test_a_read_leaves_no_transaction_open(self, pooled: AgensGraph):
        pooled.query("RETURN 1 AS x")
        with pooled._engine.connection() as conn:
            state = conn.execute(
                "SELECT state FROM pg_stat_activity WHERE pid = pg_backend_pid()"
            ).fetchone()[0]
        assert state != "idle in transaction"

    def test_a_block_that_wants_one_still_gets_it(self, pooled: AgensGraph):
        """Autocommit is the default, not a refusal to open a transaction."""
        with pooled.read_only(allow_server_programs=True):
            with pytest.raises(Exception):
                pooled.query('CREATE (:"Mark" {who: \'inside\'})')
        assert pooled.query('MATCH (n:"Mark") RETURN count(*) AS c')[0]["c"] == 0

    @pytest.mark.asyncio
    async def test_the_async_pool_agrees(self, pooled: AgensGraph):
        engine = pooled._engine
        try:
            async with engine.aconnection() as conn:
                assert conn.autocommit
        finally:
            # Closed here rather than by the fixture: the workers of an async pool
            # belong to the loop that opened it, and this one ends with the test.
            await engine._apool.close()
            engine._apool = None


_SESSION_INDEX = [
    DesiredIndex(
        label="Session", properties=("id",), unique=True, name="Session_id_idx"
    )
]


class TestReconcilingIndexesIsNotPaidPerRequest:
    """Reading every index of a graph is far dearer than reading one label's.

    A chat history is built once per session, which in a served application is once per
    request, so the graph-wide read cannot be paid each time. Reading only the labels being
    asked about is cheap enough to pay always -- and paying always is what makes the answer
    true, since the alternative is a claim about a database this process does not own.
    """

    @staticmethod
    def _catalog_reads(build) -> list:
        from agensgraph.observability import add_query_logger, remove_query_logger

        seen: list = []

        def watch(record) -> None:
            if "ag_get_propindexdef" in str(record.statement or ""):
                seen.append(record)

        add_query_logger(watch)
        try:
            build()
        finally:
            remove_query_logger(watch)
        return seen

    def test_a_second_construction_reads_only_the_label_it_asks_about(
        self, graph: AgensGraph
    ):
        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="first", graph=graph)
        reads = self._catalog_reads(
            lambda: AgensChatMessageHistory(session_id="second", graph=graph)
        )
        # One per label asked about -- the session's id, and the message's (session, seq).
        assert len(reads) == 2, "one read per label, and no more"
        for read in reads:
            # Narrowed to the label, which is what makes it affordable to ask every time.
            assert "labname::text = " in str(read.statement)
            assert read.elapsed < 0.01, (
                f"a per-label read, not a graph-wide one: {read.elapsed}"
            )

    def test_a_second_construction_writes_nothing(self, graph: AgensGraph):
        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="first", graph=graph)
        assert graph.ensure_indexes(_SESSION_INDEX) == []

    def test_an_index_dropped_behind_our_back_is_made_again(self, graph: AgensGraph):
        """Remembering that it was made once is not the same as it being there.

        Eight writers on twenty-five shared session keys made twenty-seven elements when
        this was remembered rather than looked at: the uniqueness that stops two writers
        from each creating the same session had been dropped, and nothing looked again.
        """
        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="first", graph=graph)
        graph.query('DROP PROPERTY INDEX IF EXISTS "Session_id_idx"')
        assert not [i for i in graph.connection.indexes("Session") if i.unique]

        AgensChatMessageHistory(session_id="second", graph=graph)
        assert [i for i in graph.connection.indexes("Session") if i.unique], (
            "the unique index is back"
        )

    def test_the_index_is_still_there_and_still_unique(self, graph: AgensGraph):
        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="s", graph=graph)
        found = [i for i in graph.connection.indexes("Session") if i.unique]
        assert found, "the session id carries a unique index"

    def test_a_different_set_is_still_reconciled(self, graph: AgensGraph):
        """Remembering one set says nothing about another."""
        from agensgraph.introspect import DesiredIndex

        graph.query('CREATE VLABEL IF NOT EXISTS "Remembered"')
        graph.query('DROP PROPERTY INDEX IF EXISTS "Remembered_a"')
        graph.query('DROP PROPERTY INDEX IF EXISTS "Remembered_b"')
        first = [
            DesiredIndex(label="Remembered", properties=("a",), name="Remembered_a")
        ]
        assert graph.ensure_indexes(first)
        assert graph.ensure_indexes(first) == []
        second = [
            DesiredIndex(label="Remembered", properties=("b",), name="Remembered_b")
        ]
        assert graph.ensure_indexes(second)


class TestTheAwaitedWritePathsRetryToo:
    """Making the merge key unique turns a lost race into a refusal.

    So a write path without the retry does not merely duplicate -- it does not write at
    all. Eight awaiting writers over twenty-five shared keys lost forty-three of two
    hundred appends before these paths retried.
    """

    WRITERS, KEYS = 6, 10

    @pytest.mark.asyncio
    async def test_awaiting_writers_all_land(self, graph: AgensGraph):
        from langchain_core.messages import HumanMessage

        from langchain_agensgraph.chat_message_histories.agensgraph import (
            AgensChatMessageHistory,
        )

        AgensChatMessageHistory(session_id="warm", graph=graph)
        graphs = [
            AgensGraph("shared_state_it", _conf(), create=True, refresh_schema=False)
            for _ in range(self.WRITERS)
        ]
        refused: list = []

        async def writer(w):
            for k in range(self.KEYS):
                try:
                    await AgensChatMessageHistory(
                        session_id=f"s{k}", graph=graphs[w]
                    ).aadd_messages([HumanMessage(content=f"m{w}")])
                except Exception as exc:  # noqa: BLE001 - asserted on below
                    refused.append(exc)

        await asyncio.gather(*(writer(w) for w in range(self.WRITERS)))
        for g in graphs:
            g.close()

        sessions = graph.query(
            "MATCH (s:\"Session\") WHERE s.id <> 'warm' RETURN count(*) AS c"
        )[0]["c"]
        messages = graph.query('MATCH (m:"Message") RETURN count(*) AS c')[0]["c"]
        assert sessions == self.KEYS
        # The count alone would pass even if most writers had died, which is how the
        # hole in this path went unnoticed: every message has to be there too.
        assert messages == self.WRITERS * self.KEYS, f"{refused[:1]}"


class TestATransactionTakesAConnectionOfItsOwn:
    """A transaction occupies its connection for as long as it lasts.

    So with a pool it takes one of the pool's. Taking the dedicated connection meant a
    second caller found the first one's transaction open on it and was refused -- which
    is every write through a shared engine, the arrangement the engine exists for.
    """

    def test_writers_sharing_one_engine_all_land(self, pooled: AgensGraph):
        from langchain_agensgraph.store.agensgraph import AgensStore

        AgensStore(graph=pooled)
        refused: list = []
        start = threading.Barrier(6)

        def writer(w):
            store = AgensStore(graph=pooled)
            start.wait()
            for k in range(10):
                try:
                    store.put(("ns",), f"k{k}", {"w": w})
                except Exception as exc:  # noqa: BLE001 - asserted on below
                    refused.append(exc)

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)

        held = pooled.query('MATCH (n:"StoreItem") RETURN count(*) AS c')[0]["c"]
        assert held == 10, f"{refused[:1]}"

    def test_a_transaction_does_not_occupy_the_dedicated_connection(
        self, pooled: AgensGraph
    ):
        dedicated = pooled.connection.execute("SELECT pg_backend_pid()").fetchone()[0]
        with pooled.transaction():
            inside = pooled.query("RETURN pg_backend_pid() AS pid")[0]["pid"]
        assert inside != dedicated, "a pool was available and was not used"

    def test_without_a_pool_it_uses_the_one_connection(self, graph: AgensGraph):
        dedicated = graph.connection.execute("SELECT pg_backend_pid()").fetchone()[0]
        with graph.transaction():
            inside = graph.query("RETURN pg_backend_pid() AS pid")[0]["pid"]
        assert inside == dedicated
