"""The engine: pools by kind, graphs on demand, a caller's transaction, replay, sharing."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from agensgraph import Vector
from agensgraph.errors import ConfigurationError
from psycopg import sql

from langchain_agensgraph._core import engine as engines
from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph._core.names import ident
from langchain_agensgraph._core.program import Pipeline, Program, Statement, scalar
from langchain_agensgraph.errors import ClosedError

pytestmark = pytest.mark.server


@pytest.fixture
def engine(dsn: str) -> Iterator[AgensEngine]:
    with AgensEngine(dsn, min_size=1, max_size=2) as made:
        yield made


@pytest.fixture
def graph(dsn: str, graph_name: str) -> str:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE GRAPH {}").format(sql.Identifier(graph_name)))
        conn.execute(sql.SQL("CREATE VLABEL t"))
    return graph_name


def count(label: str = "t") -> Program[int]:
    result = yield Statement(f"MATCH (n:{label}) RETURN count(n)")
    return int(scalar(result))


def create(n: int) -> Program[None]:
    yield Statement("CREATE (:t {n: %s})", (n,))


def show(name: str) -> Program[str]:
    return str(scalar((yield Statement(f"SHOW {name}"))))


def test_a_program_runs_on_the_graph_it_names(engine: AgensEngine, graph: str) -> None:
    engine.run(lambda: create(1), graph=graph)
    assert engine.run(lambda: count(), graph=graph) == 1
    with engine.connection(graph=graph) as conn:
        assert conn.label_table.graph == graph


def test_a_pipeline_applies_together_or_not_at_all(engine: AgensEngine, graph: str) -> None:
    def failing() -> Program[None]:
        yield Pipeline((Statement("CREATE (:t {n: 1})"), Statement("SELECT 1 / 0")))

    with pytest.raises(psycopg.errors.DivisionByZero):
        engine.run(failing, graph=graph)
    assert engine.run(lambda: count(), graph=graph) == 0

    def reading() -> Program[list[object]]:
        rows = yield Pipeline(
            (Statement("CREATE (:t {n: 2})"), Statement("MATCH (x:t) RETURN x.n AS n"))
        )
        return [rows[0].records, rows[1].records, rows[1].keys]

    assert engine.run(reading, graph=graph) == [[], [(2,)], ["n"]]


def test_a_name_holding_a_percent_survives_a_statement_with_parameters(
    engine: AgensEngine, graph: str
) -> None:
    def roundtrip() -> Program[object]:
        yield Statement(f"CREATE VLABEL {ident('per%cent')}")
        yield Statement(f"CREATE (:{ident('per%cent')} {{{ident('k%')}: %s}})", ("v",))
        found = yield Statement(f"MATCH (n:{ident('per%cent')}) RETURN n.{ident('k%')}")
        return scalar(found)

    assert engine.run(roundtrip, graph=graph) == "v"


def test_connections_carry_the_timeout_name_and_vectors_from_the_start(
    engine: AgensEngine,
) -> None:
    assert engine.run(lambda: show("statement_timeout")) == "30s"
    assert engine.run(lambda: show("application_name")).startswith("langchain-agensgraph/")
    with engine.connection("maintenance") as conn:
        assert conn.execute("SHOW statement_timeout").fetchone() != ("30s",)
    with engine.connection() as conn:
        (value,) = conn.execute_query("SELECT '[1,2,3]'::vector").records[0]
    assert isinstance(value, Vector)


def test_capabilities_come_from_the_server(engine: AgensEngine) -> None:
    assert engine.capabilities.release[:2] >= (2, 14)


def test_a_callers_transaction_is_joined_and_rolls_back_together(
    engine: AgensEngine, graph: str
) -> None:
    with pytest.raises(RuntimeError, match="caller"):
        with engine.transaction():
            engine.run(lambda: create(1), graph=graph)
            assert engine.run(lambda: count(), graph=graph) == 1
            raise RuntimeError("caller")
    assert engine.run(lambda: count(), graph=graph) == 0
    with engine.transaction() as conn:
        engine.run(lambda: create(2), graph=graph)
        conn.execute("CREATE (:t {n: 3})")
    assert engine.run(lambda: count(), graph=graph) == 2


def test_a_nested_transaction_is_a_savepoint(engine: AgensEngine, graph: str) -> None:
    with engine.transaction():
        engine.run(lambda: create(1), graph=graph)
        with pytest.raises(RuntimeError):
            with engine.transaction():
                engine.run(lambda: create(2), graph=graph)
                raise RuntimeError
    assert engine.run(lambda: count(), graph=graph) == 1


def test_maintenance_work_joins_a_callers_transaction_and_untrusted_work_never_does(
    engine: AgensEngine,
) -> None:
    with engine.transaction() as conn:
        assert engine.in_transaction()
        with engine.connection("maintenance") as maintenance:
            assert maintenance is conn
        with engine.connection() as same:
            assert same is conn
        with engine.connection("untrusted") as other:
            assert other.info.backend_pid != conn.info.backend_pid
    assert not engine.in_transaction()
    with engine.connection("maintenance") as maintenance:
        assert maintenance.info.backend_pid != conn.info.backend_pid


def backend() -> Program[int]:
    return int(scalar((yield Statement("SELECT pg_backend_pid()"))))


def test_another_thread_does_not_join_a_callers_transaction(engine: AgensEngine) -> None:
    seen: list[int] = []
    with engine.transaction() as conn:
        assert engine.run(backend) == conn.info.backend_pid
        thread = threading.Thread(target=lambda: seen.append(engine.run(backend)))
        thread.start()
        thread.join()
    assert seen and seen[0] != conn.info.backend_pid


async def test_a_connection_a_cancelled_program_held_is_not_lent_again(dsn: str) -> None:
    engine = AgensEngine(dsn, maintenance_max_size=1)

    def interrupted() -> Program[None]:
        # Session settings, as a concurrent index build makes, put back by a step never reached:
        # the driver retires a connection whose statement was cancelled.
        yield Statement("SELECT set_config('work_mem', '77MB', false)")
        yield Statement("SELECT pg_sleep(30)")

    def setting() -> Program[str]:
        return str(scalar((yield Statement("SHOW work_mem"))))

    task = asyncio.create_task(engine.arun(interrupted, kind="maintenance"))
    await asyncio.sleep(1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await engine.arun(setting, kind="maintenance") != "77MB"
    await engine.aclose()


async def test_a_task_started_outside_does_not_join_a_callers_transaction(
    engine: AgensEngine,
) -> None:
    inside = asyncio.Event()

    async def sibling() -> int:
        await inside.wait()
        return await engine.arun(backend)

    task = asyncio.create_task(sibling())
    async with engine.atransaction() as conn:
        assert await engine.arun(backend) == conn.info.backend_pid
        inside.set()
        assert await task != conn.info.backend_pid
    await engine.aclose()


def test_a_thread_given_the_context_does_not_join_a_callers_transaction(
    engine: AgensEngine,
) -> None:
    def elsewhere() -> tuple[bool, int]:
        return engine.in_transaction(), engine.run(backend)

    with engine.transaction() as conn:
        # LangChain's thread pool copies the context into each thread, transaction and all.
        with ThreadPoolExecutor(2) as pool:
            seen = [pool.submit(contextvars.copy_context().run, elsewhere) for _ in range(2)]
            results = [f.result() for f in seen]
        assert engine.in_transaction()
    assert all(not inside and pid != conn.info.backend_pid for inside, pid in results)


async def test_child_tasks_do_not_join_a_callers_transaction(engine: AgensEngine) -> None:
    async def child() -> tuple[bool, int]:
        return engine.in_transaction(), await engine.arun(backend)

    async with engine.atransaction() as conn:
        assert await engine.arun(backend) == conn.info.backend_pid
        results = await asyncio.gather(child(), child())
        assert engine.in_transaction()
    assert all(not inside and pid != conn.info.backend_pid for inside, pid in results)
    await engine.aclose()


def test_what_a_transaction_made_is_remembered_once_it_commits(engine: AgensEngine) -> None:
    seen: list[str] = []
    engine.remember(lambda: seen.append("now"))
    assert seen == ["now"]
    with engine.transaction():
        engine.remember(lambda: seen.append("committed"))
        with pytest.raises(RuntimeError), engine.transaction():
            engine.remember(lambda: seen.append("savepoint rolled back"))
            raise RuntimeError
        with engine.transaction():
            engine.remember(lambda: seen.append("savepoint kept"))
        assert seen == ["now"]
    assert seen == ["now", "committed", "savepoint kept"]
    with pytest.raises(RuntimeError), engine.transaction():
        engine.remember(lambda: seen.append("rolled back"))
        raise RuntimeError
    assert seen == ["now", "committed", "savepoint kept"]


async def test_what_an_awaited_transaction_made_is_remembered_once_it_commits(
    engine: AgensEngine,
) -> None:
    seen: list[str] = []
    async with engine.atransaction():
        engine.remember(lambda: seen.append("committed"))
        assert seen == []
    with pytest.raises(RuntimeError):
        async with engine.atransaction():
            engine.remember(lambda: seen.append("rolled back"))
            raise RuntimeError
    assert seen == ["committed"]
    await engine.aclose()


async def test_an_awaited_transaction_is_joined_too(engine: AgensEngine, graph: str) -> None:
    with pytest.raises(RuntimeError):
        async with engine.atransaction():
            await engine.arun(lambda: create(1), graph=graph)
            raise RuntimeError
    assert await engine.arun(lambda: count(), graph=graph) == 0
    async with engine.atransaction():
        await engine.arun(lambda: create(1), graph=graph)
    assert await engine.arun(lambda: count(), graph=graph) == 1
    await engine.aclose()


def test_a_replayable_program_survives_a_lost_connection(dsn: str) -> None:
    with AgensEngine(dsn, min_size=1, max_size=1) as engine:
        with engine.connection() as conn:
            pid = conn.info.backend_pid
        _terminate(dsn, pid)
        assert engine.run(lambda: show("server_version_num"), replay=True)


def test_a_program_not_declared_replayable_reports_the_lost_connection(dsn: str) -> None:
    with AgensEngine(dsn, min_size=1, max_size=1) as engine:
        with engine.connection() as conn:
            pid = conn.info.backend_pid
        _terminate(dsn, pid)
        with pytest.raises(psycopg.OperationalError):
            engine.run(lambda: show("server_version_num"))


def test_model_writes_need_a_writer_role(engine: AgensEngine) -> None:
    with pytest.raises(ConfigurationError, match="writer_conninfo"):
        with engine.connection("writer"):
            pass


def test_components_handed_one_conninfo_share_one_engine(dsn: str) -> None:
    first, owned = engines.acquire(dsn)
    second, _ = engines.acquire(dsn)
    assert owned and first is second
    engines.release(first)
    assert not first.closed
    engines.release(second)
    assert first.closed
    third, _ = engines.acquire(dsn)
    assert third is not first
    engines.release(third)


def test_a_closed_engine_refuses_to_lend(dsn: str) -> None:
    engine = AgensEngine(dsn)
    engine.close()
    with pytest.raises(ClosedError):
        engine.run(lambda: show("server_version_num"))


def _terminate(dsn: str, pid: int) -> None:
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        for _ in range(50):
            gone = admin.execute(
                "SELECT NOT EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid = %s)", (pid,)
            ).fetchone()
            if gone and gone[0]:
                return
            time.sleep(0.05)
