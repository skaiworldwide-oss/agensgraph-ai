"""Query logging scoped to its block, the role grants, and setup that runs once."""

from __future__ import annotations

import threading

import agensgraph
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph._core.names import companion_schema, ident
from langchain_agensgraph._core.program import (
    InTransaction,
    Pipeline,
    Program,
    Statement,
    run,
    scalar,
)
from langchain_agensgraph._core.setup import Setup
from langchain_agensgraph.observability import log_queries
from langchain_agensgraph.roles import render_grants, render_roles

pytestmark = pytest.mark.server

ROLE_PASSWORD = "agens"


@pytest.fixture
def graph(dsn: str, graph_name: str) -> str:
    store = companion_schema(graph_name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE GRAPH {}").format(sql.Identifier(graph_name)))
        conn.execute("CREATE VLABEL person")
        conn.execute("CREATE ELABEL knows")
        conn.execute("CREATE (:person {name: 'a'})-[:knows]->(:person {name: 'b'})")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(store)))
        conn.execute(f"CREATE TABLE {ident(store)}.notes (id serial PRIMARY KEY, body text)")
    return graph_name


def select(text: str) -> Program[object]:
    return scalar((yield Statement(text)))


def test_a_block_collects_its_own_statements_and_no_one_elses(engine: AgensEngine) -> None:
    other_ran = threading.Event()

    def elsewhere() -> None:
        engine.run(lambda: select("SELECT 'elsewhere'"))
        other_ran.set()

    with log_queries() as outer:
        engine.run(lambda: select("SELECT 'outer'"))
        with log_queries() as inner:
            engine.run(lambda: select("SELECT 'inner'"))
        thread = threading.Thread(target=elsewhere)
        thread.start()
        thread.join()
    engine.run(lambda: select("SELECT 'after'"))
    assert other_ran.is_set()
    texts = [record.statement for record in outer]
    assert any("'outer'" in text for text in texts)
    assert any("'inner'" in text for text in texts)
    assert not any("'elsewhere'" in text or "'after'" in text for text in texts)
    assert [record.statement for record in inner] == [t for t in texts if "'inner'" in t]


def test_settings_reach_only_the_statement_or_pipeline_that_asked(engine: AgensEngine) -> None:
    joins = "SELECT current_setting('enable_hashjoin') || current_setting('enable_mergejoin')"

    def piped() -> Program[list[object]]:
        rows = yield Pipeline(
            (
                Statement(joins),
                Statement(joins, settings={"enable_mergejoin": False}),
                Statement(joins),
            ),
            settings={"enable_hashjoin": False},
        )
        return [r.records[0][0] for r in rows]

    def alone() -> Program[object]:
        result = yield Statement(joins, settings={"enable_hashjoin": False})
        return result.records[0][0]

    with engine.transaction():
        engine.run(lambda: select("SELECT set_config('enable_mergejoin', 'on', true)"))
        assert engine.run(piped) == ["offon", "offoff", "offon"]
        assert engine.run(lambda: select(joins)) == "onon"
        assert engine.run(alone) == "offon"
        assert engine.run(lambda: select(joins)) == "onon"
    assert engine.run(alone) == "offon"


def test_a_statement_that_may_fail_leaves_a_callers_transaction_usable(
    engine: AgensEngine,
) -> None:
    def program() -> Program[object]:
        try:
            yield Statement("SELECT * FROM no_such_table_anywhere", may_fail=True)
        except psycopg.errors.UndefinedTable:
            pass
        return (yield from select("SELECT 'still usable'"))

    assert engine.run(program) == "still usable"
    with engine.transaction():
        assert engine.run(program) == "still usable"


def test_a_failed_pipeline_leaves_no_statement_its_connection_cannot_run(dsn: str) -> None:
    probe = Statement("SELECT count(*) FROM t WHERE k > %s", (0,))

    def program() -> Program[object]:
        yield Pipeline((probe,))
        # The probe is prepared by the pipeline that fails, ahead of the failure, and named
        # again after it, as a statement's settings are on each side of every statement.
        try:
            yield Pipeline((probe, Statement("INSERT INTO t VALUES (1)"), probe))
        except psycopg.errors.UniqueViolation:
            pass
        rows = yield Pipeline((probe,))
        return rows[0].records[0][0]

    with psycopg.connect(dsn, autocommit=True, prepare_threshold=2) as conn:
        conn.execute("CREATE TEMP TABLE t (k int PRIMARY KEY)")
        conn.execute("INSERT INTO t VALUES (1)")
        assert run(conn, program()) == 1  # type: ignore[arg-type]


def test_a_blocks_settings_end_with_it_and_its_level_is_for_a_transaction_of_its_own(
    engine: AgensEngine,
) -> None:
    def block() -> Program[tuple[object, object]]:
        memory = yield from select("SELECT current_setting('work_mem')")
        level = yield from select("SELECT current_setting('transaction_isolation')")
        return memory, level

    def program() -> Program[tuple[object, object, object]]:
        memory, level = yield InTransaction(
            block(), isolation="REPEATABLE READ", settings={"work_mem": "7MB"}
        )
        after = yield from select("SELECT current_setting('work_mem')")
        return memory, level, after

    default = engine.run(lambda: select("SELECT current_setting('work_mem')"))
    assert engine.run(program) == ("7MB", "repeatable read", default)
    with engine.transaction():
        assert engine.run(program) == ("7MB", "read committed", default)


def test_the_role_recipe_is_valid_sql() -> None:
    assert render_roles(password="x").count("CREATE ROLE") == 3


def test_the_grants_give_each_role_what_it_needs_and_no_more(dsn: str, graph: str) -> None:
    store = companion_schema(graph)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(render_grants(graph, writer_role="llm_rw"))
    with agensgraph.connect(_as(dsn, "app_rw"), autocommit=True) as app:
        app.graph(graph)
        app.execute("CREATE (:person {name: 'c'})")
        app.execute("MATCH (n:person {name: 'c'}) DETACH DELETE n")
        app.execute(f"INSERT INTO {ident(store)}.notes (body) VALUES ('kept')")
        app.execute(f"TRUNCATE {ident(store)}.notes")
    with agensgraph.connect(_as(dsn, "llm_ro"), autocommit=True) as llm:
        llm.graph(graph)
        assert llm.execute("MATCH (n:person) RETURN count(n)").fetchone() == (2,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            llm.execute(f"SELECT * FROM {ident(store)}.notes")
        assert llm.untrusted_role_problems(app_role="app_rw", schemas=[graph, store]) == ()
        if llm.capabilities.has("cypher_write_enforced"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                llm.execute("CREATE (:person {name: 'model'})")


def test_setup_creates_once_and_then_only_checks(engine: AgensEngine, graph: str) -> None:
    store = companion_schema(graph)
    made: list[str] = []

    def check() -> Program[bool]:
        found = yield Statement("SELECT to_regclass(%s) IS NOT NULL", (f"{ident(store)}.made",))
        return bool(scalar(found))

    def create() -> Program[None]:
        found = yield Statement("SELECT to_regclass(%s) IS NOT NULL", (f"{ident(store)}.made",))
        if not scalar(found):
            made.append("table")
            yield Statement(f"CREATE TABLE {ident(store)}.made (id int)")

    setup = Setup(engine, graph, check, create)
    setup.ensure()
    setup.ensure()
    assert made == ["table"] and setup.done
    setup.forget()
    setup.ensure()
    assert made == ["table"]
    other = Setup(engine, graph, check, create)
    other.ensure()
    assert made == ["table"]


def _as(dsn: str, role: str) -> str:
    return make_conninfo(dsn, user=role, password=ROLE_PASSWORD)
