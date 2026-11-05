"""Programs run the same on either executor: steps in order, failures thrown back in."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from langchain_agensgraph._core.program import (
    Call,
    InTransaction,
    Pipeline,
    Program,
    Statement,
    arun,
    put_back_settings,
    run,
    set_settings,
)


@dataclass
class FakeResult:
    records: list[tuple[Any, ...]]


@dataclass
class FakeColumn:
    name: str


class FakeCursor:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.text = ""
        self.description: list[FakeColumn] | None = None

    def execute(self, text: str, params: Any = None, prepare: bool | None = None) -> None:
        self.conn.sent.append(("piped", text, params))
        self.conn.prepared.append(prepare)
        self.text = text
        self.description = [FakeColumn("value")]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [(self.text,)]

    def close(self) -> None:
        pass


class FakeAsyncCursor(FakeCursor):
    async def execute(  # type: ignore[override]
        self, text: str, params: Any = None, prepare: bool | None = None
    ) -> None:
        FakeCursor.execute(self, text, params, prepare)

    async def fetchall(self) -> list[tuple[Any, ...]]:  # type: ignore[override]
        return FakeCursor.fetchall(self)

    async def close(self) -> None:  # type: ignore[override]
        pass


@dataclass
class FakeConn:
    sent: list[Any] = field(default_factory=list)
    prepared: list[bool | None] = field(default_factory=list)
    refuse: str | None = None

    def execute_query(self, text: str, params: Any = None, **kwargs: Any) -> FakeResult:
        if text == self.refuse:
            raise ValueError(f"refused {text}")
        self.sent.append((text, params, kwargs.get("settings_")))
        self.prepared.append(kwargs.get("prepare_"))
        return FakeResult([(text,)])

    @contextmanager
    def pipeline(self) -> Iterator[None]:
        self.sent.append("pipeline")
        yield
        self.sent.append("sync")

    def cursor(self, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self)

    def twice(self, value: int) -> int:
        return value * 2

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.sent.append("begin")
        try:
            yield
        except BaseException:
            self.sent.append("rollback")
            raise
        self.sent.append("commit")

    @contextmanager
    def schema_lock(self, *, graph: str | None = None) -> Iterator[None]:
        with self.transaction():
            self.sent.append(("lock", graph))
            yield


class FakeAsyncConn:
    def __init__(self, refuse: str | None = None) -> None:
        self.inner = FakeConn(refuse=refuse)
        self.sent = self.inner.sent

    async def execute_query(self, text: str, params: Any = None, **kwargs: Any) -> FakeResult:
        return self.inner.execute_query(text, params, **kwargs)

    @asynccontextmanager
    async def pipeline(self) -> AsyncIterator[None]:
        with self.inner.pipeline():
            yield

    def cursor(self, **kwargs: Any) -> FakeAsyncCursor:
        return FakeAsyncCursor(self.inner)

    async def twice(self, value: int) -> int:
        return value * 2

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        with self.inner.transaction():
            yield

    @asynccontextmanager
    async def schema_lock(self, *, graph: str | None = None) -> AsyncIterator[None]:
        with self.inner.schema_lock(graph=graph):
            yield


def counting() -> Program[list[Any]]:
    first = yield Statement("one", (1,))
    doubled = yield Call("twice", (21,))
    both = yield Pipeline((Statement("two"), Statement("three", settings={"a.b": True})))
    return [first.records[0][0], doubled, [r.records[0][0] for r in both]]


def test_steps_run_in_order_and_the_program_returns_its_value() -> None:
    conn = FakeConn()
    assert run(conn, counting()) == ["one", 42, ["two", "three"]]  # type: ignore[arg-type]
    assert conn.sent[0] == ("one", (1,), None)
    # Every statement is sent before the one Sync that ends the block, settings set just before
    # the statement that asked for them and put back just after it.
    assert conn.sent[1:] == [
        "pipeline",
        ("piped", "two", ()),
        ("piped", *set_settings({"a.b": True}, "statement")),
        ("piped", "three", ()),
        ("piped", *put_back_settings({"a.b": True}, "statement")),
        "sync",
    ]


async def test_the_awaiting_executor_runs_the_same_program() -> None:
    conn = FakeAsyncConn()
    assert await arun(conn, counting()) == ["one", 42, ["two", "three"]]  # type: ignore[arg-type]


def recovering() -> Program[str]:
    try:
        yield Statement("bad")
    except ValueError:
        result = yield Statement("fallback")
        return str(result.records[0][0])
    return "unreachable"


def test_a_failure_is_thrown_into_the_program_which_may_handle_it() -> None:
    conn = FakeConn(refuse="bad")
    assert run(conn, recovering()) == "fallback"  # type: ignore[arg-type]


def test_a_failure_the_program_does_not_handle_reaches_the_caller() -> None:
    def careless() -> Program[None]:
        yield Statement("bad")

    with pytest.raises(ValueError, match="refused bad"):
        run(FakeConn(refuse="bad"), careless())  # type: ignore[arg-type]


def inner(fail: bool) -> Program[str]:
    yield Statement("inside")
    if fail:
        raise RuntimeError("inner failed")
    return "done"


def outer(fail: bool, lock: str | None = None) -> Program[str]:
    try:
        return (yield InTransaction(inner(fail), lock=lock))
    except RuntimeError as exc:
        return f"caught: {exc}"


def test_a_transaction_commits_what_its_program_did() -> None:
    conn = FakeConn()
    assert run(conn, outer(False)) == "done"  # type: ignore[arg-type]
    assert conn.sent == ["begin", ("inside", (), None), "commit"]


def test_a_failed_transaction_rolls_back_and_the_failure_reaches_the_outer_program() -> None:
    conn = FakeConn()
    assert run(conn, outer(True)) == "caught: inner failed"  # type: ignore[arg-type]
    assert conn.sent[-1] == "rollback"


async def test_a_locked_transaction_holds_the_graphs_schema_lock() -> None:
    conn = FakeAsyncConn()
    assert await arun(conn, outer(False, lock="g")) == "done"  # type: ignore[arg-type]
    assert conn.sent[:2] == ["begin", ("lock", "g")]


def test_a_statement_adding_to_a_map_is_never_prepared() -> None:
    def writes() -> Program[None]:
        yield Statement("MATCH (n) SET n += %s", ({},))
        yield Statement("MATCH (n) RETURN n")
        yield Pipeline((Statement("MERGE (n {id: 1}) SET n += {}"), Statement("RETURN 1")))

    conn = FakeConn()
    run(conn, writes())  # type: ignore[arg-type]
    assert conn.prepared == [False, None, False, None]


def test_settings_are_sent_as_text_with_what_they_replace_kept_aside() -> None:
    text, params = set_settings({"hnsw.ef_search": 40, "enable_seqscan": False}, "statement")
    assert text == (
        "select set_config('agensgraph_ai.kept_statement_0', current_setting(%s, true), true), "
        "set_config(%s, %s, true), "
        "set_config('agensgraph_ai.kept_statement_1', current_setting(%s, true), true), "
        "set_config(%s, %s, true)"
    )
    assert params == [
        "hnsw.ef_search",
        "hnsw.ef_search",
        "40",
        "enable_seqscan",
        "enable_seqscan",
        "off",
    ]
    text, params = put_back_settings({"hnsw.ef_search": 40, "enable_seqscan": False}, "statement")
    assert text == (
        "select set_config(%s, nullif(current_setting('agensgraph_ai.kept_statement_0', true), "
        "''), true), set_config(%s, nullif(current_setting('agensgraph_ai.kept_statement_1', "
        "true), ''), true)"
    )
    assert params == ["hnsw.ef_search", "enable_seqscan"]


def test_a_pipelines_settings_hold_for_all_of_it_and_a_statements_for_it_alone() -> None:
    def piped() -> Program[Any]:
        yield Pipeline(
            (Statement("one"), Statement("two", settings={"b": 1}), Statement("three")),
            settings={"a": True},
        )

    conn = FakeConn()
    run(conn, piped())  # type: ignore[arg-type]
    assert conn.sent == [
        "pipeline",
        ("piped", *set_settings({"a": True}, "pipeline")),
        ("piped", "one", ()),
        ("piped", *set_settings({"b": 1}, "statement")),
        ("piped", "two", ()),
        ("piped", *put_back_settings({"b": 1}, "statement")),
        ("piped", "three", ()),
        ("piped", *put_back_settings({"a": True}, "pipeline")),
        "sync",
    ]
