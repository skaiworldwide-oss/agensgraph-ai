"""Each operation written once, as a program, and run by a blocking or an awaiting executor.

A program is a generator. It yields what it wants done -- a statement, a pipeline of statements, a
method of the driver's connection, or another program to run inside a transaction -- and is sent
back what came of it, or has the failure thrown into it at the ``yield``. It does no I/O itself,
so :func:`run` and :func:`arun` execute the same code on a blocking or an awaiting connection, and
the two interfaces of every component cannot drift apart::

    def count(label: str) -> Program[int]:
        result = yield Statement(f"MATCH (n:{ident(label)}) RETURN count(n)")
        return result.records[0][0]

Each step is one round trip unless it says otherwise, which is what makes the number of round
trips an operation costs readable off its program.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar

from agensgraph import AsyncConnection, Connection, Result
from agensgraph.settings import check_settings
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

T = TypeVar("T")

Params: TypeAlias = Sequence[Any] | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Statement:
    """One statement; sends back the driver's :class:`~agensgraph.Result`, rows as tuples.

    ``settings`` reach this statement alone (``hnsw.ef_search``, ``statement_timeout`` in
    milliseconds). In autocommit they go in the statement's round trip and end with its implicit
    transaction; inside a transaction they are set before it and put back after it, a round trip
    each.

    ``may_fail`` says the program expects the statement may fail and goes on after it: inside a
    transaction it runs in a savepoint, so its failure leaves the transaction usable; in
    autocommit nothing else is needed.
    """

    text: str
    params: Params | None = None
    binary: bool = False
    settings: Mapping[str, object] | None = None
    may_fail: bool = False


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Statements sent together; sends back a list of :class:`Rows`, one per statement.

    One round trip, with one Sync after the last statement: outside a transaction the statements
    share one implicit transaction, so they apply together or not at all, and a lock one of them
    takes is held by the ones after it. Settings asked of a statement are set just before it and
    put back just after it, so they reach no other statement; ``settings`` asked of the pipeline
    hold for all of its statements and are put back after the last.
    """

    statements: tuple[Statement, ...]
    settings: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Rows:
    """What one pipelined statement returned: its rows as tuples, and its column names."""

    records: list[tuple[Any, ...]]
    keys: list[str]


@dataclass(frozen=True, slots=True)
class Call:
    """A method of the driver's connection, awaited on an awaiting one; sends back its result."""

    method: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InTransaction:
    """Another program, run in a transaction of its own, or a savepoint inside an open one.

    With ``lock`` naming a graph, the graph's schema lock -- the driver's, which its own
    ``ensure_*`` methods take -- is held until the transaction ends, so two processes changing
    what the graph declares take turns. A failure rolls the transaction back and is thrown into
    the program that yielded this. Sends back what the inner program returned.

    ``isolation`` (such as ``"REPEATABLE READ"``) is the level of a transaction of its own; a
    savepoint runs at the open transaction's. ``settings`` hold for the inner program and end
    with it, a savepoint's put back as it ends. A locked transaction takes neither.
    """

    body: Generator[Step, Any, Any]
    lock: str | None = None
    isolation: str | None = None
    settings: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.lock is not None and (self.isolation or self.settings):
            raise ValueError("a locked transaction takes no isolation level or settings")


Step: TypeAlias = Statement | Pipeline | Call | InTransaction
Program: TypeAlias = Generator[Step, Any, T]


def run(conn: Connection[Any], program: Program[T]) -> T:
    """Run a program on a blocking connection, to the value it returns."""
    send: Any = None
    failure: BaseException | None = None
    while True:
        try:
            step = program.throw(failure) if failure is not None else program.send(send)
        except StopIteration as done:
            return done.value  # type: ignore[no-any-return]
        send, failure = None, None
        try:
            send = _do(conn, step)
        except Exception as exc:
            failure = exc


async def arun(conn: AsyncConnection[Any], program: Program[T]) -> T:
    """Run a program on an awaiting connection, to the value it returns."""
    send: Any = None
    failure: BaseException | None = None
    while True:
        try:
            step = program.throw(failure) if failure is not None else program.send(send)
        except StopIteration as done:
            return done.value  # type: ignore[no-any-return]
        send, failure = None, None
        try:
            send = await _ado(conn, step)
        except Exception as exc:
            failure = exc


def _do(conn: Connection[Any], step: Step) -> Any:
    match step:
        case Statement(may_fail=True) if _in_transaction(conn):
            with conn.transaction():
                return _statement(conn, step)
        case Statement():
            return _statement(conn, step)
        case Pipeline():
            statements, keep = _pipelined(step)
            # Every statement is sent before any result is read, so the block ends in one Sync;
            # reading inside it would sync after each statement and commit each on its own.
            cursors = []
            try:
                with conn.pipeline():
                    for text, params in statements:
                        cursor = conn.cursor(row_factory=tuple_row)
                        cursor.execute(text, params, prepare=_prepare(text))
                        cursors.append(cursor)
            except Exception:
                _forget_prepared(conn)
                raise
            out = []
            for i, cursor in enumerate(cursors):
                if i in keep:
                    out.append(
                        Rows(list(cursor.fetchall()), _keys(cursor))
                        if cursor.description
                        else Rows([], [])
                    )
                cursor.close()
            return out
        case Call():
            return getattr(conn, step.method)(*step.args, **step.kwargs)
        case InTransaction(lock=None):
            opening, closing = _framed(conn, step)
            with conn.transaction():
                _send(conn, opening)
                result = run(conn, step.body)
                _send(conn, closing)
                return result
        case InTransaction(lock=graph):
            with conn.schema_lock(graph=graph):
                return run(conn, step.body)
    raise TypeError(f"a program yielded {step!r}, which is not a step")


def _statement(conn: Connection[Any], step: Statement) -> Any:
    if step.settings and _in_transaction(conn):
        with conn.cursor() as cursor:
            cursor.execute(*set_settings(step.settings, "statement"))
            try:
                return conn.execute_query(
                    step.text,
                    _bound(step.params),
                    binary_=step.binary,
                    row_=tuple_row,
                    prepare_=_prepare(step.text),
                )
            finally:
                # A failed statement leaves the transaction to be rolled back, settings and all.
                if _in_transaction(conn):
                    cursor.execute(*put_back_settings(step.settings, "statement"))
    return conn.execute_query(
        step.text,
        _bound(step.params),
        binary_=step.binary,
        settings_=step.settings,
        row_=tuple_row,
        prepare_=_prepare(step.text),
    )


async def _ado(conn: AsyncConnection[Any], step: Step) -> Any:
    match step:
        case Statement(may_fail=True) if _in_transaction(conn):
            async with conn.transaction():
                return await _astatement(conn, step)
        case Statement():
            return await _astatement(conn, step)
        case Pipeline():
            statements, keep = _pipelined(step)
            cursors = []
            try:
                async with conn.pipeline():
                    for text, params in statements:
                        cursor = conn.cursor(row_factory=tuple_row)
                        await cursor.execute(text, params, prepare=_prepare(text))
                        cursors.append(cursor)
            except Exception:
                _forget_prepared(conn)
                raise
            out = []
            for i, cursor in enumerate(cursors):
                if i in keep:
                    out.append(
                        Rows(list(await cursor.fetchall()), _keys(cursor))
                        if cursor.description
                        else Rows([], [])
                    )
                await cursor.close()
            return out
        case Call():
            return await getattr(conn, step.method)(*step.args, **step.kwargs)
        case InTransaction(lock=None):
            opening, closing = _framed(conn, step)
            async with conn.transaction():
                await _asend(conn, opening)
                result = await arun(conn, step.body)
                await _asend(conn, closing)
                return result
        case InTransaction(lock=graph):
            async with conn.schema_lock(graph=graph):
                return await arun(conn, step.body)
    raise TypeError(f"a program yielded {step!r}, which is not a step")


async def _astatement(conn: AsyncConnection[Any], step: Statement) -> Any:
    if step.settings and _in_transaction(conn):
        async with conn.cursor() as cursor:
            await cursor.execute(*set_settings(step.settings, "statement"))
            try:
                return await conn.execute_query(
                    step.text,
                    _bound(step.params),
                    binary_=step.binary,
                    row_=tuple_row,
                    prepare_=_prepare(step.text),
                )
            finally:
                if _in_transaction(conn):
                    await cursor.execute(*put_back_settings(step.settings, "statement"))
    return await conn.execute_query(
        step.text,
        _bound(step.params),
        binary_=step.binary,
        settings_=step.settings,
        row_=tuple_row,
        prepare_=_prepare(step.text),
    )


def _framed(
    conn: Any, step: InTransaction
) -> tuple[list[tuple[str, Params]], list[tuple[str, Params]]]:
    """What a transaction block sends before and after its program: the isolation level, which
    only a transaction of its own takes, and the settings, put back at a savepoint's end, since
    one made in a savepoint outlives it."""
    opening: list[tuple[str, Params]] = []
    closing: list[tuple[str, Params]] = []
    if not (step.isolation or step.settings):
        return opening, closing
    fresh = not _in_transaction(conn)
    if step.isolation and fresh:
        opening.append((f"SET TRANSACTION ISOLATION LEVEL {step.isolation}", ()))
    if step.settings:
        opening.append(set_settings(step.settings, "block"))
        if not fresh:
            closing.append(put_back_settings(step.settings, "block"))
    return opening, closing


def _send(conn: Connection[Any], statements: list[tuple[str, Params]]) -> None:
    if statements:
        with conn.cursor() as cursor:
            for text, params in statements:
                cursor.execute(text, params)


async def _asend(conn: AsyncConnection[Any], statements: list[tuple[str, Params]]) -> None:
    if statements:
        async with conn.cursor() as cursor:
            for text, params in statements:
                await cursor.execute(text, params)


def _keys(cursor: Any) -> list[str]:
    return [column.name for column in cursor.description]


def _forget_prepared(conn: Connection[Any] | AsyncConnection[Any]) -> None:
    """Forget the connection's prepared statements after a pipeline failed outside a
    transaction, as psycopg does after a rollback it sends.

    The pipeline's implicit transaction rolled back, and a statement it prepared ahead of the one
    that failed went with it, while psycopg kept its name: the connection's next use of that
    text, by whoever borrows it next, failed with 26000 (psycopg 3.3, on every release). psycopg
    deallocates on the server before its next statement. Inside a transaction the rollback that
    must follow does all of this.
    """
    if conn.info.transaction_status == TransactionStatus.IDLE:
        conn._prepared.clear()


def _in_transaction(conn: Any) -> bool:
    """Whether the connection is inside a transaction block, where a setting made with
    ``set_config(..., true)`` lasts until the block ends rather than with the statement."""
    return bool(conn.info.transaction_status == TransactionStatus.INTRANS)


def _prepare(text: str) -> bool | None:
    """False for a statement holding ``+=``, which is never prepared; None leaves it to the
    connection.

    On 2.14 and 2.15 a prepared ``SET n += map`` that is planned again, as an ``ANALYZE`` makes
    it, replaces the vertex's properties with the map instead of adding the map to them.
    Unprepared, it is planned each time it runs, and adds.
    """
    return False if "+=" in text else None


def _bound(params: Params | None) -> Params:
    """Parameters for psycopg, always bound: a statement whose names were written in by
    :func:`~.names.ident` has its ``%`` doubled, which only a bound statement reads back."""
    return () if params is None else params


def _pipelined(step: Pipeline) -> tuple[list[tuple[str, Params | None]], list[int]]:
    """A pipeline's statements as sent, and which results to keep: settings set just before
    what asked for them and put back just after it."""
    sent: list[tuple[str, Params | None]] = []
    keep: list[int] = []
    if step.settings:
        sent.append(set_settings(step.settings, "pipeline"))
    for statement in step.statements:
        if statement.settings:
            sent.append(set_settings(statement.settings, "statement"))
        keep.append(len(sent))
        sent.append((statement.text, _bound(statement.params)))
        if statement.settings:
            sent.append(put_back_settings(statement.settings, "statement"))
    if step.settings:
        sent.append(put_back_settings(step.settings, "pipeline"))
    return sent, keep


_KEPT = "agensgraph_ai.kept_"


def set_settings(settings: Mapping[str, object], scope: str) -> tuple[str, list[str]]:
    """``select``: each setting's value kept aside under ``scope``, then the new value set, both
    until the transaction ends or :func:`put_back_settings` restores them."""
    calls: list[str] = []
    params: list[str] = []
    for i, (name, text) in enumerate(check_settings(settings)):
        calls.append(f"set_config('{_KEPT}{scope}_{i}', current_setting(%s, true), true)")
        calls.append("set_config(%s, %s, true)")
        params += [name, name, text]
    return "select " + ", ".join(calls), params


def put_back_settings(settings: Mapping[str, object], scope: str) -> tuple[str, list[str]]:
    """``select``: each setting back to the value :func:`set_settings` kept aside under
    ``scope``; one that had no value is reset."""
    calls = [
        f"set_config(%s, nullif(current_setting('{_KEPT}{scope}_{i}', true), ''), true)"
        for i in range(len(settings))
    ]
    return "select " + ", ".join(calls), list(settings)


def mappings(result: Result | Rows) -> list[dict[str, Any]]:
    """A result's rows as mappings of column name to value."""
    return [dict(zip(result.keys, row, strict=True)) for row in result.records]


def scalar(result: Result) -> Any:
    """The first column of a result's first row, or None when there is no row."""
    return result.records[0][0] if result.records else None


__all__ = [
    "Call",
    "InTransaction",
    "Pipeline",
    "Rows",
    "Program",
    "Statement",
    "Step",
    "arun",
    "mappings",
    "put_back_settings",
    "run",
    "scalar",
    "set_settings",
]
