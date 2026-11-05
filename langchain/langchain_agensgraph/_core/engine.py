"""Connections for every component: one engine per process and database, shared.

An :class:`AgensEngine` keeps the driver's pools, opened on first use, one for each kind of work
so that one kind cannot starve another:

==============  ==================================================================================
``main``        every ordinary call; prepared statements, the default statement timeout
``maintenance`` DDL, index builds, ``VACUUM``, purges: long work that must not hold ``main``
``untrusted``   model-written Cypher, as a role made for it (``untrusted_conninfo``)
``writer``      model-written Cypher allowed to write (``writer_conninfo``), never retried
==============  ==================================================================================

Every pool connection runs in autocommit, so a single statement costs one round trip and a
pipeline of statements one implicit transaction. A component gives the engine a program (see
:mod:`.program`) and the graph it reads, and the engine borrows a connection, moves it to that
graph only if it is elsewhere, and runs the program.

A component handed a conninfo string rather than an engine shares one with every other component
handed the same string, through a process-wide cache that closes the engine when the last
component using it closes.

Inside :meth:`AgensEngine.transaction` every call a component makes through the engine, in the
thread or task that opened it, joins the caller's transaction on the caller's connection, so
package writes and the caller's own statements commit or roll back together.
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import os
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable, Hashable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Literal, TypeAlias, TypeVar

import agensgraph
from agensgraph import AsyncConnection, Capabilities, Connection, RetryPolicy
from agensgraph.errors import ConfigurationError, Retryability, retryability
from agensgraph.retry import full_jitter
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from ..errors import ClosedError
from .names import checked
from .program import Program, arun, run

T = TypeVar("T")

Kind = Literal["main", "maintenance", "untrusted", "writer"]

# Kinds whose calls join a caller's open transaction. Maintenance work does too: on a connection
# of its own, DDL on a table the caller's transaction has written to would wait for that
# transaction's locks while the caller waits for the DDL, and neither would ever end. Its work is
# then undone with the caller's, so what it makes there is remembered as made only once the
# caller commits (AgensEngine.remember), and a concurrent index build, which cannot run in a
# transaction, is refused there. Model-written statements always run in a transaction of their own.
_JOINS_CALLER = frozenset({"main", "maintenance"})

# What the server answers when a table, a schema, or a label or other object is missing.
_GONE = frozenset({"42P01", "3F000", "42704"})

# What it answers when two writers met over one element: a concurrent update, a deadlock.
_CONFLICTS = frozenset({"40001", "40P01"})

_Binding: TypeAlias = tuple[tuple["AgensEngine", bool, object], Any, list[Callable[[], None]]]

# The open transaction blocks of the calling context, innermost last: (engine, whether it is
# blocking, the thread or task that opened it), the connection it runs on, and what is to be
# remembered once it commits.
_bound: contextvars.ContextVar[tuple[_Binding, ...]] = contextvars.ContextVar(
    "langchain_agensgraph_bound_connections", default=()
)


def _owner(*, sync: bool) -> object:
    """The thread running blocking work, or the task running awaiting work; None outside one."""
    if sync:
        return threading.get_ident()
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def package_version() -> str:
    try:
        return importlib.metadata.version("langchain-agensgraph")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class AgensEngine:
    """The pools every component of this package borrows its connections from.

    ``statement_timeout`` (seconds, ``None`` for none) applies to ``main`` connections from the
    moment they connect, so it costs no statement; maintenance work sets a budget of its own.
    ``prepare_threshold`` and ``prepared_max`` are psycopg's, for ``main``; the other kinds never
    prepare. ``configure`` is called on every new connection after the engine's own setup, and
    ``pool_kwargs`` go to the ``main`` pool.
    """

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        application_name: str | None = None,
        statement_timeout: float | None = 30.0,
        connect_timeout: int = 5,
        prepare_threshold: int | None = 5,
        prepared_max: int = 128,
        maintenance_max_size: int = 2,
        untrusted_conninfo: str | None = None,
        untrusted_max_size: int = 4,
        writer_conninfo: str | None = None,
        configure: Callable[[Connection[Any]], None] | None = None,
        aconfigure: Callable[[AsyncConnection[Any]], Any] | None = None,
        retry_policy: RetryPolicy | None = None,
        **pool_kwargs: Any,
    ) -> None:
        if min_size < 0 or max_size < 1 or min_size > max_size:
            raise ValueError("pool sizes must satisfy 0 <= min_size <= max_size and max_size >= 1")
        self.conninfo = conninfo
        self.min_size = min_size
        self.max_size = max_size
        self.application_name = application_name or f"langchain-agensgraph/{package_version()}"
        self.statement_timeout = statement_timeout
        self.connect_timeout = connect_timeout
        self.prepare_threshold = prepare_threshold
        self.prepared_max = prepared_max
        self.maintenance_max_size = maintenance_max_size
        self.untrusted_conninfo = untrusted_conninfo
        self.untrusted_max_size = untrusted_max_size
        self.writer_conninfo = writer_conninfo
        self.retry_policy = retry_policy or RetryPolicy()
        self._configure_hook = configure
        self._aconfigure_hook = aconfigure
        self._pool_kwargs = pool_kwargs
        self._pools: dict[Kind, agensgraph.ConnectionPool] = {}
        self._apools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopPools] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()
        self._capabilities: Capabilities | None = None
        self._main_role: str | None = None
        self._setups: weakref.WeakSet[Any] = weakref.WeakSet()
        self._made: set[tuple[str, Hashable]] = set()
        self._closed = False
        _ENGINES.add(self)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> AgensEngine:
        """An engine for a ``postgresql://user:password@host:port/dbname`` URL."""
        return cls(url, **kwargs)

    # -- what the server is ----------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        """What the server can do, read from the first connection the engine made.

        Free once a connection has been borrowed; before that, borrowing one is what answers.
        """
        if self._capabilities is None:
            with self.connection():
                pass
        assert self._capabilities is not None
        return self._capabilities

    async def acapabilities(self) -> Capabilities:
        if self._capabilities is None:
            async with self.aconnection():
                pass
        assert self._capabilities is not None
        return self._capabilities

    @property
    def main_role(self) -> str:
        """The role ``main`` connections log in as: the application's own role, which a role
        running model-written Cypher must not be or belong to."""
        if self._main_role is None:
            with self.connection() as conn:
                self._main_role = str(conn.info.user)
        return self._main_role

    async def amain_role(self) -> str:
        if self._main_role is None:
            async with self.aconnection() as conn:
                self._main_role = str(conn.info.user)
        return self._main_role

    # -- borrowing ---------------------------------------------------------------------------

    @contextmanager
    def connection(
        self, kind: Kind = "main", *, graph: str | None = None
    ) -> Iterator[Connection[Any]]:
        """A connection of this kind for the block, on ``graph`` if one is named.

        Inside :meth:`transaction`, the caller's own connection for the kinds that join it.
        """
        bound = self._bound(kind, sync=True)
        if bound is not None:
            on_graph(bound, graph)
            yield bound
            return
        with self._pool(kind).connection() as conn:
            self._learn(conn)
            on_graph(conn, graph)
            yield conn

    @asynccontextmanager
    async def aconnection(
        self, kind: Kind = "main", *, graph: str | None = None
    ) -> AsyncIterator[AsyncConnection[Any]]:
        """The awaiting form of :meth:`connection`."""
        bound = self._bound(kind, sync=False)
        if bound is not None:
            await aon_graph(bound, graph)
            yield bound
            return
        pool = await self._apool(kind)
        async with pool.connection() as conn:
            self._learn(conn)
            await aon_graph(conn, graph)
            yield conn

    def run(
        self,
        program: Callable[[], Program[T]],
        *,
        graph: str | None = None,
        kind: Kind = "main",
        replay: bool = False,
        merging: bool = False,
    ) -> T:
        """Run the program ``program()`` makes on a borrowed connection.

        ``replay=True`` says the program may run again from the start after a failure the driver
        classes as transient (a lost connection, a serialization failure, a deadlock): a read, or
        a write whose second run changes nothing the first did not. It is never replayed inside a
        caller's transaction, whose earlier statements the replay could not repeat. ``merging``
        says the program makes only what is missing, so another writer having made it first
        (``23505``) is a reason to run it again too.

        When the server answers that a table, schema or label is missing, every component on the
        engine checks its setup again on its next call.
        """
        try:
            return self._run(program, graph=graph, kind=kind, replay=replay, merging=merging)
        except Exception as exc:
            self._forget_if_gone(exc)
            raise

    def _run(
        self,
        program: Callable[[], Program[T]],
        *,
        graph: str | None,
        kind: Kind,
        replay: bool,
        merging: bool,
    ) -> T:
        joined = self._bound(kind, sync=True) is not None
        if not replay or joined:
            with self.connection(kind, graph=graph) as conn:
                return run(conn, program())
        policy = self.retry_policy
        previous: list[BaseException] = []
        for number in range(1, policy.attempts + 1):
            try:
                with self.connection(kind, graph=graph) as conn:
                    try:
                        result = run(conn, program())
                    except Exception as exc:
                        _recover(conn, exc)
                        raise
                policy.succeeded()
                return result
            except Exception as exc:
                delay = _again(policy, exc, number=number, merging=merging)
                if delay is None:
                    raise
                previous.append(exc)
                time.sleep(delay)
        raise policy.exhausted(previous[-1], attempts=policy.attempts, previous=previous)

    async def arun(
        self,
        program: Callable[[], Program[T]],
        *,
        graph: str | None = None,
        kind: Kind = "main",
        replay: bool = False,
        merging: bool = False,
    ) -> T:
        """The awaiting form of :meth:`run`."""
        try:
            return await self._arun(program, graph=graph, kind=kind, replay=replay, merging=merging)
        except Exception as exc:
            self._forget_if_gone(exc)
            raise

    async def _arun(
        self,
        program: Callable[[], Program[T]],
        *,
        graph: str | None,
        kind: Kind,
        replay: bool,
        merging: bool,
    ) -> T:
        joined = self._bound(kind, sync=False) is not None
        if not replay or joined:
            async with self.aconnection(kind, graph=graph) as conn:
                return await arun(conn, program())
        policy = self.retry_policy
        previous: list[BaseException] = []
        for number in range(1, policy.attempts + 1):
            try:
                async with self.aconnection(kind, graph=graph) as conn:
                    try:
                        result = await arun(conn, program())
                    except Exception as exc:
                        await _arecover(conn, exc)
                        raise
                policy.succeeded()
                return result
            except Exception as exc:
                delay = _again(policy, exc, number=number, merging=merging)
                if delay is None:
                    raise
                previous.append(exc)
                await asyncio.sleep(delay)
        raise policy.exhausted(previous[-1], attempts=policy.attempts, previous=previous)

    # -- components' setup -----------------------------------------------------------------

    def track_setup(self, setup: Any) -> None:
        """Keep a component's setup, to forget it when what it made may have gone."""
        self._setups.add(setup)

    def made(self, key: tuple[str, Hashable]) -> bool:
        """Whether a setup shared by its key, ``(graph, what it makes)``, is done."""
        return key in self._made

    def make(self, key: tuple[str, Hashable]) -> None:
        self._made.add(key)

    def unmake(self, key: tuple[str, Hashable]) -> None:
        self._made.discard(key)

    def forget_setups(self, graph: str | None = None) -> None:
        """Have every component on this engine with a setup for ``graph``, or for any graph,
        check it again on its next call."""
        self._made = {key for key in self._made if graph is not None and key[0] != graph}
        for setup in list(self._setups):
            if graph is None or setup.graph == graph:
                setup.forget()

    def _forget_if_gone(self, exc: BaseException) -> None:
        cause: BaseException | None = exc
        while cause is not None:
            if getattr(cause, "sqlstate", None) in _GONE:
                self.forget_setups()
                return
            cause = cause.__cause__

    # -- a caller's transaction --------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Connection[Any]]:
        """A transaction that every call through this engine joins until the block ends.

        The block's connection is yielded for statements of the caller's own. A failure in the
        block rolls everything back; inside another such block this is a savepoint.
        """
        held: list[Callable[[], None]] = []
        bound = self._bound("main", sync=True)
        if bound is not None:
            with _graph_forgotten_on_failure(bound), bound.transaction():
                with self._binding(bound, held, sync=True):
                    yield bound
        else:
            with self._pool("main").connection() as conn:
                self._learn(conn)
                with _graph_forgotten_on_failure(conn), conn.transaction():
                    with self._binding(conn, held, sync=True):
                        yield conn
        for remembered in held:
            self.remember(remembered)

    @asynccontextmanager
    async def atransaction(self) -> AsyncIterator[AsyncConnection[Any]]:
        """The awaiting form of :meth:`transaction`."""
        held: list[Callable[[], None]] = []
        bound = self._bound("main", sync=False)
        if bound is not None:
            with _graph_forgotten_on_failure(bound):
                async with bound.transaction():
                    with self._binding(bound, held, sync=False):
                        yield bound
        else:
            pool = await self._apool("main")
            async with pool.connection() as conn:
                self._learn(conn)
                with _graph_forgotten_on_failure(conn):
                    async with conn.transaction():
                        with self._binding(conn, held, sync=False):
                            yield conn
        for remembered in held:
            self.remember(remembered)

    @contextmanager
    def _binding(self, conn: Any, held: list[Callable[[], None]], *, sync: bool) -> Iterator[None]:
        """The block's connection bound to the calling thread or task, with what is to be
        remembered once the block commits."""
        token = _bound.set((*_bound.get(), ((self, sync, _owner(sync=sync)), conn, held)))
        try:
            yield
        finally:
            _bound.reset(token)

    def in_transaction(self) -> bool:
        """Whether the caller is inside :meth:`transaction` or :meth:`atransaction`: in the thread
        or task that opened it."""
        return any(
            engine is self and owner == _owner(sync=sync)
            for (engine, sync, owner), _, _ in _bound.get()
        )

    def remember(self, remembered: Callable[[], None]) -> None:
        """Call ``remembered`` once what the calling thread or task has done is committed.

        For what a component keeps about what it made, which a rollback would take back: outside
        a caller's transaction it is called now; inside one, when the transaction commits, and
        never if it, or the savepoint it was made in, rolls back.
        """
        for (engine, sync, owner), _, held in reversed(_bound.get()):
            if engine is self and owner == _owner(sync=sync):
                held.append(remembered)
                return
        remembered()

    def _bound(self, kind: Kind, *, sync: bool) -> Any:
        """The caller's transaction's connection, for work of ``kind`` in the thread or task
        that opened it.

        A thread pool that copies the context, as LangChain's does, or a child task, which
        always does, sees the transaction too; work there runs outside it, on a connection of its
        own, as one connection can serve only one caller at a time.
        """
        if kind not in _JOINS_CALLER:
            return None
        here = _owner(sync=sync)
        for (engine, blocking, owner), conn, _ in reversed(_bound.get()):
            if engine is self and blocking == sync and owner == here:
                return conn
        return None

    # -- pools -----------------------------------------------------------------------------

    def _pool(self, kind: Kind) -> agensgraph.ConnectionPool:
        pool = self._pools.get(kind)
        if pool is not None:
            return pool
        with self._lock:
            if self._closed:
                raise ClosedError("the engine is closed")
            pool = self._pools.get(kind)
            if pool is None:
                conninfo, options = self._spec(kind)
                pool = agensgraph.ConnectionPool(conninfo, configure=self._configure, **options)
                pool.open(wait=True, timeout=self._open_timeout())
                self._pools[kind] = pool
        return pool

    async def _apool(self, kind: Kind) -> agensgraph.AsyncConnectionPool:
        if self._closed:
            raise ClosedError("the engine is closed")
        loop = asyncio.get_running_loop()
        held = self._apools.get(loop)
        if held is None:
            held = self._apools.setdefault(loop, _LoopPools())
        pool = held.pools.get(kind)
        if pool is not None:
            return pool
        async with held.lock:
            pool = held.pools.get(kind)
            if pool is None:
                conninfo, options = self._spec(kind)
                pool = agensgraph.AsyncConnectionPool(
                    conninfo, configure=self._aconfigure, **options
                )
                await pool.open(wait=True, timeout=self._open_timeout())
                held.pools[kind] = pool
        return pool

    def _open_timeout(self) -> float:
        return float(max(self.connect_timeout, 1) * 3)

    def _spec(self, kind: Kind) -> tuple[str, dict[str, Any]]:
        """The conninfo and pool options for a kind."""
        conninfo = self.conninfo
        if kind == "untrusted" and self.untrusted_conninfo is not None:
            conninfo = self.untrusted_conninfo
        if kind == "writer":
            if self.writer_conninfo is None:
                raise ConfigurationError(
                    "model-written Cypher that may write needs writer_conninfo: a role allowed "
                    "exactly the writes a model may make, and nothing else"
                )
            conninfo = self.writer_conninfo
        kwargs: dict[str, Any] = {
            "autocommit": True,
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout,
            "prepare_threshold": self.prepare_threshold if kind == "main" else None,
        }
        if self.statement_timeout is not None and kind == "main":
            kwargs["options"] = _with_option(
                conninfo, f"-c statement_timeout={int(self.statement_timeout * 1000)}"
            )
        options: dict[str, Any] = {"kwargs": kwargs, "name": f"langchain-agensgraph-{kind}"}
        if kind == "main":
            options.update(min_size=self.min_size, max_size=self.max_size)
            options.update(self._pool_kwargs)
        elif kind == "maintenance":
            options.update(min_size=0, max_size=self.maintenance_max_size, max_idle=60.0)
        elif kind == "untrusted":
            options.update(min_size=0, max_size=self.untrusted_max_size)
        else:
            options.update(min_size=0, max_size=max(1, self.untrusted_max_size // 2))
        return conninfo, options

    def _configure(self, conn: Connection[Any]) -> None:
        conn.prepared_max = self.prepared_max
        conn.register_vectors()
        if self._configure_hook is not None:
            self._configure_hook(conn)

    async def _aconfigure(self, conn: AsyncConnection[Any]) -> None:
        conn.prepared_max = self.prepared_max
        await conn.register_vectors()
        if self._aconfigure_hook is not None:
            await self._aconfigure_hook(conn)

    def _learn(self, conn: Connection[Any] | AsyncConnection[Any]) -> None:
        if self._capabilities is None:
            self._capabilities = conn.capabilities

    # -- closing -----------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close every blocking pool. Awaiting pools need :meth:`aclose` on their own loop."""
        with self._lock:
            self._closed = True
            pools, self._pools = list(self._pools.values()), {}
        for pool in pools:
            pool.close()

    async def aclose(self) -> None:
        """Close this loop's awaiting pools and every blocking pool."""
        self._closed = True
        held = self._apools.pop(asyncio.get_running_loop(), None)
        if held is not None:
            for pool in list(held.pools.values()):
                await pool.close()
        self.close()

    def __enter__(self) -> AgensEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> AgensEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _forget(self) -> None:
        """Drop every pool without closing it: after a fork, they belong to the parent."""
        self._pools = {}
        self._apools = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        parts = conninfo_to_dict(self.conninfo)
        where = f"{parts.get('host', '')}:{parts.get('port', '')}/{parts.get('dbname', '')}"
        return f"AgensEngine({where!r}{', closed' if self._closed else ''})"


class _LoopPools:
    """The awaiting pools of one event loop, and the lock that makes each once."""

    def __init__(self) -> None:
        self.pools: dict[Kind, agensgraph.AsyncConnectionPool] = {}
        self.lock = asyncio.Lock()


def on_graph(conn: Connection[Any], graph: str | None) -> None:
    """Move a connection to a graph, unless it is known to be there already."""
    if graph is not None and conn.label_table.graph != graph:
        conn.graph(graph)


async def aon_graph(conn: AsyncConnection[Any], graph: str | None) -> None:
    if graph is not None and conn.label_table.graph != graph:
        await conn.graph(graph)


@contextmanager
def _graph_forgotten_on_failure(conn: Connection[Any] | AsyncConnection[Any]) -> Iterator[None]:
    """Forget which graph a connection reads when the block around a transaction fails.

    Moving to a graph inside a transaction is undone by its rollback. A server that reports the
    graph path tells the driver so; an older one does not, and the driver would go on believing the
    connection reads the graph it no longer does. Forgetting costs the next call one move.
    """
    try:
        yield
    except BaseException:
        if not conn.capabilities.has("graph_path_reported"):
            conn.label_table.invalidate()
        raise


def _recover(conn: Connection[Any], failure: BaseException) -> None:
    try:
        conn.recover(failure)
    except Exception:  # noqa: BLE001 - the original failure is the one to report
        pass


async def _arecover(conn: AsyncConnection[Any], failure: BaseException) -> None:
    try:
        await conn.recover(failure)
    except Exception:  # noqa: BLE001 - the original failure is the one to report
        pass


def _again(
    policy: RetryPolicy, failure: BaseException, *, number: int, merging: bool
) -> float | None:
    """How long to wait before running a replayable program again; None to give up.

    The driver's policy decides, within the process's retry allowance, which holds retries back
    from a server in trouble. A merging program meeting another writer of the same elements is
    the exception: that writer committed, so running again finds what it made, and the failure
    says nothing about the server. It spends nothing of the allowance, which concurrent writers of
    the same keys would otherwise empty in one burst, and is bounded by the policy's attempts
    alone.
    """
    if merging and _met_another_writer(failure):
        return full_jitter(number) if number < policy.attempts else None
    attempt = policy.decide(failure, number=number)
    return attempt.delay if attempt.retry else None


def _met_another_writer(failure: BaseException) -> bool:
    """Whether a failure is another writer's work meeting this one's: a concurrent update of the
    same element (``40001``, which AgensGraph's SET and MERGE report), a deadlock (``40P01``), or
    the other having made what this one was making (``23505``, ``42P07``)."""
    if getattr(failure, "sqlstate", None) in _CONFLICTS:
        return True
    return (
        retryability(failure) is Retryability.FATAL
        and retryability(failure, merging=True) is Retryability.SAFE
    )


def _with_option(conninfo: str, option: str) -> str:
    """``options`` for a connection: the conninfo's own, then this one."""
    try:
        given = conninfo_to_dict(conninfo).get("options")
    except Exception:  # noqa: BLE001 - a malformed conninfo is reported when it is used
        given = None
    return f"{given} {option}" if given else option


# -- one engine per conninfo, shared -------------------------------------------------------

_ENGINES: weakref.WeakSet[AgensEngine] = weakref.WeakSet()
_shared: dict[str, tuple[AgensEngine, int]] = {}
_shared_lock = threading.Lock()


def acquire(engine: AgensEngine | str) -> tuple[AgensEngine, bool]:
    """The engine a component uses, and whether the component holds a share of it.

    An engine passed in is the caller's to close. A conninfo string resolves to the process's
    shared engine for that string, made on first use and closed by :func:`release` when the last
    component holding a share releases it.
    """
    if isinstance(engine, AgensEngine):
        if engine.closed:
            raise ClosedError("the engine is closed")
        return engine, False
    if not isinstance(engine, str):
        raise TypeError(f"engine must be an AgensEngine or a conninfo string, not {engine!r}")
    key = make_conninfo(engine)
    with _shared_lock:
        held = _shared.get(key)
        if held is None or held[0].closed:
            shared = AgensEngine(engine)
            _shared[key] = (shared, 1)
            return shared, True
        _shared[key] = (held[0], held[1] + 1)
        return held[0], True


def release(engine: AgensEngine) -> None:
    """Give back a share of a shared engine, closing it when no component holds one."""
    if _last_share(engine):
        engine.close()


async def arelease(engine: AgensEngine) -> None:
    """:func:`release`, closing this loop's awaiting pools too when it is the last share."""
    if _last_share(engine):
        await engine.aclose()


def _last_share(engine: AgensEngine) -> bool:
    """Give back a share of a shared engine; whether it was the last one."""
    with _shared_lock:
        for key, (shared, count) in list(_shared.items()):
            if shared is engine:
                if count > 1:
                    _shared[key] = (shared, count - 1)
                    return False
                del _shared[key]
                return True
    return False


def bind(
    engine: AgensEngine | str | None, graph: Any, graph_name: str | None
) -> tuple[AgensEngine, bool, str]:
    """The engine and graph a component works with, from what it was given.

    ``graph`` is an object with ``engine`` and ``graph_name``, such as an ``AgensGraph``; without
    it, ``engine`` and ``graph_name`` are both required. Returns the engine, whether the component
    holds a share of it (see :func:`acquire`), and the graph's name.
    """
    if graph is not None:
        name = graph.graph_name
        if graph_name is not None and graph_name != name:
            raise ValueError(f"graph_name {graph_name!r} differs from the graph's {name!r}")
        return graph.engine, False, checked(name, "graph name")
    if engine is None:
        raise ValueError("pass engine= (an AgensEngine or a conninfo string) or graph=")
    if graph_name is None:
        raise ValueError("graph_name is required")
    made, owned = acquire(engine)
    return made, owned, checked(graph_name, "graph name")


def _after_fork() -> None:
    global _shared_lock
    for engine in list(_ENGINES):
        engine._forget()
    _shared_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
