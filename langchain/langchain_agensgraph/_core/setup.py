"""Setup that runs once, the first time a component needs it, and costs one read after that.

A component's constructor sends nothing. Its first call runs ``check``, a program answering whether
everything the component needs exists -- one catalog read -- and only when something is missing
runs ``create``, on a maintenance connection, in a transaction holding the graph's schema lock: the
driver's, which its own ``ensure_*`` methods take too, so two processes setting up one graph take
turns. ``create`` must therefore look again before making anything, since another process may have
made it meanwhile.

Inside a caller's transaction setup runs in that transaction, as maintenance work does, and is
remembered as done only once the transaction commits: a rollback takes what it made with it, so the
next call checks again.

Components that make the same thing on one engine share whether it is done, so one made per
request, as a chat history is, pays no check after the first.

Once done, setup is not run again until it is forgotten, which happens to every component on an
engine when the objects may have gone: for a graph that ``AgensGraph.drop_graph`` drops, and for
all when the server answers that a table, schema or label is missing, as after a drop elsewhere.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Iterable, Mapping

import psycopg

from ..errors import UnknownLabel
from .engine import AgensEngine
from .names import ident, qualified
from .program import InTransaction, Program, Statement, scalar

_PRESENT = (
    "SELECT count(*) FROM pg_catalog.pg_class c "
    "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
    "JOIN unnest(%s::text[], %s::text[]) AS wanted(nspname, relname) "
    "ON n.nspname = wanted.nspname AND c.relname = wanted.relname"
)


def present(relations: Iterable[tuple[str, str]]) -> Program[bool]:
    """Whether every relation, a ``(schema, name)`` pair, exists.

    Read from the catalog tables, not by looking each name up with ``to_regclass``: a name a
    session looked up and did not find can stay not found for the rest of its transaction, so a
    first call inside a caller's transaction would not see what setup has just made on another
    connection.
    """
    wanted = sorted(set(relations))
    found = yield Statement(
        _PRESENT, ([schema for schema, _ in wanted], [name for _, name in wanted])
    )
    return bool(scalar(found) == len(wanted))


def graph_exists(graph: str) -> Program[bool]:
    found = yield Statement("SELECT EXISTS (SELECT 1 FROM ag_graph WHERE graphname = %s)", (graph,))
    return bool(scalar(found))


def made_graph(graph: str, *, create: bool) -> Program[None]:
    """The graph made when it is missing and ``create`` says to; refused when it is missing and
    ``create`` does not."""
    if (yield from graph_exists(graph)):
        return
    if not create:
        raise UnknownLabel(
            f"the graph {graph!r} does not exist; pass create_graph=True to create it"
        )
    yield Statement(f"CREATE GRAPH IF NOT EXISTS {ident(graph)}")


class Migrations:
    """The statements that make a component's tables, each numbered and applied once, its number
    recorded in ``table`` of ``schema`` when it is.

    Append-only: a number is never reused, so a change to what an applied step made is a new step.
    :meth:`check`, :meth:`create` and :attr:`key` are a :class:`Setup`'s.
    """

    def __init__(self, schema: str, table: str, steps: Mapping[int, str]) -> None:
        self.schema = schema
        self.table = table
        self.steps = dict(sorted(steps.items()))
        self._recorded = qualified(schema, table)

    @property
    def key(self) -> Hashable:
        """What these migrations make, the same for every component that makes it."""
        return (self.schema, self.table, tuple(self.steps.items()))

    def check(self) -> Program[bool]:
        """Whether every step is recorded: one statement, which fails, as an answer, while the
        table does not exist yet."""
        try:
            found = yield Statement(
                f"SELECT count(*) FROM {self._recorded} WHERE v = ANY(%s)",
                (list(self.steps),),
                may_fail=True,
            )
        except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName):
            return False
        return bool(scalar(found) == len(self.steps))

    def create(self) -> Program[None]:
        """Apply, in order, every step not recorded."""
        yield Statement(f"CREATE SCHEMA IF NOT EXISTS {ident(self.schema)}")
        yield Statement(f"CREATE TABLE IF NOT EXISTS {self._recorded} (v integer PRIMARY KEY)")
        done = yield Statement(f"SELECT v FROM {self._recorded}")
        applied = {row[0] for row in done.records}
        for number, step in self.steps.items():
            if number not in applied:
                yield Statement(step)
                yield Statement(f"INSERT INTO {self._recorded} (v) VALUES (%s)", (number,))


class Setup:
    """A component's setup on one engine and graph.

    ``key`` names what ``create`` makes, for components that make the same thing: the engine
    then keeps whether it is done once for all of them. ``forgotten`` is called when the setup is
    forgotten, for whatever else the component keeps about objects that may have gone with it.
    """

    def __init__(
        self,
        engine: AgensEngine,
        graph: str,
        check: Callable[[], Program[bool]],
        create: Callable[[], Program[None]],
        *,
        key: Hashable | None = None,
        forgotten: Callable[[], None] | None = None,
    ) -> None:
        self.engine = engine
        self.graph = graph
        self._check = check
        self._create = create
        self._key = None if key is None else (graph, key)
        self._forgotten = forgotten
        self._done = False
        self._lock = threading.Lock()
        engine.track_setup(self)

    @property
    def done(self) -> bool:
        if self._key is not None:
            return self.engine.made(self._key)
        return self._done

    def ensure(self) -> None:
        """Make what is missing, once."""
        if self.done:
            return
        with self._lock:
            if self.done:
                return
            if not self.engine.run(self._check, replay=True):
                self.engine.run(self._locked, kind="maintenance")
            self.engine.remember(self._made)

    async def aensure(self) -> None:
        """The awaiting form of :meth:`ensure`. Two first calls at once both check, and the
        second to take the lock finds nothing left to make."""
        if self.done:
            return
        if not await self.engine.arun(self._check, replay=True):
            await self.engine.arun(self._locked, kind="maintenance")
        self.engine.remember(self._made)

    def verify(self, missing: Callable[[], Exception]) -> None:
        """Check, once, and make nothing: raise ``missing()`` when something is missing."""
        if self.done:
            return
        if not self.engine.run(self._check, replay=True):
            raise missing()
        self.engine.remember(self._made)

    async def averify(self, missing: Callable[[], Exception]) -> None:
        """The awaiting form of :meth:`verify`."""
        if self.done:
            return
        if not await self.engine.arun(self._check, replay=True):
            raise missing()
        self.engine.remember(self._made)

    def forget(self) -> None:
        """Check again on the next call."""
        self._done = False
        if self._key is not None:
            self.engine.unmake(self._key)
        if self._forgotten is not None:
            self._forgotten()

    def _made(self) -> None:
        self._done = True
        if self._key is not None:
            self.engine.make(self._key)

    def _locked(self) -> Program[None]:
        yield InTransaction(self._create(), lock=self.graph)
