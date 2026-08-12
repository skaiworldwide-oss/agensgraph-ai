"""
Copyright (c) 2025, SKAI Worldwide Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from hashlib import md5
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import agensgraph
from agensgraph import Edge, Path, RetryPolicy, SparseVector, Vector, Vertex
from agensgraph.cypher import quote_identifier
from agensgraph.errors import BatchFailed, is_retryable
from agensgraph.introspect import DesiredIndex, reconcile_indexes
from agensgraph.retry import TokenBucket
from psycopg import sql
from psycopg.errors import DuplicateObject, DuplicateTable, UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.graph_document import GraphDocument
from langchain_agensgraph.graphs.graph_store import GraphStore

if TYPE_CHECKING:
    from langchain_agensgraph.engine import AgensEngine

_T = TypeVar("_T")

LIST_LIMIT = 128
"""List-valued result properties longer than this are dropped when sanitize=True."""

_IN_TRANSACTION: contextvars.ContextVar[Mapping[int, Any]] = contextvars.ContextVar(
    "agensgraph_in_transaction", default={}
)
"""The connection the calling thread or task holds a transaction on, per graph.

The connection and not merely the fact of one: a transaction belongs to one connection,
so every statement inside the block has to run on *that* one. With a pool there is no
reason for it to be the dedicated connection -- and if it is, every caller holding a
transaction at once piles onto the same connection and they refuse each other.

A transaction belongs to one connection, so a statement made while one is open has to
run on that connection and must not commit, since the block around it owns the commit.
Which is true *of the caller*, not of the graph: a graph is shared, and a second caller
reaching it while the first is mid-transaction has a statement of its own to run, which
belongs on a connection of its own. Holding this in a context variable is what scopes it
to the caller -- a thread sees only what it set, and so does a task, since a task is
given a copy of the context it was created in.
"""

_IN_ASYNC_TRANSACTION: contextvars.ContextVar[Mapping[int, Any]] = (
    contextvars.ContextVar("agensgraph_in_async_transaction", default={})
)
"""The same, for the async connection, which is a different connection.

Awaited work and blocking work run in one context when one calls the other, so a single
set would offer an awaiting caller the blocking connection and the other way round.
"""

DEFAULT_SCHEMA_SAMPLE = 100
"""How many elements of each label are read to learn what properties it holds.

A label's shape does not need every row to establish, and reading every row of a label
full of embeddings to learn that one key holds an array is minutes rather than
milliseconds.
"""

ENHANCED_SCHEMA_SAMPLE = 1000
"""The larger sample ``enhanced_schema=True`` asks for."""


_UNSET: Any = object()
"""Distinguishes "no timeout has been stated on this transaction" from "no timeout"."""


class _Pin:
    """The connection a caller's transaction is on, and the timeout it is carrying.

    The timeout is recorded here rather than against the graph because a pinned connection
    may be one of the pool's, and what one borrowed connection carries says nothing about
    what the next one does. Recording it per transaction also means nothing has to be put
    back afterwards: a transaction states its timeout with ``SET LOCAL``, so the value ends
    when the transaction does, whether it commits or rolls back.
    """

    __slots__ = ("conn", "applied")

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.applied: Any = _UNSET


def checked_names(**names: Optional[str]) -> None:
    """Refuse a caller-supplied label or key that would not reach the server intact.

    The server's lexer stops at a null byte, so a name holding one is quoted into a
    statement that ends somewhere other than where it appears to: a label written
    ``Ses\0sion`` is composed as ``"Ses"`` and the server is asked about a different
    label than the caller named. The driver refuses such a name rather than quoting it,
    and this asks it to, at the point the name arrives rather than at the statement it
    ends up in.
    """
    for role, name in names.items():
        if name is None:
            continue
        try:
            quote_identifier(name)
        except ValueError as exc:
            raise ValueError(f"{role}: {exc}") from exc


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("langchain-agensgraph")
        except PackageNotFoundError:
            return "dev"
    except Exception:
        return "dev"


def _with_application_name(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``conf`` with ``application_name`` set if not provided."""
    out = dict(conf)
    if not out.get("application_name"):
        out["application_name"] = f"langchain-agensgraph/{_package_version()}"
    return out


def _sanitize_value(value: Any, *, list_limit: int = LIST_LIMIT) -> Any:
    """Recursively drop oversized lists from a result value.

    A list with more than ``list_limit`` elements is removed entirely, so embeddings and
    other large arrays do not end up serialized into an LLM prompt.

    A graph element is described by its identity, its label and its properties, and only
    the properties are walked -- which is also the only part of it that can hold a large
    list.
    """
    if isinstance(value, (Vertex, Edge)):
        return _sanitize_value(agensgraph.to_builtins(value), list_limit=list_limit)
    if isinstance(value, Path):
        return {
            "vertices": [
                _sanitize_value(v, list_limit=list_limit) for v in value.vertices
            ],
            "edges": [_sanitize_value(e, list_limit=list_limit) for e in value.edges],
        }
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            sv = _sanitize_value(v, list_limit=list_limit)
            if sv is not None or v is None:
                out[k] = sv
        return out
    if isinstance(value, Vector):
        # The value most likely to flood a prompt, and the one the length rule below does
        # not see: a vector is not a `list`, though it compares equal to one.
        return None if len(value) > list_limit else list(value)
    if isinstance(value, SparseVector):
        # Measured by the numbers it actually holds, which is what would be written out.
        return None if len(value) > list_limit else value.to_dict()
    if isinstance(value, list):
        if len(value) > list_limit:
            return None
        return [_sanitize_value(v, list_limit=list_limit) for v in value]
    return value


class AgensQueryException(Exception):
    """Exception for the Agensgraph queries.

    The server's own account of the failure is the ``__cause__`` where there was one, so
    an ``except`` clause can still reach the psycopg class and its SQLSTATE.
    """

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception.get("message", "unknown")
            # Both spellings are read. Every raise site writes "detail"; this class used
            # to read only "details", so the server's own account of the failure was
            # captured and then thrown away on every error.
            self.details = (
                exception.get("detail") or exception.get("details") or "unknown"
            )
        else:
            self.message = exception
            self.details = "unknown"
        super().__init__(self.message)

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class AgensGraph(GraphStore):
    """
    Agensgraph wrapper for graph operations.

    Args:
        graph_name (str): the name of the graph to connect to or create conf (Dict[str,
        Any]): the connection config, passed to the driver create (bool): if True and
        graph doesn't exist, attempt to create it

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions. Failure to do so
        may result in data corruption or loss, since the calling code may attempt
        commands that would result in deletion, mutation of data if appropriately
        prompted or reading sensitive data if such data is present in the database. The
        best way to guard against such negative outcomes is to (as appropriate) limit
        the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.

    A vertex comes back as an ``agensgraph.Vertex``, an edge as an ``Edge`` and a path
    as a ``Path``, each carrying its label, its identity and its properties.
    """

    # Precompiled regex for checking chars in graph labels.
    label_regex: Pattern = re.compile("[^0-9a-zA-Z]+")

    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        create: bool = False,
        schema_cache_ttl: float = 0.0,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        engine: Optional["AgensEngine"] = None,
        enhanced_schema: bool = False,
        refresh_schema: bool = True,
        retry_attempts: int = 3,
    ) -> None:
        """Create a new Agensgraph Graph instance.

        Args:
            graph_name: Name of the AgensGraph graph to use. conf: connection kwargs, as
            ``psycopg.connect`` takes them. create: Create the graph if it does not
            exist. schema_cache_ttl: When > 0, ``refresh_schema`` short-circuits if it
                ran within the last ``schema_cache_ttl`` seconds. Set to 0 (default) to
                disable caching.
            timeout: Default per-query statement timeout in seconds. ``None``
                disables it. Can be overridden per call via ``query(..., timeout=)``.
            sanitize: When True, list-valued result properties longer than
                ``LIST_LIMIT`` (128) are stripped from query results so large arrays
                (e.g. embeddings) do not flood an LLM context.
            engine: Optional :class:`~langchain_agensgraph.engine.AgensEngine`.
                When provided, ``query``/``aquery`` borrow pooled connections so
                concurrent callers don't serialize on one connection.
            enhanced_schema: When True, the schema is read from a larger sample of
                each label, which describes a label whose shape varies more closely.
            refresh_schema: When True (default), describe the graph eagerly in
                ``__init__``. Set False to avoid the round trips it costs.
            retry_attempts: How many times a write that merges on a unique key is made
                before giving up. Counts the first try, so three means one attempt and at
                most two retries. Raise it where many writers share keys.
        """
        # Before the graph is created rather than after: a name holding a null byte is
        # composed into a statement that ends at the byte, so the server creates a graph
        # by the shorter name and refusing it afterwards refuses nothing.
        checked_names(graph_name=graph_name)
        self.graph_name = graph_name
        # How many times a write that merges on a unique key is made before giving up.
        # Concurrent graph writes surface as a serialisation failure by design, so under
        # heavy contention a caller may want more than the three that suit most.
        self.retry_attempts = retry_attempts
        self._engine = engine
        self._conf = _with_application_name(conf)
        # Autocommit, so the connection holds no snapshot and no locks between calls.
        # `add_graph_documents` opens a transaction explicitly where it needs one.
        self.connection = agensgraph.connect(**self._conf, autocommit=True)
        # Held for as long as anything is using the shared connection. One connection does
        # one thing at a time, so two callers opening a transaction on it would interleave
        # their BEGIN and COMMIT. A pool gives each caller its own connection and does not
        # queue here.
        self._exclusive = threading.RLock()
        self._aexclusive = asyncio.Lock()
        # How much retrying this graph may do. The driver's process-wide allowance exists
        # to stop a retry count being multiplied by a stack of layers that each retry;
        # this is the only such layer here, so it holds its own and `retry_attempts`
        # governs what this graph does rather than what the process has left.
        self._allowance = TokenBucket()
        self.schema_cache_ttl = schema_cache_ttl
        self.timeout = timeout
        self.sanitize = sanitize
        self.enhanced_schema = enhanced_schema
        self._schema_refreshed_at: float = 0.0
        self._aconn: Optional[agensgraph.AsyncConnection] = None
        # Whether a caller has asked for vectors to be carried as themselves, so that the
        # async connection is told the same thing whenever it is opened.
        self._vectors_registered = False
        # What identifies this graph for the record of which indexes agree on it: the
        # server, the database and the graph, so two graphs of the same name on different
        # servers are not taken for one.
        self._where: Tuple[Any, ...] = (
            self._conf.get("host"),
            self._conf.get("port"),
            self._conf.get("dbname"),
            graph_name,
        )
        # What statement timeout each dedicated connection is currently carrying, so that
        # a caller asking for the same one every time pays for it once rather than per
        # query. One per connection: the value is session state, so what the blocking
        # connection carries says nothing about what the async one does.
        self._applied_timeout: Optional[float] = None
        self._aapplied_timeout: Optional[float] = None
        # Populated by refresh_schema(); initialized so attribute access is safe even
        # when the eager refresh is skipped (refresh_schema=False).
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        existing = {graph.name for graph in self.connection.graphs()}
        if graph_name not in existing:
            if not create:
                raise AgensQueryException(
                    {
                        "message": (
                            f'Graph "{graph_name}" does not exist in the database '
                            f'and "create" is set to False'
                        ),
                        "detail": f"known graphs: {sorted(existing)}",
                    }
                )
            self._run(
                sql.SQL("CREATE GRAPH IF NOT EXISTS {name}").format(
                    name=sql.Identifier(graph_name)
                )
            )

        # Selects the graph and fills the label table the composite rendering resolves a
        # label name through, both in one call.
        self.connection.graph(graph_name)

        if refresh_schema:
            self.refresh_schema()

    # ---------- connection handling ----------

    @property
    def capabilities(self) -> agensgraph.Capabilities:
        """What this server can do, read from its startup packet at no round trip."""
        return self.connection.capabilities

    @property
    def _in_transaction(self) -> bool:
        """Does the calling thread or task hold a transaction on this graph?"""
        return id(self) in _IN_TRANSACTION.get()

    @property
    def _pin(self) -> Optional[_Pin]:
        """The transaction this caller holds on this graph, if it holds one."""
        return _IN_TRANSACTION.get().get(id(self))

    @property
    def _pinned_connection(self) -> Any:
        """The connection this caller's transaction is on, if it holds one."""
        pin = self._pin
        return None if pin is None else pin.conn

    @property
    def _in_async_transaction(self) -> bool:
        """The same, for the async connection."""
        return id(self) in _IN_ASYNC_TRANSACTION.get()

    @property
    def _apin(self) -> Optional[_Pin]:
        return _IN_ASYNC_TRANSACTION.get().get(id(self))

    @property
    def _apinned_connection(self) -> Any:
        pin = self._apin
        return None if pin is None else pin.conn

    @contextmanager
    def _pinned(self, conn: Any) -> Iterator[None]:
        """Say that statements from here belong on ``conn``."""
        token = _IN_TRANSACTION.set(
            {**_IN_TRANSACTION.get(), id(self): _Pin(conn)}
        )
        try:
            yield
        finally:
            _IN_TRANSACTION.reset(token)

    @asynccontextmanager
    async def _apinned(self, conn: Any) -> AsyncIterator[None]:
        """Async sibling of :meth:`_pinned`."""
        token = _IN_ASYNC_TRANSACTION.set(
            {**_IN_ASYNC_TRANSACTION.get(), id(self): _Pin(conn)}
        )
        try:
            yield
        finally:
            _IN_ASYNC_TRANSACTION.reset(token)

    async def amerging(
        self, write: Callable[[], Any], *, attempts: Optional[int] = None
    ) -> Any:
        """Async sibling of :meth:`merging`.

        The awaited write paths need this for the same reason the blocking ones do, and
        more so: a unique merge key turns a lost race from a duplicate element into a
        refusal, so an awaited path without it does not write at all.
        """
        attempts = attempts if attempts is not None else self.retry_attempts
        policy = RetryPolicy(attempts=attempts, bucket=self._allowance)
        previous: List[BaseException] = []
        for number in range(1, attempts + 1):
            try:
                result = await write()
            except Exception as exc:
                cause = exc.__cause__ if exc.__cause__ is not None else exc
                lost_the_race = isinstance(cause, UniqueViolation)
                if not (lost_the_race or is_retryable(cause, wrote=True)):
                    raise
                decision = policy.decide(cause, number=number, wrote=True)
                if not (decision.retry or lost_the_race) or number == attempts:
                    final = policy.exhausted(exc, attempts=number, previous=previous)
                    if final is exc:
                        raise final
                    raise final from exc
                previous.append(exc)
                await asyncio.sleep(decision.delay)
                continue
            policy.succeeded()
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    @contextmanager
    def _borrowed(self) -> Iterator[Any]:
        """A connection to hold a transaction on for the length of a block.

        A caller already inside one keeps the connection it is on, so a component's write
        joins the transaction that encloses it rather than opening a second one. Taking a
        fresh connection instead is what makes a nested write escape: the enclosing
        transaction rolls back and the inner write, committed elsewhere, survives it. On
        the same connection the driver marks the inner block off with a savepoint, so it
        can fail on its own and still be undone with the outer one.

        Otherwise one of the pool's, because a transaction occupies its connection for as
        long as it lasts. Without a pool there is only the shared connection, and it is
        taken exclusively -- one connection can hold one transaction, so concurrent callers
        queue for it here rather than corrupting each other's.
        """
        held = self._pinned_connection
        if held is not None:
            yield held
            return
        if self._engine is not None:
            with self._engine.connection(self.graph_name) as conn:
                yield conn
            return
        with self._exclusive:
            yield self.connection

    @asynccontextmanager
    async def _aborrowed(self) -> AsyncIterator[Any]:
        """Async sibling of :meth:`_borrowed`."""
        held = self._apinned_connection
        if held is not None:
            yield held
            return
        if self._engine is not None:
            async with self._engine.aconnection(self.graph_name) as conn:
                yield conn
            return
        async with self._aexclusive:
            yield await self._aconn_get()

    @contextmanager
    def _dedicated(self) -> Iterator[Any]:
        """The connection this graph owns, held for the length of a block.

        The work that is about the handle rather than about a caller's statement -- reading
        the catalogs, creating labels, registering how a type is carried -- belongs here
        rather than on a borrowed connection, because some of it is session state that
        would be left on whichever connection happened to serve it. Held exclusively for
        the same reason a transaction is: another caller may have one open on it.
        """
        if self._pinned_connection is self.connection:
            yield self.connection
            return
        with self._exclusive:
            yield self.connection

    @contextmanager
    def _acquire(self, timeout: Optional[float] = None) -> Iterator[Any]:
        """Yield the connection a statement should run on.

        The caller's own transaction when it holds one, whichever connection that is on --
        asking only whether it is the shared one leaves a statement inside a pooled
        transaction with no timeout at all, which is where generated Cypher runs.

        Otherwise a pooled connection when an engine is configured, spending the deadline
        from the pool's own budget, which costs one round trip per borrow rather than one
        per statement. Failing that the shared connection, held for the statement: a
        statement sent while another caller has a transaction open on it would otherwise
        run inside that transaction.
        """
        pin = self._pin
        if pin is not None:
            self._apply_timeout(pin, timeout)
            yield pin.conn
            return
        if self._engine is not None:
            with self._engine.connection(self.graph_name, deadline=timeout) as conn:
                yield conn
            return
        with self._exclusive:
            self._apply_timeout(None, timeout)
            yield self.connection

    @asynccontextmanager
    async def _aacquire(self, timeout: Optional[float] = None) -> AsyncIterator[Any]:
        """Async sibling of :meth:`_acquire`."""
        pin = self._apin
        if pin is not None:
            await self._aapply_timeout(pin, timeout)
            yield pin.conn
            return
        if self._engine is not None:
            async with self._engine.aconnection(
                self.graph_name, deadline=timeout
            ) as conn:
                yield conn
            return
        async with self._aexclusive:
            conn = await self._aconn_get()
            await self._aapply_timeout(None, timeout)
            yield conn

    @contextmanager
    def _untimed(self) -> Iterator[None]:
        """Run the driver's own work without the caller's statement timeout.

        A timeout is about the statements a caller sends, not about reading the catalogs
        or writing a batch. Without this a store built with a small ``timeout`` cancels
        its own schema read during construction -- the timeout is session state and
        outlives the statement it was set for.

        Nothing is restored on the way out: the next caller statement compares what it
        wants against what the connection carries and sets it again if they differ, so
        putting it back here would be a round trip for nothing.
        """
        if self._applied_timeout is not None:
            with self._dedicated() as conn:
                conn.execute(self._timeout_statement(None))
            self._applied_timeout = None
        yield

    def _apply_timeout(self, pin: Optional[_Pin], timeout: Optional[float]) -> None:
        """Put a statement timeout on the connection, and only when it changes.

        Issued only when the wanted value differs from what the connection is already
        carrying, so a caller passing the same timeout on every query pays for it once.

        Inside a transaction the value is stated for that transaction alone. A plain ``SET``
        there would be rolled back with the transaction while the record of it was not, so
        the next statement would compare what it wants against a timeout the connection no
        longer has and run unbounded; and on a pooled connection it would outlive the
        borrow and quietly govern whoever borrows next.
        """
        wanted = timeout if timeout is not None else self.timeout
        if pin is not None:
            if wanted == pin.applied:
                return
            if wanted is None and pin.applied is _UNSET and self._applied_timeout is None:
                # Nothing wants a timeout and nothing has ever set one on this graph, so
                # there is nothing to clear. Stating it anyway is a round trip inside every
                # transaction, paid by callers who never asked for one at all.
                pin.applied = None
                return
            pin.conn.execute(self._timeout_statement(wanted, local=True))
            pin.applied = wanted
            return
        if wanted == self._applied_timeout:
            return
        self.connection.execute(self._timeout_statement(wanted))
        self._applied_timeout = wanted

    async def _aapply_timeout(
        self, pin: Optional[_Pin], timeout: Optional[float]
    ) -> None:
        wanted = timeout if timeout is not None else self.timeout
        if pin is not None:
            if wanted == pin.applied:
                return
            if wanted is None and pin.applied is _UNSET and self._aapplied_timeout is None:
                # Nothing wants a timeout and nothing has ever set one on this graph, so
                # there is nothing to clear. Stating it anyway is a round trip inside every
                # transaction, paid by callers who never asked for one at all.
                pin.applied = None
                return
            await pin.conn.execute(self._timeout_statement(wanted, local=True))
            pin.applied = wanted
            return
        if wanted == self._aapplied_timeout:
            return
        conn = await self._aconn_get()
        await conn.execute(self._timeout_statement(wanted))
        self._aapplied_timeout = wanted

    @staticmethod
    def _timeout_statement(wanted: Optional[float], *, local: bool = False) -> str:
        """The statement that sets, or clears, the statement timeout.

        The value is inlined because ``SET`` takes no parameter; it is an integer by
        then, so there is nothing a caller could put there.
        """
        scope = "set local" if local else "set"
        if wanted is None:
            return f"{scope} statement_timeout = default"
        return f"{scope} statement_timeout = {int(wanted * 1000)}"

    async def _aconn_get(self) -> agensgraph.AsyncConnection:
        """Return a lazily-opened async connection bound to ``graph_path``.

        Built on demand so users that never call any ``a*`` method do not pay the cost
        of a second connection.
        """
        if self._aconn is None or self._aconn.closed:
            self._aconn = await agensgraph.AsyncConnection.connect(
                **self._conf, autocommit=True
            )
            await self._aconn.graph(self.graph_name)
            if self._vectors_registered:
                await self._aconn.register_vectors()
            self._aapplied_timeout = None
        return self._aconn

    def _run(self, statement: Any, params: Any = None) -> None:
        """One statement on the dedicated connection, for setup rather than a caller."""
        with self._dedicated() as conn:
            conn.execute(statement, params)

    def create_labels(self, vertices: Sequence[str] = (), edges: Sequence[str] = ()) -> None:
        """Create the labels a component needs, in one burst rather than one at a time.

        Creating a label is a statement whose cost is the round trip, not the work, and a
        component needs several before it can write anything. They go together, since none
        of their results is wanted and a batch that fails names the batch.

        Sent every time rather than remembered. A record of what this process has created
        is a claim about a database it does not own: a graph dropped and made again leaves
        the record saying the labels are there, the creates are skipped, and the next
        statement fails on a label that does not exist. The whole burst is one round trip,
        which is cheap enough not to trade for a claim that can be wrong.

        The label table is read again afterwards, once for the whole burst. A label is
        carried on the wire as a number, and the name it stands for is looked up in a table
        the connection holds; a label created after that table was read is a number the
        connection cannot name, and a row carrying one fails to decode. Only binary reads
        show it -- the text form spells the label out -- which is why it is worth stating:
        registering vectors turns binary on, so a store is exactly where it would appear.
        """
        wanted = sorted(
            {("v", name) for name in vertices} | {("e", name) for name in edges}
        )
        if not wanted:
            return
        with self._dedicated() as conn:
            statements = [
                sql.SQL(
                    "CREATE VLABEL IF NOT EXISTS {l}"
                    if kind == "v"
                    else "CREATE ELABEL IF NOT EXISTS {l}"
                )
                .format(l=sql.Identifier(name))
                .as_string(conn)
                for kind, name in wanted
            ]
            try:
                conn.pipeline_batch([(one, None) for one in statements])
            except (BatchFailed, DuplicateObject, DuplicateTable, UniqueViolation):
                # `IF NOT EXISTS` looks and then creates, and another builder can create
                # it in between: of eight components built at once, two were told the
                # label was already there. What was wanted is what is there now, so the
                # labels are read back and only a real absence is raised.
                made = {label.name for label in conn.labels()}
                if [name for _, name in wanted if name not in made]:
                    raise
            conn.refresh_labels()

    def register_vectors(self) -> None:
        """Let this graph's connections carry vectors as themselves.

        An embedding sent as its decimal spelling is 21,504 bytes at 1,536 dimensions
        against 6,152, and the server has to parse it. Registering is per connection, so
        it is remembered and done again for the async one whenever that is opened -- a
        caller registering on one connection means it for the graph, not for whichever
        connection happened to be open at the time.
        """
        self._vectors_registered = True
        with self._dedicated() as conn:
            conn.register_vectors()

    def merging(self, write: Callable[[], _T], *, attempts: Optional[int] = None) -> _T:
        """Run a write that merges on a unique key, making it again if it lost a race.

        Two writers merging the same key both look, both find nothing, and both create.
        The unique index over that key refuses the second, and a serialisable conflict
        between them reads the same way. Neither is a mistake in what the caller asked
        for: the element the merge asked to exist does exist by then, so the write is made
        again and the second attempt matches what its rival created.

        The driver decides how long to wait and reports how many attempts were made. It
        does not call a unique violation retryable by itself, and it is right not to --
        for a statement that is not a merge on the violated key it means the caller's data
        conflicts, and another attempt gets the same answer. Here the key is the one the
        merge matches on, which is what makes losing the race recoverable.
        """
        attempts = attempts if attempts is not None else self.retry_attempts
        policy = RetryPolicy(attempts=attempts, bucket=self._allowance)
        previous: List[BaseException] = []
        for number in range(1, attempts + 1):
            try:
                result = write()
            except Exception as exc:
                cause = exc.__cause__ if exc.__cause__ is not None else exc
                lost_the_race = isinstance(cause, UniqueViolation)
                if not (lost_the_race or is_retryable(cause, wrote=True)):
                    raise
                decision = policy.decide(cause, number=number, wrote=True)
                if not (decision.retry or lost_the_race) or number == attempts:
                    final = policy.exhausted(exc, attempts=number, previous=previous)
                    if final is exc:
                        # The failure itself, annotated. Chaining it to itself would
                        # make `__cause__` point at its own exception, and anything
                        # walking that chain to the root walks forever.
                        raise final
                    raise final from exc
                previous.append(exc)
                time.sleep(decision.delay)
                continue
            policy.succeeded()
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    def _held_indexes(self, conn: Any, desired: Sequence[Any]) -> List[Any]:
        """What the graph holds on the labels being asked about.

        Per label rather than for the whole graph, which is the difference between
        0.36 milliseconds and 31 -- the graph-wide read starts from every index the graph
        has and this one starts from the label. Since nothing here ever drops an index it
        was not asked about, the labels named are the only ones whose contents can decide
        anything.
        """
        labels = sorted({one.label for one in desired})
        return [index for label in labels for index in conn.indexes(label)]

    def _reclaim_index_names(self, conn: Any, desired: Sequence[Any], held: List[Any]) -> None:
        """Free the name of an index that is not the one being asked for.

        Only for an index this package named and is about to describe differently; one
        that already covers what is wanted is left where it is, so this does nothing on
        every run after the first.
        """
        by_name = {index.name for index in held}
        for one in desired:
            name = getattr(one, "name", None)
            if not name or name not in by_name:
                continue
            if not reconcile_indexes([one], held):
                continue  # what is there is what is wanted
            self._run(
                sql.SQL("DROP PROPERTY INDEX IF EXISTS {n}").format(
                    n=sql.Identifier(name)
                )
            )

    def ensure_indexes(self, desired: Sequence[Any]) -> List[str]:
        """Make this graph's indexes the ones asked for, and say what changed.

        The driver reads what is there and emits only what is missing, so a second call
        does nothing and a call that finds an index built differently -- not unique, when
        uniqueness is what keeps two writers from creating the same element twice --
        rebuilds it rather than leaving it as it found it. Creating each one blind with
        ``IF NOT EXISTS`` cannot do that: an index of the right name is taken for the
        right index.

        Reconciling is by what an index covers, so one built on different properties is a
        different index -- and if the name this one wants is already held by such an index,
        the create fails. The name is this package's, so the index holding it is dropped
        and the name reused.

        A label that already holds duplicates cannot take a unique index, and the server
        says so plainly; what it cannot say is where they came from, so that is said here.

        The state is read from the server every time rather than remembered. A component
        is built once per request, so this has to be cheap, and reading only the labels
        being asked about makes it so. Remembering would be a claim about a database this
        process does not own: an index dropped behind such an answer is never noticed, and
        the uniqueness that stops two writers creating the same element is gone with it.
        """
        try:
            with self._dedicated() as conn:
                statements = reconcile_indexes(
                    desired, self._held_indexes(conn, desired)
                )
                if not statements:
                    return []
                # Something has to be built, so the graph-wide read is worth paying for
                # now: an index name belongs to the graph rather than to a label, and one
                # held by a label nobody asked about is invisible to the per-label read
                # that makes the ordinary case cheap -- leaving the create to fail on a
                # name this package owns, and the label with no index at all.
                self._reclaim_index_names(conn, desired, conn.indexes())
                try:
                    return conn.ensure_indexes(desired)
                except (DuplicateObject, DuplicateTable, UniqueViolation):
                    # Somebody else created it between the read and the create. Eight
                    # components built at once each looked, each found it missing, and
                    # each tried; the losers are told either that the name is taken or --
                    # since two backends creating one name collide over the catalogs --
                    # that a unique constraint was violated, which is the same answer a
                    # label holding real duplicates gives. Rather than tell those apart by
                    # how they are spelled, the state is read again: either what was
                    # wanted is there now, whoever made it, or it is not and the failure
                    # was about the data, which the caller is told about below.
                    if reconcile_indexes(desired, self._held_indexes(conn, desired)):
                        raise
                    return []
        except UniqueViolation as exc:
            wanted = ", ".join(
                f"{d.label}({', '.join(str(p) for p in d.properties)})"
                for d in desired
                if getattr(d, "unique", False)
            )
            raise AgensQueryException(
                {
                    "message": (
                        "This graph already holds elements that a unique index would "
                        f"refuse: {wanted}. Duplicates like these are what an earlier "
                        "version left behind, having created the index without "
                        "uniqueness. Remove them and construct this again."
                    ),
                    "detail": str(exc),
                }
            ) from exc

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Run several statements so that they land together, or not at all.

        Pinned, so that a concurrent caller's statement is not swept into it, and nested
        as a savepoint when one is already open.

        The connection is yielded because some of what belongs in a transaction is said to
        a connection rather than sent as a statement -- a search option, for one, which
        means nothing except on the connection the search then runs on.

        The caller's timeout is stated as the transaction opens rather than per statement,
        because a caller holding the connection can send statements this never sees: a
        batch given straight to the connection is one, and it would otherwise run
        unbounded on a graph that asked for a bound. Nothing is sent when no timeout was
        asked for, which is the ordinary case.
        """
        with self._borrowed() as conn, self._pinned(conn), conn.transaction():
            if self.timeout is not None:
                self._apply_timeout(self._pin, None)
            yield conn

    @asynccontextmanager
    async def atransaction(self) -> AsyncIterator[Any]:
        """Async sibling of :meth:`transaction`."""
        async with self._aborrowed() as conn:
            async with self._apinned(conn), conn.transaction():
                if self.timeout is not None:
                    await self._aapply_timeout(self._apin, None)
                yield conn

    @contextmanager
    def read_only(self, *, allow_server_programs: bool = False) -> Iterator[None]:
        """Run statements in a transaction the server will not let write.

        For Cypher that came from a model. The server refuses the write, so a statement
        is refused however it is spelled, including the SQL forms Cypher has no words
        for.

        Statements inside the block run on the dedicated connection, since a transaction
        belongs to one connection.

        ``COPY ... TO PROGRAM`` runs a command on the server's host, and a read-only
        transaction does not refuse it: it takes rows out of the database rather than
        putting any in, so there is no write to refuse. What stops it is not holding
        ``pg_execute_server_program``, so that is what the driver asks about, and it
        declines to open the transaction for a role that holds it. This defaults to the
        driver's answer rather than overriding it -- a caller who has decided the role's
        privileges are acceptable passes ``allow_server_programs=True`` and says so.
        """
        with self._borrowed() as conn, self._pinned(conn), conn.read_only_transaction(
            allow_server_programs=allow_server_programs
        ):
            yield

    @asynccontextmanager
    async def aread_only(
        self, *, allow_server_programs: bool = False
    ) -> AsyncIterator[None]:
        """Async sibling of :meth:`read_only`.

        Statements inside the block run on the dedicated async connection, which is a
        different connection from the one :meth:`read_only` uses, so the two boundaries
        are independent and an awaited statement gets the same one a blocking statement
        gets.
        """
        async with self._aborrowed() as conn:
            async with self._apinned(conn), conn.read_only_transaction(
                allow_server_programs=allow_server_programs
            ):
                yield

    # ---------- querying ----------

    def _shape(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The rows as a caller sees them.

        With ``sanitize`` off this is the identity, which is the point: the driver has
        already built the values and there is nothing left to convert. A property map is
        decoded on first access, so a row nobody reads the properties of never decodes
        them.
        """
        if not self.sanitize:
            return rows
        return [_sanitize_value(row) for row in rows]

    def query(
        self,
        query: Any,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and converting the
        result.

        Args:
            query (str): a cypher query to be executed params (dict): parameters for the
            query timeout (Optional[float]): statement timeout in seconds for this
                call. Falls back to the instance ``timeout``. ``None`` disables.

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set. A
            vertex is an :class:`agensgraph.Vertex`, an edge an :class:`agensgraph.Edge`
            and a path an :class:`agensgraph.Path`.
        """
        with self._acquire(timeout) as conn:
            try:
                result = conn.execute_query(query, params, row_=dict_row)
                if not self._in_transaction and not conn.autocommit:
                    conn.commit()
            except Exception as exc:
                if not self._in_transaction and not conn.autocommit:
                    conn.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(exc),
                    }
                ) from exc
            return self._shape(result.records)

    async def aquery(
        self,
        query: Any,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Async sibling of :meth:`query`."""
        async with self._aacquire(timeout) as conn:
            try:
                result = await conn.execute_query(query, params, row_=dict_row)
                if not self._in_async_transaction and not conn.autocommit:
                    await conn.commit()
            except Exception as exc:
                if not self._in_async_transaction and not conn.autocommit:
                    await conn.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(exc),
                    }
                ) from exc
            return self._shape(result.records)

    def query_many(self, statements: Sequence[Any]) -> List[List[Dict[str, Any]]]:
        """Run a burst of **reads** without waiting for each in turn.

        For the shapes whose cost is round trips rather than work: a lookup per id, a
        walk up a chain of ancestors.

        Reads only. A pipeline attributes a failure to the wrong statement, and the
        driver recovers by running the batch again singly, which would apply a write
        twice.
        """
        pairs = [
            (item, None) if not isinstance(item, tuple) else item for item in statements
        ]
        with self._acquire() as conn:
            try:
                results = conn.pipeline_query(pairs)
            except Exception as exc:
                raise AgensQueryException(
                    {"message": "Error executing pipelined reads", "detail": str(exc)}
                ) from exc
        # A pipelined cursor is built without a row factory, so its rows are tuples. The
        # column names come back beside them, which is what makes them dicts here and
        # keeps this method's answer the same shape as `query`'s.
        return [
            self._shape([dict(zip(result.keys, record)) for record in result.records])
            for result in results
        ]

    # ---------- schema ----------

    def refresh_schema(self, *, force: bool = False) -> None:
        """Refresh the graph schema information.

        Read from the catalogs. The labels, their inheritance and their counts are
        catalog rows; which label joins which comes from the catalog the server keeps
        for its own planner; and the property names come from a bounded sample of each
        label.

        When ``schema_cache_ttl > 0`` and a refresh happened more recently than that,
        this is a no-op unless ``force=True``.
        """
        if (
            not force
            and self.schema_cache_ttl > 0
            and (time.monotonic() - self._schema_refreshed_at) < self.schema_cache_ttl
        ):
            return

        sample = (
            ENHANCED_SCHEMA_SAMPLE if self.enhanced_schema else DEFAULT_SCHEMA_SAMPLE
        )
        # The triple catalog is filled by a gather, and `auto_gather_graphmeta` is off by
        # default, so a refresh asks for one. The driver gathers only when the server
        # reports the catalog is not current, so a read-mostly graph pays a catalog read
        # and no write.
        with self._untimed(), self._dedicated() as conn:
            description = conn.describe(sample=sample, refresh=True)

        self.structured_schema = self._structured(description)
        if self.enhanced_schema:
            self._augment_with_examples(self.structured_schema)
        self.schema = self._rendered(description, self.structured_schema)
        self._schema_refreshed_at = time.monotonic()

    def _augment_with_examples(
        self, structured: Dict[str, Any], sample: int = 25, limit: int = 3
    ) -> None:
        """Attach a few example values to each property, for a Text2Cypher prompt.

        The types alone say a property holds a string; the examples say whether that
        string is a name, a country code or a URL, which is what decides how a model
        writes the predicate against it.

        Every label is asked in one pipelined burst rather than a round trip each, so the
        cost is one wait however many labels there are. Each query is bounded by
        ``sample``, so this reads a fixed number of elements per label rather than the
        label.
        """
        wanted = [
            (label, kind)
            for section, kind in (("node_props", "v"), ("rel_props", "e"))
            for label in structured[section]
        ]
        if not wanted:
            return
        statements = [
            (
                sql.SQL(
                    "MATCH ()-[n:{l}]->() WITH n LIMIT %s RETURN properties(n) AS p"
                    if kind == "e"
                    else "MATCH (n:{l}) WITH n LIMIT %s RETURN properties(n) AS p"
                ).format(l=sql.Identifier(label)),
                (sample,),
            )
            for label, kind in wanted
        ]
        try:
            results = self.query_many(statements)
        except AgensQueryException:
            # A label that cannot be read leaves the schema without its examples, which
            # is a smaller loss than a schema that cannot be built at all.
            return
        for (label, kind), rows in zip(wanted, results):
            found: Dict[str, List[Any]] = {}
            for row in rows:
                for key, value in (row.get("p") or {}).items():
                    if value is None:
                        continue
                    # An example is for a model to read, and the schema it lands in goes
                    # into every prompt. A property holding an embedding would put three
                    # of them there, so a value too large to be an example is not one --
                    # the same measure the results themselves are held to.
                    example = _sanitize_value(value)
                    if example is None:
                        continue
                    bucket = found.setdefault(key, [])
                    if example not in bucket and len(bucket) < limit:
                        bucket.append(example)
            section = "rel_props" if kind == "e" else "node_props"
            for entry in structured[section][label]:
                if entry["property"] in found:
                    entry["examples"] = found[entry["property"]]

    def _structured(self, description: agensgraph.GraphDescription) -> Dict[str, Any]:
        """The schema as a mapping, in the shape a chain reads."""
        by_kind = {label.name: label.kind for label in description.labels}
        counts = description.counts
        node_props: Dict[str, List[Dict[str, str]]] = {}
        rel_props: Dict[str, List[Dict[str, str]]] = {}
        for label, shapes in description.properties.items():
            # A label that holds nothing is left out. It exists in the catalogs -- the
            # description reads them rather than the graph, so it finds every label ever
            # created, including ones whose elements have all been deleted -- but telling
            # a model a label is there when nothing is in it invites a query that matches
            # nothing.
            if not counts.get(label):
                continue
            entries = [{"property": shape.name, "type": shape.kind} for shape in shapes]
            if by_kind.get(label) == "e":
                rel_props[label] = entries
            else:
                node_props[label] = entries
        return {
            "node_props": node_props,
            "rel_props": rel_props,
            # A triple whose edge label holds nothing is left out for the same reason: the
            # catalog remembers a pairing the graph no longer has any edges for.
            "relationships": [
                {"start": triple.start, "type": triple.edge, "end": triple.end}
                for triple in description.triples
                if counts.get(triple.edge)
            ],
            "counts": dict(description.counts),
            "metadata": {
                "agensgraph_version": self.capabilities.version,
                "meta_gathered": description.meta_gathered,
            },
        }

    @staticmethod
    def _rendered(
        description: agensgraph.GraphDescription, structured: Dict[str, Any]
    ) -> str:
        """The schema as the string a prompt carries.

        The counts are here because they cost nothing -- they came from the catalogs
        with everything else -- and a model choosing between two ways of matching writes
        a better query when it knows which label holds ten rows and which holds ten
        million.
        """
        triples = [
            '(:"{start}")-[:"{type}"]->(:"{end}")'.format(**rel)
            for rel in structured["relationships"]
        ]
        counts = structured["counts"]
        return f"""
        Node properties are the following:
        {[{"labels": label, "properties": props}
          for label, props in structured["node_props"].items()]}
        Relationship properties are the following:
        {[{"type": label, "properties": props}
          for label, props in structured["rel_props"].items()]}
        The relationships are the following:
        {triples}
        Element counts are the following:
        {counts}
        """

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph (computed lazily if never refreshed)."""
        if not self.schema:
            self.refresh_schema()
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """The structured schema of the Graph (computed lazily if never refreshed)."""
        if not self.structured_schema:
            self.refresh_schema()
        return self.structured_schema

    # ---------- lifecycle ----------

    async def aclose(self) -> None:
        """Close the async connection (if any). Safe to call multiple times."""
        if self._aconn is not None and not self._aconn.closed:
            await self._aconn.close()
            self._aconn = None

    def close(self) -> None:
        """Close the sync connection. Idempotent.

        Note: if an async connection was opened it should be closed with :meth:`aclose`
        from within an event loop; ``close`` only closes the synchronous connection.
        """
        if getattr(self, "connection", None) is not None and not self.connection.closed:
            self.connection.close()

    def __enter__(self) -> "AgensGraph":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "AgensGraph":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
        self.close()

    # ---------- writing ----------

    @staticmethod
    def clean_graph_labels(label: str) -> str:
        """
        remove any disallowed characters from a label and replace with '_'

        Args:
            label (str): the original label

        Returns:
            str: the sanitized version of the label
        """
        return re.sub(AgensGraph.label_regex, "_", label)

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """
        insert a list of graph documents into the graph

        Args:
            graph_documents (List[GraphDocument]): the list of documents to be inserted
            include_source (bool): if True add nodes for the sources
                with MENTIONS edges to the entities they mention

        Returns:
            None

        Elements are written a label at a time rather than one statement per element.
        Vertices go through the driver's ``upsert_vertices``, which reads which keys are
        already there, copies the rows whose key is not, and updates the rest as one
        statement -- about twelve times a MERGE each. Edges go as one ``UNWIND`` per
        edge label, which is twenty times a statement each and still idempotent.

        All statements for one call commit atomically: if any insert fails partway
        through, the entire batch is rolled back so the graph never holds orphan nodes
        or edges.
        """
        vertices, edges = self._collect(graph_documents, include_source)
        self._ensure_graph_doc_labels(vertices, edges)

        # `_untimed` outside the transaction, so the statement it issues is not one the
        # transaction can take back with it on the way out. The whole batch is made again
        # if it loses a race: it merges on the key of every element it writes, and two
        # sources being read at once is the ordinary case rather than an exotic one.
        with self._untimed():
            self.merging(lambda: self._write_graph_documents(vertices, edges))

    def _write_graph_documents(
        self,
        vertices: Dict[str, Dict[str, Dict[str, Any]]],
        edges: Dict[Tuple[str, str, str], List[Tuple[str, str, Dict[str, Any]]]],
    ) -> None:
        """Write a batch's vertices and edges, so that all of it lands or none does.

        On whichever connection a transaction round this one is already using, so that a
        caller wrapping an ingest in one gets what they asked for. Taking the graph's own
        connection regardless put the batch in a transaction of its own on another
        connection, where an enclosing rollback could not reach it and the elements
        survived it.
        """
        with self._borrowed() as conn, self._pinned(conn), conn.transaction():
            for label, rows in vertices.items():
                # `update` rather than `skip`: re-reading a source is meant to refresh
                # what it says about an element it has seen before.
                conn.upsert_vertices(
                    label, "id", list(rows.values()), on_existing="update"
                )
            for triple, pairs in edges.items():
                self._write_edges(conn, triple, pairs)

    def _collect(
        self, graph_documents: List[GraphDocument], include_source: bool
    ) -> Tuple[
        Dict[str, Dict[str, Dict[str, Any]]],
        Dict[Tuple[str, str, str], List[Tuple[str, str, Dict[str, Any]]]],
    ]:
        """Group a batch by label, so a label is written once rather than an element.

        A vertex seen twice in one batch is written once, last description winning,
        which is what a per-row MERGE did by running twice.

        Edges are keyed by the whole triple -- start label, edge label, end label --
        rather than by the edge label alone, so the statement that writes them can name
        both endpoints' labels. Matching an endpoint by ``id`` without its label would
        find an element of any label that happened to share the id.
        """
        vertices: Dict[str, Dict[str, Dict[str, Any]]] = {}
        edges: Dict[Tuple[str, str, str], List[Tuple[str, str, Dict[str, Any]]]] = {}

        def remember(label: str, identity: str, properties: Dict[str, Any]) -> None:
            """Merge what this mention says about an element into what is already known.

            Merged rather than replaced, because the same element is mentioned twice:
            once in ``doc.nodes``, carrying its properties, and again as an endpoint of
            a relationship, where it is a bare id and a type. Replacing would let the
            second mention erase the first one's properties.
            """
            seen = vertices.setdefault(label, {})
            known = seen.setdefault(identity, {"id": identity})
            known.update(properties)
            known["id"] = identity

        for doc in graph_documents:
            for node in doc.nodes:
                remember(self.clean_graph_labels(node.type), node.id, node.properties)
            for edge in doc.relationships:
                start = self.clean_graph_labels(edge.source.type)
                end = self.clean_graph_labels(edge.target.type)
                remember(start, edge.source.id, edge.source.properties)
                remember(end, edge.target.id, edge.target.properties)
                label = self.clean_graph_labels(edge.type).upper()
                edges.setdefault((start, label, end), []).append(
                    (edge.source.id, edge.target.id, edge.properties)
                )
            if include_source:
                if not doc.source.metadata.get("id"):
                    doc.source.metadata["id"] = md5(
                        doc.source.page_content.encode("utf-8")
                    ).hexdigest()
                source_label = self.clean_graph_labels(doc.source.type)
                remember(source_label, doc.source.metadata["id"], doc.source.metadata)
                for node in doc.nodes:
                    node_label = self.clean_graph_labels(node.type)
                    edges.setdefault(
                        (source_label, "MENTIONS", node_label), []
                    ).append((doc.source.metadata["id"], node.id, {}))
        return vertices, edges

    def _ensure_graph_doc_labels(
        self,
        vertices: Dict[str, Dict[str, Dict[str, Any]]],
        edges: Dict[Tuple[str, str, str], List[Tuple[str, str, Dict[str, Any]]]],
    ) -> None:
        """Create the labels a batch needs, and the unique index its upsert requires.

        The index is unique. Two writers merging on a key with no uniqueness behind it
        each create an element rather than finding one, and the driver refuses to upsert
        on such a key.

        Both are asked for as a set rather than a label at a time. A batch over thirty
        labels of which one is new sent eighty-nine statements of schema for that one
        label, because each was created blind; here the labels this process has not created
        go together in one burst, and the indexes are reconciled against what the graph
        holds, so a batch that changes nothing sends nothing.
        """
        self.create_labels(
            vertices=tuple(vertices),
            edges=tuple({label for _, label, _ in edges}),
        )
        self.ensure_indexes(
            [
                DesiredIndex(
                    label=label,
                    properties=("id",),
                    unique=True,
                    name=f"{label}_id_idx",
                )
                for label in vertices
            ]
        )

    def _write_edges(
        self,
        conn: Any,
        triple: Tuple[str, str, str],
        pairs: List[Tuple[str, str, Dict[str, Any]]],
    ) -> None:
        """Every edge joining one pair of labels, in one statement.

        ``MERGE``, so reading a source in twice does not make a second edge for every
        one already there.

        Both endpoints are matched by label as well as by id, which the unique index on
        each label's ``id`` answers with a probe.

        On the connection the batch's vertices were written on, which has to be said
        rather than assumed: the endpoints this matches are still uncommitted, so a
        statement sent anywhere else cannot see them, matches nothing, and merges nothing.
        It does not fail -- a ``MERGE`` that matches no pattern creates no edge and reports
        success -- so an ingest through a pool wrote every vertex and not one edge, and
        said nothing about it.
        """
        if not pairs:
            return
        start_label, label, end_label = triple
        rows = [{"f": start, "t": end, "p": props or {}} for start, end, props in pairs]
        # Only the edges that carry properties are written to. Most extracted edges carry
        # none, and `SET r += {}` still rewrites the row -- so re-reading a source rewrote
        # every edge it had ever produced to say nothing about any of them.
        assignment = (
            "\n                   SET r += row.p"
            if any(row["p"] for row in rows)
            else ""
        )
        conn.execute(
            sql.SQL(
                """UNWIND %(rows)s AS row
                   MATCH (f:{start} {{id: row.f}}), (t:{end} {{id: row.t}}) MERGE
                   (f)-[r:{label}]->(t)""" + assignment
            ).format(
                start=sql.Identifier(start_label),
                end=sql.Identifier(end_label),
                label=sql.Identifier(label),
            ),
            {"rows": Jsonb(rows)},
        )
