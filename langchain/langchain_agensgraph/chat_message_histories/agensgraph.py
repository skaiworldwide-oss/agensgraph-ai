"""A LangChain chat message history in the companion schema of an AgensGraph graph.

Two tables in ``"<graph>_store"``: ``chat_session``, one row per session holding the next sequence
number and how many messages it has, and ``chat_message``, one row per message, keyed by session
and sequence number and unique by session and message id. A message is kept as LangChain's own
``message_to_dict``, so it comes back as it went in, ``additional_kwargs`` and all.

Every call is one round trip. Adding messages is a pipeline, so one implicit transaction: the first
statement reserves a block of sequence numbers on the session's row, which also locks it, so two
writers to one session take turns; the second inserts the messages, skipping any whose id the
session already holds, and adds to the count only what it inserted. Adding the same messages twice
therefore changes nothing the second time. A message without an id is given one; the caller's
message is not changed.

Tables are made on first use, under the graph's schema lock; the graph itself need not exist.

A Cypher statement reads a session's messages next to the graph through
:meth:`AgensChatMessageHistory.cypher_subquery`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from agensgraph import Jsonb
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from .._core.component import Component
from .._core.engine import AgensEngine, bind
from .._core.messages import message_fields
from .._core.names import companion_schema, qualified
from .._core.program import Pipeline, Program, Statement
from .._core.setup import Migrations, Setup


class _Tables:
    def __init__(self, graph: str) -> None:
        self.schema = companion_schema(graph)
        self.sessions = qualified(self.schema, "chat_session")
        self.messages = qualified(self.schema, "chat_message")
        self.migrations = [
            f"CREATE TABLE IF NOT EXISTS {self.sessions} (session_id text PRIMARY KEY, "
            "created_at timestamptz NOT NULL DEFAULT now(), "
            "updated_at timestamptz NOT NULL DEFAULT now(), next_seq bigint NOT NULL, "
            "message_count bigint NOT NULL)",
            f"CREATE TABLE IF NOT EXISTS {self.messages} (session_id text NOT NULL, "
            "seq bigint NOT NULL, message_id text NOT NULL, message jsonb NOT NULL, "
            "created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (session_id, seq), "
            "UNIQUE (session_id, message_id))",
        ]

    @property
    def reserve(self) -> str:
        """Reserve ``%s`` sequence numbers for a session, making the session if it is new."""
        return (
            f"INSERT INTO {self.sessions} AS s (session_id, next_seq, message_count) "
            "VALUES (%s, %s, 0) ON CONFLICT (session_id) DO UPDATE "
            "SET next_seq = s.next_seq + EXCLUDED.next_seq, updated_at = now()"
        )

    @property
    def insert(self) -> str:
        """Insert messages at the numbers just reserved, and count what was inserted."""
        return (
            f"WITH inserted AS (INSERT INTO {self.messages} (session_id, seq, message_id, message) "
            f"SELECT %s, (SELECT next_seq FROM {self.sessions} WHERE session_id = %s) - %s "
            "+ u.ord - 1, u.id, u.message "
            "FROM unnest(%s::text[], %s::jsonb[]) WITH ORDINALITY AS u(id, message, ord) "
            "ON CONFLICT (session_id, message_id) DO NOTHING RETURNING 1) "
            f"UPDATE {self.sessions} SET message_count = message_count + "
            "(SELECT count(*) FROM inserted) WHERE session_id = %s"
        )

    @property
    def read(self) -> str:
        return f"SELECT message FROM {self.messages} WHERE session_id = %s ORDER BY seq"

    @property
    def read_last(self) -> str:
        return (
            f"SELECT message FROM (SELECT seq, message FROM {self.messages} "
            "WHERE session_id = %s ORDER BY seq DESC LIMIT %s) last ORDER BY seq"
        )

    @property
    def clear_messages(self) -> str:
        return f"DELETE FROM {self.messages} WHERE session_id = %s"

    @property
    def clear_session(self) -> str:
        return f"DELETE FROM {self.sessions} WHERE session_id = %s"


class AgensChatMessageHistory(Component, BaseChatMessageHistory):
    """The messages of one session, in the order they were added.

    ``window`` limits what :attr:`messages` returns to the last that many. ``engine`` is an
    :class:`AgensEngine` or a conninfo string; ``graph`` an ``AgensGraph`` whose engine and graph
    name are used, ``graph_name`` the graph otherwise.
    """

    _what = "chat history"

    def __init__(
        self,
        session_id: str,
        engine: AgensEngine | str | None = None,
        *,
        graph: Any = None,
        graph_name: str | None = None,
        window: int | None = None,
        setup: bool = True,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if window is not None and window < 1:
            raise ValueError(f"window must be positive, not {window}")
        self.session_id = session_id
        self.window = window
        self._engine, self._owns_engine, self.graph_name = bind(engine, graph, graph_name)
        self._tables = _Tables(self.graph_name)
        self._setup_wanted = setup
        migrations = Migrations(
            self._tables.schema, "chat_migrations", dict(enumerate(self._tables.migrations))
        )
        self._setup = Setup(
            self._engine,
            self.graph_name,
            migrations.check,
            migrations.create,
            key=migrations.key,
        )

    # -- reading -------------------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        self._ensure()
        return self._engine.run(self._read, replay=True)

    async def aget_messages(self) -> list[BaseMessage]:
        await self._aensure()
        return await self._engine.arun(self._read, replay=True)

    def _read(self) -> Program[list[BaseMessage]]:
        if self.window is None:
            found = yield Statement(self._tables.read, (self.session_id,))
        else:
            found = yield Statement(self._tables.read_last, (self.session_id, self.window))
        return messages_from_dict([row[0] for row in found.records])

    def cypher_subquery(self, session: str = "%s") -> str:
        """A SQL subquery a Cypher statement can hold, giving a session's messages as a list.

        Each element is a map of ``seq``, ``id``, ``type`` (``human``, ``ai``, …), ``text`` (always
        a string), ``content`` (a string or a list of blocks) and ``message``, the whole
        ``message_to_dict`` form, in the order they were added. ``session`` is the SQL for the
        session id: a placeholder, by default, or an expression. For example::

            graph.query(
                f"UNWIND [1] AS a WITH {history.cypher_subquery()} AS messages "
                'UNWIND messages AS m MATCH (p:"Person") WHERE m.text CONTAINS p.name '
                "RETURN p.name AS who, m.seq AS seq",
                (history.session_id,),
            )
        """
        return (
            "(SELECT coalesce(jsonb_agg(jsonb_build_object('seq', m.seq, 'id', m.message_id, "
            f"{message_fields('m.message')}) ORDER BY m.seq), '[]') "
            f"FROM {self._tables.messages} m WHERE m.session_id = {session})"
        )

    # -- writing -------------------------------------------------------------------------------

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages after those already held; one already held, by id, is skipped."""
        rows = self._rows(messages)
        if not rows:
            return
        self._ensure()
        self._engine.run(lambda: self._add(rows), replay=True)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        rows = self._rows(messages)
        if not rows:
            return
        await self._aensure()
        await self._engine.arun(lambda: self._add(rows), replay=True)

    def _rows(self, messages: Sequence[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
        rows: dict[str, dict[str, Any]] = {}
        for message in messages:
            if not isinstance(message, BaseMessage):
                raise TypeError(f"a message must be a BaseMessage, not {type(message).__name__}")
            encoded = message_to_dict(message)
            key = message.id or str(uuid.uuid4())
            encoded["data"] = {**encoded["data"], "id": key}
            rows.setdefault(key, encoded)
        return list(rows.items())

    def _add(self, rows: list[tuple[str, dict[str, Any]]]) -> Program[None]:
        tables = self._tables
        count = len(rows)
        yield Pipeline(
            (
                Statement(tables.reserve, (self.session_id, count)),
                Statement(
                    tables.insert,
                    (
                        self.session_id,
                        self.session_id,
                        count,
                        [key for key, _ in rows],
                        [Jsonb(message) for _, message in rows],
                        self.session_id,
                    ),
                ),
            )
        )

    def clear(self) -> None:
        """Remove the session's messages, and the session."""
        self._ensure()
        self._engine.run(self._clear, replay=True)

    async def aclear(self) -> None:
        await self._aensure()
        await self._engine.arun(self._clear, replay=True)

    def _clear(self) -> Program[None]:
        yield Pipeline(
            (
                Statement(self._tables.clear_messages, (self.session_id,)),
                Statement(self._tables.clear_session, (self.session_id,)),
            )
        )

    # -- setup and closing ---------------------------------------------------------------------

    def _ensure(self) -> None:
        self._open()
        if self._setup_wanted:
            self._setup.ensure()

    async def _aensure(self) -> None:
        self._open()
        if self._setup_wanted:
            await self._setup.aensure()

    def setup(self) -> None:
        """Make the tables now rather than on the first call."""
        self._open()
        self._setup.ensure()

    async def asetup(self) -> None:
        self._open()
        await self._setup.aensure()
