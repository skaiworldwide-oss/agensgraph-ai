"""Chat message history backed by AgensGraph.

Messages for a session are stored as an ordered chain of ``Message`` vertices
linked from a ``Session`` vertex::

    (:Session {id})-[:HAS_MESSAGE]->(:Message {seq, data})

``data`` is the LangChain ``message_to_dict`` form, so any ``BaseMessage`` subtype
round-trips losslessly through ``messages_from_dict``. ``seq`` is a monotonic
per-session ordinal used to preserve order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agensgraph.introspect import DesiredIndex
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from psycopg import sql
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import AgensGraph, checked_names


class AgensChatMessageHistory(BaseChatMessageHistory):
    """Persist LangChain chat messages in an AgensGraph graph.

    Args:
        session_id: Identifier for the conversation. graph: An existing
        :class:`AgensGraph` to reuse. If omitted, ``conf``
            (and ``graph_name``) are used to build one.
        conf: psycopg connection kwargs (used only when ``graph`` is None). graph_name:
        Graph to store messages in (default ``chat_history``). session_node_label:
        Vertex label for sessions (default ``Session``). message_node_label: Vertex
        label for messages (default ``Message``). relationship: Edge label linking
        session to messages
            (default ``HAS_MESSAGE``).
        window: If set, ``messages`` returns only the most recent ``window``
            messages (still in chronological order).
    """

    def __init__(
        self,
        session_id: str,
        *,
        graph: Optional[AgensGraph] = None,
        conf: Optional[Dict[str, Any]] = None,
        graph_name: str = "chat_history",
        session_node_label: str = "Session",
        message_node_label: str = "Message",
        relationship: str = "HAS_MESSAGE",
        window: Optional[int] = None,
    ) -> None:
        if graph is None:
            if conf is None:
                raise ValueError(
                    "AgensChatMessageHistory requires either `graph` or `conf`."
                )
            # Nothing here reads the schema, and describing a graph counts every vertex
            # and every edge in it, for a component that never looks at the answer.
            graph = AgensGraph(
                graph_name, conf, create=True, refresh_schema=False
            )
        self._graph = graph
        self.session_id = session_id
        self.session_node_label = session_node_label
        self.message_node_label = message_node_label
        self.relationship = relationship
        self.window = window
        # Whether this session has been given the number it counts from. A session
        # written before it carried one is counted once, here rather than on every append.
        self._numbered = False

        # Ensure the labels exist (AgensGraph requires labels before MERGE/CREATE in
        # some paths; IF NOT EXISTS makes this idempotent).
        checked_names(
            session_node_label=session_node_label,
            message_node_label=message_node_label,
            relationship=relationship,
        )
        self._graph.create_labels(
            vertices=(self.session_node_label, self.message_node_label),
            edges=(self.relationship,),
        )
        # A session is its id, and every lookup and append merges on it, so the index
        # over it is unique -- that is what stops two requests for the same session from
        # each finding nothing and each creating one. Without it a per-session read is
        # also a scan of every session.
        self._graph.ensure_indexes(
            [
                DesiredIndex(
                    label=self.session_node_label,
                    properties=("id",),
                    unique=True,
                    name=f"{self.session_node_label}_id_idx",
                ),
                # A conversation is read far more often than it is added to, and it is
                # read by its last few messages. Keyed this way a window is the first rows
                # of an index range, whatever the length of the conversation.
                DesiredIndex(
                    label=self.message_node_label,
                    properties=("session", "seq"),
                    name=f"{self.message_node_label}_session_seq",
                ),
            ]
        )

    # ---- query builders (shared by sync + async) ----

    def _select_query(self) -> Any:
        """The session's messages, or the last ``window`` of them.

        Reached by the session a message carries rather than by walking the edges out of
        the session vertex. Both answer the same question, but the edge walk has to visit
        every message of the session before it can take the last few, so a window would
        cost the whole conversation.

        That is what the ``(session, seq)`` index is for. It orders as well as narrows, so
        a window is the first rows of an index range and nothing is sorted.
        """
        order = "ORDER BY m.seq"
        limit = ""
        if self.window is not None:
            # take the last `window` by seq desc, caller re-sorts ascending
            order = "ORDER BY m.seq DESC"
            limit = f"LIMIT {int(self.window)}"
        return sql.SQL(
            "MATCH (m:{ml}) WHERE m.session = %(sid)s "
            "RETURN m.data AS data, m.seq AS seq " + order + " " + limit
        ).format(ml=sql.Identifier(self.message_node_label))

    def _append_query(self) -> Any:
        """Append messages, numbering them from where the session left off.

        The session carries the next number, so appending costs the messages appended
        rather than the messages already there. Counting them instead would make each
        append dearer than the last, which over a conversation is quadratic in its length.

        A session written before it carried the number has its messages counted once to
        establish it. That is the only time the count is taken.

        One statement, so the number is read and moved on by the statement that writes.
        Uniqueness on ``(session, seq)`` is not expressible as a property index, which is
        per label rather than per session, so two appenders committing at the same instant
        can still collide. Both messages are written; only their order between the two is
        unsettled.
        """
        return sql.SQL(
            "MERGE (s:{sl} {{id: %(sid)s}}) "
            "WITH s, coalesce(s.next_seq, 0) AS base "
            "SET s.next_seq = base + %(added)s, s.named = true "
            "WITH s, base "
            "UNWIND %(rows)s AS row "
            "CREATE (s)-[:{rl}]->(:{ml} "
            "{{seq: base + row.i, data: row.data, session: %(sid)s}})"
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    def _repair_needed_query(self) -> Any:
        """Whether a session is missing anything, asked of the session alone.

        Two things can be missing, and they arrived at different times: the number the
        session counts from, and the session name each message carries. A session can have
        the first and not the second, and that session reads back empty -- every message is
        there and none says which session it belongs to -- so the two are asked about
        separately rather than taken as one condition.

        Asked of the session vertex and nothing else. Counting the messages to decide
        whether they need repairing would cost the whole conversation on every request
        that builds a history, to answer a yes or no -- so the answer is stored as one: a
        session repaired, or written already carrying both, says so on itself.

        A read, so the counts are allowed: on this server an aggregate followed by a write
        clause in one statement is refused with an internal error, which is also why
        neither repair can be made by the statement that appends.
        """
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}}) "
            "RETURN count(s) AS sessions, count(s.next_seq) AS numbered, "
            "count(s.named) AS named"
        ).format(sl=sql.Identifier(self.session_node_label))

    def _count_messages_query(self) -> Any:
        """How many messages a session holds, for the one that never carried a number.

        Only asked when ``next_seq`` is missing, which is once in the life of a session
        written before it existed.
        """
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}})-[:{rl}]->(m:{ml}) RETURN count(m) AS cnt"
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    def _number_query(self) -> Any:
        """Give a session the number it now carries, counted once."""
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}}) WHERE s.next_seq IS NULL "
            "SET s.next_seq = %(cnt)s"
        ).format(sl=sql.Identifier(self.session_node_label))

    def _mark_named_query(self) -> Any:
        """Record that this session's messages carry it, so nobody asks again."""
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}}) SET s.named = true"
        ).format(sl=sql.Identifier(self.session_node_label))

    def _name_session_query(self) -> Any:
        """Tell a session's messages which session they belong to.

        A message written before it carried its session is reachable only by the edge from
        the session vertex, which is what the read here no longer does. Walked once and
        never again, and asked for on its own account rather than alongside the numbering:
        the two arrived at different times, so a session can carry the number and not the
        names, and that is exactly the session whose messages cannot be found.
        """
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}})-[:{rl}]->(m:{ml}) SET m.session = %(sid)s"
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    def _clear_query(self, delete_session_node: bool) -> Any:
        if delete_session_node:
            return sql.SQL(
                "MATCH (s:{sl} {{id: %(sid)s}}) "
                "OPTIONAL MATCH (s)-[:{rl}]->(m:{ml}) "
                "DETACH DELETE s, m"
            ).format(
                sl=sql.Identifier(self.session_node_label),
                ml=sql.Identifier(self.message_node_label),
                rl=sql.Identifier(self.relationship),
            )
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}})-[:{rl}]->(m:{ml}) DETACH DELETE m"
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    def _rows_to_messages(self, rows: List[Dict[str, Any]]) -> List[BaseMessage]:
        if self.window is not None:
            rows = list(reversed(rows))  # _select_query returned desc; restore order
        data = []
        for r in rows:
            d = r["data"]
            data.append(d)
        return messages_from_dict(data)

    @staticmethod
    def _offsets(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        """Each message with its offset from wherever the session's numbering resumes."""
        return [
            {"i": i, "data": message_to_dict(m)} for i, m in enumerate(messages)
        ]

    @property
    def messages(self) -> List[BaseMessage]:
        rows = self._graph.query(self._select_query(), {"sid": self.session_id})
        if not rows and not self._numbered:
            # Nothing came back, which is either a session with no messages or one
            # written before a message carried the session it belongs to. Asked only
            # when the read found nothing, so a conversation that has any messages is
            # still one statement.
            if self._number_existing_messages():
                rows = self._graph.query(
                    self._select_query(), {"sid": self.session_id}
                )
        return self._rows_to_messages(rows)

    def _number_existing_messages(self) -> bool:
        """Establish the session's next number, for one written before it carried it.

        Done once per history object, and it reads nothing once the number is there. The
        count and the write are separate statements because on this server an aggregate
        followed by a write clause in one statement is refused.
        """
        if self._numbered:
            return False
        self._numbered = True
        rows = self._graph.query(self._repair_needed_query(), {"sid": self.session_id})
        if not rows or not int(rows[0]["sessions"]):
            return False  # no such session yet
        numbered = int(rows[0]["numbered"])
        named = int(rows[0]["named"])
        if numbered and named:
            return False  # it carries both already
        count = (
            int(self._graph.query(
                self._count_messages_query(), {"sid": self.session_id}
            )[0]["cnt"])
            if not numbered
            else 0
        )
        with self._graph.transaction():
            if not named:
                self._graph.query(
                    self._name_session_query(), {"sid": self.session_id}
                )
                self._graph.query(
                    self._mark_named_query(), {"sid": self.session_id}
                )
            if not numbered:
                self._graph.query(
                    self._number_query(), {"sid": self.session_id, "cnt": count}
                )
        return True

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        self._number_existing_messages()
        self._graph.merging(
            lambda: self._graph.query(
                self._append_query(),
                {
                    "sid": self.session_id,
                    "rows": Jsonb(self._offsets(messages)),
                    "added": len(messages),
                },
            )
        )

    def clear(self, delete_session_node: bool = False) -> None:
        self._graph.query(
            self._clear_query(delete_session_node), {"sid": self.session_id}
        )

    # ---- async API ----

    async def aget_messages(self) -> List[BaseMessage]:
        rows = await self._graph.aquery(self._select_query(), {"sid": self.session_id})
        if not rows and not self._numbered:
            if await self._anumber_existing_messages():
                rows = await self._graph.aquery(
                    self._select_query(), {"sid": self.session_id}
                )
        return self._rows_to_messages(rows)

    async def _anumber_existing_messages(self) -> bool:
        """Async sibling of :meth:`_number_existing_messages`."""
        if self._numbered:
            return False
        self._numbered = True
        rows = await self._graph.aquery(
            self._repair_needed_query(), {"sid": self.session_id}
        )
        if not rows or not int(rows[0]["sessions"]):
            return False
        numbered = int(rows[0]["numbered"])
        named = int(rows[0]["named"])
        if numbered and named:
            return False
        count = 0
        if not numbered:
            counted = await self._graph.aquery(
                self._count_messages_query(), {"sid": self.session_id}
            )
            count = int(counted[0]["cnt"])
        async with self._graph.atransaction():
            if not named:
                await self._graph.aquery(
                    self._name_session_query(), {"sid": self.session_id}
                )
                await self._graph.aquery(
                    self._mark_named_query(), {"sid": self.session_id}
                )
            if not numbered:
                await self._graph.aquery(
                    self._number_query(), {"sid": self.session_id, "cnt": count}
                )
        return True

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        await self._anumber_existing_messages()
        await self._graph.amerging(
            lambda: self._graph.aquery(
                self._append_query(),
                {
                    "sid": self.session_id,
                    "rows": Jsonb(self._offsets(messages)),
                    "added": len(messages),
                },
            )
        )

    async def aclear(self, delete_session_node: bool = False) -> None:
        await self._graph.aquery(
            self._clear_query(delete_session_node), {"sid": self.session_id}
        )


__all__ = ["AgensChatMessageHistory"]
