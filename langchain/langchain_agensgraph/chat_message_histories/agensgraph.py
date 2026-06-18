"""Chat message history backed by AgensGraph.

Messages for a session are stored as an ordered chain of ``Message`` vertices
linked from a ``Session`` vertex::

    (:Session {id})-[:HAS_MESSAGE]->(:Message {seq, data})

``data`` is the LangChain ``message_to_dict`` form, so any ``BaseMessage``
subtype round-trips losslessly through ``messages_from_dict``. ``seq`` is a
monotonic per-session ordinal used to preserve order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from psycopg import sql
from psycopg.types.json import Jsonb

from langchain_agensgraph.graphs.agensgraph import AgensGraph


class AgensChatMessageHistory(BaseChatMessageHistory):
    """Persist LangChain chat messages in an AgensGraph graph.

    Args:
        session_id: Identifier for the conversation.
        graph: An existing :class:`AgensGraph` to reuse. If omitted, ``conf``
            (and ``graph_name``) are used to build one.
        conf: psycopg connection kwargs (used only when ``graph`` is None).
        graph_name: Graph to store messages in (default ``chat_history``).
        session_node_label: Vertex label for sessions (default ``Session``).
        message_node_label: Vertex label for messages (default ``Message``).
        relationship: Edge label linking session to messages
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
            graph = AgensGraph(graph_name, conf, create=True)
        self._graph = graph
        self.session_id = session_id
        self.session_node_label = session_node_label
        self.message_node_label = message_node_label
        self.relationship = relationship
        self.window = window

        # Ensure the labels exist (AgensGraph requires labels before MERGE/CREATE
        # in some paths; IF NOT EXISTS makes this idempotent).
        self._graph.query(
            sql.SQL("CREATE VLABEL IF NOT EXISTS {l}").format(
                l=sql.Identifier(self.session_node_label)
            )
        )
        self._graph.query(
            sql.SQL("CREATE VLABEL IF NOT EXISTS {l}").format(
                l=sql.Identifier(self.message_node_label)
            )
        )
        self._graph.query(
            sql.SQL("CREATE ELABEL IF NOT EXISTS {l}").format(
                l=sql.Identifier(self.relationship)
            )
        )
        # Index the session id so per-session lookups/appends are index scans
        # rather than a seq scan over every session.
        self._graph.query(
            sql.SQL("CREATE PROPERTY INDEX IF NOT EXISTS {name} ON {l} (id)").format(
                name=sql.Identifier(f"{self.session_node_label}_id_idx"),
                l=sql.Identifier(self.session_node_label),
            )
        )

    # ---- query builders (shared by sync + async) ----

    def _select_query(self) -> Any:
        order = "ORDER BY m.seq"
        limit = ""
        if self.window is not None:
            # take the last `window` by seq desc, caller re-sorts ascending
            order = "ORDER BY m.seq DESC"
            limit = f"LIMIT {int(self.window)}"
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}})-[:{rl}]->(m:{ml}) "
            "RETURN m.data AS data, m.seq AS seq " + order + " " + limit
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    def _count_query(self) -> Any:
        # Messages are append-only, so the next sequence number is simply the
        # current count. (Counting avoids relying on max() over a jsonb
        # property, whose ordering is not reliably numeric in AgensGraph.)
        return sql.SQL(
            "MATCH (s:{sl} {{id: %(sid)s}})-[:{rl}]->(m:{ml}) "
            "RETURN count(m) AS cnt"
        ).format(
            sl=sql.Identifier(self.session_node_label),
            ml=sql.Identifier(self.message_node_label),
            rl=sql.Identifier(self.relationship),
        )

    @staticmethod
    def _next_seq(rows: List[Dict[str, Any]]) -> int:
        if rows and rows[0].get("cnt") is not None:
            return int(rows[0]["cnt"])
        return 0

    def _insert_query(self) -> Any:
        return sql.SQL(
            "MERGE (s:{sl} {{id: %(sid)s}}) "
            "WITH s "
            "UNWIND %(rows)s AS row "
            "CREATE (s)-[:{rl}]->(:{ml} {{seq: row.seq, data: row.data}})"
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

    def _message_rows(self, messages: Sequence[BaseMessage], start_seq: int):
        return [
            {"seq": start_seq + i, "data": message_to_dict(m)}
            for i, m in enumerate(messages)
        ]

    # ---- sync API ----

    @property
    def messages(self) -> List[BaseMessage]:
        rows = self._graph.query(self._select_query(), {"sid": Jsonb(self.session_id)})
        return self._rows_to_messages(rows)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        rows = self._graph.query(self._count_query(), {"sid": Jsonb(self.session_id)})
        start = self._next_seq(rows)
        self._graph.query(
            self._insert_query(),
            {"sid": Jsonb(self.session_id), "rows": Jsonb(self._message_rows(messages, start))},
        )

    def clear(self, delete_session_node: bool = False) -> None:
        self._graph.query(
            self._clear_query(delete_session_node), {"sid": Jsonb(self.session_id)}
        )

    # ---- async API ----

    async def aget_messages(self) -> List[BaseMessage]:
        rows = await self._graph.aquery(self._select_query(), {"sid": Jsonb(self.session_id)})
        return self._rows_to_messages(rows)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        rows = await self._graph.aquery(self._count_query(), {"sid": Jsonb(self.session_id)})
        start = self._next_seq(rows)
        await self._graph.aquery(
            self._insert_query(),
            {"sid": Jsonb(self.session_id), "rows": Jsonb(self._message_rows(messages, start))},
        )

    async def aclear(self, delete_session_node: bool = False) -> None:
        await self._graph.aquery(
            self._clear_query(delete_session_node), {"sid": Jsonb(self.session_id)}
        )


__all__ = ["AgensChatMessageHistory"]
