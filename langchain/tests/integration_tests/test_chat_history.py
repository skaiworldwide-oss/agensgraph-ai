"""Integration tests for AgensChatMessageHistory."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_agensgraph import AgensChatMessageHistory, AgensGraph


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


@pytest.fixture
def graph():
    g = AgensGraph("chat_history", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    yield g
    g.close()


def test_roundtrip_preserves_order_and_kwargs(graph):
    h = AgensChatMessageHistory("s1", graph=graph)
    h.add_messages(
        [
            SystemMessage(content="be brief"),
            HumanMessage(content="hi", additional_kwargs={"locale": "en"}),
            AIMessage(content="hello"),
        ]
    )
    msgs = h.messages
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert [m.content for m in msgs] == ["be brief", "hi", "hello"]
    assert msgs[1].additional_kwargs["locale"] == "en"


def test_append_keeps_growing(graph):
    h = AgensChatMessageHistory("s2", graph=graph)
    h.add_message(HumanMessage(content="one"))
    h.add_message(AIMessage(content="two"))
    h.add_message(HumanMessage(content="three"))
    assert [m.content for m in h.messages] == ["one", "two", "three"]


def test_sessions_are_isolated(graph):
    a = AgensChatMessageHistory("alice", graph=graph)
    b = AgensChatMessageHistory("bob", graph=graph)
    a.add_message(HumanMessage(content="from alice"))
    b.add_message(HumanMessage(content="from bob"))
    assert [m.content for m in a.messages] == ["from alice"]
    assert [m.content for m in b.messages] == ["from bob"]


def test_clear(graph):
    h = AgensChatMessageHistory("s3", graph=graph)
    h.add_message(HumanMessage(content="x"))
    assert h.messages
    h.clear()
    assert h.messages == []


def test_window_returns_recent_in_order(graph):
    h = AgensChatMessageHistory("s4", graph=graph)
    for i in range(5):
        h.add_message(HumanMessage(content=f"m{i}"))
    windowed = AgensChatMessageHistory("s4", graph=graph, window=2)
    assert [m.content for m in windowed.messages] == ["m3", "m4"]


def test_order_holds_past_ten_messages(graph):
    # Guards against lexicographic (vs numeric) ordering of the seq property.
    h = AgensChatMessageHistory("s6", graph=graph)
    for i in range(12):
        h.add_message(HumanMessage(content=f"m{i}"))
    assert [m.content for m in h.messages] == [f"m{i}" for i in range(12)]


async def test_async_roundtrip(graph):
    h = AgensChatMessageHistory("s5", graph=graph)
    await h.aadd_messages([HumanMessage(content="async hi"), AIMessage(content="yo")])
    msgs = await h.aget_messages()
    assert [m.content for m in msgs] == ["async hi", "yo"]
    await h.aclear()
    assert await h.aget_messages() == []


class TestAppendingDoesNotGetDearerWithDepth:
    """A session carries the number it counts from.

    Counting the messages already there to number the next one makes each append dearer
    than the last, which over a conversation is quadratic in its length.
    """

    def test_a_session_written_without_the_number_keeps_its_order(self, graph):
        """A session written without the number is counted once and carries on."""
        from psycopg.types.json import Jsonb

        graph.query('CREATE (s:"Session" {id: \'legacy\'})')
        for i in range(5):
            graph.query(
                'MATCH (s:"Session" {id: \'legacy\'}) '
                'CREATE (s)-[:"HAS_MESSAGE"]->(:"Message" {seq: %(i)s, data: %(d)s})',
                {
                    "i": i,
                    "d": Jsonb({"type": "human", "data": {"content": f"old {i}"}}),
                },
            )
        history = AgensChatMessageHistory(session_id="legacy", graph=graph)
        assert [m.content for m in history.messages] == [f"old {i}" for i in range(5)]

        history.add_user_message("new one")
        assert [m.content for m in history.messages] == [
            *[f"old {i}" for i in range(5)],
            "new one",
        ]
        seqs = [
            r["seq"]
            for r in graph.query(
                'MATCH (s:"Session" {id: \'legacy\'})-[:"HAS_MESSAGE"]->(m:"Message") '
                "RETURN m.seq AS seq ORDER BY m.seq"
            )
        ]
        assert seqs == sorted(set(seqs)), "a number was handed out twice"

    def test_the_number_is_established_once(self, graph):
        history = AgensChatMessageHistory(session_id="counted", graph=graph)
        history.add_user_message("first")
        counts: list = []

        from agensgraph.observability import add_query_logger

        add_query_logger(
            lambda r: counts.append(r) if "count(" in str(r.statement or "") else None
        )
        history.add_user_message("second")
        history.add_user_message("third")
        assert counts == []

    def test_messages_keep_their_order_across_appends(self, graph):
        history = AgensChatMessageHistory(session_id="ordered", graph=graph)
        for i in range(12):
            history.add_user_message(f"m{i}")
        assert [m.content for m in history.messages] == [f"m{i}" for i in range(12)]

    @pytest.mark.asyncio
    async def test_the_async_path_numbers_the_same_way(self, graph):
        history = AgensChatMessageHistory(session_id="async-ordered", graph=graph)
        await history.aadd_messages([HumanMessage(content=f"a{i}") for i in range(5)])
        await history.aadd_messages([HumanMessage(content="a5")])
        got = await history.aget_messages()
        assert [m.content for m in got] == [f"a{i}" for i in range(6)]
