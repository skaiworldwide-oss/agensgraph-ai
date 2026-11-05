"""Chat history against a server: order, windows, replays, concurrency, lossless messages."""

from __future__ import annotations

import threading

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph._core.names import companion_schema, ident
from langchain_agensgraph.chat_message_histories import AgensChatMessageHistory
from langchain_agensgraph.observability import log_queries
from tests import datagen

pytestmark = pytest.mark.server


def history(
    engine: AgensEngine, graph_name: str, session: str, **kwargs: object
) -> AgensChatMessageHistory:
    return AgensChatMessageHistory(session, engine, graph_name=graph_name, **kwargs)  # type: ignore[arg-type]


def test_histories_on_one_engine_check_their_tables_once(
    engine: AgensEngine, graph_name: str
) -> None:
    history(engine, graph_name, "a").add_messages([HumanMessage("hi")])
    # A history made per request, as RunnableWithMessageHistory makes one, pays no check.
    with log_queries() as statements:
        assert history(engine, graph_name, "b").messages == []
    assert len([s for s in statements if s.statement]) == 1


def test_messages_come_back_in_order_and_as_they_went_in(
    engine: AgensEngine, graph_name: str
) -> None:
    sessions = datagen.chat_sessions(2, 6, seed=4)
    for session_id, messages in sessions.items():
        chat = history(engine, graph_name, session_id)
        chat.add_messages(messages[:3])
        chat.add_messages(messages[3:])
        assert chat.messages == messages
    first, second = (history(engine, graph_name, s) for s in sessions)
    first.clear()
    assert first.messages == []
    assert second.messages == sessions["session-1"]


def test_every_kind_of_message_survives_the_round_trip(
    engine: AgensEngine, graph_name: str
) -> None:
    messages = [
        SystemMessage("be brief", id="m0"),
        HumanMessage("weather?", id="m1", additional_kwargs={"lang": "en"}),
        AIMessage(
            "",
            id="m2",
            tool_calls=[
                {"name": "weather", "args": {"city": "Seoul"}, "id": "c1", "type": "tool_call"}
            ],
            response_metadata={"model": "fake"},
        ),
        ToolMessage("sunny", tool_call_id="c1", id="m3"),
    ]
    chat = history(engine, graph_name, "kinds")
    chat.add_messages(messages)
    assert chat.messages == messages


def test_a_message_without_an_id_is_given_one_on_a_copy(
    engine: AgensEngine, graph_name: str
) -> None:
    given = HumanMessage("no id")
    chat = history(engine, graph_name, "ids")
    chat.add_messages([given])
    assert given.id is None
    (stored,) = chat.messages
    assert stored.content == "no id" and stored.id


def test_adding_the_same_messages_again_changes_nothing(
    dsn: str, engine: AgensEngine, graph_name: str
) -> None:
    messages = datagen.chat_sessions(1, 4, seed=1)["session-0"]
    chat = history(engine, graph_name, "replay")
    chat.add_messages(messages)
    chat.add_messages(messages)
    chat.add_messages(messages[2:])
    assert chat.messages == messages
    with psycopg.connect(dsn, autocommit=True) as conn:
        count = conn.execute(
            f"SELECT message_count FROM {ident(companion_schema(graph_name))}.chat_session "
            "WHERE session_id = 'replay'"
        ).fetchone()
    assert count == (4,)


def test_a_window_reads_the_last_messages(engine: AgensEngine, graph_name: str) -> None:
    messages = datagen.chat_sessions(1, 300, seed=2)["session-0"]
    chat = history(engine, graph_name, "long")
    for start in range(0, 300, 37):
        chat.add_messages(messages[start : start + 37])
    assert chat.messages == messages
    assert history(engine, graph_name, "long", window=10).messages == messages[-10:]


def test_a_first_call_in_a_callers_transaction_finds_the_tables_setup_made(
    engine: AgensEngine, graph_name: str
) -> None:
    with engine.transaction():
        chat = history(engine, graph_name, "s")
        chat.add_messages([HumanMessage("hi", id="m1")])
        assert [m.content for m in chat.messages] == ["hi"]
    assert [m.content for m in history(engine, graph_name, "s").messages] == ["hi"]


def test_concurrent_writers_lose_nothing_and_keep_their_own_order(
    engine: AgensEngine, graph_name: str
) -> None:
    history(engine, graph_name, "busy").setup()

    def write(writer: int) -> None:
        chat = history(engine, graph_name, "busy")
        for n in range(10):
            chat.add_messages([HumanMessage(f"{writer}:{n}", id=f"w{writer}m{n}")])

    threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = [str(m.content) for m in history(engine, graph_name, "busy").messages]
    assert len(stored) == 80 and len(set(stored)) == 80
    for writer in range(8):
        own = [s for s in stored if s.startswith(f"{writer}:")]
        assert own == [f"{writer}:{n}" for n in range(10)]


async def test_the_awaiting_interface_does_the_same(engine: AgensEngine, graph_name: str) -> None:
    messages = datagen.chat_sessions(1, 4, seed=7)["session-0"]
    chat = history(engine, graph_name, "async")
    await chat.aadd_messages(messages)
    assert await chat.aget_messages() == messages
    await chat.aclear()
    assert await chat.aget_messages() == []
    await engine.aclose()
