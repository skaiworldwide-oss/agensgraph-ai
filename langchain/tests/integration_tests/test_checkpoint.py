"""Integration tests for AgensSaver (LangGraph checkpointer)."""

import os
from typing import Annotated, TypedDict

import pytest

from langchain_agensgraph import AgensGraph, AgensSaver


def _conf():
    return {
        "dbname": os.getenv("AGENSGRAPH_DB"),
        "user": os.getenv("AGENSGRAPH_USER"),
        "password": os.getenv("AGENSGRAPH_PASSWORD"),
        "host": os.getenv("AGENSGRAPH_HOST", "localhost"),
        "port": int(os.getenv("AGENSGRAPH_PORT", 5432)),
    }


@pytest.fixture
def saver():
    g = AgensGraph("checkpoints", _conf(), create=True)
    g.query("MATCH (n) DETACH DELETE n")
    s = AgensSaver(graph=g)
    yield s
    g.close()


def _empty_checkpoint(cid: str, channel_values: dict, versions: dict):
    return {
        "v": 1,
        "id": cid,
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": channel_values,
        "channel_versions": versions,
        "versions_seen": {},
        "pending_sends": [],
    }


def test_put_get_roundtrip(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    ckpt = _empty_checkpoint("c1", {"messages": ["hi"], "count": 1}, {"messages": "1", "count": "1"})
    new_cfg = saver.put(cfg, ckpt, {"source": "input", "step": 1}, {"messages": "1", "count": "1"})
    assert new_cfg["configurable"]["checkpoint_id"] == "c1"

    tup = saver.get_tuple({"configurable": {"thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": "c1"}})
    assert tup is not None
    assert tup.checkpoint["id"] == "c1"
    assert tup.checkpoint["channel_values"] == {"messages": ["hi"], "count": 1}
    assert tup.metadata["source"] == "input"


def test_get_latest_without_id(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "t2", "checkpoint_ns": ""}}
    saver.put(cfg, _empty_checkpoint("c1", {"x": 1}, {"x": "1"}), {"step": 1}, {"x": "1"})
    # second checkpoint with parent
    cfg2 = {"configurable": {"thread_id": "t2", "checkpoint_ns": "", "checkpoint_id": "c1"}}
    saver.put(cfg2, _empty_checkpoint("c2", {"x": 2}, {"x": "2"}), {"step": 2}, {"x": "2"})

    latest = saver.get_tuple({"configurable": {"thread_id": "t2", "checkpoint_ns": ""}})
    assert latest.checkpoint["id"] == "c2"
    assert latest.checkpoint["channel_values"] == {"x": 2}
    assert latest.parent_config["configurable"]["checkpoint_id"] == "c1"


def test_put_writes_surface_as_pending(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}}
    saver.put(cfg, _empty_checkpoint("c1", {"x": 1}, {"x": "1"}), {}, {"x": "1"})
    wcfg = {"configurable": {"thread_id": "t3", "checkpoint_ns": "", "checkpoint_id": "c1"}}
    saver.put_writes(wcfg, [("messages", "partial-a"), ("messages", "partial-b")], task_id="task-1")
    tup = saver.get_tuple(wcfg)
    assert tup.pending_writes
    channels = [c for _, c, _ in tup.pending_writes]
    values = [v for _, _, v in tup.pending_writes]
    assert channels == ["messages", "messages"]
    assert values == ["partial-a", "partial-b"]


def test_list_orders_and_limits(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}
    for cid in ["c1", "c2", "c3"]:
        saver.put(cfg, _empty_checkpoint(cid, {"x": cid}, {"x": cid}), {}, {"x": cid})
    listed = list(saver.list({"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}, limit=2))
    ids = [t.checkpoint["id"] for t in listed]
    assert ids == ["c3", "c2"]  # descending, limited


def test_delete_thread(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "t5", "checkpoint_ns": ""}}
    saver.put(cfg, _empty_checkpoint("c1", {"x": 1}, {"x": "1"}), {}, {"x": "1"})
    saver.delete_thread("t5")
    assert saver.get_tuple({"configurable": {"thread_id": "t5", "checkpoint_ns": "", "checkpoint_id": "c1"}}) is None


from langgraph.graph.message import add_messages  # noqa: E402


class _AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    counter: int


def test_end_to_end_stategraph_resume(saver: AgensSaver):
    from langgraph.graph import START, StateGraph

    State = _AgentState

    def step(state: State):
        return {"messages": ["tick"], "counter": state.get("counter", 0) + 1}

    builder = StateGraph(State)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.set_finish_point("step")
    app = builder.compile(checkpointer=saver)

    thread = {"configurable": {"thread_id": "agent-1"}}
    app.invoke({"messages": [], "counter": 0}, thread)
    app.invoke({"messages": []}, thread)  # resumes, counter should continue

    state = app.get_state(thread)
    assert state.values["counter"] == 2
    assert len(state.values["messages"]) == 2

    # A brand new saver/app over the same store still sees the persisted state.
    state2 = app.get_state(thread)
    assert state2.values["counter"] == 2


async def test_async_put_get(saver: AgensSaver):
    cfg = {"configurable": {"thread_id": "at1", "checkpoint_ns": ""}}
    await saver.aput(cfg, _empty_checkpoint("c1", {"x": 9}, {"x": "1"}), {"step": 1}, {"x": "1"})
    tup = await saver.aget_tuple({"configurable": {"thread_id": "at1", "checkpoint_ns": "", "checkpoint_id": "c1"}})
    assert tup.checkpoint["channel_values"] == {"x": 9}
