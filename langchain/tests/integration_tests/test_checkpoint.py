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


class TestPruneCopyAndRunDeletion:
    """The lifecycle operations BaseCheckpointSaver leaves to the implementation."""

    @staticmethod
    def _cfg(thread_id: str, checkpoint_id: str | None = None):
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return cfg

    def _write(self, saver: AgensSaver, thread_id: str, cid: str, run_id=None):
        md = {"source": "loop", "step": 1}
        if run_id:
            md["run_id"] = run_id
        ckpt = _empty_checkpoint(cid, {"messages": [cid]}, {"messages": "1"})
        saver.put(self._cfg(thread_id), ckpt, md, {"messages": "1"})
        saver.put_writes(self._cfg(thread_id, cid), [("messages", cid)], f"task-{cid}")

    def test_prune_keep_latest_keeps_only_the_newest(self, saver: AgensSaver):
        for cid in ("c1", "c2", "c3"):
            self._write(saver, "t1", cid)
        saver.prune(["t1"])
        remaining = [t.checkpoint["id"] for t in saver.list(self._cfg("t1"))]
        assert remaining == ["c3"]
        # the surviving checkpoint is still readable in full
        assert saver.get_tuple(self._cfg("t1")).checkpoint["id"] == "c3"

    def test_prune_keep_latest_is_per_namespace(self, saver: AgensSaver):
        for cid in ("c1", "c2"):
            self._write(saver, "t1", cid)
        cfg_ns = {"configurable": {"thread_id": "t1", "checkpoint_ns": "sub"}}
        saver.put(cfg_ns, _empty_checkpoint("s1", {}, {}), {"source": "loop"}, {})
        saver.put(cfg_ns, _empty_checkpoint("s2", {}, {}), {"source": "loop"}, {})
        saver.prune(["t1"])
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t1"))] == ["c2"]
        assert [t.checkpoint["id"] for t in saver.list(cfg_ns)] == ["s2"]

    def test_prune_delete_strategy_removes_everything(self, saver: AgensSaver):
        for cid in ("c1", "c2"):
            self._write(saver, "t1", cid)
        saver.prune(["t1"], strategy="delete")
        assert list(saver.list(self._cfg("t1"))) == []

    def test_prune_rejects_an_unknown_strategy(self, saver: AgensSaver):
        with pytest.raises(ValueError):
            saver.prune(["t1"], strategy="keep_everything")

    def test_prune_leaves_other_threads_alone(self, saver: AgensSaver):
        self._write(saver, "t1", "c1")
        self._write(saver, "t1", "c2")
        self._write(saver, "t2", "d1")
        saver.prune(["t1"])
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t2"))] == ["d1"]

    def test_copy_thread_carries_the_whole_chain(self, saver: AgensSaver):
        for cid in ("c1", "c2", "c3"):
            self._write(saver, "src", cid)
        saver.copy_thread("src", "dst")
        src = [t.checkpoint["id"] for t in saver.list(self._cfg("src"))]
        dst = [t.checkpoint["id"] for t in saver.list(self._cfg("dst"))]
        assert dst == src == ["c3", "c2", "c1"]
        # the copy is independent of its source
        saver.delete_thread("src")
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("dst"))] == dst

    def test_copy_thread_carries_channel_values_and_writes(self, saver: AgensSaver):
        self._write(saver, "src", "c1")
        saver.copy_thread("src", "dst")
        tup = saver.get_tuple(self._cfg("dst"))
        assert tup.checkpoint["channel_values"] == {"messages": ["c1"]}
        assert tup.pending_writes

    def test_delete_for_runs_removes_only_that_run(self, saver: AgensSaver):
        self._write(saver, "t1", "c1", run_id="run-a")
        self._write(saver, "t1", "c2", run_id="run-b")
        saver.delete_for_runs(["run-a"])
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t1"))] == ["c2"]

    def test_delete_for_runs_spans_threads(self, saver: AgensSaver):
        self._write(saver, "t1", "c1", run_id="run-a")
        self._write(saver, "t2", "d1", run_id="run-a")
        self._write(saver, "t2", "d2", run_id="run-b")
        saver.delete_for_runs(["run-a"])
        assert list(saver.list(self._cfg("t1"))) == []
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t2"))] == ["d2"]

    def test_empty_inputs_are_no_ops(self, saver: AgensSaver):
        self._write(saver, "t1", "c1")
        saver.delete_for_runs([])
        saver.prune([])
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t1"))] == ["c1"]

    @pytest.mark.asyncio
    async def test_async_prune_copy_and_run_deletion(self, saver: AgensSaver):
        self._write(saver, "t1", "c1", run_id="run-a")
        self._write(saver, "t1", "c2", run_id="run-b")
        await saver.aprune(["t1"])
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t1"))] == ["c2"]
        await saver.acopy_thread("t1", "t9")
        assert [t.checkpoint["id"] for t in saver.list(self._cfg("t9"))] == ["c2"]
        await saver.adelete_for_runs(["run-b"])
        assert list(saver.list(self._cfg("t1"))) == []
