"""The checkpointer against a server: LangGraph's conformance suite through both interfaces, what
the suite leaves out, and real graphs run on it."""

from __future__ import annotations

import asyncio
import math
import operator
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import psycopg
import pytest
from agensgraph.numbers import read_numbers_exactly, reading_numbers_exactly
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    message_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import Capability
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.types import ERROR, INTERRUPT, TASKS, _DeltaSnapshot
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from psycopg import sql

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph._core.names import companion_schema
from langchain_agensgraph._core.program import Program, Statement, scalar
from langchain_agensgraph.checkpoint import AgensSaver
from langchain_agensgraph.checkpoint import agensgraph as saver_module
from langchain_agensgraph.errors import ClosedError
from langchain_agensgraph.graphs import AgensGraph, GraphDocument, Node
from langchain_agensgraph.observability import log_queries

pytestmark = pytest.mark.server


@pytest.fixture
async def aengine(dsn: str) -> AsyncIterator[AgensEngine]:
    async with AgensEngine(dsn) as made:
        yield made


@pytest.fixture
def saver(engine: AgensEngine, graph_name: str) -> AgensSaver:
    return AgensSaver(engine, graph_name=graph_name)


def config(
    thread_id: Any, checkpoint_id: str | None = None, *, ns: str | None = ""
) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if ns is not None:
        configurable["checkpoint_ns"] = ns
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def checkpoint(
    values: dict[str, Any] | None = None, versions: ChannelVersions | None = None, *, v: int = 4
) -> Checkpoint:
    values = values or {}
    return Checkpoint(
        v=v,
        id=str(uuid6(clock_seq=-1)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values=values,
        channel_versions=versions if versions is not None else dict.fromkeys(values, "1"),
        versions_seen={},
        updated_channels=None,
    )


def chain(
    saver: AgensSaver, thread_id: str, steps: Sequence[dict[str, Any]], *, ns: str = "", **meta: Any
) -> list[RunnableConfig]:
    """One checkpoint per step, each the child of the one before and every value under a new
    version; their configs."""
    saved: list[RunnableConfig] = []
    for step, values in enumerate(steps):
        parent = saved[-1]["configurable"]["checkpoint_id"] if saved else None
        made = checkpoint(values, {k: saver.get_next_version(None, None) for k in values})
        saved.append(
            saver.put(
                config(thread_id, parent, ns=ns),
                made,
                {"source": "loop", "step": step, **meta},
                made["channel_versions"],
            )
        )
    return saved


def ids(found: Iterator[CheckpointTuple] | list[CheckpointTuple]) -> list[str]:
    return [t.config["configurable"]["checkpoint_id"] for t in found]


def count(dsn: str, graph_name: str, table: str, thread_id: str) -> int:
    schema = companion_schema(graph_name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        found = conn.execute(
            sql.SQL("SELECT count(*) FROM {}.{} WHERE thread_id = %s").format(
                sql.Identifier(schema), sql.Identifier(table)
            ),
            (thread_id,),
        ).fetchone()
    assert found is not None
    return int(found[0])


# -- conformance -----------------------------------------------------------------------------


class Blocking(BaseCheckpointSaver[str]):
    """A saver's blocking methods behind the awaiting ones the conformance suite calls, each run
    on a worker thread."""

    def __init__(self, saver: AgensSaver) -> None:
        super().__init__(serde=saver.serde)
        self.saver = saver

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.saver.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        found = await asyncio.to_thread(
            lambda: [*self.saver.list(config, filter=filter, before=before, limit=limit)]
        )
        for each in found:
            yield each

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.saver.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.saver.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.saver.delete_thread, thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await asyncio.to_thread(self.saver.delete_for_runs, run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await asyncio.to_thread(self.saver.copy_thread, source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await asyncio.to_thread(self.saver.prune, thread_ids, strategy=strategy)


@pytest.mark.parametrize("interface", ["awaiting", "blocking"])
async def test_the_conformance_suite_passes_every_capability(
    aengine: AgensEngine, graph_name: str, interface: str
) -> None:
    saver = AgensSaver(aengine, graph_name=graph_name)

    @checkpointer_test(name=f"AgensSaver ({interface})")
    async def made() -> AsyncIterator[BaseCheckpointSaver[str]]:
        yield saver if interface == "awaiting" else Blocking(saver)

    report = await validate(made)
    assert set(report.results) == {capability.value for capability in Capability}
    for name, result in report.results.items():
        assert result.detected, name
        assert result.passed, (name, result.failures)
        assert result.tests_passed > 0, name


# -- setup -----------------------------------------------------------------------------------


def test_the_first_call_makes_the_tables_and_the_graph_need_not_exist(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    assert saver.get_tuple(config("t")) is None
    schema = companion_schema(graph_name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        graphs = conn.execute("SELECT count(*) FROM ag_graph WHERE graphname = %s", (graph_name,))
        assert graphs.fetchone() == (0,)
        applied = conn.execute(
            sql.SQL("SELECT v FROM {}.checkpoint_migrations ORDER BY v").format(
                sql.Identifier(schema)
            )
        )
        assert [v for (v,) in applied] == [*range(len(saver._tables.migrations))]
        indexes = conn.execute(
            "SELECT indexname::text FROM pg_indexes WHERE schemaname = %s ORDER BY 1", (schema,)
        )
        assert [name for (name,) in indexes] == [
            "checkpoint_blobs_pkey",
            "checkpoint_items_pkey",
            "checkpoint_migrations_pkey",
            "checkpoint_threads_pkey",
            "checkpoint_writes_pkey",
            "checkpoints_pkey",
            "checkpoints_run_id_idx",
        ]


def test_setup_is_checked_once_and_can_be_left_out(engine: AgensEngine, graph_name: str) -> None:
    unmade = AgensSaver(engine, graph_name=graph_name, setup=False)
    with pytest.raises(psycopg.errors.UndefinedTable):
        unmade.get_tuple(config("t"))
    unmade.setup()
    assert unmade.get_tuple(config("t")) is None
    again = AgensSaver(engine, graph_name=graph_name)
    again.setup()
    assert again._setup.done


def test_a_first_call_inside_a_callers_transaction_joins_it(
    engine: AgensEngine, saver: AgensSaver
) -> None:
    with pytest.raises(RuntimeError), engine.transaction() as conn:
        saver.put(config("t"), checkpoint({"x": 1}), {}, {"x": "1"})
        conn.execute("SELECT 1")
        raise RuntimeError("roll back")
    assert saver.get_tuple(config("t")) is None
    with engine.transaction():
        saver.put(config("t"), checkpoint({"x": 1}), {}, {"x": "1"})
    assert saver.get_tuple(config("t")) is not None


def test_each_call_sends_the_statements_the_module_says(
    dsn: str, engine: AgensEngine, graph_name: str
) -> None:
    AgensSaver(engine, graph_name=graph_name).setup()
    saver = AgensSaver(engine, graph_name=graph_name)

    def sent(call: Any) -> int:
        """How many statements the call sent; the pool's empty liveness check is not one."""
        with log_queries() as statements:
            call()
        return len([s for s in statements if s.statement])

    made = checkpoint({"items": [1], "text": "inline"})
    # The tables were made through this engine, so a saver's first call checks nothing; on
    # another engine the first call reads which migrations are recorded, once.
    assert sent(lambda: saver.get_tuple(config("t"))) == 1
    with AgensEngine(dsn) as other:
        fresh = AgensSaver(other, graph_name=graph_name)
        assert sent(lambda: fresh.get_tuple(config("t"))) == 2
        assert sent(lambda: fresh.get_tuple(config("t"))) == 1
    saved = saver.put(config("t"), made, {"run_id": "r"}, made["channel_versions"])
    assert sent(lambda: saver.put(saved, checkpoint(), {}, {})) == 1
    assert sent(lambda: saver.put(saved, made, {}, made["channel_versions"])) == 2
    assert sent(lambda: saver.put_writes(saved, [("a", 1), ("b", 2), ("c", 3)], "task")) == 1
    assert sent(lambda: [*saver.list(config("t"), filter={"run_id": "r"}, limit=5)]) == 1
    assert sent(lambda: saver.copy_thread("t", "u")) == 1
    assert sent(lambda: saver.delete_for_runs(["r"])) == 1
    assert sent(lambda: saver.delete_thread("u")) == 5
    assert sent(lambda: saver.delete_for_runs([])) == 0
    assert sent(lambda: saver.prune(["t"], strategy="delete")) == 5

    # A declared channel adds the pointer and one statement per channel written, to the same
    # pipeline. A put whose parent is where this saver left the pointer writes only the channels
    # with a new version; any other, a fork here, writes every declared channel.
    queried = AgensSaver(engine, graph_name=graph_name, queryable_channels={"items": "append"})
    first = queried.put(config("q"), made, {}, made["channel_versions"])
    same = checkpoint({"items": [1]})
    second = queried.put(first, same, {}, {})
    assert sent(lambda: queried.put(second, checkpoint({"items": [1]}), {}, {})) == 2
    latest = queried.get_tuple(config("q"))
    assert latest is not None
    changed = checkpoint({"items": [1, 2]}, {"items": "2"})
    assert sent(lambda: queried.put(latest.config, changed, {}, {"items": "2"})) == 4
    assert sent(lambda: queried.put(first, checkpoint({"items": [1]}), {}, {})) == 3


# -- reads and writes ------------------------------------------------------------------------


def test_values_are_inline_or_blobs_and_read_back_by_version(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    first = checkpoint(
        {
            "text": "a",
            "n": 1,
            "f": 1.5,
            "flag": True,
            "none": None,
            "items": [1, 2],
            "map": {"k": 1},
        }
    )
    saved = saver.put(
        config("t"), first, {"source": "input", "step": -1}, first["channel_versions"]
    )
    assert count(dsn, graph_name, "checkpoint_blobs", "t") == 2
    second = checkpoint({"items": [1, 2, 3], "map": {"k": 1}}, {"items": "2", "map": "1"})
    saver.put(saved, second, {"source": "loop", "step": 0}, {"items": "2"})
    assert count(dsn, graph_name, "checkpoint_blobs", "t") == 3
    found = saver.get_tuple(config("t"))
    assert found is not None
    assert found.checkpoint["channel_values"] == {"items": [1, 2, 3], "map": {"k": 1}}
    assert found.parent_config == saved
    older = saver.get_tuple(saved)
    assert older is not None
    assert older.checkpoint["channel_values"] == first["channel_values"]


def test_a_delta_snapshot_is_a_blob_behind_an_inline_marker(saver: AgensSaver) -> None:
    made = checkpoint({"delta": _DeltaSnapshot(["a", "b"])})
    saver.put(config("t"), made, {}, made["channel_versions"])
    found = saver.get_tuple(config("t"))
    assert found is not None
    assert found.checkpoint["channel_values"] == {"delta": _DeltaSnapshot(["a", "b"])}


def test_metadata_loses_null_characters_and_a_filter_finds_it_without_them(
    saver: AgensSaver,
) -> None:
    made = checkpoint()
    saver.put(config("t"), made, {"source": "loop", "note": "a\x00b", "step": 0}, {})
    found = saver.get_tuple(config("t"))
    assert found is not None and found.metadata["note"] == "ab"
    assert ids(saver.list(None, filter={"note": "a\x00b"})) == [made["id"]]


def test_writes_are_read_in_task_path_order_and_their_index_decides_replacement(
    saver: AgensSaver,
) -> None:
    saved = saver.put(config("t"), checkpoint(), {}, {})
    saver.put_writes(saved, [("a", 1), ("b", 2)], "task-z", "~path-2")
    saver.put_writes(saved, [("c", 3)], "task-a", "~path-1")
    saver.put_writes(saved, [("a", 9)], "task-z", "~path-2")
    saver.put_writes(saved, [(ERROR, "first"), (ERROR, "second")], "task-e")
    saver.put_writes(saved, [(INTERRUPT, "kept"), ("d", 4), (INTERRUPT, "dropped")], "task-i")
    found = saver.get_tuple(saved)
    assert found is not None
    assert found.pending_writes == [
        ("task-e", ERROR, "second"),
        ("task-i", INTERRUPT, "kept"),
        ("task-i", "d", 4),
        ("task-a", "c", 3),
        ("task-z", "a", 1),
        ("task-z", "b", 2),
    ]
    saver.put_writes(saved, [(ERROR, "replaced")], "task-e")
    saver.put_writes(saved, [], "task-none")
    found = saver.get_tuple(saved)
    assert found is not None and ("task-e", ERROR, "replaced") in (found.pending_writes or [])


def test_a_checkpoint_before_format_4_reads_the_sends_its_parent_holds(
    saver: AgensSaver,
) -> None:
    for thread_id in ("t1", "t2"):
        parent = checkpoint({"x": "a"}, v=1)
        saved = saver.put(config(thread_id), parent, {}, {"x": "1"})
        saver.put_writes(saved, [(TASKS, f"{thread_id}-send-1")], "task-1", "~1")
        saver.put_writes(saved, [(TASKS, f"{thread_id}-send-0")], "task-0", "~0")
        child = checkpoint({"x": "a"}, {"x": "1"}, v=1)
        saver.put(saved, child, {}, {})
    found = saver.get_tuple(config("t1"))
    assert found is not None
    assert found.checkpoint["channel_values"][TASKS] == ["t1-send-0", "t1-send-1"]
    assert found.checkpoint["channel_versions"][TASKS] == "1"
    listed = {
        t.config["configurable"]["thread_id"]: t.checkpoint["channel_values"].get(TASKS)
        for t in saver.list(None)
        if t.parent_config is not None
    }
    assert listed == {"t1": ["t1-send-0", "t1-send-1"], "t2": ["t2-send-0", "t2-send-1"]}


def test_list_reads_every_thread_namespace_and_bound(saver: AgensSaver) -> None:
    root = chain(saver, "t1", [{"x": 1}, {"x": 2}, {"x": 3}])
    child = chain(saver, "t1", [{"y": 1}], ns="child:1")
    other = chain(saver, "t2", [{"z": 1}])
    newest_first = [c["configurable"]["checkpoint_id"] for c in root[::-1]]
    assert ids(saver.list(config("t1"))) == newest_first
    assert len([*saver.list(config("t1", ns=None))]) == 4
    assert len([*saver.list(None)]) == 5
    assert ids(saver.list(config("t1"), before=root[2], limit=1)) == newest_first[1:2]
    assert ids(saver.list(config("t1"), filter={"step": 1})) == newest_first[1:2]
    assert ids(saver.list(config("t1", ns="child:1"))) == [
        child[0]["configurable"]["checkpoint_id"]
    ]
    assert ids(saver.list(other[0])) == [other[0]["configurable"]["checkpoint_id"]]


# -- deleting, copying and pruning ------------------------------------------------------------


def test_delete_thread_takes_every_row_of_the_thread_only(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    kept = chain(saver, "t2", [{"x": [1]}])
    gone = chain(saver, "t1", [{"x": [1]}, {"x": [2]}])
    saver.put_writes(gone[-1], [("x", 3)], "task")
    chain(saver, "t1", [{"y": [1]}], ns="child:1")
    saver.delete_thread("t1")
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        assert count(dsn, graph_name, table, "t1") == 0
    assert saver.get_tuple(kept[0]) is not None


def test_delete_for_runs_takes_their_checkpoints_writes_and_the_blobs_only_they_used(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    first = chain(saver, "t1", [{"x": [1], "kept": [0]}], run_id="run-a")
    shared = saver.get_tuple(first[0])
    assert shared is not None
    later = checkpoint({"x": [2], "kept": [0]}, {**shared.checkpoint["channel_versions"], "x": "9"})
    second = saver.put(first[0], later, {"run_id": "run-b"}, {"x": "9"})
    child = chain(saver, "t1", [{"y": [1]}], run_id="run-c")
    elsewhere = chain(saver, "t2", [{"x": [3]}], run_id="run-a")
    saver.put_writes(first[0], [("x", 9)], "task")
    with pytest.raises(TypeError):
        saver.delete_for_runs("run-a")
    saver.delete_for_runs(["run-a", "run-unknown"])
    saver.delete_for_runs([])
    assert set(ids(saver.list(None))) == {
        second["configurable"]["checkpoint_id"],
        child[0]["configurable"]["checkpoint_id"],
    }
    assert saver.get_tuple(elsewhere[0]) is None
    assert count(dsn, graph_name, "checkpoint_writes", "t1") == 0
    # x's first version went with its only checkpoint; "kept" is still read by the second.
    assert count(dsn, graph_name, "checkpoint_blobs", "t1") == 3
    assert count(dsn, graph_name, "checkpoint_blobs", "t2") == 0
    found = saver.get_tuple(second)
    assert found is not None
    assert found.parent_config is None
    assert found.checkpoint["channel_values"] == {"x": [2], "kept": [0]}


def test_copy_thread_copies_the_whole_chain_and_refuses_an_occupied_target(
    saver: AgensSaver,
) -> None:
    source = chain(saver, "src", [{"x": [1]}, {"x": [2]}])
    chain(saver, "src", [{"y": "sub"}], ns="child:1")
    saver.put_writes(source[-1], [("x", 3)], "task", "~1")
    saver.copy_thread("src", "dst")
    for ns in ("", "child:1"):
        copied = [*saver.list(config("dst", ns=ns))]
        original = [*saver.list(config("src", ns=ns))]
        assert len(copied) == len(original) > 0
        for made, kept in zip(copied, original, strict=True):
            assert (made.checkpoint, made.metadata, made.pending_writes) == (
                kept.checkpoint,
                kept.metadata,
                kept.pending_writes,
            )
            assert made.config == {
                "configurable": {**kept.config["configurable"], "thread_id": "dst"}
            }
            if kept.parent_config is None:
                assert made.parent_config is None
            else:
                assert made.parent_config == {
                    "configurable": {**kept.parent_config["configurable"], "thread_id": "dst"}
                }
    saver.delete_thread("src")
    assert len([*saver.list(config("dst", ns=None))]) == 3
    chain(saver, "src", [{"x": [9]}])
    with pytest.raises(ValueError, match="already holds checkpoints"):
        saver.copy_thread("src", "dst")
    assert len([*saver.list(config("dst", ns=None))]) == 3


def test_prune_keep_latest_keeps_each_namespaces_head_and_its_blobs(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    root = chain(saver, "t", [{"x": [1]}, {"x": [2]}, {"x": [3]}])
    saver.put_writes(root[0], [("x", 0)], "old-task")
    saver.put_writes(root[-1], [("x", 4)], "new-task")
    first_root = root[0]["configurable"]["checkpoint_id"]
    call = chain(saver, "t", [{"y": [1]}], ns="call:task", parents={"": first_root})
    kept_sub = chain(saver, "t", [{"z": [1]}, {"z": [2]}], ns="sub")
    with pytest.raises(TypeError):
        saver.prune("t")
    with pytest.raises(ValueError, match="Unknown prune strategy"):
        saver.prune(["t"], strategy="keep_everything")
    saver.prune(["t"])
    assert ids(saver.list(config("t"))) == [root[-1]["configurable"]["checkpoint_id"]]
    head = saver.get_tuple(config("t"))
    assert head is not None and head.parent_config is None
    assert head.pending_writes == [("new-task", "x", 4)]
    assert saver.get_tuple(call[0]) is None
    assert ids(saver.list(config("t", ns="sub"))) == [kept_sub[-1]["configurable"]["checkpoint_id"]]
    assert count(dsn, graph_name, "checkpoint_blobs", "t") == 2
    assert count(dsn, graph_name, "checkpoint_writes", "t") == 1


def test_prune_refuses_a_thread_using_a_delta_channel_and_delete_removes_it(
    saver: AgensSaver,
) -> None:
    chain(saver, "plain", [{"x": [1]}, {"x": [2]}])
    chain(saver, "counted", [{"x": [1]}, {"x": [2]}], counters_since_delta_snapshot={"x": [1, 1]})
    snapshot = checkpoint({"d": _DeltaSnapshot([1])})
    saved = saver.put(config("marked"), snapshot, {}, snapshot["channel_versions"])
    saver.put(saved, checkpoint({}, {"d": "2"}), {}, {"d": "2"})
    for thread_id in ("counted", "marked"):
        with pytest.raises(NotImplementedError, match=thread_id):
            saver.prune(["plain", thread_id])
    assert len([*saver.list(config("plain"))]) == 2
    saver.prune(["plain", "counted", "marked"], strategy="delete")
    assert [*saver.list(None)] == []


# -- delta channel history ----------------------------------------------------------------------


def delta_thread(saver: AgensSaver, thread_id: str) -> list[RunnableConfig]:
    """A snapshot, then checkpoints that version the channel and store nothing for it."""
    first = checkpoint({"delta": _DeltaSnapshot(["seed"]), "plain": "kept"})
    saved = [saver.put(config(thread_id), first, {}, first["channel_versions"])]
    saver.put_writes(saved[0], [("delta", "w0"), ("other", "x")], "task-b", "~b")
    saver.put_writes(saved[0], [("delta", "w0-early")], "task-a", "~a")
    for step in range(1, 5):
        made = checkpoint({}, {"delta": f"{step + 1}", "plain": "1"})
        saved.append(saver.put(saved[-1], made, {}, {"delta": f"{step + 1}"}))
        saver.put_writes(saved[-1], [("delta", f"w{step}")], f"task-{step}", f"~{step}")
    return saved


@pytest.mark.parametrize("page", [2, 1024])
def test_delta_history_agrees_with_the_walk_the_base_class_does(
    saver: AgensSaver, monkeypatch: pytest.MonkeyPatch, page: int
) -> None:
    monkeypatch.setattr(saver_module, "DELTA_PAGE_SIZE", page)
    saved = delta_thread(saver, "t")
    channels = ["delta", "plain", "missing"]
    for target in (saved[-1], saved[2], saved[0], config("t")):
        ours = saver.get_delta_channel_history(config=target, channels=channels)
        base = BaseCheckpointSaver.get_delta_channel_history(
            saver, config=target, channels=channels
        )
        assert ours == base, target
    latest = saver.get_delta_channel_history(config=config("t"), channels=["delta"])
    assert latest["delta"]["seed"] == _DeltaSnapshot(["seed"])
    assert [w[2] for w in latest["delta"]["writes"]] == ["w0-early", "w0", "w1", "w2", "w3"]
    assert saver.get_delta_channel_history(config=config("none"), channels=["delta"]) == {
        "delta": {"writes": []}
    }
    assert saver.get_delta_channel_history(config=config("t"), channels=[]) == {}


async def test_the_awaiting_delta_history_is_the_blocking_one(
    aengine: AgensEngine, graph_name: str
) -> None:
    saver = AgensSaver(aengine, graph_name=graph_name)
    saved = await asyncio.to_thread(delta_thread, saver, "t")
    for target in (saved[-1], config("t")):
        assert await saver.aget_delta_channel_history(
            config=target, channels=["delta", "plain"]
        ) == await asyncio.to_thread(
            lambda: saver.get_delta_channel_history(config=target, channels=["delta", "plain"])
        )


# -- binding and closing -------------------------------------------------------------------------


def test_a_saver_is_bound_by_engine_conninfo_or_graph(dsn: str, graph_name: str) -> None:
    with AgensSaver.from_conn_string(dsn, graph_name=graph_name) as made:
        made.put(config("t"), checkpoint(), {}, {})
        own = made.engine
    assert own.closed
    shared = AgensSaver(dsn, graph_name=graph_name)
    assert len([*shared.list(None)]) == 1
    through = AgensSaver(graph=SimpleNamespace(engine=shared.engine, graph_name=graph_name))
    assert through.engine is shared.engine
    shared.close()
    assert shared.engine.closed
    with pytest.raises(ClosedError):
        shared.get_tuple(config("t"))
    with pytest.raises(ValueError, match="graph_name is required"):
        AgensSaver(dsn)
    with pytest.raises(ValueError, match="differs"):
        AgensSaver(graph=SimpleNamespace(engine=shared.engine, graph_name="a"), graph_name="b")


async def test_an_awaiting_saver_from_a_conninfo_closes_its_pools(
    dsn: str, graph_name: str
) -> None:
    async with AgensSaver.afrom_conn_string(dsn, graph_name=graph_name) as made:
        await made.aput(config("t"), checkpoint(), {}, {})
        assert [t async for t in made.alist(None)] != []
        own = made.engine
    assert own.closed


# -- a real graph ----------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turns: Annotated[int, operator.add]


def respond(state: State) -> dict[str, Any]:
    return {"messages": [AIMessage(f"echo: {state['messages'][-1].content}")], "turns": 1}


def app(checkpointer: BaseCheckpointSaver[Any]) -> Any:
    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


def contents(state: Any) -> list[str]:
    return [str(m.content) for m in state.values["messages"]]


def shape(history: Sequence[Any]) -> list[tuple[Any, ...]]:
    """What a history says, apart from ids and times."""
    return [
        (s.metadata["source"], s.metadata["step"], contents(s), s.values.get("turns"), s.next)
        for s in history
    ]


def run_blocking(checkpointer: BaseCheckpointSaver[Any]) -> dict[str, Any]:
    graph = app(checkpointer)
    thread: RunnableConfig = {"configurable": {"thread_id": "conversation"}}
    graph.invoke({"messages": [HumanMessage("hi")]}, thread)
    graph.invoke({"messages": [HumanMessage("again")]}, thread)
    after_two = graph.get_state(thread)
    graph.update_state(thread, {"messages": [HumanMessage("edited")]})
    history = [*graph.get_state_history(thread)]
    older = next(s for s in history if s.metadata["step"] == 1)
    graph.invoke({"messages": [HumanMessage("fork")]}, older.config)
    forked = graph.get_state(thread)
    return {
        "after_two": (contents(after_two), after_two.values["turns"]),
        "history": shape(history),
        "forked": (contents(forked), forked.values["turns"], forked.parent_config is not None),
        "everything": shape([*graph.get_state_history(thread)]),
        "older_kept": contents(graph.get_state(older.config)),
    }


async def run_awaiting(checkpointer: BaseCheckpointSaver[Any]) -> dict[str, Any]:
    graph = app(checkpointer)
    thread: RunnableConfig = {"configurable": {"thread_id": "conversation"}}
    await graph.ainvoke({"messages": [HumanMessage("hi")]}, thread)
    await graph.ainvoke({"messages": [HumanMessage("again")]}, thread)
    after_two = await graph.aget_state(thread)
    await graph.aupdate_state(thread, {"messages": [HumanMessage("edited")]})
    history = [s async for s in graph.aget_state_history(thread)]
    older = next(s for s in history if s.metadata["step"] == 1)
    await graph.ainvoke({"messages": [HumanMessage("fork")]}, older.config)
    forked = await graph.aget_state(thread)
    return {
        "after_two": (contents(after_two), after_two.values["turns"]),
        "history": shape(history),
        "forked": (contents(forked), forked.values["turns"], forked.parent_config is not None),
        "everything": shape([s async for s in graph.aget_state_history(thread)]),
        "older_kept": contents(await graph.aget_state(older.config)),
    }


def test_a_graph_runs_on_the_blocking_interface_as_on_the_reference_saver(
    saver: AgensSaver,
) -> None:
    ours = run_blocking(saver)
    assert ours == run_blocking(InMemorySaver())
    assert ours["after_two"] == (["hi", "echo: hi", "again", "echo: again"], 2)
    assert ours["forked"] == (["hi", "echo: hi", "fork", "echo: fork"], 2, True)
    assert ours["older_kept"] == ["hi", "echo: hi"]


async def test_a_graph_runs_on_the_awaiting_interface_as_on_the_reference_saver(
    aengine: AgensEngine, graph_name: str
) -> None:
    ours = await run_awaiting(AgensSaver(aengine, graph_name=graph_name))
    assert ours == await run_awaiting(InMemorySaver())
    assert ours["after_two"] == (["hi", "echo: hi", "again", "echo: again"], 2)
    assert ours["forked"] == (["hi", "echo: hi", "fork", "echo: fork"], 2, True)


# -- what jsonb can hold, ids that are not strings, numbers, leftovers ---------------------------


def test_a_value_jsonb_cannot_hold_is_a_blob(dsn: str, saver: AgensSaver, graph_name: str) -> None:
    values = {
        "nul": "a\x00b",
        "nan": math.nan,
        "inf": -math.inf,
        "surrogate": "a\ud800",
        "text": "plain",
        "f": 0.5,
        "big": 2**80,
    }
    made = checkpoint(values)
    saver.put(config("t"), made, {}, made["channel_versions"])
    assert count(dsn, graph_name, "checkpoint_blobs", "t") == 4
    found = saver.get_tuple(config("t"))
    assert found is not None
    got = found.checkpoint["channel_values"]
    assert got["nul"] == "a\x00b" and math.isnan(got["nan"]) and got["inf"] == -math.inf
    assert (got["text"], got["f"], got["big"]) == ("plain", 0.5, 2**80)
    assert "surrogate" in got


def test_a_large_blob_goes_as_a_binary_parameter(saver: AgensSaver) -> None:
    tables = saver._tables
    for text in (tables.upsert_blobs, tables.upsert_writes, tables.insert_writes):
        assert "%b::bytea[]" in text
    big = bytes(range(256)) * 12_000
    made = checkpoint({"big": big})
    saved = saver.put(config("t"), made, {}, made["channel_versions"])
    saver.put_writes(saved, [("big", big)], "task")
    found = saver.get_tuple(saved)
    assert found is not None
    assert found.checkpoint["channel_values"]["big"] == big
    assert found.pending_writes == [("task", "big", big)]


def test_a_thread_id_that_is_not_a_string_is_read_as_its_text(saver: AgensSaver) -> None:
    for thread in (7, uuid.uuid4()):
        made = checkpoint({"x": [1]})
        saved = saver.put(config(thread), made, {}, made["channel_versions"])
        assert saved["configurable"]["thread_id"] == thread
        saver.put_writes(saved, [("x", 2)], "task")
        found = saver.get_tuple(config(thread))
        assert found is not None and found.config["configurable"]["thread_id"] == str(thread)
        assert found.pending_writes == [("task", "x", 2)]
        assert ids(saver.list(config(thread))) == [made["id"]]
        history = saver.get_delta_channel_history(config=config(thread), channels=["x"])
        assert history == {"x": {"writes": []}}
        saver.delete_thread(thread)
        assert saver.get_tuple(config(thread)) is None


def test_floats_stay_floats_when_the_driver_reads_numbers_exactly(saver: AgensSaver) -> None:
    was = reading_numbers_exactly()
    read_numbers_exactly(True)
    try:
        made = checkpoint({"f": 0.1, "n": 3})
        saved = saver.put(config("t"), made, {"score": 0.25}, made["channel_versions"])
        saver.put(saved, checkpoint({}, {"f": "2", "n": "1"}), {}, {"f": "2"})
        found = saver.get_tuple(saved)
        assert found is not None
        assert type(found.checkpoint["channel_values"]["f"]) is float
        assert type(found.metadata["score"]) is float  # type: ignore[typeddict-item]
        history = saver.get_delta_channel_history(config=config("t"), channels=["f"])
        assert type(history["f"]["seed"]) is float
    finally:
        read_numbers_exactly(was)


def test_a_copy_after_deleting_runs_meets_no_leftover_blob(
    dsn: str, saver: AgensSaver, graph_name: str
) -> None:
    chain(saver, "src", [{"x": [1]}], run_id="r1")
    saver.copy_thread("src", "dst")
    saver.delete_for_runs(["r1"])
    assert count(dsn, graph_name, "checkpoint_blobs", "src") == 0
    assert count(dsn, graph_name, "checkpoint_blobs", "dst") == 0
    chain(saver, "src", [{"x": [2]}], run_id="r2")
    schema = sql.Identifier(companion_schema(graph_name))
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "INSERT INTO {}.checkpoint_blobs SELECT 'dst', checkpoint_ns, channel, version, "
                "type, blob FROM {}.checkpoint_blobs WHERE thread_id = 'src'"
            ).format(schema, schema)
        )
    saver.copy_thread("src", "dst")
    found = saver.get_tuple(config("dst"))
    assert found is not None and found.checkpoint["channel_values"] == {"x": [2]}


def test_a_deleted_middle_run_orphans_its_child_and_a_missing_parent_ends_a_walk(
    dsn: str, saver: AgensSaver, graph_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(saver_module, "DELTA_PAGE_SIZE", 2)
    saved: list[RunnableConfig] = []
    for step in range(6):
        made = checkpoint({}, {"d": f"{step + 1}"}) if step else checkpoint({"d": [0]})
        parent = saved[-1] if saved else config("t")
        saved.append(saver.put(parent, made, {"run_id": f"r{step}"}, made["channel_versions"]))
        saver.put_writes(saved[-1], [("d", step)], f"task-{step}")
    saver.delete_for_runs(["r4"])
    orphan = saver.get_tuple(saved[5])
    assert orphan is not None and orphan.parent_config is None
    base = BaseCheckpointSaver.get_delta_channel_history
    assert saver.get_delta_channel_history(config=saved[5], channels=["d"]) == base(
        saver, config=saved[5], channels=["d"]
    )
    target, parent = (c["configurable"]["checkpoint_id"] for c in (saved[3], saved[2]))
    schema = sql.Identifier(companion_schema(graph_name))
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "UPDATE {}.checkpoints SET parent_checkpoint_id = %s WHERE checkpoint_id = %s"
            ).format(schema),
            (parent + "x", target),
        )
    with log_queries() as statements:
        ours = saver.get_delta_channel_history(config=saved[3], channels=["d"])
    assert len([s for s in statements if "ver_0" in s.statement]) == 1
    assert ours == base(saver, config=saved[3], channels=["d"]) == {"d": {"writes": []}}


def test_a_stored_none_is_a_seed(saver: AgensSaver) -> None:
    first = checkpoint({"d": None})
    saved = saver.put(config("t"), first, {}, first["channel_versions"])
    saver.put_writes(saved, [("d", "w")], "task")
    later = saver.put(saved, checkpoint({}, {"d": "2"}), {}, {"d": "2"})
    ours = saver.get_delta_channel_history(config=later, channels=["d"])
    assert ours == {"d": {"writes": [("task", "d", "w")], "seed": None}}
    assert ours == BaseCheckpointSaver.get_delta_channel_history(
        saver, config=later, channels=["d"]
    )


def test_before_naming_no_checkpoint_bounds_nothing(saver: AgensSaver) -> None:
    chain(saver, "t", [{"x": 1}, {"x": 2}])
    assert len([*saver.list(config("t"), before=config("t"))]) == 2


def test_prune_inside_a_callers_transaction_joins_it(
    engine: AgensEngine, saver: AgensSaver
) -> None:
    root = chain(saver, "t", [{"x": [1]}, {"x": [2]}, {"x": [3]}])
    first = saver.get_tuple(root[0])
    assert first is not None
    with pytest.raises(RuntimeError), engine.transaction():
        # Saving the first checkpoint again locks its row, which the prune deletes.
        saver.put(config("t"), first.checkpoint, first.metadata, {})
        saver.prune(["t"])
        assert ids(saver.list(config("t"))) == [root[-1]["configurable"]["checkpoint_id"]]
        raise RuntimeError("roll back")
    assert len([*saver.list(config("t"))]) == 3
    with engine.transaction():
        saver.prune(["t"], strategy="delete")
    assert [*saver.list(None)] == []


# -- queryable channels --------------------------------------------------------------------------

CHANNELS = {"messages": "messages", "todo": "append", "plan": "last_value"}


class Chat(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    todo: Annotated[list[str], operator.add]
    plan: str


def answer(state: Chat) -> dict[str, Any]:
    last = state["messages"][-1] if state["messages"] else HumanMessage("nothing")
    return {
        "messages": [AIMessage(f"echo: {last.content}", id=f"ai-{uuid.uuid4().hex}")],
        "todo": [str(last.content)],
        "plan": f"answered {last.content}",
    }


def chat(checkpointer: BaseCheckpointSaver[Any]) -> Any:
    builder = StateGraph(Chat)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def run(thread: str, run_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread, "run_id": run_id}}


def item_query(saver: AgensSaver, channel: str) -> str:
    return f"SELECT {saver.cypher_subquery(channel, namespace='%s')}"


def one(text: str, params: Sequence[Any]) -> Program[Any]:
    return scalar((yield Statement(text, params)))


def items_of(engine: AgensEngine, saver: AgensSaver, thread: str) -> dict[str, Any]:
    return {
        channel: engine.run(lambda: one(item_query(saver, channel), (thread, "")))
        for channel in CHANNELS
    }


async def aitems_of(engine: AgensEngine, saver: AgensSaver, thread: str) -> dict[str, Any]:
    return {
        channel: await engine.arun(lambda: one(item_query(saver, channel), (thread, "")))
        for channel in CHANNELS
    }


def expected(values: dict[str, Any]) -> dict[str, Any]:
    """What the items of a state's channels read as."""
    messages = values.get("messages", [])
    return {
        "messages": [
            {
                "position": position,
                "id": message.id,
                "type": message.type,
                "text": message.text,
                "content": message.content,
                "message": message_to_dict(message),
            }
            for position, message in enumerate(messages)
        ],
        "todo": values.get("todo", []),
        "plan": [values["plan"]] if "plan" in values else [],
    }


def test_declared_channels_follow_the_latest_checkpoint(
    engine: AgensEngine, graph_name: str
) -> None:
    saver = AgensSaver(engine, graph_name=graph_name, queryable_channels=CHANNELS)
    graph = chat(saver)
    thread: RunnableConfig = {"configurable": {"thread_id": "t"}}

    def check(name: str = "t") -> None:
        state = graph.get_state({"configurable": {"thread_id": name}})
        assert items_of(engine, saver, name) == expected(state.values)

    graph.invoke({"messages": [HumanMessage("hi", id="h1")]}, run("t", "run-1"))
    check()
    graph.invoke({"messages": [HumanMessage("again", id="h2")]}, run("t", "run-2"))
    check()
    graph.invoke({"messages": [RemoveMessage(id="h1")]}, run("t", "run-3"))
    check()
    assert "h1" not in [m["id"] for m in items_of(engine, saver, "t")["messages"]]

    # The latest run deleted, the items are those of the newest checkpoint left, which still
    # holds the message the deleted run removed.
    saver.delete_for_runs(["run-3"])
    check()
    assert "h1" in [m["id"] for m in items_of(engine, saver, "t")["messages"]]

    older = next(s for s in graph.get_state_history(thread) if s.metadata["step"] == 1)
    graph.update_state(older.config, {"messages": [HumanMessage("forked", id="f1")]})
    check()
    assert [m["id"] for m in items_of(engine, saver, "t")["messages"]][-1] == "f1"

    saver.copy_thread("t", "copy")
    assert items_of(engine, saver, "copy") == items_of(engine, saver, "t")
    saver.delete_thread("t")
    assert items_of(engine, saver, "t") == {"messages": [], "todo": [], "plan": []}
    check("copy")
    graph.invoke({"messages": [HumanMessage("on the copy", id="c1")]}, run("copy", "run-4"))
    check("copy")


async def test_declared_channels_follow_the_awaiting_interface(
    aengine: AgensEngine, graph_name: str
) -> None:
    saver = AgensSaver(aengine, graph_name=graph_name, queryable_channels=CHANNELS)
    graph = chat(saver)
    thread: RunnableConfig = {"configurable": {"thread_id": "t"}}

    async def check() -> None:
        state = await graph.aget_state(thread)
        assert await aitems_of(aengine, saver, "t") == expected(state.values)

    await graph.ainvoke({"messages": [HumanMessage("hi", id="h1")]}, run("t", "run-1"))
    await check()
    await graph.ainvoke({"messages": [HumanMessage("again", id="h2")]}, run("t", "run-2"))
    await check()
    older = [s async for s in graph.aget_state_history(thread) if s.metadata["step"] == 1][0]
    await graph.aupdate_state(older.config, {"messages": [HumanMessage("forked", id="f1")]})
    await check()
    await graph.ainvoke({"messages": [HumanMessage("more", id="h3")]}, run("t", "run-3"))
    await check()
    await saver.adelete_for_runs(["run-3"])
    await check()
    assert (await aitems_of(aengine, saver, "t"))["messages"][-1]["id"] == "f1"
    await saver.adelete_thread("t")
    assert await aitems_of(aengine, saver, "t") == {"messages": [], "todo": [], "plan": []}


def batched(state: list[AnyMessage], writes: Sequence[Any]) -> list[AnyMessage]:
    for write in writes:
        state = add_messages(state, write)  # type: ignore[assignment]
    return state


class DeltaChat(TypedDict):
    messages: Annotated[list[AnyMessage], DeltaChannel(batched, snapshot_frequency=3)]


def delta_answer(state: DeltaChat) -> dict[str, Any]:
    return {"messages": [AIMessage(f"echo {len(state['messages'])}", id=uuid.uuid4().hex)]}


def test_deleting_runs_and_writing_their_threads_items_again_commit_together(
    engine: AgensEngine, graph_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    saver = AgensSaver(engine, graph_name=graph_name, queryable_channels=CHANNELS)
    graph = chat(saver)
    thread: RunnableConfig = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("hi", id="h1")]}, run("t", "run-1"))
    graph.invoke({"messages": [HumanMessage("again", id="h2")]}, run("t", "run-2"))
    kept = len([*saver.list(thread)])
    claim = saver._claim

    def failing(target: Any, before: Any = ()) -> Any:
        raise RuntimeError("the connection went while the items were written again")
        yield

    # Only the delete says which threads to write again, so a failure writing them takes the
    # delete back with it: a second attempt finds the run still there and finishes the job.
    monkeypatch.setattr(saver, "_claim", failing)
    with pytest.raises(RuntimeError, match="the connection went"):
        saver.delete_for_runs(["run-2"])
    assert len([*saver.list(thread)]) == kept
    monkeypatch.setattr(saver, "_claim", claim)
    saver.delete_for_runs(["run-2"])
    assert len([*saver.list(thread)]) < kept
    assert items_of(engine, saver, "t") == expected(graph.get_state(thread).values)


def test_a_delta_channel_is_rebuilt_from_its_history(engine: AgensEngine, graph_name: str) -> None:
    saver = AgensSaver(engine, graph_name=graph_name, queryable_channels={"messages": "messages"})
    builder = StateGraph(DeltaChat)
    builder.add_node("answer", delta_answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    thread: RunnableConfig = {"configurable": {"thread_id": "t"}}
    query = item_query(saver, "messages")
    snapshots = 0
    for turn in range(5):
        graph.invoke({"messages": [HumanMessage(f"turn {turn}", id=f"h{turn}")]}, thread)
        state = graph.get_state(thread)
        stored = engine.run(lambda: one(query, ("t", "")))
        assert stored == expected(state.values)["messages"]
        latest = saver.get_tuple(thread)
        assert latest is not None
        snapshots += isinstance(latest.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    assert len(stored) == 10
    assert snapshots < 5


def test_a_cypher_statement_reads_a_threads_messages_next_to_the_graph(
    engine: AgensEngine, graph_name: str
) -> None:
    graph = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    ada = Node(id="ada", type="Person", properties={"name": "Ada"})
    graph.add_graph_documents([GraphDocument(nodes=[ada], relationships=[])])
    saver = AgensSaver(engine, graph_name=graph_name, queryable_channels=CHANNELS)
    app = chat(saver)
    blocks = [{"type": "reasoning", "summary": []}, {"type": "text", "text": "Ada wrote it."}]
    app.invoke({"messages": [HumanMessage("Who is Ada?", id="m1")]}, run("s", "run-1"))
    app.invoke({"messages": [AIMessage(blocks, id="m2")]}, run("s", "run-2"))  # type: ignore[arg-type]
    app.invoke({"messages": [HumanMessage("Ada again", id="m3")]}, run("other", "run-3"))
    rows = graph.query(
        f'MATCH (p:"Person") WITH p, {saver.cypher_subquery("messages")} AS messages '
        "UNWIND messages AS m WITH p, m WHERE m.text CONTAINS p.name "
        "RETURN p.name AS who, m.id AS id, m.type AS type, m.position AS position, "
        "m.text AS text ORDER BY position",
        ("s",),
    )
    assert [(r["who"], r["type"], r["position"]) for r in rows] == [
        ("Ada", "human", 0),
        ("Ada", "ai", 1),
        ("Ada", "ai", 2),
        ("Ada", "ai", 3),
    ]
    assert (rows[0]["id"], rows[0]["text"]) == ("m1", "Who is Ada?")
    assert (rows[2]["id"], rows[2]["text"]) == ("m2", "Ada wrote it.")


class Cipher:
    """A cipher that keeps the text as it is."""

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "none", plaintext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        return ciphertext


def test_plaintext_items_are_refused_beside_an_encrypting_serializer(
    engine: AgensEngine, graph_name: str
) -> None:
    encrypted = EncryptedSerializer(Cipher())
    for serde in (encrypted, SimpleNamespace(serde=encrypted)):
        with pytest.raises(ValueError, match="allow_plaintext"):
            AgensSaver(
                engine,
                graph_name=graph_name,
                serde=serde,  # type: ignore[arg-type]
                queryable_channels=CHANNELS,
            )
    AgensSaver(engine, graph_name=graph_name, serde=encrypted)
    allowed = AgensSaver(
        engine,
        graph_name=graph_name,
        serde=encrypted,
        queryable_channels=CHANNELS,
        allow_plaintext=True,
    )
    chat(allowed).invoke({"messages": [HumanMessage("hi", id="h1")]}, run("t", "run-1"))
    assert [m["id"] for m in items_of(engine, allowed, "t")["messages"]][0] == "h1"
    with pytest.raises(ValueError, match="'reducer'"):
        AgensSaver(engine, graph_name=graph_name, queryable_channels={"x": "reducer"})
    with pytest.raises(ValueError, match="not one of the queryable channels"):
        AgensSaver(engine, graph_name=graph_name).cypher_subquery("messages")
