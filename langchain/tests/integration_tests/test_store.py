"""The store against a server, compared read for read with LangGraph's ``InMemoryStore``."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langgraph.store.base import GetOp, InvalidNamespaceError, PutOp, SearchItem
from langgraph.store.memory import InMemoryStore

from langchain_agensgraph._core.engine import AgensEngine
from langchain_agensgraph.store import AgensStore, AsyncAgensStore
from tests import datagen
from tests.goldens.golden import FixedEmbeddings

pytestmark = pytest.mark.server

DIMS = 6

EXTRA: list[tuple[tuple[str, ...], str, dict[str, Any]]] = [
    (("users", "alice", "prefs"), "theme", {"text": "dark mode", "n": 1, "meta": {"lang": "en"}}),
    (("users", "alice"), "profile", {"text": "alice likes graphs", "n": 7, "tags": ["t1"]}),
    (("users", "bob"), "profile", {"text": "bob likes tables", "n": 2.5, "flag": None}),
    (("users", "bob", "prefs"), "theme", {"text": "light mode", "n": "3", "tags": ["t1", "t2"]}),
    (("user_1",), "k", {"text": "underscore namespace", "n": 0}),
    (("docs",), "readme", {"text": "the readme", "n": -1, "meta": {"lang": "ko", "v": 2}}),
]
"""Every item has ``n``: ``InMemoryStore`` raises on a numeric comparison with a missing key,
where this store matches nothing."""

FILTERS: list[dict[str, Any] | None] = [
    None,
    {"n": 1},
    {"n": {"$gt": 2}},
    {"n": {"$lte": 2.5}},
    {"n": {"$ne": 7}},
    {"tags": ["t1"]},
    {"meta": {"lang": "en"}},
    {"flag": None},
    {"n": {"$eq": "3"}},
]


def items() -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    return datagen.store_items(40, namespaces=4, depth=3, seed=3) + EXTRA


def embedding() -> FixedEmbeddings:
    return FixedEmbeddings({}, DIMS)


def plain(found: Any) -> Any:
    """A read with its timestamps left out, which two stores never share."""
    if found is None:
        return None
    if isinstance(found, list):
        return [plain(item) for item in found]
    out = (found.namespace, found.key, found.value)
    if isinstance(found, SearchItem) and found.score is not None:
        return (*out, round(found.score, 5))
    return out


def loaded(store: Any) -> Any:
    for namespace, key, value in items():
        store.put(namespace, key, value)
    return store


def test_every_read_matches_the_in_memory_store(engine: AgensEngine, graph_name: str) -> None:
    ours = loaded(AgensStore(engine, graph_name=graph_name))
    theirs = loaded(InMemoryStore())
    # A key absent from a namespace that exists: InMemoryStore remembers any namespace it is
    # asked about, and would list it afterwards.
    for namespace, key, _ in [*items(), (("users", "bob"), "absent", {})]:
        assert plain(ours.get(namespace, key)) == plain(theirs.get(namespace, key))
    for prefix in [(), ("users",), ("users", "alice"), ("user_1",), ("ns1-0",), ("nobody",)]:
        for where in FILTERS:
            got = sorted(plain(ours.search(prefix, filter=where, limit=100)), key=repr)
            want = sorted(plain(theirs.search(prefix, filter=where, limit=100)), key=repr)
            assert got == want, (prefix, where)
    for kwargs in [
        {},
        {"prefix": ("users",)},
        {"suffix": ("prefs",)},
        {"prefix": ("users", "*", "prefs")},
        {"max_depth": 1},
        {"max_depth": 2, "prefix": ("users",)},
    ]:
        got = sorted(ours.list_namespaces(limit=100, **kwargs))
        assert got == sorted(theirs.list_namespaces(limit=100, **kwargs)), kwargs


def test_paging_reads_newest_first_and_namespaces_in_byte_order(
    engine: AgensEngine, graph_name: str
) -> None:
    store = AgensStore(engine, graph_name=graph_name)
    for key in "abcdef":
        store.put(("p",), key, {"k": key})
    assert [i.key for i in store.search(("p",), limit=3, offset=1)] == ["e", "d", "c"]
    store.put(("a-x",), "k", {})
    store.put(("a", "b"), "k", {})
    assert store.list_namespaces(prefix=("a",)) == [("a", "b")]
    assert store.list_namespaces(limit=2) == [("a-x",), ("a", "b")]


def test_a_batch_reads_before_it_writes_and_keeps_the_last_write(
    engine: AgensEngine, graph_name: str
) -> None:
    store = AgensStore(engine, graph_name=graph_name)
    store.put(("b",), "k", {"v": 0})
    first = store.get(("b",), "k")
    assert first is not None
    results = store.batch(
        [
            PutOp(("b",), "k", {"v": 1}),
            GetOp(("b",), "k"),
            PutOp(("b",), "k", {"v": 2}),
            PutOp(("b",), "new", {"v": 3}),
            GetOp(("b",), "new"),
        ]
    )
    assert results[1] is not None and results[1].value == {"v": 0} and results[4] is None
    after = store.get(("b",), "k")
    assert after is not None and after.value == {"v": 2}
    assert after.created_at == first.created_at and after.updated_at > first.updated_at
    store.delete(("b",), "k")
    assert store.get(("b",), "k") is None


def test_a_first_call_in_a_callers_transaction_finds_the_tables_setup_made(
    engine: AgensEngine, graph_name: str
) -> None:
    with engine.transaction():
        store = AgensStore(engine, graph_name=graph_name)
        store.put(("docs",), "k", {"v": 1})
        assert plain(store.get(("docs",), "k")) == (("docs",), "k", {"v": 1})
    assert plain(AgensStore(engine, graph_name=graph_name).get(("docs",), "k")) is not None


def test_a_namespace_label_holding_a_period_is_refused(
    engine: AgensEngine, graph_name: str
) -> None:
    store = AgensStore(engine, graph_name=graph_name)
    with pytest.raises(InvalidNamespaceError):
        store.batch([PutOp(("a.b",), "k", {})])


def test_a_semantic_search_matches_the_in_memory_store(
    engine: AgensEngine, graph_name: str
) -> None:
    index = {"dims": DIMS, "embed": embedding(), "fields": ["text"]}
    ours = loaded(AgensStore(engine, graph_name=graph_name, index=index))  # type: ignore[arg-type]
    theirs = loaded(InMemoryStore(index=index))  # type: ignore[arg-type]
    for query, prefix, where in [
        ("graphs", (), None),
        ("dark mode", ("users",), None),
        ("tables", ("users",), {"n": {"$gt": 1}}),
    ]:
        got = plain(ours.search(prefix, query=query, filter=where, limit=5))
        assert got == plain(theirs.search(prefix, query=query, filter=where, limit=5)), query
    ours.put(("nolook",), "k", {"text": "skipped"}, index=False)
    theirs.put(("nolook",), "k", {"text": "skipped"}, index=False)
    assert plain(ours.search(("nolook",), query="skipped")) == plain(
        theirs.search(("nolook",), query="skipped")
    )


def test_an_item_expires_and_reading_it_extends_its_life(
    engine: AgensEngine, graph_name: str
) -> None:
    store = AgensStore(
        engine,
        graph_name=graph_name,
        ttl={"default_ttl": 0.005, "refresh_on_read": True, "omit_expired": True},
    )
    store.put(("t",), "short", {"v": 1})
    store.put(("t",), "kept", {"v": 2}, ttl=None)
    time.sleep(0.15)
    assert store.get(("t",), "kept") is not None
    store.put(("t",), "refreshed", {"v": 3}, ttl=0.01)
    time.sleep(0.4)
    assert store.get(("t",), "refreshed") is not None
    time.sleep(0.4)
    assert store.get(("t",), "refreshed") is not None
    assert store.get(("t",), "short") is None
    assert store.sweep_ttl() == 1
    assert {i.key for i in store.search(("t",))} == {"refreshed", "kept"}


def test_the_sweeper_thread_removes_what_has_expired(engine: AgensEngine, graph_name: str) -> None:
    store = AgensStore(engine, graph_name=graph_name, ttl={"default_ttl": 0.001})
    store.put(("s",), "gone", {"v": 1})
    store.start_ttl_sweeper(sweep_interval_minutes=0.002)
    deadline = time.monotonic() + 5
    while store.search(("s",)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert store.search(("s",)) == []
    assert store.stop_ttl_sweeper(timeout=5)


async def test_the_awaiting_store_batches_concurrent_calls(
    engine: AgensEngine, graph_name: str
) -> None:
    store = AsyncAgensStore(engine, graph_name=graph_name)
    await asyncio.gather(*(store.aput(("a",), f"k{i}", {"i": i}) for i in range(20)))
    found = await asyncio.gather(*(store.aget(("a",), f"k{i}") for i in range(20)))
    assert [f.value["i"] for f in found if f is not None] == list(range(20))
    assert len(await store.asearch(("a",), filter={"i": {"$gte": 10}}, limit=50)) == 10
    assert await store.alist_namespaces() == [("a",)]
    await store.aclose()
    await engine.aclose()


async def test_the_blocking_store_awaits_natively_too(engine: AgensEngine, graph_name: str) -> None:
    store = AgensStore(engine, graph_name=graph_name)
    await store.aput(("x",), "k", {"v": 1})
    found = await store.aget(("x",), "k")
    assert found is not None and found.value == {"v": 1}
    await engine.aclose()
