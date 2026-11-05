"""The rewrite against the released 0.3.1: the recorded calls replayed on the same data.

``tests/goldens/record.py`` recorded what 0.3.1 returned on 2.17 and 2.18. Each component here makes
the same calls through the 0.4 API on every leg, and each answer must match the recording, except
the calls in :data:`CHANGED`, whose answers differ on purpose (CHANGELOG.md says how).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.base import empty_checkpoint

from langchain_agensgraph import (
    AgensChatMessageHistory,
    AgensEngine,
    AgensGraph,
    AgensgraphVector,
    AgensSaver,
    AgensStore,
    GraphDocument,
    Node,
    Relationship,
)
from tests import datagen
from tests.goldens import golden

pytestmark = pytest.mark.server

QUERY = "graph vector search filter"
Replay = list[tuple[str, dict[str, Any], Any]]


def _documents(result: Any) -> Any:
    return [pair[0] for pair in result]


def _ids(result: Any) -> Any:
    return sorted(doc["id"] for doc in result)


def _unanswered(result: Any) -> Any:
    return None


def _keys(result: Any) -> Any:
    return sorted(item["key"] for item in result)


def _count(result: Any) -> Any:
    return len(result)


def _without_source_text(schema: Any) -> Any:
    props = schema["node_props"]
    kept = [p for p in props.get("Document", []) if p["property"] != "text"]
    return {**schema, "node_props": {**props, "Document": kept}}


CHANGED: dict[tuple[str, ...], tuple[str, Callable[[Any], Any]]] = {
    ("graph", "structured_schema"): (
        "a source document's vertex carries its text",
        _without_source_text,
    ),
    ("store", "search_prefix"): (
        "a search without a query reads newest first, as InMemoryStore does; the items match",
        _keys,
    ),
    ("store", "search_filter"): (
        "a search without a query reads newest first, as InMemoryStore does; the items match",
        _keys,
    ),
    ("store", "search_prefix_page"): ("the page is taken newest first", _count),
    ("store", "search_page"): (
        "an empty prefix searches every namespace, as InMemoryStore does, where 0.3.1 refused it",
        _unanswered,
    ),
    ("vectorstore", "similarity_search_with_score"): (
        "a score is the raw distance; the documents are the same",
        _documents,
    ),
    ("vectorstore", "hybrid_search"): (
        "a tie in the fused score is broken by id; the hits are the same",
        _ids,
    ),
    ("vectorstore", "similarity_search_filter", '{"note": {"$exists": true}}'): (
        "$exists is answered, where 0.3.1 refused it",
        _unanswered,
    ),
    ("vectorstore", "similarity_search_filter", '{"note": {"$exists": false}}'): (
        "$exists is answered, where 0.3.1 refused it",
        _unanswered,
    ),
}
"""The calls whose answers differ from 0.3.1 on purpose: the reason, and the part of the answer
that is still compared. A key names a component and a step, and for one call of a step its
filter as JSON."""


def _changed(component: str, step: str, args: dict[str, Any]) -> Callable[[Any], Any] | None:
    where = json.dumps(args.get("filter"), sort_keys=True) if "filter" in args else None
    found = CHANGED.get((component, step, where or "")) or CHANGED.get((component, step))
    return found[1] if found else None


def attempt(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - a refusal is an answer
        return {"raises": type(exc).__name__}


def vectorstore(engine: AgensEngine, graph_name: str) -> Replay:
    spec = golden.DATASETS["vectorstore"]
    chunks = datagen.chunks(spec["chunks"], spec["dims"], seed=spec["seed"])
    embedding = golden.FixedEmbeddings({c.text: c.embedding for c in chunks}, spec["dims"])
    texts = [c.text for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]
    ids = [c.id for c in chunks]
    out: Replay = []
    store = AgensgraphVector(
        embedding, engine=engine, graph_name=graph_name, create_graph=True, search_mode="exact"
    )
    out.append(("add_texts", {"rows": len(ids)}, store.add_texts(texts, metadatas, ids=ids)))
    out.append(("similarity_search", {"query": QUERY, "k": 5}, store.similarity_search(QUERY, k=5)))
    out.append(
        (
            "similarity_search_with_score",
            {"query": QUERY, "k": 5},
            store.similarity_search_with_score(QUERY, k=5),
        )
    )
    out.append(
        (
            "similarity_search_by_vector",
            {"chunk": 0, "k": 5},
            store.similarity_search_by_vector(embedding.vector(texts[0]), k=5),
        )
    )
    for where in (
        {"s10": 3},
        {"lang": {"$in": ["ko", "ur"]}},
        {"year": {"$gte": 2020}},
        {"price": {"$lt": 10}},
        {"flag": True},
        {"note": {"$exists": True}},
        {"note": {"$exists": False}},
        {"lang": {"$ne": "en"}},
        {"lang": {"$nin": ["en", "ko"]}},
        {"price": {"$between": [10, 20]}},
        {"note": {"$like": "n1%"}},
        {"$and": [{"s10": 3}, {"year": {"$lt": 2010}}]},
        {"$or": [{"s1": 7}, {"s1": 8}]},
    ):
        out.append(
            (
                "similarity_search_filter",
                {"query": QUERY, "k": 5, "filter": where},
                attempt(lambda where=where: store.similarity_search(QUERY, k=5, filter=where)),
            )
        )
    out.append(
        (
            "max_marginal_relevance_search",
            {"query": QUERY, "k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            store.max_marginal_relevance_search(QUERY, k=5, fetch_k=20, lambda_mult=0.5),
        )
    )
    out.append(
        (
            "as_retriever",
            {"query": QUERY, "k": 3},
            store.as_retriever(search_kwargs={"k": 3}).invoke(QUERY),
        )
    )
    out.append(
        ("get_by_ids", {"ids": [*ids[:3], "absent"]}, store.get_by_ids([*ids[:3], "absent"]))
    )
    out.append(("delete", {"ids": ids[:10]}, store.delete(ids[:10])))
    out.append(
        (
            "similarity_search_after_delete",
            {"query": QUERY, "k": 5},
            store.similarity_search(QUERY, k=5),
        )
    )
    hybrid = AgensgraphVector.from_texts(
        texts,
        embedding,
        metadatas=metadatas,
        ids=ids,
        engine=engine,
        graph_name=graph_name,
        label="Hybrid",
        search_type="hybrid",
        search_mode="exact",
    )
    out.append(
        (
            "hybrid_search",
            {"query": datagen.RARE_WORD, "k": 5},
            hybrid.similarity_search(datagen.RARE_WORD, k=5),
        )
    )
    return out


def graph(engine: AgensEngine, graph_name: str) -> Replay:
    spec = golden.DATASETS["graph"]
    docs = datagen.graph_documents(
        spec["documents"], entities=spec["entities"], per_doc=spec["per_doc"], seed=spec["seed"]
    )
    graph_docs = []
    for doc in docs:
        nodes = {
            e.id: Node(id=e.id, type=e.type, properties=dict(e.properties)) for e in doc.entities
        }
        links = [
            Relationship(
                source=nodes[link.source],
                target=nodes[link.target],
                type=link.type,
                properties=dict(link.properties),
            )
            for link in doc.links
        ]
        graph_docs.append(
            GraphDocument(
                nodes=list(nodes.values()), relationships=links, source=Document(doc.source)
            )
        )
    counts = {
        "vertices": "MATCH (n) RETURN label(n) AS label, count(*) AS n ORDER BY label",
        "edges": "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type",
        "neighbours": (
            "MATCH (a {id: 'e1'})-[r]->(b) RETURN type(r) AS type, b.id AS b, r.weight AS weight "
            "ORDER BY type, b"
        ),
    }
    out: Replay = []
    store = AgensGraph(engine, graph_name=graph_name, create_graph=True)
    store.add_graph_documents(graph_docs, include_source=True)
    out.append(("add_graph_documents", {"include_source": True}, None))
    for name, text in counts.items():
        out.append((f"query_{name}", {"query": text}, store.query(text)))
    store.refresh_schema(force=True)
    out.append(("structured_schema", {}, _schema(store.get_structured_schema)))
    store.add_graph_documents(graph_docs, include_source=True)
    out.append(
        (
            "query_vertices_after_readd",
            {"query": counts["vertices"]},
            store.query(counts["vertices"]),
        )
    )
    out.append(
        ("query_edges_after_readd", {"query": counts["edges"]}, store.query(counts["edges"]))
    )
    return out


def _schema(schema: dict[str, Any]) -> dict[str, Any]:
    def ordered(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ordered(item) for key, item in value.items()}
        if isinstance(value, list):
            items = [ordered(item) for item in value]
            return sorted(items, key=lambda item: repr(golden.canonical(item)))
        return value

    return ordered({key: value for key, value in schema.items() if key != "metadata"})


def chat(engine: AgensEngine, graph_name: str) -> Replay:
    spec = golden.DATASETS["chat"]
    sessions = datagen.chat_sessions(spec["sessions"], spec["messages"], seed=spec["seed"])
    out: Replay = []
    for session_id, messages in sessions.items():
        history = AgensChatMessageHistory(session_id, engine, graph_name=graph_name)
        history.add_messages(messages[:4])
        history.add_messages(messages[4:])
        out.append(("messages", {"session": session_id}, history.messages))
    first = next(iter(sessions))
    windowed = AgensChatMessageHistory(first, engine, graph_name=graph_name, window=3)
    out.append(("window", {"session": first, "window": 3}, windowed.messages))
    AgensChatMessageHistory(first, engine, graph_name=graph_name).clear()
    for session_id in sessions:
        out.append(
            (
                "messages_after_clear",
                {"session": session_id, "cleared": first},
                AgensChatMessageHistory(session_id, engine, graph_name=graph_name).messages,
            )
        )
    return out


CHECKPOINT_IDS = [f"1f0b0000-0000-6000-8000-00000000000{i}" for i in range(1, 5)]


def _checkpoint(number: int, values: dict[str, Any], versions: dict[str, str]) -> Any:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = CHECKPOINT_IDS[number]
    checkpoint["ts"] = datetime(2026, 1, 1, 0, 0, number, tzinfo=timezone.utc).isoformat()
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = versions
    return checkpoint


def _tuple(found: Any) -> Any:
    if found is None:
        return None
    return {
        "config": found.config,
        "checkpoint": {k: v for k, v in found.checkpoint.items() if k != "ts"},
        "metadata": found.metadata,
        "parent_config": found.parent_config,
        "pending_writes": [list(write) for write in found.pending_writes or []],
    }


def checkpointer(engine: AgensEngine, graph_name: str) -> Replay:
    saver = AgensSaver(engine, graph_name=graph_name)
    out: Replay = []
    config: Any = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    for step in range(3):
        values = {"count": step + 1, "messages": [f"m{i}" for i in range(step + 1)]}
        versions = {"count": f"{step + 1:032}.0", "messages": f"{step + 1:032}.0"}
        metadata: Any = {"source": "loop", "step": step, "run_id": f"r{step // 2}"}
        config = saver.put(config, _checkpoint(step, values, versions), metadata, versions)
        out.append(("put", {"thread": "t1", "step": step}, config))
    saver.put_writes(config, [("messages", "m-pending"), ("count", 9)], "task-1", "path-1")
    thread: Any = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    out.append(("get_tuple_latest", {"thread": "t1"}, _tuple(saver.get_tuple(thread))))
    first: Any = {"configurable": {**thread["configurable"], "checkpoint_id": CHECKPOINT_IDS[0]}}
    out.append(("get_tuple_by_id", {"thread": "t1", "id": 0}, _tuple(saver.get_tuple(first))))
    out.append(("list", {"thread": "t1"}, [_tuple(t) for t in saver.list(thread)]))
    out.append(
        (
            "list_limit",
            {"thread": "t1", "limit": 2},
            [_tuple(t) for t in saver.list(thread, limit=2)],
        )
    )
    before: Any = {"configurable": {**thread["configurable"], "checkpoint_id": CHECKPOINT_IDS[2]}}
    out.append(
        (
            "list_before",
            {"thread": "t1", "before": 2},
            [_tuple(t) for t in saver.list(thread, before=before)],
        )
    )
    out.append(
        (
            "list_filter",
            {"thread": "t1", "filter": {"source": "loop", "step": 1}},
            [_tuple(t) for t in saver.list(thread, filter={"source": "loop", "step": 1})],
        )
    )
    child: Any = {"configurable": {"thread_id": "t2", "checkpoint_ns": "child"}}
    versions = {"count": f"{1:032}.0"}
    saver.put(
        child, _checkpoint(3, {"count": 1}, versions), {"source": "input", "step": -1}, versions
    )
    out.append(
        ("list_namespace", {"thread": "t2", "ns": "child"}, [_tuple(t) for t in saver.list(child)])
    )
    saver.delete_thread("t1")
    out.append(
        ("list_after_delete_thread", {"thread": "t1"}, [_tuple(t) for t in saver.list(thread)])
    )
    out.append(("list_other_thread_kept", {"thread": "t2"}, [_tuple(t) for t in saver.list(child)]))
    return out


def _item(item: Any) -> Any:
    if item is None:
        return None
    return {"namespace": list(item.namespace), "key": item.key, "value": item.value}


def store(engine: AgensEngine, graph_name: str) -> Replay:
    spec = golden.DATASETS["store"]
    items = datagen.store_items(
        spec["items"], namespaces=spec["namespaces"], depth=spec["depth"], seed=spec["seed"]
    )
    spaces = sorted({namespace for namespace, _, _ in items})
    out: Replay = []
    kept = AgensStore(engine, graph_name=graph_name)
    for namespace, key, value in items:
        kept.put(namespace, key, value)
    for namespace, key, _ in items[:3]:
        out.append(("get", {"namespace": namespace, "key": key}, _item(kept.get(namespace, key))))
    out.append(
        (
            "get_absent",
            {"namespace": spaces[0], "key": "absent"},
            _item(kept.get(spaces[0], "absent")),
        )
    )
    prefix = spaces[1][:1]
    out.append(
        (
            "search_prefix",
            {"prefix": prefix, "limit": 50},
            [_item(i) for i in kept.search(prefix, limit=50)],
        )
    )
    out.append(
        (
            "search_filter",
            {"prefix": prefix, "filter": {"tags": ["t1"]}, "limit": 50},
            [_item(i) for i in kept.search(prefix, filter={"tags": ["t1"]}, limit=50)],
        )
    )
    out.append(
        (
            "search_page",
            {"prefix": [], "limit": 5, "offset": 5},
            attempt(lambda: [_item(i) for i in kept.search((), limit=5, offset=5)]),
        )
    )
    out.append(
        (
            "search_prefix_page",
            {"prefix": prefix, "limit": 3, "offset": 2},
            [_item(i) for i in kept.search(prefix, limit=3, offset=2)],
        )
    )
    out.append(("list_namespaces", {}, kept.list_namespaces()))
    out.append(("list_namespaces_prefix", {"prefix": prefix}, kept.list_namespaces(prefix=prefix)))
    suffix = spaces[2][-1:]
    out.append(("list_namespaces_suffix", {"suffix": suffix}, kept.list_namespaces(suffix=suffix)))
    out.append(("list_namespaces_depth", {"max_depth": 2}, kept.list_namespaces(max_depth=2)))
    out.append(
        ("list_namespaces_page", {"limit": 2, "offset": 1}, kept.list_namespaces(limit=2, offset=1))
    )
    namespace, key, _ = items[0]
    kept.delete(namespace, key)
    out.append(
        ("get_after_delete", {"namespace": namespace, "key": key}, _item(kept.get(namespace, key)))
    )
    vectors = golden.DATASETS["vectorstore"]
    semantic = AgensStore(
        engine,
        graph_name=graph_name,
        index={
            "dims": vectors["dims"],
            "embed": golden.FixedEmbeddings({}, vectors["dims"]),
            "fields": ["text"],
        },
    )
    for namespace, key, value in items:
        semantic.put(namespace, key, value)
    hits = semantic.search(spaces[0][:1], query=QUERY, limit=3)
    out.append(
        (
            "search_query",
            {"prefix": spaces[0][:1], "query": QUERY, "limit": 3},
            [{**_item(hit), "score": hit.score} for hit in hits],
        )
    )
    return out


COMPONENTS: dict[str, Callable[[AgensEngine, str], Replay]] = {
    "vectorstore": vectorstore,
    "graph": graph,
    "chat": chat,
    "checkpointer": checkpointer,
    "store": store,
}


@pytest.mark.parametrize("component", list(COMPONENTS))
def test_the_rewrite_answers_as_the_release_did(
    engine: AgensEngine, graph_name: str, dsn: str, component: str
) -> None:
    replayed = [
        json.loads(golden.line(component, step, args, result))
        for step, args, result in COMPONENTS[component](engine, graph_name)
    ]
    recorded = golden.recorded(component)
    assert [(e["step"], e["args"]) for e in replayed] == [(e["step"], e["args"]) for e in recorded]
    differ = []
    for answer, entry in zip(replayed, recorded, strict=True):
        compared = _changed(component, entry["step"], entry["args"]) or (lambda result: result)
        if not golden.matches(compared(entry["result"]), compared(answer["result"])):
            differ.append(
                f"{entry['step']} {json.dumps(entry['args'])}:\n"
                f"  0.3.1: {entry['result']}\n  0.4:   {answer['result']}"
            )
    assert not differ, "\n".join(differ)
