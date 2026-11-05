"""Record what the released 0.3.1 returns for scripted calls, on each server it supports.

Run through ``make goldens``, which installs 0.3.1 and agensgraph-python 2.0.0 from PyPI into a
throwaway environment, outside this project::

    uv run --no-project --with langchain-agensgraph==0.3.1 --with agensgraph-python==2.0.0 \
        python -P tests/goldens/record.py 2.17=postgresql://... 2.18.6=postgresql://...

Each conninfo reaches a superuser. The recorder works in a database of its own, ``lcag_goldens``,
dropped afterwards. 0.3.1 runs on 2.17 and later only. It writes ``0.3.1.jsonl`` when every leg
gave the same answers, and otherwise names the calls that differ and writes nothing.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import datagen  # noqa: E402
import golden  # noqa: E402
import psycopg  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langgraph.checkpoint.base import empty_checkpoint  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.conninfo import conninfo_to_dict  # noqa: E402

import langchain_agensgraph  # noqa: E402
from langchain_agensgraph import (  # noqa: E402
    AgensChatMessageHistory,
    AgensGraph,
    AgensgraphVector,
    GraphDocument,
    Node,
    Relationship,
)
from langchain_agensgraph.checkpoint import AgensSaver  # noqa: E402
from langchain_agensgraph.store import AgensStore  # noqa: E402
from langchain_agensgraph.vectorstores.agensgraph_vector import SearchType  # noqa: E402

DB = "lcag_goldens"
QUERY = "graph vector search filter"

Record = Callable[[str, dict[str, Any], Any], None]


def _conf(conninfo: str) -> dict[str, Any]:
    parts = conninfo_to_dict(conninfo)
    return {
        "host": parts.get("host", "localhost"),
        "port": int(parts.get("port", 5432)),
        "user": parts.get("user"),
        "password": parts.get("password"),
        "dbname": DB,
    }


def attempt(call: Callable[[], Any]) -> Any:
    """What a call returns, or the name of the exception it raises."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - a refusal is a recorded outcome
        return {"raises": type(exc).__name__}


@contextmanager
def _graph(conf: dict[str, Any], name: str) -> Iterator[AgensGraph]:
    graph = AgensGraph(name, conf, create=True)
    try:
        yield graph
    finally:
        graph.close()


def vectorstore(conf: dict[str, Any], record: Record) -> None:
    spec = golden.DATASETS["vectorstore"]
    chunks = datagen.chunks(spec["chunks"], spec["dims"], seed=spec["seed"])
    embedding = golden.FixedEmbeddings({c.text: c.embedding for c in chunks}, spec["dims"])
    texts = [c.text for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]
    ids = [c.id for c in chunks]
    # ef_search above the row count makes the index search exact, so every run agrees.
    options = {"hnsw.ef_search": 1000}
    with _graph(conf, "golden_vectors") as graph:
        store = AgensgraphVector(embedding, graph=graph, search_options=options)
        record("add_texts", {"rows": len(ids)}, store.add_texts(texts, metadatas, ids=ids))
        record("similarity_search", {"query": QUERY, "k": 5}, store.similarity_search(QUERY, k=5))
        record(
            "similarity_search_with_score",
            {"query": QUERY, "k": 5},
            store.similarity_search_with_score(QUERY, k=5),
        )
        record(
            "similarity_search_by_vector",
            {"chunk": 0, "k": 5},
            store.similarity_search_by_vector(embedding.vector(texts[0]), k=5),
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
            record(
                "similarity_search_filter",
                {"query": QUERY, "k": 5, "filter": where},
                attempt(lambda where=where: store.similarity_search(QUERY, k=5, filter=where)),
            )
        record(
            "max_marginal_relevance_search",
            {"query": QUERY, "k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            store.max_marginal_relevance_search(QUERY, k=5, fetch_k=20, lambda_mult=0.5),
        )
        record(
            "as_retriever",
            {"query": QUERY, "k": 3},
            store.as_retriever(search_kwargs={"k": 3}).invoke(QUERY),
        )
        record("get_by_ids", {"ids": [*ids[:3], "absent"]}, store.get_by_ids([*ids[:3], "absent"]))
        record("delete", {"ids": ids[:10]}, store.delete(ids[:10]))
        record(
            "similarity_search_after_delete",
            {"query": QUERY, "k": 5},
            store.similarity_search(QUERY, k=5),
        )
        store.close()
    with _graph(conf, "golden_hybrid") as graph:
        hybrid = AgensgraphVector.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            graph=graph,
            search_type=SearchType.HYBRID,
            search_options=options,
        )
        record(
            "hybrid_search",
            {"query": datagen.RARE_WORD, "k": 5},
            hybrid.similarity_search(datagen.RARE_WORD, k=5),
        )
        hybrid.close()


def graph_store(conf: dict[str, Any], record: Record) -> None:
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
    with _graph(conf, "golden_graph") as graph:
        graph.add_graph_documents(graph_docs, include_source=True)
        record("add_graph_documents", {"include_source": True}, None)
        for name, text in counts.items():
            record(f"query_{name}", {"query": text}, graph.query(text))
        graph.refresh_schema()
        record("structured_schema", {}, _schema(graph.get_structured_schema))
        graph.add_graph_documents(graph_docs, include_source=True)
        record(
            "query_vertices_after_readd",
            {"query": counts["vertices"]},
            graph.query(counts["vertices"]),
        )
        record("query_edges_after_readd", {"query": counts["edges"]}, graph.query(counts["edges"]))


def _schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema with every list in a fixed order: the server returns labels in catalog order."""

    def ordered(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ordered(item) for key, item in value.items()}
        if isinstance(value, list):
            items = [ordered(item) for item in value]
            return sorted(items, key=lambda item: repr(golden.canonical(item)))
        return value

    return ordered({key: value for key, value in schema.items() if key != "metadata"})


def chat(conf: dict[str, Any], record: Record) -> None:
    spec = golden.DATASETS["chat"]
    sessions = datagen.chat_sessions(spec["sessions"], spec["messages"], seed=spec["seed"])
    with _graph(conf, "golden_chat") as graph:
        for session_id, messages in sessions.items():
            history = AgensChatMessageHistory(session_id, graph=graph)
            history.add_messages(messages[:4])
            history.add_messages(messages[4:])
            record("messages", {"session": session_id}, history.messages)
        first = next(iter(sessions))
        windowed = AgensChatMessageHistory(first, graph=graph, window=3)
        record("window", {"session": first, "window": 3}, windowed.messages)
        AgensChatMessageHistory(first, graph=graph).clear()
        for session_id in sessions:
            record(
                "messages_after_clear",
                {"session": session_id, "cleared": first},
                AgensChatMessageHistory(session_id, graph=graph).messages,
            )


CHECKPOINT_IDS = [f"1f0b0000-0000-6000-8000-00000000000{i}" for i in range(1, 5)]


def _checkpoint(number: int, values: dict[str, Any], versions: dict[str, str]) -> dict[str, Any]:
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


def checkpointer(conf: dict[str, Any], record: Record) -> None:
    with _graph(conf, "golden_checkpoints") as graph:
        saver = AgensSaver(graph=graph)
        config: dict[str, Any] = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        for step in range(3):
            values = {"count": step + 1, "messages": [f"m{i}" for i in range(step + 1)]}
            versions = {"count": f"{step + 1:032}.0", "messages": f"{step + 1:032}.0"}
            metadata = {"source": "loop", "step": step, "run_id": f"r{step // 2}"}
            config = saver.put(config, _checkpoint(step, values, versions), metadata, versions)
            record("put", {"thread": "t1", "step": step}, config)
        saver.put_writes(config, [("messages", "m-pending"), ("count", 9)], "task-1", "path-1")
        thread = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        record("get_tuple_latest", {"thread": "t1"}, _tuple(saver.get_tuple(thread)))
        first = {"configurable": {**thread["configurable"], "checkpoint_id": CHECKPOINT_IDS[0]}}
        record("get_tuple_by_id", {"thread": "t1", "id": 0}, _tuple(saver.get_tuple(first)))
        record("list", {"thread": "t1"}, [_tuple(t) for t in saver.list(thread)])
        record(
            "list_limit",
            {"thread": "t1", "limit": 2},
            [_tuple(t) for t in saver.list(thread, limit=2)],
        )
        before = {"configurable": {**thread["configurable"], "checkpoint_id": CHECKPOINT_IDS[2]}}
        record(
            "list_before",
            {"thread": "t1", "before": 2},
            [_tuple(t) for t in saver.list(thread, before=before)],
        )
        record(
            "list_filter",
            {"thread": "t1", "filter": {"source": "loop", "step": 1}},
            [_tuple(t) for t in saver.list(thread, filter={"source": "loop", "step": 1})],
        )
        child: dict[str, Any] = {"configurable": {"thread_id": "t2", "checkpoint_ns": "child"}}
        versions = {"count": f"{1:032}.0"}
        saver.put(
            child, _checkpoint(3, {"count": 1}, versions), {"source": "input", "step": -1}, versions
        )
        record(
            "list_namespace",
            {"thread": "t2", "ns": "child"},
            [_tuple(t) for t in saver.list(child)],
        )
        saver.delete_thread("t1")
        record(
            "list_after_delete_thread", {"thread": "t1"}, [_tuple(t) for t in saver.list(thread)]
        )
        record("list_other_thread_kept", {"thread": "t2"}, [_tuple(t) for t in saver.list(child)])


def _item(item: Any) -> Any:
    if item is None:
        return None
    return {"namespace": list(item.namespace), "key": item.key, "value": item.value}


def store(conf: dict[str, Any], record: Record) -> None:
    spec = golden.DATASETS["store"]
    items = datagen.store_items(
        spec["items"], namespaces=spec["namespaces"], depth=spec["depth"], seed=spec["seed"]
    )
    spaces = sorted({namespace for namespace, _, _ in items})
    with _graph(conf, "golden_store") as graph:
        kept = AgensStore(graph=graph)
        for namespace, key, value in items:
            kept.put(namespace, key, value)
        for namespace, key, _ in items[:3]:
            record("get", {"namespace": namespace, "key": key}, _item(kept.get(namespace, key)))
        record(
            "get_absent",
            {"namespace": spaces[0], "key": "absent"},
            _item(kept.get(spaces[0], "absent")),
        )
        prefix = spaces[1][:1]
        record(
            "search_prefix",
            {"prefix": prefix, "limit": 50},
            [_item(i) for i in kept.search(prefix, limit=50)],
        )
        record(
            "search_filter",
            {"prefix": prefix, "filter": {"tags": ["t1"]}, "limit": 50},
            [_item(i) for i in kept.search(prefix, filter={"tags": ["t1"]}, limit=50)],
        )
        record(
            "search_page",
            {"prefix": [], "limit": 5, "offset": 5},
            attempt(lambda: [_item(i) for i in kept.search((), limit=5, offset=5)]),
        )
        record(
            "search_prefix_page",
            {"prefix": prefix, "limit": 3, "offset": 2},
            [_item(i) for i in kept.search(prefix, limit=3, offset=2)],
        )
        record("list_namespaces", {}, kept.list_namespaces())
        record("list_namespaces_prefix", {"prefix": prefix}, kept.list_namespaces(prefix=prefix))
        suffix = spaces[2][-1:]
        record("list_namespaces_suffix", {"suffix": suffix}, kept.list_namespaces(suffix=suffix))
        record("list_namespaces_depth", {"max_depth": 2}, kept.list_namespaces(max_depth=2))
        record(
            "list_namespaces_page",
            {"limit": 2, "offset": 1},
            kept.list_namespaces(limit=2, offset=1),
        )
        namespace, key, _ = items[0]
        kept.delete(namespace, key)
        record(
            "get_after_delete",
            {"namespace": namespace, "key": key},
            _item(kept.get(namespace, key)),
        )
    vectors = golden.DATASETS["vectorstore"]
    embedding = golden.FixedEmbeddings({}, vectors["dims"])
    with _graph(conf, "golden_store_vectors") as graph:
        semantic = AgensStore(
            graph=graph, index={"dims": vectors["dims"], "embed": embedding, "fields": ["text"]}
        )
        for namespace, key, value in items:
            semantic.put(namespace, key, value)
        hits = semantic.search(spaces[0][:1], query=QUERY, limit=3)
        record(
            "search_query",
            {"prefix": spaces[0][:1], "query": QUERY, "limit": 3},
            [{**_item(hit), "score": hit.score} for hit in hits],
        )


COMPONENTS: dict[str, Callable[[dict[str, Any], Record], None]] = {
    "vectorstore": vectorstore,
    "graph": graph_store,
    "chat": chat,
    "checkpointer": checkpointer,
    "store": store,
}


def record_leg(conninfo: str) -> list[str]:
    """Every scripted call made on one server, as recorded lines."""
    with psycopg.connect(conninfo, dbname="postgres", autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(DB)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB)))
    lines: list[str] = []
    try:
        with psycopg.connect(conninfo, dbname=DB, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conf = _conf(conninfo)
        for component, run in COMPONENTS.items():

            def record(
                step: str, args: dict[str, Any], result: Any, component: str = component
            ) -> None:
                lines.append(golden.line(component, step, args, result))

            run(conf, record)
    finally:
        with psycopg.connect(conninfo, dbname="postgres", autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(DB))
            )
    return lines


def main(pairs: list[str]) -> None:
    released = Path(langchain_agensgraph.__file__).resolve()
    if langchain_agensgraph.__version__ != "0.3.1" or "site-packages" not in released.parts:
        raise SystemExit(f"record from the released 0.3.1, not {released}")
    if not pairs:
        raise SystemExit("usage: record.py LEG=CONNINFO ...")
    legs = dict(pair.split("=", 1) for pair in pairs)
    recorded = {leg: record_leg(conninfo) for leg, conninfo in legs.items()}
    first, *others = recorded
    differing = [
        f"{leg}: {json.loads(line)['component']} {json.loads(line)['step']}"
        for leg in others
        for line, expected in zip(recorded[leg], recorded[first], strict=True)
        if line != expected
    ]
    if differing:
        raise SystemExit(f"0.3.1 answered differently than on {first}:\n" + "\n".join(differing))
    golden.RECORDING.write_text("".join(line + "\n" for line in recorded[first]))
    print(f"{len(recorded[first])} calls recorded, alike on {', '.join(legs)}")


if __name__ == "__main__":
    main(sys.argv[1:])
