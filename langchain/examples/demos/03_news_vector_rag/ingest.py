"""News vector RAG — ingest.

Streams real news articles from Hugging Face (CC-News), chunks them, and loads them into an
AgensgraphVector store set up for HYBRID search (pgvector HNSW plus a stored ``tsvector`` per
chunk), with per-chunk metadata (domain, date, title, url) for filtered retrieval. With
``DEMOS_FAKE=1`` the articles are the 50 invented ones of ``_common/fixtures/news.jsonl``.

    cd langchain
    .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
    NEWS_LIMIT=2000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, Iterator, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings

from _common import agens, config, console
from _common.datautil import batched, env_int, fixture, stream_hf
from _common.models import get_embeddings
from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.vectorstores import SearchType

GRAPH = "news"
NODE_LABEL = "Article"
DATASET = "vblagoje/cc_news"
# The metadata keys rag.py filters on, each given an index of its own.
FILTER_KEYS = ("domain", "date")


def chunks(limit: int, chunk_chars: int) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (text, metadata) chunks from the articles until `limit` chunks.

    How many chunks an article yields is not known until it is read, so the stream is opened
    unbounded and stopped here. It is closed on the way out rather than left to the collector:
    the reader downloads on a thread of its own, and one still fetching when the interpreter
    finalizes calls back into an interpreter that will not have it.
    """
    source = fixture("news") if config.fake() else stream_hf(DATASET, limit=None)
    n = 0
    try:
        for rec in source:
            text = (rec.get("text") or "").strip()
            if len(text) < 100:
                continue
            # A field the record does not carry is left out rather than stored as "".
            # An empty string is a value: it satisfies IS NOT NULL, it is the minimum of
            # any range of dates, and a filter of `date >= min(date)` built from it
            # therefore selects everything.
            base = {
                key: value
                for key, value in (
                    ("domain", rec.get("domain", "")),
                    ("date", (rec.get("date") or "")[:10]),  # YYYY-MM-DD, sorts as text
                    ("title", (rec.get("title") or "")[:200]),
                    ("url", rec.get("url", "")),
                )
                if value
            }
            for i in range(0, len(text), chunk_chars):
                piece = text[i:i + chunk_chars].strip()
                if len(piece) < 100:
                    continue
                yield piece, {**base, "chunk": i // chunk_chars}
                n += 1
                if n >= limit:
                    return
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def news_store(embedding: Embeddings, graph_name: str = GRAPH, **kwargs: Any) -> AgensgraphVector:
    """The store: chunks are Article vertices, searched by embedding and by keyword."""
    return agens.make_vector(
        embedding,
        graph_name=graph_name,
        label=NODE_LABEL,
        search_type=SearchType.HYBRID,
        filter_keys=FILTER_KEYS,
        **kwargs,
    )


def ingest(
    store: AgensgraphVector, items: Iterable[Tuple[str, Dict[str, Any]]], batch: int = 1000
) -> int:
    """Embed and store the chunks a batch at a time, then build the HNSW index once."""
    total = 0
    for part in batched(items, batch):
        store.add_texts([text for text, _ in part], metadatas=[meta for _, meta in part])
        total += len(part)
        if total % (batch * 10) == 0:
            print(f"    ... {total:,} chunks")
    store.build_index()
    return total


def main() -> None:
    limit = env_int("NEWS_LIMIT", 100000)
    chunk_chars = env_int("NEWS_CHUNK_CHARS", 900)
    batch = env_int("NEWS_BATCH", 1000)
    config.require_openai_key()

    console.section(f"News vector RAG — ingest  (NEWS_LIMIT={limit:,} chunks)")
    try:
        if os.getenv("NEWS_RESET"):
            agens.make_graph(GRAPH).drop_graph()
            print(f"[reset] dropped graph {GRAPH!r} and its companion schema")

        console.sub("chunking + embedding into AgensgraphVector (HYBRID)")
        # The first write creates the graph, the label and its indexes, the vector table and
        # the keyword table; index_build="manual" leaves the HNSW index to one build at the end.
        store = news_store(get_embeddings(), create_graph=True, index_build="manual")
        wall = time.perf_counter()
        with console.timer("embed + insert + index") as t:
            total = ingest(store, chunks(limit, chunk_chars), batch)

        console.sub("done")
        print(f"  ingested {total:,} chunks ({total / max(t.seconds, 1e-9):,.0f} chunks/s); "
              f"wall {time.perf_counter() - wall:.1f}s")
        print("\nNext:  .venv/bin/python examples/demos/03_news_vector_rag/rag.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
