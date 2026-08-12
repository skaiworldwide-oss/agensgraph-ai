"""News vector RAG — ingest.

Streams real news articles from Hugging Face (CC-News), chunks them, and loads
them into an AgensgraphVector store configured for HYBRID search (pgvector HNSW
+ a fulltext keyword index), with per-chunk metadata (domain, date, title, url)
for filtered retrieval.

    cd langchain
    .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
    NEWS_LIMIT=2000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py   # quick
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from _common import agens, config, console
from _common.datautil import env_int, stream_hf
from _common.models import get_embeddings

from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.vectorstores.agensgraph_vector import SearchType

GRAPH = "news"
NODE_LABEL = "Article"
DATASET = "vblagoje/cc_news"


def _chunks(limit: int, chunk_chars: int):
    """Yield (text, metadata) chunks from streamed news until `limit` chunks.

    How many chunks an article yields is not known until it is read, so the stream is
    opened unbounded and stopped here. It is closed on the way out rather than left to
    the collector: the reader downloads on a thread of its own, and one still fetching
    when the interpreter finalizes calls back into an interpreter that will not have it.
    """
    source = stream_hf(DATASET, limit=None)
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
        source.close()


def _declare_label(dims: int) -> None:
    """Create the label before the store writes to it, with two keys promoted.

    All of a label's properties live in one jsonb column, so reading any one of them
    reassembles the whole map -- and this map holds a 1,536-dimension embedding, so the
    heap is small and the TOAST behind it is not. A promoted key is a column of its own,
    which reading never touches the map for, and which an index can be built on directly.

    `embedding` is promoted so the distance ranks a column, and `domain` because it is
    what `rag.py` filters on: a filter on the map costs a read of every element's
    properties, where the same filter on a column is an index scan.

    Declaring it here rather than after loading matters -- adding a promoted column to a
    label that already holds elements rewrites the table, reading every map once.
    """
    with psycopg.connect(**config.conf(), autocommit=True) as c:
        c.execute(f'CREATE GRAPH IF NOT EXISTS "{GRAPH}"')
        c.execute(f'SET graph_path = "{GRAPH}"')
        c.execute(
            f'CREATE VLABEL IF NOT EXISTS "{NODE_LABEL}" '
            f"(embedding vector({dims}) GENERATED, domain text GENERATED)"
        )
        c.execute(
            f'CREATE INDEX IF NOT EXISTS "{NODE_LABEL}_embedding_idx" '
            f'ON "{GRAPH}"."{NODE_LABEL}" USING hnsw (embedding vector_cosine_ops)'
        )
        c.execute(
            f'CREATE INDEX IF NOT EXISTS "{NODE_LABEL}_domain_idx" '
            f'ON "{GRAPH}"."{NODE_LABEL}" (domain)'
        )


def main() -> None:
    limit = env_int("NEWS_LIMIT", 100000)
    chunk_chars = env_int("NEWS_CHUNK_CHARS", 900)
    batch = env_int("NEWS_BATCH", 1000)
    reset = bool(os.getenv("NEWS_RESET"))
    config.require_openai_key()

    console.section(f"News vector RAG — ingest  (NEWS_LIMIT={limit:,} chunks)")
    if reset:
        with psycopg.connect(**config.conf(), autocommit=True) as c:
            c.execute('DROP GRAPH IF EXISTS "%s" CASCADE' % GRAPH)
        print(f"[reset] dropped graph {GRAPH!r}")

    console.sub("label with promoted columns for the embedding and the filtered key")
    _declare_label(dims=len(get_embeddings().embed_query("dimension probe")))

    console.sub("streaming CC-News + chunking + embedding into AgensgraphVector (HYBRID)")

    store = None
    texts: list[str] = []
    metas: list[dict] = []
    total = 0
    embed_seconds = 0.0
    wall = time.perf_counter()

    def flush():
        nonlocal store, total, embed_seconds
        if not texts:
            return
        t0 = time.perf_counter()
        if store is None:
            # First batch creates the graph, the HNSW vector index, and (because
            # search_type=HYBRID) the fulltext keyword index.
            store = AgensgraphVector.from_texts(
                texts,
                embedding=get_embeddings(),
                metadatas=metas,
                engine=agens.get_engine(),
                graph_name=GRAPH,
                node_label=NODE_LABEL,
                search_type=SearchType.HYBRID,
            )
        else:
            store.add_texts(texts, metadatas=metas)
        embed_seconds += time.perf_counter() - t0
        total += len(texts)
        if total % (batch * 10) == 0:
            print(f"    ... {total:,} chunks")

    for text, meta in _chunks(limit, chunk_chars):
        texts.append(text)
        metas.append(meta)
        if len(texts) >= batch:
            flush()
            texts, metas = [], []
    flush()

    console.sub("done")
    print(f"  ingested {total:,} chunks in {embed_seconds:.1f}s embed+insert "
          f"({total / embed_seconds:,.0f} chunks/s); wall {time.perf_counter() - wall:.1f}s")
    print("\nNext:  .venv/bin/python examples/demos/03_news_vector_rag/rag.py")
    agens.close()


if __name__ == "__main__":
    main()
