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

from langchain_agensgraph import AgensgraphVector
from langchain_agensgraph.vectorstores.agensgraph_vector import SearchType

from _common import agens, config, console
from _common.datautil import env_int, stream_hf
from _common.models import get_embeddings

GRAPH = "news"
NODE_LABEL = "Article"
DATASET = "vblagoje/cc_news"


def _chunks(limit: int, chunk_chars: int):
    """Yield (text, metadata) chunks from streamed news until `limit` chunks."""
    n = 0
    for rec in stream_hf(DATASET, limit=None):
        text = (rec.get("text") or "").strip()
        if len(text) < 100:
            continue
        base = {
            "domain": rec.get("domain", ""),
            "date": (rec.get("date") or "")[:10],   # YYYY-MM-DD (lexicographically sortable)
            "title": (rec.get("title") or "")[:200],
            "url": rec.get("url", ""),
        }
        for i in range(0, len(text), chunk_chars):
            piece = text[i:i + chunk_chars].strip()
            if len(piece) < 100:
                continue
            yield piece, {**base, "chunk": i // chunk_chars}
            n += 1
            if n >= limit:
                return


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
    # from_texts(engine=...) now creates the graph itself (no pre-creation needed).
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
