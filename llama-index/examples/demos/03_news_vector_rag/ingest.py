"""News vector RAG — ingest (CC-News → AgensgraphVectorStore).

Streams real news articles, chunks them, embeds the chunks with OpenAI (in
parallel), and stores them in an ``AgensgraphVectorStore`` (HNSW cosine) with
metadata {domain, date, title, url}. Property indexes are created on the
filterable keys so metadata-filtered search uses an index scan, not a seq scan.

    cd llama-index
    .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
    NEWS_LIMIT=3000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py  # dry run

Knobs: NEWS_LIMIT (chunks, default 100000), NEWS_CHUNK_SIZE (tokens/chunk, 256),
NEWS_BATCH (embed + add batch, 1000), EMBED_CONCURRENCY (10), NEWS_RESET=1.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from _common import agens, config, console
from _common.datautil import env_int, stream_hf
from _common.models import get_embed_model

GRAPH = "news"
NODE_LABEL = "Article"
DATASET = "vblagoje/cc_news"

LIMIT = env_int("NEWS_LIMIT", 100_000)          # number of CHUNKS to ingest
CHUNK_SIZE = env_int("NEWS_CHUNK_SIZE", 256)     # tokens per chunk
BATCH = env_int("NEWS_BATCH", 1000)
EMBED_CONCURRENCY = env_int("EMBED_CONCURRENCY", 10)
RESET = os.getenv("NEWS_RESET", "").strip() not in ("", "0", "false", "False")


def _docs():
    """Yield LlamaIndex Documents from the CC-News stream."""
    # stream more articles than chunks needed; each article makes 1-several chunks.
    for rec in stream_hf(DATASET, limit=None):
        text = (rec.get("text") or "").strip()
        if len(text) < 200:
            continue
        date = (rec.get("date") or "")[:10]  # YYYY-MM-DD (lexicographically sortable)
        yield Document(
            text=text,
            metadata={
                "domain": rec.get("domain") or "",
                "date": date,
                "title": (rec.get("title") or "").strip(),
                "url": rec.get("url") or "",
            },
            excluded_embed_metadata_keys=["domain", "date", "title", "url"],
            excluded_llm_metadata_keys=["url"],
        )


def chunk_until_limit(splitter) -> list:
    """Chunk streamed articles until we have LIMIT chunks."""
    nodes: list = []
    with console.timer("stream + chunk") as t:
        for doc in _docs():
            nodes.extend(splitter.get_nodes_from_documents([doc]))
            if len(nodes) >= LIMIT:
                break
    nodes = nodes[:LIMIT]
    print("  " + t.rate(len(nodes), "chunks"))
    return nodes


async def _embed_and_store(store, nodes, embed_model) -> int:
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    batches = [nodes[i:i + BATCH] for i in range(0, len(nodes), BATCH)]
    done = 0

    async def run(batch):
        nonlocal done
        async with sem:
            vecs = await embed_model.aget_text_embedding_batch(
                [n.get_content(metadata_mode="none") for n in batch]
            )
            for n, v in zip(batch, vecs):
                n.embedding = v
            await store.async_add(batch)
        done += len(batch)
        print(f"  ... embedded + stored {done:,}/{len(nodes):,}")

    await asyncio.gather(*(run(b) for b in batches))
    await agens.aclose()  # close async pool inside this loop
    return done


def _reset_graph() -> None:
    console.sub(f"reset: dropping graph '{GRAPH}'")
    with psycopg.connect(config.url(), autocommit=True) as conn:
        conn.execute(f"DROP GRAPH IF EXISTS {GRAPH} CASCADE")


def main() -> None:
    config.require_openai_key()
    console.section(f"CC-News → AgensgraphVectorStore '{GRAPH}'  (target {LIMIT:,} chunks)")
    if RESET:
        _reset_graph()

    store = agens.make_vector_store(graph_name=GRAPH, node_label=NODE_LABEL)
    embed_model = get_embed_model()
    try:
        # Index the metadata keys we filter on, up front (HNSW can't serve the
        # filter predicate). Done before ingest so the embed phase can close the
        # engine's async pool at the end of its event loop.
        console.sub("property indexes for filtered search")
        for key in ("domain", "date"):
            store.create_property_index(key)
            print(f"  indexed {key}")

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=20)
        nodes = chunk_until_limit(splitter)

        tokens = sum(len(n.get_content(metadata_mode="none")) for n in nodes) / 4
        console.kv("chunks", f"{len(nodes):,}")
        console.kv("est. tokens", f"~{tokens/1e6:.1f}M  (~${tokens/1e6 * 0.02:.2f})")

        with console.timer("embed + store") as t:
            n = asyncio.run(_embed_and_store(store, nodes, embed_model))
        print("  " + t.rate(n, "chunks"))

        console.section("done")
        console.kv("graph", GRAPH)
        console.kv("chunks", f"{n:,}")
        print("\nNext: .venv/bin/python examples/demos/03_news_vector_rag/rag.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
