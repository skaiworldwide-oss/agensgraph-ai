"""02 · incremental — grow a knowledge graph over time, and watch entities merge.

A real corpus arrives in waves. LightRAG ingests each wave incrementally: new
documents extend the SAME graph, entities that recur across documents are merged
into one node (degree + description grow), and the doc-status pipeline tracks
every document (with a per-wave `track_id`). Re-submitting a document is a no-op.

This runs on CC-News (entities — people, companies, places — recur naturally
across articles) in the `lightrag_news` database.

    cd lightrag
    NEWS_LIMIT=40 .venv/bin/python examples/demos/02_incremental/ingest.py   # tiny dry-run
    .venv/bin/python examples/demos/02_incremental/ingest.py                 # ~600 articles

Knobs: NEWS_LIMIT (articles total, split into waves), NEWS_WAVES (default 2),
NEWS_RESET=1 (drop & rebuild).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.datautil import env_int, stream_hf
from _common.models import count_tokens, print_cost_estimate
from _common.rag import build_rag, reset_rag

DB = "lightrag_news"
NEWS = "vblagoje/cc_news"
LIMIT = env_int("NEWS_LIMIT", 600)
WAVES = env_int("NEWS_WAVES", 2)
CHARS = env_int("NEWS_CHARS", 2500)
CHUNK_TOKENS = 1200
GLEANING = 1

try:
    from lightrag.utils import GRAPH_FIELD_SEP
except Exception:  # pragma: no cover
    GRAPH_FIELD_SEP = "<SEP>"


def load_articles():
    for i, rec in enumerate(stream_hf(NEWS, limit=LIMIT * 2)):
        text = (rec.get("text") or "").strip()
        if len(text) < 400:
            continue
        yield {
            "id": f"news-{i}",
            "url": rec.get("url") or f"news-{i}",
            "text": text[:CHARS],
        }


async def kg_size(rag) -> int:
    return len(await rag.chunk_entity_relation_graph.get_all_labels())


def _source_count(node: dict) -> int:
    return len([s for s in (node.get("source_id") or "").split(GRAPH_FIELD_SEP) if s])


async def top_cross_document(rag, scan: int = 60, top: int = 8):
    """Entities that recur across the most documents — i.e. merged across docs.

    LightRAG merges an entity seen in many documents into one node, accumulating
    its source chunks and edges. Ranking the most-connected entities by source-
    chunk count surfaces exactly those cross-document merges.
    """
    g = rag.chunk_entity_relation_graph
    rows = []
    for name in await g.get_popular_labels(limit=scan):
        node = await g.get_node(name)
        rows.append((name, await g.node_degree(name), _source_count(node)))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:top]


async def main() -> None:
    config.require_openai_key()
    config.ensure_db(DB)

    console.section(f"Collecting up to {LIMIT} CC-News articles → {WAVES} waves")
    articles = []
    for a in load_articles():
        articles.append(a)
        if len(articles) >= LIMIT:
            break
    total_tokens = sum(count_tokens(a["text"]) for a in articles)
    console.kv("articles", len(articles))
    print_cost_estimate(total_tokens, chunk_token_size=CHUNK_TOKENS, gleaning=GLEANING)

    per = max(1, len(articles) // WAVES)
    waves = [articles[i:i + per] for i in range(0, len(articles), per)]

    rag = build_rag(DB, chunk_token_size=CHUNK_TOKENS, gleaning=GLEANING)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    try:
        if env_int("NEWS_RESET", 0):
            console.sub("NEWS_RESET=1 — dropping existing graph + stores")
            await reset_rag(rag)

        for w, wave in enumerate(waves, 1):
            track_id = f"wave-{w}"
            console.section(f"Wave {w}/{len(waves)} — ingest {len(wave)} articles (track_id={track_id})")
            with console.timer(f"wave {w} ingest"):
                await rag.ainsert([a["text"] for a in wave],
                                  ids=[a["id"] for a in wave],
                                  file_paths=[a["url"] for a in wave],
                                  track_id=track_id)
            counts = await rag.doc_status.get_all_status_counts()
            # CC-News contains many duplicate articles; LightRAG detects them
            # (by filename / content hash) and records them as `failed` rather
            # than re-extracting — that's the dedup pipeline at work.
            console.kv("processed", counts.get("processed"))
            console.kv("failed (incl. duplicate docs skipped)", counts.get("failed"))
            console.kv("entities in graph (grows each wave)", f"{await kg_size(rag):,}")
            console.kv(f"docs tagged {track_id}", len(await rag.doc_status.get_docs_by_track_id(track_id)))

        console.section("Cross-document entity merging")
        print("  Entities seen in many documents are merged into one node, accumulating")
        print("  source chunks + edges. The most cross-document entities:\n")
        rows = await top_cross_document(rag)
        console.table([(n, d, s) for n, d, s in rows],
                      headers=["entity", "degree", "source documents"])

        console.section("Duplicate handling — re-submitting a processed document")
        from lightrag.base import DocStatus
        proc_ids = set((await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)).keys())
        dup = next((a for a in articles if a["id"] in proc_ids), articles[0])
        before = await rag.doc_status.get_by_id(dup["id"])
        await rag.ainsert([dup["text"]], ids=[dup["id"]], file_paths=[dup["url"]], track_id="resubmit")
        after = await rag.doc_status.get_by_id(dup["id"])
        same = (before or {}).get("updated_at") == (after or {}).get("updated_at")
        print(f"  re-submitted an already-processed document ({dup['id']}); it was "
              f"{'NOT re-processed (detected as duplicate)' if same else 're-processed'}.")

        console.section("Doc-status pagination (newest first)")
        rows, total = await rag.doc_status.get_docs_paginated(page=1, page_size=5,
                                                              sort_field="updated_at", sort_direction="desc")
        console.kv("total documents", total)
        for doc_id, status in rows:
            fp = getattr(status, "file_path", "?")
            print(f"   {doc_id}  status={getattr(status, 'status', '?')}  {fp}")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
