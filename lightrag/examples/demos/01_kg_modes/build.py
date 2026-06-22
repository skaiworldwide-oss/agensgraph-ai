"""01 · build — turn Wikipedia articles into a knowledge graph with LightRAG.

LightRAG calls the LLM on every chunk to extract entities + relationships and
merges them across documents, building one connected KG in AgensGraph (graph +
vector + KV + doc-status, all in the `lightrag_wiki` database).

    cd lightrag
    WIKI_LIMIT=20 .venv/bin/python examples/demos/01_kg_modes/build.py   # tiny dry-run first
    .venv/bin/python examples/demos/01_kg_modes/build.py                 # ~1000 articles

Knobs: WIKI_LIMIT (articles), WIKI_CHARS (lead chars/article), WIKI_BATCH
(articles per insert), WIKI_RESET=1 (drop & rebuild). Insert is LLM-extraction-
bound — the script prints a cost estimate before it spends anything.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.datautil import batched, env_int, stream_hf
from _common.models import count_tokens, print_cost_estimate
from _common.rag import build_rag, reset_rag

DB = "lightrag_wiki"
WIKI = ("wikimedia/wikipedia", "20231101.en")
LIMIT = env_int("WIKI_LIMIT", 1000)
CHARS = env_int("WIKI_CHARS", 2500)
BATCH = env_int("WIKI_BATCH", 50)
CHUNK_TOKENS = 1200
GLEANING = 1


def load_docs():
    """Yield (id, title, lead_text) for entity-rich Wikipedia articles."""
    for rec in stream_hf(WIKI[0], config=WIKI[1], limit=LIMIT * 2):
        text = (rec.get("text") or "").strip()
        title = (rec.get("title") or "").strip()
        if len(text) < 400 or not title:   # skip stubs/redirects
            continue
        yield (f"wiki-{rec.get('id')}", title, text[:CHARS])


async def main() -> None:
    config.require_openai_key()
    config.ensure_db(DB)

    console.section(f"Collecting up to {LIMIT} Wikipedia articles (lead {CHARS} chars)")
    docs = []
    for d in load_docs():
        docs.append(d)
        if len(docs) >= LIMIT:
            break
    total_tokens = sum(count_tokens(t) for _, _, t in docs)
    console.kv("articles", len(docs))
    console.kv("total tokens", f"{total_tokens:,}")
    print_cost_estimate(total_tokens, chunk_token_size=CHUNK_TOKENS, gleaning=GLEANING)

    rag = build_rag(DB, chunk_token_size=CHUNK_TOKENS, gleaning=GLEANING)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    try:
        if env_int("WIKI_RESET", 0):
            console.sub("WIKI_RESET=1 — dropping existing graph + stores")
            await reset_rag(rag)

        console.section("Extracting the knowledge graph (LLM per chunk)")
        done = 0
        with console.timer("total ingest") as t:
            for batch in batched(docs, BATCH):
                ids = [d[0] for d in batch]
                titles = [d[1] for d in batch]
                texts = [d[2] for d in batch]
                with console.timer(f"insert {len(batch)} articles"):
                    await rag.ainsert(texts, ids=ids, file_paths=titles)
                done += len(batch)
                print(f"  progress: {done}/{len(docs)} articles")
        print("  " + t.rate(len(docs), "articles"))

        console.section("Result")
        counts = await rag.doc_status.get_all_status_counts()
        console.kv("doc status", counts)
        labels = await rag.chunk_entity_relation_graph.get_all_labels()
        console.kv("entities (graph nodes)", f"{len(labels):,}")
        popular = await rag.chunk_entity_relation_graph.get_popular_labels(limit=15)
        print("  most-connected entities: " + ", ".join(popular))
        print("\n  KG built. Explore the modes with: python examples/demos/01_kg_modes/ask.py")
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
