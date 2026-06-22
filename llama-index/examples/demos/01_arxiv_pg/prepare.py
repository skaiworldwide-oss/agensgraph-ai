"""arXiv property graph — prepare (ingest + embed).

Builds a structured property graph in AgensGraph through the LlamaIndex
``AgensPropertyGraphStore`` — deterministically (no LLM extraction), so it scales:

    (Paper {id,title,abstract,year}) -[AUTHORED_BY]-> (Author {name})
    (Paper)                          -[IN_CATEGORY]-> (Category {name})

Every node is physically stored on one ``"__Node__"`` vertex label with its type
in a ``labels`` list (that's how the store models multiple labels); the helpers
here just hand it ``EntityNode``/``Relation`` objects. Paper nodes are then
embedded (title+abstract, OpenAI) in parallel and the vectors written back, so
the HNSW ``entity`` index serves ``vector_query``.

    cd llama-index
    .venv/bin/python examples/demos/01_arxiv_pg/prepare.py
    ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_pg/prepare.py  # dry run

Knobs: ARXIV_LIMIT (papers, default 50000), ARXIV_BATCH (UNWIND batch, 1000),
EMBED_CONCURRENCY (parallel OpenAI requests, 10), EMBED_BATCH (texts per request,
256), ARXIV_RESET=1 (drop & rebuild the graph first).
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import psycopg
from llama_index.core.graph_stores.types import EntityNode, Relation

from _common import agens, config, console
from _common.datautil import batched, env_int, stream_hf
from _common.models import EMBED_DIM, get_embed_model

GRAPH = "arxiv"
DATASET = "UniverseTBD/arxiv-abstracts-large"
MAX_AUTHORS = 20  # cap fan-out from pathological author lists

LIMIT = env_int("ARXIV_LIMIT", 50_000)
BATCH = env_int("ARXIV_BATCH", 1000)
EMBED_CONCURRENCY = env_int("EMBED_CONCURRENCY", 10)
EMBED_BATCH = env_int("EMBED_BATCH", 256)
RESET = os.getenv("ARXIV_RESET", "").strip() not in ("", "0", "false", "False")


def _norm(rec: dict) -> dict | None:
    """Normalize a raw arXiv record to {id, title, abstract, year, authors, categories}."""
    pid = (rec.get("id") or "").strip()
    title = " ".join((rec.get("title") or "").split())
    abstract = " ".join((rec.get("abstract") or "").split())
    if not pid or not title:
        return None

    # authors: prefer the parsed [[last, first, suffix], ...] form, else split the string.
    authors: list[str] = []
    parsed = rec.get("authors_parsed")
    if isinstance(parsed, list) and parsed:
        for p in parsed:
            name = " ".join(part for part in (p[1] if len(p) > 1 else "", p[0]) if part).strip()
            if name:
                authors.append(name)
    else:
        raw = rec.get("authors") or ""
        for chunk in raw.replace(" and ", ", ").split(","):
            name = " ".join(chunk.split())
            if name:
                authors.append(name)
    authors = authors[:MAX_AUTHORS]

    categories = [c for c in (rec.get("categories") or "").split() if c]

    year = None
    upd = rec.get("update_date")
    if upd is not None:
        if hasattr(upd, "year"):  # datetime / date
            year = int(upd.year)
        else:
            s = str(upd)
            if len(s) >= 4 and s[:4].isdigit():
                year = int(s[:4])

    return {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "categories": categories,
    }


def _reset_graph() -> None:
    console.sub(f"reset: dropping graph '{GRAPH}'")
    with psycopg.connect(config.url(), autocommit=True) as conn:
        conn.execute(f"DROP GRAPH IF EXISTS {GRAPH} CASCADE")


def ingest(store) -> tuple[int, dict[str, str]]:
    """Stream + upsert papers/authors/categories. Returns (#papers, {id: embed_text})."""
    papers = 0
    nodes_total = 0
    edges_total = 0
    texts: dict[str, str] = {}

    with console.timer("graph ingest") as t:
        for batch in batched((_norm(r) for r in stream_hf(DATASET, limit=LIMIT)), BATCH):
            batch = [b for b in batch if b]
            if not batch:
                continue

            entities: dict[str, EntityNode] = {}
            relations: list[Relation] = []
            for p in batch:
                entities[p["id"]] = EntityNode(
                    name=p["id"],
                    label="Paper",
                    properties={"title": p["title"], "abstract": p["abstract"],
                                **({"year": p["year"]} if p["year"] is not None else {})},
                )
                texts[p["id"]] = f"{p['title']}\n\n{p['abstract']}"
                for a in p["authors"]:
                    entities.setdefault(a, EntityNode(name=a, label="Author"))
                    relations.append(Relation(label="AUTHORED_BY", source_id=p["id"], target_id=a))
                for c in p["categories"]:
                    entities.setdefault(c, EntityNode(name=c, label="Category"))
                    relations.append(Relation(label="IN_CATEGORY", source_id=p["id"], target_id=c))

            store.upsert_nodes(list(entities.values()))
            store.upsert_relations(relations)

            papers += len(batch)
            nodes_total += len(entities)
            edges_total += len(relations)
            print(f"  ... {papers:,} papers  ({nodes_total:,} node upserts, {edges_total:,} edges)")

    print("  " + t.rate(papers, "papers"))
    console.kv("node upserts", f"{nodes_total:,}")
    console.kv("edges", f"{edges_total:,}")
    return papers, texts


async def _embed(texts: dict[str, str]) -> list[tuple[str, list[float]]]:
    embed_model = get_embed_model(embed_batch_size=EMBED_BATCH)
    items = list(texts.items())
    chunks = [items[i:i + EMBED_BATCH] for i in range(0, len(items), EMBED_BATCH)]
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    done = 0
    out: list[tuple[str, list[float]]] = []

    async def run(chunk):
        nonlocal done
        async with sem:
            vecs = await embed_model.aget_text_embedding_batch([txt for _, txt in chunk])
        done_local = [(cid, v) for (cid, _), v in zip(chunk, vecs)]
        done += len(done_local)
        print(f"  ... embedded {done:,}/{len(items):,}")
        return done_local

    for fut in asyncio.as_completed([asyncio.create_task(run(c)) for c in chunks]):
        out.extend(await fut)
    return out


async def _store_embeddings(store, pairs: list[tuple[str, list[float]]]) -> None:
    for chunk in batched(pairs, BATCH):
        await store.aupsert_nodes(
            [EntityNode(name=cid, label="Paper", embedding=v) for cid, v in chunk]
        )


async def _embed_phase(store, texts: dict[str, str]) -> int:
    """Embed + store, then close the async pool inside this same loop."""
    pairs = await _embed(texts)
    await _store_embeddings(store, pairs)
    await agens.aclose()  # async pool workers live in this loop — close them here
    return len(pairs)


def embed_papers(store, texts: dict[str, str]) -> None:
    tokens = sum(len(t) for t in texts.values()) / 4  # ~4 chars/token
    console.kv("papers to embed", f"{len(texts):,}")
    console.kv("est. tokens", f"~{tokens/1e6:.1f}M  (~${tokens/1e6 * 0.02:.2f} @ text-embedding-3-small)")
    with console.timer("embed + store") as t:
        n = asyncio.run(_embed_phase(store, texts))
    print("  " + t.rate(n, "embeddings"))


def main() -> None:
    config.require_openai_key()
    console.section(f"arXiv → AgensPropertyGraphStore '{GRAPH}'  (limit={LIMIT:,})")
    if RESET:
        _reset_graph()

    # vector_dimension is required so the HNSW 'entity' index is created up front.
    store = agens.make_pg_store(GRAPH, vector_dimension=EMBED_DIM)
    try:
        papers, texts = ingest(store)
        if papers:
            console.sub("embedding papers (OpenAI, parallel)")
            embed_papers(store, texts)
        console.section("done")
        console.kv("graph", GRAPH)
        console.kv("papers", f"{papers:,}")
        print("\nNext: .venv/bin/python examples/demos/01_arxiv_pg/query.py")
    finally:
        agens.close()


if __name__ == "__main__":
    main()
