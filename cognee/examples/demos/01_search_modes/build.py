"""01 · build — turn Wikipedia into cognee memory (knowledge graph + embeddings).

`cognee.add` ingests the articles into a dataset; `cognee.cognify` runs the ECL
pipeline — chunk, LLM-extract entities + relationships, summarize, embed — and
stores the whole thing in AgensGraph (graph + pgvector, the `cognee_wiki` database).

    cd cognee
    WIKI_LIMIT=15 .venv/bin/python examples/demos/01_search_modes/build.py   # tiny dry-run first
    .venv/bin/python examples/demos/01_search_modes/build.py                 # ~350 articles

Knobs: WIKI_LIMIT (articles), WIKI_CHARS (lead chars/article), WIKI_RESET=0 (keep
existing memory). cognify is LLM-extraction-bound — a cost estimate prints first.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console
from _common.datautil import env_int, stream_hf
from _common.models import count_tokens, print_cost_estimate

DB = "cognee_wiki"
DATASET = "wiki"
WIKI = ("wikimedia/wikipedia", "20231101.en")
LIMIT = env_int("WIKI_LIMIT", 350)
CHARS = env_int("WIKI_CHARS", 2000)


def load_docs():
    """Yield entity-rich Wikipedia lead sections."""
    for rec in stream_hf(WIKI[0], config=WIKI[1], limit=LIMIT * 2):
        text = (rec.get("text") or "").strip()
        title = (rec.get("title") or "").strip()
        if len(text) < 400 or not title:
            continue
        yield f"# {title}\n\n{text[:CHARS]}"


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.ensure_db(DB)
    config.configure(DB)

    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine

    console.section(f"Collecting up to {LIMIT} Wikipedia articles (lead {CHARS} chars)")
    docs = []
    for d in load_docs():
        docs.append(d)
        if len(docs) >= LIMIT:
            break
    total_tokens = sum(count_tokens(d) for d in docs)
    console.kv("articles", len(docs))
    console.kv("total tokens", f"{total_tokens:,}")
    print_cost_estimate(total_tokens)

    if env_int("WIKI_RESET", 1):
        console.sub("WIKI_RESET=1 — pruning existing memory")
        await config.aprune()

    console.section("Building cognee memory (add + cognify)")
    with console.timer("add"):
        await cognee.add(docs, dataset_name=DATASET)
    with console.timer("cognify (LLM extraction + summaries + embeddings)") as t:
        await cognee.cognify([DATASET])
    print("  " + t.rate(len(docs), "articles"))

    console.section("Result — knowledge graph in AgensGraph")
    metrics = await (await get_graph_engine()).get_graph_metrics(include_optional=False)
    console.kv("nodes", metrics.get("num_nodes"))
    console.kv("edges", metrics.get("num_edges"))
    console.kv("mean degree", round(metrics.get("mean_degree") or 0, 2))
    print("\n  Memory built. Explore the search modes with: "
          "python examples/demos/01_search_modes/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
