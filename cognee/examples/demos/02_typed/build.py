"""02 · build — ontology-guided extraction: make the graph follow YOUR vocabulary.

cognify with the default extractor invents whatever entity types the LLM picks
("organization", "thing", …). Give cognee a domain **ontology** and it aligns
extracted entities to your classes instead: the ontology's classes become
`EntityType` nodes and matched entities are flagged `ontology_valid`. Same robust
graph build as demo 1, now grounded in a schema you control (`ontology.ttl`).

    cd cognee
    TYPED_LIMIT=15 .venv/bin/python examples/demos/02_typed/build.py   # tiny dry-run
    .venv/bin/python examples/demos/02_typed/build.py                 # ~120 articles

Knobs: TYPED_LIMIT (articles), TYPED_CHARS, TYPED_RESET=0 (keep existing memory).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console
from _common.datautil import env_int, stream_hf
from _common.models import count_tokens, print_cost_estimate

DB = "cognee_typed"
DATASET = "typed"
ONTOLOGY = str(pathlib.Path(__file__).resolve().parent / "ontology.ttl")
WIKI = ("wikimedia/wikipedia", "20231101.en")
LIMIT = env_int("TYPED_LIMIT", 120)
CHARS = env_int("TYPED_CHARS", 2000)


def load_docs():
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

    console.section(f"Collecting up to {LIMIT} Wikipedia articles")
    console.kv("ontology", ONTOLOGY)
    docs = []
    for d in load_docs():
        docs.append(d)
        if len(docs) >= LIMIT:
            break
    total_tokens = sum(count_tokens(d) for d in docs)
    console.kv("articles", len(docs))
    print_cost_estimate(total_tokens)

    if env_int("TYPED_RESET", 1):
        console.sub("TYPED_RESET=1 — pruning existing memory")
        await config.aprune()

    console.section("Building cognee memory with ontology-guided extraction")
    with console.timer("add"):
        await cognee.add(docs, dataset_name=DATASET)
    with console.timer("cognify (ontology-guided)"):
        await cognee.cognify([DATASET], ontology_file_path=ONTOLOGY)

    console.section("Result — entities aligned to the ontology")
    nodes, edges = await (await get_graph_engine()).get_graph_data()
    from collections import Counter
    entity_types = sorted(
        {(p.get("name")) for _, p in nodes if p.get("type") == "EntityType"}
    )
    valid = sum(1 for _, p in nodes if p.get("ontology_valid"))
    console.kv("nodes / edges", f"{len(nodes)} / {len(edges)}")
    console.kv("ontology-aligned entities", valid)
    console.kv("EntityType nodes (the vocabulary in use)", ", ".join(entity_types) or "—")
    print("\n  Built. Query the typed graph with: python examples/demos/02_typed/ask.py")


if __name__ == "__main__":
    asyncio.run(main())
