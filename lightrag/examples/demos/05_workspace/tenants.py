"""05 · workspace — many tenants, one AgensGraph database, fully isolated.

LightRAG's `workspace` partitions storage so several tenants can share one
database without seeing each other's data. With lightrag-agensgraph that means:
the relational stores (vector / KV / doc-status) partition by a `workspace`
column, and the graph is given a per-workspace name — so two tenants' knowledge
graphs never mix. This demo ingests distinct, made-up corpora for two tenants
into one `lightrag_tenants` database and proves the isolation both ways.

    cd lightrag
    .venv/bin/python examples/demos/05_workspace/tenants.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.rag import build_rag, reset_rag

DB = "lightrag_tenants"

TENANTS = {
    "acme": [
        "Acme Corporation manufactures gadgets in the desert. Wile E. Coyote is "
        "Acme's most loyal customer, ordering rockets and traps. Acme ships fast.",
        "Wile E. Coyote uses Acme products to chase the Road Runner across canyons.",
    ],
    "globex": [
        "Globex Corporation is a multinational led by Hank Scorpio. Globex is based "
        "in Cypress Creek and builds ambitious engineering projects.",
        "Hank Scorpio gave his new employee a house in Cypress Creek near Globex HQ.",
    ],
}
# An entity that belongs to each tenant — used to probe for leakage.
PROBE = {"acme": "Wile E. Coyote", "globex": "Hank Scorpio"}


async def main() -> None:
    config.require_openai_key()
    config.ensure_db(DB)

    rags = {}
    for ws in TENANTS:
        rag = build_rag(DB, workspace=ws)
        await rag.initialize_storages()
        rags[ws] = rag
    await initialize_pipeline_status()
    try:
        console.section(f"Two tenants in one database ({DB})")
        for ws, rag in rags.items():
            print(f"  workspace {ws!r} → graph '{rag.chunk_entity_relation_graph.graph_name}'")

        for ws, rag in rags.items():
            await reset_rag(rag)
            with console.timer(f"ingest tenant {ws}"):
                await rag.ainsert(TENANTS[ws], file_paths=[f"{ws}-doc-{i}" for i in range(len(TENANTS[ws]))])

        console.section("Each tenant sees only its own entities")
        labels = {ws: set(await rag.chunk_entity_relation_graph.get_all_labels()) for ws, rag in rags.items()}
        for ws in TENANTS:
            console.kv(f"{ws} entities", ", ".join(sorted(labels[ws])))
        overlap = labels["acme"] & labels["globex"]
        console.kv("shared entities", overlap or "∅ (fully isolated)")

        console.section("Cross-tenant probe — graph isolation")
        for ws, rag in rags.items():
            other = "globex" if ws == "acme" else "acme"
            foreign = PROBE[other]
            present = await rag.chunk_entity_relation_graph.has_node(foreign)
            print(f"  tenant {ws}: has node {foreign!r} (the other tenant's entity)? {present}  "
                  f"{'<-- LEAK' if present else 'OK — isolated'}")

        console.section("Cross-tenant probe — retrieval isolation")
        for ws, rag in rags.items():
            other = "globex" if ws == "acme" else "acme"
            q = f"Who is {PROBE[other]}?"
            ans = await rag.aquery(q, param=QueryParam(mode="mix", enable_rerank=False))
            text = str(ans).strip().replace("\n", " ")
            console.sub(f"tenant {ws} asked: {q}")
            print("  " + text[:300] + (" …" if len(text) > 300 else ""))
    finally:
        for rag in rags.values():
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
