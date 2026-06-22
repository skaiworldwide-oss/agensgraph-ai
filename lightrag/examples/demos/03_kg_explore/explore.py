"""03 · explore — the extracted KG is a first-class, queryable graph.

LightRAG's graph isn't a black box behind the retriever — it's a real property
graph in AgensGraph you can inspect and traverse. This reuses demo 1's
`lightrag_wiki` graph to show the graph-store API (popular entities, label
search, degree, subgraph export) and then answers a multi-hop question that
needs the graph to connect entities — contrasted with naive (chunks-only) RAG.

    cd lightrag
    .venv/bin/python examples/demos/03_kg_explore/explore.py
    .venv/bin/python examples/demos/03_kg_explore/explore.py "Entity A" "Entity B"

Run demo 1 (01_kg_modes/build.py) first — this reads its graph.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lightrag import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

from _common import config, console
from _common.rag import build_rag

DB = "lightrag_wiki"


async def main() -> None:
    config.require_openai_key()
    rag = build_rag(DB)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    g = rag.chunk_entity_relation_graph
    try:
        labels = await g.get_all_labels()
        console.section("The extracted knowledge graph")
        console.kv("total entities", f"{len(labels):,}")

        console.sub("most-connected entities (get_popular_labels)")
        popular = await g.get_popular_labels(limit=12)
        rows = [(name, await g.node_degree(name)) for name in popular]
        console.table(rows, headers=["entity", "degree"])

        term = sys.argv[1] if len(sys.argv) > 1 else (popular[0].split()[0] if popular else "the")
        console.sub(f"label search (search_labels {term!r})")
        print("  " + ", ".join(await g.search_labels(term, limit=10) or ["(none)"]))

        # Ego-network export around the most-connected entity.
        hub = popular[0] if popular else None
        if hub:
            console.sub(f"subgraph export around '{hub}' (get_knowledge_graph, depth 2)")
            kg = await g.get_knowledge_graph(hub, max_depth=2, max_nodes=40)
            console.kv("nodes", len(kg.nodes))
            console.kv("edges", len(kg.edges))
            console.kv("truncated", getattr(kg, "is_truncated", False))
            for e in kg.edges[:10]:
                print(f"   ({e.source}) -[{getattr(e, 'type', 'REL')}]- ({e.target})")

        # Multi-hop: connect two hubs. The graph modes can stitch a path across
        # documents; naive (chunks only) usually can't.
        e1 = sys.argv[1] if len(sys.argv) > 1 else (popular[0] if popular else "")
        e2 = sys.argv[2] if len(sys.argv) > 2 else (popular[1] if len(popular) > 1 else "")
        question = f"How are '{e1}' and '{e2}' connected? Explain any path between them."
        console.section(f"Multi-hop: {question}")
        for mode in ("naive", "mix"):
            with console.timer(f"{mode} answer"):
                ans = await rag.aquery(question, param=QueryParam(mode=mode, enable_rerank=False))
            text = str(ans).strip().replace("\n", " ")
            console.sub(mode)
            print("  " + text[:600] + (" …" if len(text) > 600 else ""))
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
