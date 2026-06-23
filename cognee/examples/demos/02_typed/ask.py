"""02 · ask — query the ontology-aligned knowledge graph.

Shows that the graph follows the ontology vocabulary: the `EntityType` nodes are
your ontology's classes, entities matched to them are `ontology_valid`, and
INSIGHTS surfaces typed entity→relation→entity triplets.

    cd cognee
    .venv/bin/python examples/demos/02_typed/ask.py
    .venv/bin/python examples/demos/02_typed/ask.py "your question"

Run build.py first — this reads the `cognee_typed` memory it built.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console

DB = "cognee_typed"


def _name(node) -> str:
    if isinstance(node, dict):
        return str(node.get("name") or (node.get("text") or "")[:40] or node.get("id") or "?")
    return str(node)[:40]


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.configure(DB)

    import cognee
    from cognee.modules.search.types import SearchType
    from cognee.infrastructure.databases.graph import get_graph_engine

    nodes, _ = await (await get_graph_engine()).get_graph_data()
    entity_types = {p.get("name") for _, p in nodes if p.get("type") == "EntityType"}
    aligned = [p.get("name") for _, p in nodes if p.get("ontology_valid") and p.get("type") == "Entity"]

    console.section("The graph aligns to the ontology")
    console.kv("EntityType nodes total", len(entity_types))
    console.kv("entities aligned to the ontology (ontology_valid)", len(aligned))
    print("  aligned: " + ", ".join(str(a) for a in aligned[:15]))

    question = sys.argv[1] if len(sys.argv) > 1 else (
        f"What is {aligned[0]} and what is it connected to?" if aligned
        else "What are the main entities and how are they related?"
    )
    console.section(f"INSIGHTS (typed triplets) — Q: {question}")
    triplets = await config.search(query_text=question, query_type=SearchType.INSIGHTS)
    for r in (triplets or [])[:8]:
        if isinstance(r, (tuple, list)) and len(r) == 3:
            src, edge, tgt = r
            rel = edge.get("relationship_name") if isinstance(edge, dict) else str(edge)
            print(f"   ({_name(src)}) -[{rel}]-> ({_name(tgt)})")

    console.section("GRAPH_COMPLETION — grounded answer")
    answer = await config.search(query_text=question, query_type=SearchType.GRAPH_COMPLETION)
    text = answer[0] if isinstance(answer, (list, tuple)) and answer else answer
    print("  " + str(text).strip().replace("\n", " ")[:600])


if __name__ == "__main__":
    asyncio.run(main())
