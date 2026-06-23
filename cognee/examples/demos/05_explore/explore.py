"""05 · explore — the cognee graph is a first-class, queryable AgensGraph graph.

cognee's memory isn't a black box: it's a real property graph in AgensGraph you
can measure, traverse, query with Cypher, and visualize. This reuses demo 1's
`cognee_wiki` graph.

    cd cognee
    .venv/bin/python examples/demos/05_explore/explore.py

Run 01_search_modes/build.py first — this reads its graph.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console

DB = "cognee_wiki"

# Top entities by degree, straight from AgensGraph Cypher (cognee stores every
# node on the "__Node__" vlabel with a `name`, edges on the "DIRECTED" elabel).
TOP_ENTITIES = """
MATCH (n:"__Node__")
WHERE n.name IS NOT NULL
OPTIONAL MATCH (n)-[r]-()
WITH n.name AS name, count(r) AS degree
RETURN name, degree ORDER BY degree DESC LIMIT 12
"""


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.configure(DB)

    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine

    g = await get_graph_engine()

    console.section("Graph metrics (get_graph_metrics)")
    m = await g.get_graph_metrics(include_optional=False)
    for k in ("num_nodes", "num_edges", "mean_degree", "edge_density", "num_selfloops"):
        console.kv(k, m.get(k))

    console.section("What's in the graph (node types)")
    nodes, edges = await g.get_graph_data()
    for t, c in Counter(p.get("type") for _, p in nodes).most_common():
        console.kv(t or "?", c)

    console.section("Connectivity")
    try:
        disconnected = await g.get_disconnected_nodes()
        console.kv("isolated nodes (no edges)", len(disconnected))
    except Exception as e:
        print(f"  get_disconnected_nodes: {e}")

    console.section("Top entities by degree — raw AgensGraph Cypher (query)")
    try:
        rows = await g.query(TOP_ENTITIES)
        console.table(
            [(r.get("name"), r.get("degree")) for r in rows if isinstance(r, dict)][:12],
            headers=["entity", "degree"],
        )
    except Exception as e:
        print(f"  query failed: {e}")

    console.section("Visualize the knowledge graph (HTML)")
    out = str(config.DATA_DIR / "wiki_graph.html")
    try:
        await cognee.visualize_graph(out)
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  visualize_graph failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
