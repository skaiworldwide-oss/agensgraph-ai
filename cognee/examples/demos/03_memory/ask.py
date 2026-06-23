"""03 · ask — query the unified memory built from multiple datasets.

`build.py` grew one memory incrementally: it cognified an `encyclopedia` dataset,
then *added* a `news` dataset (the graph grew, no rebuild). Here we query that one
unified memory — a single search draws on **both** datasets' knowledge.

(Wikipedia chunks start with "# Title"; CC-News chunks are plain text, so you can
see both datasets show up in the results.)

    cd cognee
    .venv/bin/python examples/demos/03_memory/ask.py

Run build.py first — this reads the `cognee_memory` it built.

Note: cognee's `datasets=[...]` search argument is a read-permission scope, not a
retrieval filter — retrieval runs over the whole unified memory regardless. For
hard per-tenant isolation, use a separate database per tenant (as the other demos
do, one DB each).
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console

DB = "cognee_memory"


def _origin(text: str) -> str:
    return "encyclopedia" if str(text).lstrip().startswith("#") else "news"


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.configure(DB)

    import cognee
    from cognee.modules.search.types import SearchType
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.engine.models.node_set import NodeSet

    g = await get_graph_engine()
    m = await g.get_graph_metrics(include_optional=False)
    console.section("One unified memory (encyclopedia + news, built incrementally)")
    console.kv("nodes / edges", f"{m['num_nodes']} / {m['num_edges']}")

    # Each cognee.add(..., node_set=[tag]) tags its nodes; get_nodeset_subgraph
    # pulls back just that slice of the unified memory.
    console.section("node_set — the slice of memory tagged by each dataset")
    for tag in ("reference", "current_events"):
        try:
            ns_nodes, ns_edges = await g.get_nodeset_subgraph(NodeSet, [tag])
            console.kv(f"'{tag}'", f"{len(ns_nodes)} nodes, {len(ns_edges)} edges")
        except Exception as e:
            print(f"  '{tag}': {e}")

    console.section("A single query retrieves from BOTH datasets (CHUNKS)")
    hits = await config.search(query_text="notable people, places, and recent events",
                               query_type=SearchType.CHUNKS)
    origins = Counter()
    for h in (hits or [])[:8]:
        text = h.get("text", "") if isinstance(h, dict) else str(h)
        origins[_origin(text)] += 1
        print(f"   [{_origin(text)}] {text.strip().splitlines()[0][:80]}")
    console.kv("retrieved from", dict(origins))

    console.section("GRAPH_COMPLETION over the whole memory")
    ans = await config.search(query_text="What kinds of topics does this memory cover?",
                              query_type=SearchType.GRAPH_COMPLETION)
    text = ans[0] if isinstance(ans, (list, tuple)) and ans else ans
    print("  " + str(text).strip().replace("\n", " ")[:600])


if __name__ == "__main__":
    asyncio.run(main())
