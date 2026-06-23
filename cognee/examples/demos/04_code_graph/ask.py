"""04 · ask — query the code knowledge graph.

  CODE     — semantic code search: find where something is implemented
  INSIGHTS — the dependency/relationship triplets between code entities
  visualize_graph — write an interactive HTML view of the code graph

    cd cognee
    .venv/bin/python examples/demos/04_code_graph/ask.py
    .venv/bin/python examples/demos/04_code_graph/ask.py "how are sessions handled?"

Run build.py first — this reads the `cognee_code` graph it built.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console

DB = "cognee_code"
DEFAULT_QUESTION = "How does the library send an HTTP request?"


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

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    console.section(f"CODE search — Q: {question}")
    code = await config.search(query_text=question, query_type=SearchType.CODE)
    for r in (code if isinstance(code, (list, tuple)) else [code])[:3]:
        if isinstance(r, dict):
            fname = str(r.get("name", "?")).split("/")[-1]
            snippet = " ".join((r.get("content") or "").split())[:110]
            print(f"   {fname}: {snippet}")
        else:
            print("   " + str(r).strip().replace("\n", " ")[:200])

    console.section("Code graph structure (the entities cognee extracted)")
    from collections import Counter
    from cognee.infrastructure.databases.graph import get_graph_engine
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    for t, c in Counter(p.get("type") for _, p in nodes).most_common():
        console.kv(t or "?", c)

    console.section("Visualize the code graph (HTML)")
    out = str(config.DATA_DIR / "code_graph.html")
    try:
        await cognee.visualize_graph(out)
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  visualize_graph failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
