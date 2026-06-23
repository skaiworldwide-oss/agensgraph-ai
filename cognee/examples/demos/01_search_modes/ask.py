"""01 · ask — the same question, answered many ways from one cognee memory.

cognee's strength is that you build the memory once, then query it through
different `SearchType`s. This runs one question through six of them so the
difference is visible:

  GRAPH_COMPLETION       — graph-aware answer (entities + relationships + chunks)
  RAG_COMPLETION         — plain chunk RAG, no graph (the baseline)
  GRAPH_COMPLETION_COT   — graph answer with explicit chain-of-thought
  INSIGHTS               — entity→relation→entity triplets (graph, no LLM)
  CHUNKS                 — raw matching text chunks (vector search)
  SUMMARIES              — pre-computed summaries

    cd cognee
    .venv/bin/python examples/demos/01_search_modes/ask.py
    .venv/bin/python examples/demos/01_search_modes/ask.py "your question"

Run build.py first — this reads the `cognee_wiki` memory it built.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _common import config, console

DB = "cognee_wiki"
DEFAULT_QUESTION = "What is anarchism, and what ideas, people, and movements is it connected to?"

MODES = [
    ("GRAPH_COMPLETION", "graph-aware answer (KG + chunks)"),
    ("RAG_COMPLETION", "plain chunk RAG, no graph — the baseline"),
    ("GRAPH_COMPLETION_COT", "graph answer with chain-of-thought"),
    ("INSIGHTS", "entity → relation → entity triplets (graph, no LLM)"),
    ("CHUNKS", "raw matching text chunks (vector search)"),
    ("SUMMARIES", "pre-computed summaries"),
]


def _name(node) -> str:
    if isinstance(node, dict):
        return str(node.get("name") or (node.get("text") or "")[:40] or node.get("id") or "?")
    return str(node)[:40]


def render(results) -> None:
    # Some modes (e.g. RAG_COMPLETION) return a bare answer string rather than a
    # list — normalize so we never iterate a string character by character.
    if isinstance(results, (str, bytes)) or not isinstance(results, (list, tuple)):
        results = [results] if results else []
    if not results:
        print("  (no results)")
        return
    for r in results[:4]:
        if isinstance(r, (tuple, list)) and len(r) == 3:                 # INSIGHTS triplet
            src, edge, tgt = r
            rel = edge.get("relationship_name") if isinstance(edge, dict) else str(edge)
            print(f"   ({_name(src)}) -[{rel}]-> ({_name(tgt)})")
        elif isinstance(r, dict):                                        # CHUNKS / nodes
            print(f"   {str(r.get('text') or r.get('name') or r)[:160]}")
        else:                                                            # completion string
            print("   " + str(r).strip().replace("\n", " ")[:600])


async def main() -> None:
    config.require_openai_key()
    config.quiet()
    config.configure(DB)

    import cognee
    from cognee.modules.search.types import SearchType

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    console.section(f"Q: {question}")

    for name, blurb in MODES:
        console.sub(f"{name} — {blurb}")
        with console.timer(name):
            results = await config.search(query_text=question, query_type=getattr(SearchType, name))
        render(results)


if __name__ == "__main__":
    asyncio.run(main())
