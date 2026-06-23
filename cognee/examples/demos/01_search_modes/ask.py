"""01 · ask — the same question, answered many ways from one cognee memory.

cognee's strength is that you build the memory once, then query it through
different `SearchType`s. This runs one question through nine of them so the
difference is visible, then queries the graph directly with Cypher:

  GRAPH_COMPLETION                    — graph-aware answer (entities + relationships + chunks)
  GRAPH_SUMMARY_COMPLETION            — graph answer, condensed to a summary
  GRAPH_COMPLETION_COT                — graph answer with explicit chain-of-thought
  GRAPH_COMPLETION_CONTEXT_EXTENSION  — graph answer with extra retrieved context
  RAG_COMPLETION                      — plain chunk RAG, no graph (the baseline)
  INSIGHTS                            — entity→relation→entity triplets (graph, no LLM)
  CHUNKS                              — raw matching text chunks (vector search)
  SUMMARIES                           — pre-computed summaries
  NATURAL_LANGUAGE                    — your question → generated Cypher → graph rows
  CYPHER                              — a Cypher query you write → graph rows

(CODE search is its own demo — see 04_code_graph.)

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

# Modes that answer the same natural-language question — the contrast is the point.
MODES = [
    ("GRAPH_COMPLETION", "graph-aware answer (KG + chunks)"),
    ("GRAPH_SUMMARY_COMPLETION", "graph answer, condensed to a summary"),
    ("GRAPH_COMPLETION_COT", "graph answer with chain-of-thought"),
    ("GRAPH_COMPLETION_CONTEXT_EXTENSION", "graph answer with extra retrieved context"),
    ("RAG_COMPLETION", "plain chunk RAG, no graph — the baseline"),
    ("INSIGHTS", "entity → relation → entity triplets (graph, no LLM)"),
    ("CHUNKS", "raw matching text chunks (vector search)"),
    ("SUMMARIES", "pre-computed summaries"),
    ("NATURAL_LANGUAGE", "your question → generated Cypher → graph rows"),
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

    # CYPHER is different from the modes above: instead of a question, you pass a
    # Cypher query and get rows straight from the AgensGraph-backed graph.
    console.section("CYPHER — query the graph directly (you write the Cypher)")
    cypher = 'MATCH (n:"__Node__") WHERE n.name IS NOT NULL RETURN n.name AS name LIMIT 5'
    print(f"  {cypher}")
    rows = await config.search(query_text=cypher, query_type=SearchType.CYPHER)
    for r in (rows or [])[:5]:
        print(f"   {r}")


if __name__ == "__main__":
    asyncio.run(main())
