# 01 · Search modes — build a memory once, query it many ways

This is the flagship demo, and the best place to start. `cognee.add` +
`cognee.cognify` turn a few hundred Wikipedia articles into a **knowledge graph +
embeddings**, both stored in one AgensGraph database (`cognee_wiki`). Then you ask
the **same question through every search mode** to see what a graph-backed memory
gives you that plain vector RAG does not.

📓 **Just want to read it?** Open [`search_modes.ipynb`](./search_modes.ipynb) — it's
already executed, with real outputs, so you can follow along without running anything.

## Run it

```bash
# from the cognee/ directory of this repo
WIKI_LIMIT=15 .venv/bin/python examples/demos/01_search_modes/build.py   # start tiny: a few cents, ~2 min
.venv/bin/python examples/demos/01_search_modes/build.py                 # the full ~350 articles
.venv/bin/python examples/demos/01_search_modes/ask.py                   # query it
.venv/bin/python examples/demos/01_search_modes/ask.py "your own question"
```

`build.py` prints a **cost estimate before it spends anything** (cognify makes LLM
calls per chunk), so always do the tiny `WIKI_LIMIT=15` run first. Knobs: `WIKI_LIMIT`
(number of articles), `WIKI_CHARS` (lead chars per article), `WIKI_RESET=0` (add to
the existing memory instead of rebuilding).

## The search modes

Same call, different `query_type`:

```python
from cognee.modules.search.types import SearchType
await cognee.search(query_text="What is anarchism connected to?",
                    query_type=SearchType.GRAPH_COMPLETION)
```

`ask.py` runs your question through all of these — the **contrast** is the lesson:

| mode | what it returns |
|------|-----------------|
| `GRAPH_COMPLETION` | an answer grounded in the **graph** (entities + relationships + chunks) |
| `GRAPH_SUMMARY_COMPLETION` | the same graph answer, condensed to a summary |
| `GRAPH_COMPLETION_COT` | the graph answer with explicit **chain-of-thought** reasoning |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | the graph answer with extra retrieved context folded in |
| `RAG_COMPLETION` | an answer from text **chunks only** — the plain-RAG baseline |
| `INSIGHTS` | entity → relation → entity **triplets** straight from the graph (no LLM) |
| `CHUNKS` | the raw matching text chunks (vector search) |
| `SUMMARIES` | the pre-computed per-document summaries |
| `NATURAL_LANGUAGE` | your question turned into Cypher, run on the graph → rows |
| `CYPHER` | a **Cypher query you write** (not a question), run on the graph → rows |

The graph modes (`GRAPH_*`) draw on relationships *across* documents; `INSIGHTS`
shows the actual edges; `RAG_COMPLETION`/`CHUNKS` are the no-graph baseline you'd get
from ordinary vector search; and `NATURAL_LANGUAGE`/`CYPHER` query the graph directly.

(The eleventh mode, `CODE`, is for code graphs — see [04_code_graph](../04_code_graph).)

## What you get

One AgensGraph database holding an LLM-built knowledge graph **and** its embeddings —
built once with `cognify`, then queried ten different ways — with no separate graph
DB, vector DB, or document store to run alongside it.
