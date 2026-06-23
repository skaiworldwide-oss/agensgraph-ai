# 01 · Search modes — one memory, queried many ways

The flagship demo. `cognee.add` + `cognee.cognify` turn Wikipedia articles into a
knowledge graph + embeddings in AgensGraph (the `cognee_wiki` database). Then ask
the **same question through six `SearchType`s** to see what cognee's memory layer
gives you beyond plain RAG.

📓 **Guided tour:** [`search_modes.ipynb`](./search_modes.ipynb) walks through it with
real outputs — open it after `build.py`.

## Run

```bash
# from cognee/
WIKI_LIMIT=15 .venv/bin/python examples/demos/01_search_modes/build.py   # tiny dry-run (cents, ~2 min)
.venv/bin/python examples/demos/01_search_modes/build.py                 # ~350 articles
.venv/bin/python examples/demos/01_search_modes/ask.py
.venv/bin/python examples/demos/01_search_modes/ask.py "your question"
```

Knobs: `WIKI_LIMIT` (articles), `WIKI_CHARS` (lead chars/article), `WIKI_RESET=0`.
cognify is LLM-extraction-bound — `build.py` prints a cost estimate first.

## The six modes

```python
from cognee.modules.search.types import SearchType
await cognee.search(query_text="...", query_type=SearchType.GRAPH_COMPLETION)
```

| mode | what it returns |
|------|-----------------|
| `GRAPH_COMPLETION` | an answer grounded in the **graph** (entities + relationships + chunks) |
| `RAG_COMPLETION` | an answer from text **chunks only** — the plain-RAG baseline |
| `GRAPH_COMPLETION_COT` | the graph answer with explicit **chain-of-thought** reasoning |
| `INSIGHTS` | entity → relation → entity **triplets** straight from the graph (no LLM) |
| `CHUNKS` | the raw matching text chunks (vector search) |
| `SUMMARIES` | the pre-computed per-document summaries |

`ask.py` runs all six on one question. The contrast is the point: `GRAPH_*`
answers draw on relationships across documents, `INSIGHTS` shows the actual graph
edges, and `RAG_COMPLETION`/`CHUNKS` are the no-graph baseline.

## What you get

One AgensGraph database holding an LLM-built knowledge graph **and** its
embeddings — built once with `cognify`, then queried six ways — with no separate
graph DB, vector DB, or document store to run.
