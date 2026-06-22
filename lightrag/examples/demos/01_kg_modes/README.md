# 01 · KG modes — auto-built knowledge graph + dual-level retrieval

The flagship demo. Feed LightRAG raw Wikipedia text; it calls the LLM on every
chunk to extract entities and relationships, merges them across articles, and
stores the whole knowledge graph in AgensGraph (graph + vectors + KV +
doc-status, one `lightrag_wiki` database). Then ask the **same question in all
five query modes** to see LightRAG's signature **dual-level retrieval**.

📓 **Guided tour:** [`kg_modes.ipynb`](./kg_modes.ipynb) walks through it with real
outputs — open it after `build.py`.

## Run

```bash
# from lightrag/
WIKI_LIMIT=20 .venv/bin/python examples/demos/01_kg_modes/build.py   # tiny dry-run (cents, ~1 min)
.venv/bin/python examples/demos/01_kg_modes/build.py                 # ~1000 articles
.venv/bin/python examples/demos/01_kg_modes/ask.py
.venv/bin/python examples/demos/01_kg_modes/ask.py "your question"
```

Knobs: `WIKI_LIMIT` (articles), `WIKI_CHARS` (lead chars/article), `WIKI_BATCH`,
`WIKI_RESET=1`. **Insert is LLM-extraction-bound** — `build.py` prints a cost
estimate before it spends anything; start tiny.

## The five modes

```python
from lightrag import QueryParam
for mode in ["naive", "local", "global", "hybrid", "mix"]:
    print(await rag.aquery(question, QueryParam(mode=mode, enable_rerank=False)))
```

| mode | what it retrieves |
|------|-------------------|
| `naive` | text chunks by vector similarity only — the baseline, no graph |
| `local` | low-level (entity) keywords → specific entities + their facts |
| `global` | high-level (theme) keywords → cross-document relationships |
| `hybrid` | local + global merged |
| `mix` | hybrid KG retrieval **plus** raw chunks (LightRAG's default) |

`ask.py` also calls `aquery_data(...)` to print, per mode, the **high-level vs
low-level keywords** the query extractor produced and how many entities /
relationships / chunks each mode pulled — the dual-level split made visible — and
`only_need_context=True` to show the retrieved context without an LLM answer.

## What you get

One AgensGraph database holding an LLM-built knowledge graph, queryable five ways
— from a plain chunk lookup to graph-aware, multi-document reasoning — with no
separate graph DB, vector DB, KV store, or status store to run.
