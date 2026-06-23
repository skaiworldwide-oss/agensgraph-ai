# Build with AgensGraph + cognee

[cognee](https://github.com/topoteretes/cognee) is an **AI memory** framework: you
`add` your data, `cognify` it into a **knowledge graph + embeddings**, then
`search` that memory many different ways. `cognee-agensgraph` lets **one AgensGraph
(PostgreSQL) database back both** of cognee's stores at once:

- the **knowledge graph** (entities + relationships) → a Cypher graph, and
- the **embeddings** → `pgvector` HNSW tables.

(cognee keeps its small bookkeeping — datasets, users — in a local SQLite file; the
knowledge graph and vectors live in AgensGraph.)

These demos run on real public datasets (Wikipedia, CC-News, a Python repo) and use
**OpenAI** for all embeddings + LLM calls (nothing runs on a local model).

> **Cost note — read this first.** `cognify()` runs LLM entity/relationship
> extraction *and* summarization on every chunk, so cost and time scale with the
> corpus (heavier than plain RAG). The flagship demo defaults to ~300–400
> documents (~30–60 min, a few dollars). **Every build prints a cost estimate
> before it spends anything**, sizes are env-overridable, and you should start with
> a tiny `*_LIMIT` dry-run.

## What you'll need

- A running **AgensGraph** with the `vector` extension (the local dev instance on
  `localhost:55432` works out of the box — each demo creates its own database).
- An **OpenAI API key**.
- The `cognee` venv (already set up at `cognee/.venv`).

## Quickstart

From the `cognee/` directory of this repo:

```bash
# 1. install the demo extras into the cognee venv
.venv/bin/python -m pip install -r examples/demos/requirements-demos.txt
#    (or: uv pip install --python .venv/bin/python -r examples/demos/requirements-demos.txt)

# 2. your OpenAI key
cp examples/demos/.env.example examples/demos/.env     # then edit: OPENAI_API_KEY=...

# 3. run the flagship demo — start tiny (a few cents, ~1 min)
WIKI_LIMIT=15 .venv/bin/python examples/demos/01_search_modes/build.py
.venv/bin/python examples/demos/01_search_modes/ask.py
```

Each demo creates its **own database** (`cognee_wiki`, `cognee_memory`, …) on first
run, so their knowledge graphs never collide.

## The setup

The `_common/` helpers point cognee at AgensGraph in one call:

```python
from _common import config
config.configure("cognee_wiki")          # graph + vectors → one AgensGraph database
# under the hood:
#   cognee.config.set_graph_db_config({"graph_database_url": dsn, "graph_database_provider": "agensgraph"})
#   cognee.config.set_vector_db_config({"vector_db_url": dsn, "vector_db_provider": "agensgraph"})
```

Then the cognee loop:

```python
import cognee
await cognee.add(texts, dataset_name="wiki")
await cognee.cognify(["wiki"])                                   # builds the KG + embeddings
await cognee.search(query_text="...", query_type=SearchType.GRAPH_COMPLETION)
```

## The demos

Each folder is a focused, runnable example with its own README and a pre-executed
notebook. They showcase what cognee does that plain vector RAG does **not**.

| # | What it shows | Run |
|---|----------------|-----|
| [**01_search_modes**](01_search_modes) | build a KG from Wikipedia, then query it ten ways — graph completion, chain-of-thought, RAG baseline, INSIGHTS triplets, CHUNKS, SUMMARIES, natural-language → Cypher, raw Cypher | `build.py` → `ask.py` |
| [**02_typed**](02_typed) | domain-structured extraction — guide cognify with an ontology so the graph follows your vocabulary | `build.py` → `ask.py` |
| [**03_memory**](03_memory) | a memory grown from multiple named datasets — incremental `add`, `node_set` tags, one unified query | `build.py` → `ask.py` |
| [**04_code_graph**](04_code_graph) | turn a Python repo into a code knowledge graph; `SearchType.CODE` + visualize | `build.py` → `ask.py` |
| [**05_explore**](05_explore) | inspect the AgensGraph-backed KG — metrics, slices, traversal, raw Cypher, HTML graph | `explore.py` |

## Configuration

- **Connection** comes from `AGENS_USER / AGENS_PASSWORD / AGENS_HOST / AGENS_PORT`
  (or `AGENSGRAPH_*`). Under the dev instance's trust auth, only `OPENAI_API_KEY` is
  actually required. The per-demo database is chosen by the demo, not the `.env`.
- **Models** `gpt-4o-mini` + `text-embedding-3-small` (1536-d); override with
  `DEMO_LLM_MODEL` / `DEMO_EMBED_MODEL` / `DEMO_EMBED_DIM`.
- `_common/` wires it all up once (one AgensGraph database for graph + vectors,
  OpenAI models, a cost estimate, the `configure()` helper).
