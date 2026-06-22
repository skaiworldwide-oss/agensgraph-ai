# Build with AgensGraph + LightRAG

[LightRAG](https://github.com/HKUDS/LightRAG) turns a pile of documents into a
**knowledge graph** (entities + relationships, extracted by an LLM) and answers
questions with **dual-level retrieval** over that graph. `lightrag-agensgraph`
lets a **single AgensGraph (PostgreSQL) database** serve all four LightRAG
storage roles at once:

| LightRAG storage | AgensGraph backing |
|---|---|
| Graph (entities/relationships) | a Cypher graph (`base` / `DIRECTED`) |
| Vector (entity/relation/chunk embeddings) | `pgvector` HNSW tables |
| Key-value (docs, chunks, LLM cache) | a JSONB table |
| Doc-status (the ingestion pipeline) | an indexed status table |

One database, one pooled connection — no separate graph DB, vector DB, KV store
and status store to operate.

These demos run on real public datasets (Wikipedia, CC-News) and use **OpenAI**
for all embeddings + LLM calls (nothing runs on a local model).

> **Cost note — read this first.** Unlike plain vector RAG, LightRAG calls the LLM
> on *every chunk* at insert time to extract entities and relationships. Cost and
> time scale with the corpus (~2–3 `gpt-4o-mini` calls/chunk). The demos default
> to ~1,000 documents (≈$8–15, ≈45–90 min); **every build script prints a cost
> estimate before it spends anything**, and all sizes are env-overridable — start
> with a tiny `*_LIMIT` dry-run.

## Quickstart

From the `lightrag/` directory of this repo:

```bash
# 1. install the integration + demo extras (uv)
uv venv .venv --python 3.13
uv pip install -e . -r examples/demos/requirements-demos.txt

# 2. your OpenAI key (AgensGraph defaults to the local dev instance on :55432)
cp examples/demos/.env.example examples/demos/.env     # then edit: OPENAI_API_KEY=...

# 3. run the flagship demo — start tiny (a few cents, ~1 min)
WIKI_LIMIT=20 .venv/bin/python examples/demos/01_kg_modes/build.py
.venv/bin/python examples/demos/01_kg_modes/ask.py
```

Each demo creates its **own database** (`lightrag_wiki`, `lightrag_news`, …) on
first run, so their knowledge graphs never collide.

## The building block

One factory wires LightRAG to all four AgensGraph storages on a shared pool:

```python
from _common.rag import open_rag

async with open_rag("lightrag_wiki") as rag:        # ensures the DB + the 4 stores
    await rag.ainsert(documents, file_paths=titles) # LLM extracts the KG
    print(await rag.aquery("...", QueryParam(mode="mix")))
```

## The demos

Each folder is a focused, runnable example with its own README and a pre-executed
notebook. They showcase what LightRAG does that plain vector RAG does **not**.

| # | What it shows | Run |
|---|----------------|-----|
| [**01_kg_modes**](01_kg_modes) | auto-built KG + the 5 query modes (naive/local/global/hybrid/mix) side by side — LightRAG's dual-level retrieval | `build.py` → `ask.py` |
| [**02_incremental**](02_incremental) | incremental ingestion, the doc-status pipeline, and entities merging across documents | `ingest.py` |
| [**03_kg_explore**](03_kg_explore) | exploring the extracted KG (popular labels, search, subgraph export) + multi-hop answers | `explore.py` |
| [**04_curation**](04_curation) | curating the KG: merge duplicate entities, edit, and delete by entity or document | `curate.py` |
| [**05_workspace**](05_workspace) | multi-tenancy — two isolated tenants in one database | `tenants.py` |

## Configuration

- **Connection** comes from `AGENSGRAPH_USER / AGENSGRAPH_PASSWORD / AGENSGRAPH_HOST
  / AGENSGRAPH_PORT` (or the `AGENS_*` fallback). Under the dev instance's trust
  auth, only `OPENAI_API_KEY` is actually required. The per-demo database is chosen
  by the demo, not the `.env`.
- **Models** `gpt-4o-mini` + `text-embedding-3-small` (1536-dim); override with
  `DEMO_LLM_MODEL` / `DEMO_EMBED_MODEL` (+ `DEMO_EMBED_DIM`).
- `_common/` wires all of this up once (one shared pool, OpenAI models, cost
  estimate, the `open_rag` factory) so each demo stays focused on its capability.
