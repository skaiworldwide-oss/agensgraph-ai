# llama-index-agensgraph — demo suite

Runnable, realistic demos that exercise every capability of the
`llama-index-agensgraph` integration against a live AgensGraph database, on big
public datasets. **All embeddings + LLM calls go through OpenAI** — nothing runs
on a local model.

These mirror the `langchain-agensgraph` demos on the same datasets (arXiv,
Wikipedia, CC-News) so the two integrations are directly comparable.

## Capability → demo matrix

| Capability | Demo |
|------------|------|
| `AgensPropertyGraphStore` — structured build, `upsert_nodes/relations`, HNSW `vector_query`, `get_rel_map`, `structured_query` | **01_arxiv_pg** |
| `PropertyGraphIndex.from_existing` over a populated store (GraphRAG) | **01_arxiv_pg** |
| LLM-built `PropertyGraphIndex.from_documents` (`SchemaLLMPathExtractor`) + `enhanced_schema` | **02_wikipedia_pgindex** |
| PG retriever stack: `LLMSynonymRetriever`, `VectorContextRetriever`, `TextToCypherRetriever` (AgensGraph dialect) | **02_wikipedia_pgindex** |
| `AgensgraphVectorStore` via `VectorStoreIndex` — semantic search, metadata filters, RRF hybrid, cited RAG | **03_news_vector_rag** |
| One shared `AgensEngine` backing both stores under a `RouterQueryEngine` / agent | **04_router** |

## Setup

```bash
cd /home/taha-linux/Desktop/skai/agensgraph-ai
uv venv llama-index/.venv --python 3.13
uv pip install --python llama-index/.venv/bin/python -e ./llama-index \
    -r llama-index/examples/demos/requirements-demos.txt

# dedicated database (clean isolation from the langchain demos)
createdb -h 127.0.0.1 -p 55432 -U "$(whoami)" llamaindex_demos
psql -h 127.0.0.1 -p 55432 -U "$(whoami)" -d llamaindex_demos -c 'CREATE EXTENSION IF NOT EXISTS vector'

cp llama-index/examples/demos/.env.example llama-index/examples/demos/.env   # add OPENAI_API_KEY
```

Run everything with the demo venv, from the `llama-index/` dir:

```bash
cd llama-index
.venv/bin/python examples/demos/01_arxiv_pg/prepare.py
.venv/bin/python examples/demos/01_arxiv_pg/query.py
```

## Scale & cost

Big-by-default (~50k arXiv papers, ~100k news chunks, ~500 Wikipedia articles;
≈$2–4 OpenAI total). Every demo prints a cost estimate and supports a small
`*_LIMIT` for a cheap dry-run, e.g.:

```bash
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_pg/prepare.py
```

All knobs are documented in `.env.example` and each demo's README.

## Notes

- **Database:** `llamaindex_demos` (separate from the langchain demos' `agensgraph_demos`).
- **Connection env vars:** `AGENS_DB / AGENS_USER / AGENS_PASSWORD / AGENS_HOST / AGENS_PORT`
  (or a single `AGENS_URL`); `AGENSGRAPH_*` is accepted as a fallback.
- **At scale the bottleneck is OpenAI embedding latency**, not AgensGraph — graph
  ingest is a few seconds; embeddings dominate wall-clock (demos embed in parallel).
- Performance/correctness issues found while building these were fixed in the
  library where safe (see the commit history).
