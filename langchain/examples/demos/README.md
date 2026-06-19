# langchain-agensgraph demos

Runnable, big-data demos that exercise every capability of
[`langchain-agensgraph`](../../) against a real AgensGraph database, on real
public datasets, using **OpenAI for all embeddings and LLM calls** (nothing runs
on a local model).

## What each demo shows

| Demo | Capabilities | Dataset |
|------|--------------|---------|
| [`01_arxiv_graphrag`](01_arxiv_graphrag/) | `AgensGraph` + `AgensgraphVector` + `from_existing_graph` + shared `AgensEngine`; batched ingest; **graph + vector in one DB**; hybrid GraphRAG | arXiv metadata (~50k papers) |
| [`02_wikipedia_kg`](02_wikipedia_kg/) | `LLMGraphTransformer` → `add_graph_documents`; `enhanced_schema`; natural-language **Text2Cypher** QA | Wikipedia articles |
| [`03_news_vector_rag`](03_news_vector_rag/) | `AgensgraphVector` at scale: HNSW, **RRF hybrid search**, MongoDB-style **metadata filters**, `effective_search_ratio`, RAG | CC-News (~100k chunks) |
| [`04_chat_memory_agent`](04_chat_memory_agent/) | `AgensSaver` (LangGraph checkpointer) + `AgensChatMessageHistory`; **conversation resumes across processes** | (uses demo 03's vectors) |

Plus [`bench/`](bench/) — EXPLAIN proofs that the hot paths use the right
indexes, and a throughput summary.

## Setup

```bash
# from the repo root
uv venv langchain/.venv --python 3.13
uv pip install --python langchain/.venv/bin/python -e ./langchain \
    -r langchain/examples/demos/requirements-demos.txt

# credentials
cp langchain/examples/demos/.env.example langchain/examples/demos/.env
#   then edit .env and set OPENAI_API_KEY
```

`.env` only needs `OPENAI_API_KEY`. The AgensGraph connection defaults to the
local dev instance (`localhost:55432`, database `agensgraph_demos`, trust auth);
override with `AGENSGRAPH_*` / `AGENSGRAPH_URL` if yours differs.

## Running

Run the scripts by path from the `langchain/` directory (each adds the demos
root to `sys.path` so the shared `_common` package resolves):

```bash
cd langchain
.venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py   # ingest
.venv/bin/python examples/demos/01_arxiv_graphrag/query.py     # query
```

## Scale & cost knobs

Demos default to "big". Lower these in `.env` for a quick, near-free dry run.
Costs assume `text-embedding-3-small` ($0.02 / 1M tokens) + `gpt-4o-mini`.

| Env var | Default | Controls | Rough full-run cost |
|---------|---------|----------|---------------------|
| `ARXIV_LIMIT` | 50000 | papers ingested by demo 01 | ~$0.20 embeddings |
| `NEWS_LIMIT` | 100000 | chunks ingested by demo 03 | ~$0.40 embeddings |
| `WIKI_LIMIT` | 500 | articles (LLM-extracted) by demo 02 | ~$0.30–0.60 LLM |
| `DEMO_EMBED_MODEL` | text-embedding-3-small | embedding model | — |
| `DEMO_LLM_MODEL` | gpt-4o-mini | chat/extraction model | — |

A full pass over all four demos is roughly **$1–3** of OpenAI usage.

## Notes

- Datasets stream from the Hugging Face Hub (no full download); the cache lives
  under `examples/demos/.data/` and is git-ignored.
- Every demo shares **one** `AgensEngine` connection pool (`_common/agens.py`).
- No local embedding/LLM runtime is imported anywhere — only the OpenAI API.
