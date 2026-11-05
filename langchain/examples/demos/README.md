# langchain-agensgraph demos

Runnable, big-data demos that exercise every capability of
[`langchain-agensgraph`](../../) against a real AgensGraph database, on real
public datasets, using **OpenAI for all embeddings and LLM calls** (nothing runs
on a local model). With `DEMOS_FAKE=1` they run instead on deterministic fake
models and small fixtures, with no API key.

## What each demo shows

| Demo | Capabilities | Dataset |
|------|--------------|---------|
| [`01_arxiv_graphrag`](01_arxiv_graphrag/) | `AgensgraphVector` whose chunks are the graph's `Paper` vertices + `AgensGraph.add_graph_documents`; one shared `AgensEngine`; **graph + vector in one graph**; hybrid GraphRAG | arXiv metadata (~50k papers) |
| [`02_wikipedia_kg`](02_wikipedia_kg/) | `LLMGraphTransformer` → `add_graph_documents`; `enhanced_schema`; natural-language **Text2Cypher** QA, run as the untrusted role | Wikipedia articles |
| [`03_news_vector_rag`](03_news_vector_rag/) | `AgensgraphVector` at scale: HNSW, **RRF hybrid search**, MongoDB-style **metadata filters**, `ef_search`, RAG | CC-News (~100k chunks) |
| [`04_chat_memory_agent`](04_chat_memory_agent/) | `AgensSaver` (LangGraph checkpointer) + `AgensStore` + `AgensChatMessageHistory` and its `cypher_subquery`; **conversation resumes across processes** | (uses demo 03's vectors) |
| [`05_graph_retrievers`](05_graph_retrievers/) | `AgensVectorRetriever`, `AgensGraphContextRetriever` and `AgensText2CypherRetriever` side by side on a film catalog, then against demo 01's arXiv graph | an invented catalog; arXiv |

Plus [`bench/`](bench/) — EXPLAIN proofs that the hot paths use the right
indexes, and retriever throughput — and [`smoke.py`](smoke.py), which proves each
demo's core path runs.

## Setup

```bash
# from the repo root
uv venv langchain/.venv --python 3.13
uv pip install --python langchain/.venv/bin/python -e "./langchain[langgraph]" \
    -r langchain/examples/demos/requirements-demos.txt

# credentials
cp langchain/examples/demos/.env.example langchain/examples/demos/.env
#   then edit .env and set OPENAI_API_KEY and AGENSGRAPH_UNTRUSTED_URL
```

The AgensGraph connection defaults to the local dev instance (`localhost:55432`,
database `agensgraph_demos`, trust auth); override with `AGENSGRAPH_*` /
`AGENSGRAPH_URL` if yours differs. That role owns the demos' graphs. The server
has to be AgensGraph 2.14.1 or later, with the `vector` extension (pgvector)
created in the database.

## Roles

Cypher a model wrote runs as a role made for it, never as the role that owns
the graphs. Create the roles once per cluster, as a superuser, from what
`render_roles` prints:

```python
from langchain_agensgraph.roles import render_roles
print(render_roles(password="..."))   # app_rw, llm_ro and llm_rw
```

Then set `AGENSGRAPH_UNTRUSTED_URL` to `llm_ro`'s URL. Every demo shares one
engine, `AgensEngine(AGENSGRAPH_URL, untrusted_conninfo=AGENSGRAPH_UNTRUSTED_URL)`
(`_common/agens.py`). Before a demo runs model-written Cypher on a graph it applies
`render_grants(graph)` as the owner (`agens.grant`), which lets `llm_ro` read
the graph's labels and nothing else.

## Fake mode

With `DEMOS_FAKE=1`, `_common/models.py` returns langchain_core's
`DeterministicFakeEmbedding` and the fake chat model of `_common/fake.py`, and
each demo reads a fixture of 50 records from `_common/fixtures/` instead of
downloading its dataset: invented arXiv-style papers, short encyclopedia leads
written for the fixture, and invented news articles. The answers mean nothing —
the fake embeddings carry no meaning — but every path runs, with no API key.

## Smoke test

```bash
cd langchain
AGENSGRAPH_URL=postgresql://owner:secret@localhost:55432/agensgraph_demos \
AGENSGRAPH_UNTRUSTED_URL=postgresql://llm_ro:secret@localhost:55432/agensgraph_demos \
    .venv/bin/python examples/demos/smoke.py
```

`smoke.py` turns fake mode on, runs each demo's core path (ingest, then one
query, retrieval or agent turn) through the demo's own functions on graphs of
its own (`smoke_*`), checks what came back, and drops those graphs afterwards.
It prints one line per demo and exits 1 if any failed.

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
Costs assume `text-embedding-3-small` ($0.02 / 1M tokens) + `gpt-6-luna` ($0.10 / 1M input,
$0.50 / 1M output).

| Env var | Default | Controls | Rough full-run cost |
|---------|---------|----------|---------------------|
| `ARXIV_LIMIT` | 50000 | papers ingested by demo 01 | ~$0.20 embeddings |
| `NEWS_LIMIT` | 100000 | chunks ingested by demo 03 | ~$0.40 embeddings |
| `WIKI_LIMIT` | 500 | articles (LLM-extracted) by demo 02 | ~$0.30–0.60 LLM |
| `DEMO_EMBED_MODEL` | text-embedding-3-small | embedding model | — |
| `DEMO_LLM_MODEL` | gpt-6-luna | chat/extraction model | — |
| `DEMOS_CACHE_DIR` | unset | a folder that keeps embeddings and answers, so a rerun pays nothing twice | — |

A full pass over all four demos is roughly **$1–3** of OpenAI usage.

## Notes

- Datasets stream from the Hugging Face Hub (no full download); the cache lives
  under `examples/demos/.data/` and is git-ignored.
- A graph's vectors, keyword index, LangGraph state and chat history live in its
  companion schema `"<graph>_store"`; `AgensGraph.drop_graph()` drops both.
- No local embedding/LLM runtime is imported anywhere — only the OpenAI API, or
  the fakes.
