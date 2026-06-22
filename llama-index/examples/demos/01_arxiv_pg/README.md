# 01 · arXiv PropertyGraphStore (graph + vector)

A structured property graph built **deterministically** (no LLM extraction, so it
scales) through the LlamaIndex `AgensPropertyGraphStore`, plus pgvector HNSW over
the same entities:

```
(Paper {id,title,abstract,year}) -[AUTHORED_BY]-> (Author {name})
(Paper)                          -[IN_CATEGORY]-> (Category {name})
```

Every node lives on one `"__Node__"` vertex label with its type in a `labels`
list (how the store models labels); Paper entities are embedded (title+abstract,
OpenAI) so the HNSW `entity` index serves `vector_query`.

## Run

```bash
cd llama-index
.venv/bin/python examples/demos/01_arxiv_pg/prepare.py          # ingest + embed (ARXIV_LIMIT, default 50000)
.venv/bin/python examples/demos/01_arxiv_pg/query.py            # analytics + vector + expansion + GraphRAG
.venv/bin/python examples/demos/01_arxiv_pg/query.py "your question"

# quick dry run:
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_pg/prepare.py
```

Knobs: `ARXIV_LIMIT` (papers), `ARXIV_BATCH` (UNWIND batch), `EMBED_CONCURRENCY`
(parallel OpenAI requests), `EMBED_BATCH` (texts/request), `ARXIV_RESET=1`.

## What it demonstrates

- **`prepare.py`** — batched `upsert_nodes`/`upsert_relations` (UNWIND + MERGE on
  the `id` btree), then **parallel** OpenAI embedding written back with
  `aupsert_nodes`; the HNSW `entity` index is created up front (`vector_dimension`).
- **`query.py`** — four capabilities over the one store:
  - **(a) analytics** via `structured_query` — top authors, largest categories,
    papers per year. Note the idiom: `MATCH (n:"__Node__") WHERE 'Author' IN n.labels`.
  - **(b) semantic search** via `vector_query` (HNSW) with cosine scores.
  - **(c) graph expansion** via `get_rel_map` over the vector hits (shared authors / categories).
  - **(d) GraphRAG** — `PropertyGraphIndex.from_existing(...)` + a
    `VectorContextRetriever` query engine for a grounded, source-backed answer.

## The end result

One AgensGraph graph serves **analytical Cypher, vector search, graph expansion,
and GraphRAG** over the same Paper entities — no separate graph DB + vector DB.

## Notes

- At scale the wall-clock is dominated by **OpenAI embedding latency**, not
  AgensGraph (graph ingest is seconds). Embedding runs in parallel
  (`EMBED_CONCURRENCY`) up to your account's rate limit.
- Building this demo surfaced — and the library now fixes — a bug where entity
  embeddings were silently dropped on upsert for entities without a source chunk,
  plus several scale fixes (lazy schema refresh, indexed vector/graph lookups).
