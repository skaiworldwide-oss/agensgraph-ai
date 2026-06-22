# 01 · arXiv GraphRAG

**Graph + vector in one AgensGraph database.** Streams real arXiv metadata from
Hugging Face and builds, in a single graph:

```
(:Paper {id,title,abstract,year})
   -[:AUTHORED_BY]-> (:Author {name})
   -[:IN_CATEGORY]-> (:Category {name})
   -[:UPDATED_IN]->  (:Year {year})
```

plus a **pgvector HNSW** index over the same `Paper` nodes (their title+abstract
embedded with OpenAI). One shared `AgensEngine` pool serves both.

## Run

```bash
cd langchain
.venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py          # ingest (ARXIV_LIMIT, default 50000)
.venv/bin/python examples/demos/01_arxiv_graphrag/query.py            # analytics + vector + GraphRAG
.venv/bin/python examples/demos/01_arxiv_graphrag/query.py "your question"

# quick dry run:
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py
```

Knobs: `ARXIV_LIMIT` (papers), `ARXIV_BATCH` (UNWIND batch size, default 1000),
`ARXIV_RESET=1` (drop & rebuild the graph first).

## Explore it interactively

[`arxiv_graphrag.ipynb`](arxiv_graphrag.ipynb) is a runnable, pre-executed tour of
everything here — graph stats, Cypher analytics, vector search, and GraphRAG —
with real outputs from the 50k graph. Open it after running `prepare.py`.

## The end result: what you can do after loading

One AgensGraph database now serves **three modes over the same `Paper` nodes**:

- **Analytical Cypher** — collaboration networks, most-prolific authors, category
  and per-year trends, paths between authors (questions a plain vector store
  can't answer).
- **Semantic search** — pgvector HNSW over the abstracts, with distance scores
  and metadata (arXiv id, year).
- **GraphRAG** — retrieve relevant papers by similarity, **expand through the
  graph** to papers sharing an author, and ground an LLM answer with citations.

No separate graph DB + vector DB to keep in sync — it's one database, one
connection pool.

## Measured at the 50k default (dev box: AgensGraph 2.17 / PostgreSQL 17.10)

| Metric | Value |
|--------|-------|
| Graph DB ingest | 50,000 papers + **274,756 edges in 13.87s** (3,604 papers/s, 19,804 edges/s) |
| Final graph | 138,619 nodes (50k papers · 88,455 authors · 147 categories · 17 years) |
| Embedding → HNSW | ~14 min (58 papers/s) — **100% OpenAI latency**, ~8.7M tokens ≈ $0.17 |
| Vector search | `similarity_search_with_score(k=5)` ≈ 2.3s (incl. query embedding) |
| GraphRAG | vector seeds → 8 graph-expanded papers → grounded LLM answer in ~4.5s |

## What it demonstrates

- **`prepare.py`** — batched Cypher `UNWIND` ingest through the shared engine
  (~3,600 papers/s, ~19,800 edges/s at 50k), with explicit property indexes so
  every `MERGE` is an index lookup, not a sequential scan; then
  `AgensgraphVector.from_existing_graph(...)` embeds the `Paper` nodes in place
  and builds the HNSW cosine index. The DB does its part in ~14s; the rest is
  OpenAI embedding latency.
- **`query.py`**
  - **(a) graph analytics** — pure Cypher: most prolific authors, largest
    categories, top co-authorship pairs, papers per year.
  - **(b) vector semantic search** — HNSW similarity over abstracts.
  - **(c) hybrid GraphRAG** — vector-retrieve seed papers, **expand through the
    graph** (papers sharing an author), then ask `gpt-4o-mini` for a grounded,
    citation-bearing answer.

## Notes

- At the default scale the run is dominated by **OpenAI embedding latency** and
  **HF streaming**, not by AgensGraph (DB ingest is a few seconds). Expect
  roughly 15–20 min for a full 50k run; use `ARXIV_LIMIT` to shrink it.
- Graph expansion in step (c) gets richer as the corpus grows (more shared
  authors/categories); at a few thousand papers it is sparse.
- **Larger workloads (millions of papers):** graph ingest stays fast, but
  embedding throughput is bounded by **OpenAI API latency / your account's rate
  limit** (parallelism via `EMBED_CONCURRENCY` only helps up to that ceiling),
  and the **pgvector HNSW index build needs a large `maintenance_work_mem`**
  (set `ARXIV_BUILD_MEM` close to the vector-data size) — otherwise the build is
  I/O-bound and slow.
