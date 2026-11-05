# 01 · arXiv GraphRAG

**Graph + vector in one AgensGraph graph.** Streams real arXiv metadata from
Hugging Face and builds, in a single graph:

```
(:Paper {id,title,year,text})
   -[:AUTHORED_BY]-> (:Author {name})
   -[:IN_CATEGORY]-> (:Category {name})
   -[:UPDATED_IN]->  (:Year {year})
```

The `Paper` vertices are the vector store's own chunks (title + abstract
embedded with OpenAI); their embeddings live in the companion schema
`arxiv_store`, under a **pgvector HNSW** index. One shared `AgensEngine` serves
both.

## Run

```bash
cd langchain
.venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py          # ingest (ARXIV_LIMIT, default 50000)
.venv/bin/python examples/demos/01_arxiv_graphrag/query.py            # analytics + vector + GraphRAG
.venv/bin/python examples/demos/01_arxiv_graphrag/query.py "your question"

# quick dry run:
ARXIV_LIMIT=2000 ARXIV_RESET=1 .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py

# no key: the 50 invented papers of _common/fixtures/arxiv.jsonl, on the fake models
DEMOS_FAKE=1 .venv/bin/python examples/demos/01_arxiv_graphrag/prepare.py
```

Knobs: `ARXIV_LIMIT` (papers), `ARXIV_BATCH` (papers per write, default 1000),
`ARXIV_RESET=1` (drop the graph and its companion schema first),
`EMBED_CONCURRENCY` (embedding requests in flight, default 10), `EMBED_BATCH`
(texts per request, default 100).

## Explore it interactively

[`arxiv_graphrag.ipynb`](arxiv_graphrag.ipynb) is a runnable tour of everything
here — graph stats, Cypher analytics, vector search, and GraphRAG. Open it after
running `prepare.py`.

## The end result: what you can do after loading

One AgensGraph graph now serves **three modes over the same `Paper` vertices**:

- **Analytical Cypher** — collaboration networks, most-prolific authors, category
  and per-year trends, paths between authors (questions a plain vector store
  can't answer).
- **Semantic search** — HNSW over the abstracts, with relevance scores and
  metadata (arXiv id, year).
- **GraphRAG** — retrieve relevant papers by similarity, **expand through the
  graph** to papers sharing an author, and ground an LLM answer with citations.

No separate graph DB + vector DB to keep in sync — it's one database, one
engine.

## What it demonstrates

- **`prepare.py`** — each batch of papers is embedded with many requests in
  flight and stored with `AgensgraphVector.add_embeddings` (label `Paper`, the
  arXiv id as the chunk id, `filter_keys=("id", "year")`); then
  `AgensGraph.add_graph_documents` writes the authors, categories and years and
  the edges to them, finding each Paper vertex again by `id`, one round trip per
  batch. Every label it writes gets a unique property index on `id`, so each
  `MERGE` is an index lookup. The store is made with `index_build="manual"`, and
  `build_index()` builds the HNSW index once at the end, sizing
  `maintenance_work_mem` for the build itself. A re-run embeds only papers not
  already stored.
- **`query.py`**
  - **(a) graph analytics** — pure Cypher: most prolific authors, largest
    categories, top co-authorship pairs, papers per year.
  - **(b) vector semantic search** — `AgensgraphVector.from_existing_index` over
    the Paper chunks, `similarity_search_with_relevance_scores`.
  - **(c) hybrid GraphRAG** — vector-retrieve seed papers, **expand through the
    graph** (papers sharing an author), then ask the chat model for a grounded,
    citation-bearing answer.

## Notes

- At the default scale the run is dominated by **OpenAI embedding latency** and
  **HF streaming**, not by AgensGraph; use `ARXIV_LIMIT` to shrink it.
- Graph expansion in step (c) gets richer as the corpus grows (more shared
  authors/categories); at a few thousand papers it is sparse.
- **Larger workloads (millions of papers):** graph ingest stays fast, but
  embedding throughput is bounded by **OpenAI API latency / your account's rate
  limit** (`EMBED_CONCURRENCY` only helps up to that ceiling).
