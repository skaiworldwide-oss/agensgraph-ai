# 03 · News VectorStoreIndex RAG (filters + hybrid + citations)

`AgensgraphVectorStore` as a first-class LlamaIndex vector store at scale: real
news articles (CC-News) chunked, embedded (OpenAI), and stored with metadata
`{domain, date, title, url}`, then queried four ways through a `VectorStoreIndex`.

## Run

```bash
cd llama-index
.venv/bin/python examples/demos/03_news_vector_rag/ingest.py    # chunk + embed + store (NEWS_LIMIT, default 100000 chunks)
.venv/bin/python examples/demos/03_news_vector_rag/rag.py       # semantic + filtered + hybrid + cited RAG
.venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"

# quick dry run:
NEWS_LIMIT=3000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
```

Knobs: `NEWS_LIMIT` (chunks), `NEWS_CHUNK_SIZE` (tokens/chunk), `NEWS_BATCH`
(embed+add batch), `EMBED_CONCURRENCY`, `NEWS_RESET=1`.

## What it demonstrates

- **`ingest.py`** — `SentenceSplitter` chunking, **parallel** OpenAI embedding,
  `async_add` into the store; `create_property_index("domain")`/`("date")` so
  metadata-filtered search uses an index scan, not a seq scan.
- **`rag.py`** — over `VectorStoreIndex.from_vector_store(...)`:
  - **(a) plain semantic search** (`VectorIndexRetriever`).
  - **(b) metadata-filtered** — `MetadataFilters` (`IN` domain `AND` `GTE` date),
    flowing through to `metadata_filters_to_cypher`; asserts all hits respect it.
  - **(c) hybrid RRF** — a *separate* `hybrid_search=True` store instance (hybrid
    is incompatible with filters), `vector_store_query_mode="hybrid"`.
  - **(d) cited RAG** — `CitationQueryEngine` with `[N]` inline source markers.

## The end result

Semantic, filtered, hybrid, and cited retrieval over a large news corpus, all
backed by one AgensGraph graph with an HNSW index and metadata property indexes.

## Notes

- **Hybrid ⊥ filters:** `AgensgraphVectorStore(hybrid_search=True)` raises if a
  query also carries `MetadataFilters` — so the filtered and hybrid paths use
  separate store instances over the same `Article` nodes.
- Wall-clock at 100k chunks is dominated by **OpenAI embedding latency**; ingest
  embeds in parallel and `async_add`s in batches.
