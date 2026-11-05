# 03 · News vector RAG (hybrid + metadata filters)

The deep **`AgensgraphVector`** showcase. Streams real news (CC-News) into a
pgvector **HNSW** store set up for **HYBRID** search (vector + a stored
`tsvector` per chunk), with per-chunk metadata (`domain`, `date`, `title`,
`url`), then retrieves five ways — culminating in a LangChain **LCEL RAG chain**.

## Run

```bash
cd langchain
.venv/bin/python examples/demos/03_news_vector_rag/ingest.py     # NEWS_LIMIT chunks (default 100000)
.venv/bin/python examples/demos/03_news_vector_rag/rag.py        # the five retrieval modes
.venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"

# quick, near-free dry run:
NEWS_LIMIT=2000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
```

Knobs: `NEWS_LIMIT` (chunks), `NEWS_CHUNK_CHARS` (chars/chunk, default 900),
`NEWS_BATCH` (embed/insert batch, default 1000), `NEWS_RESET=1` (rebuild).
With `DEMOS_FAKE=1` the articles are the 50 invented ones of
`_common/fixtures/news.jsonl`.

## What it demonstrates

- **`ingest.py`** — `AgensgraphVector(..., label="Article",
  search_type=SearchType.HYBRID, filter_keys=("domain", "date"))`: the chunks are
  `Article` vertices, their embeddings and keyword vectors rows of the companion
  schema `news_store`, and `domain` and `date` get an index each. Batched
  `add_texts` streams the chunks in; `build_index()` builds the HNSW index once
  at the end.
- **`rag.py`** — five retrieval modes over one store:
  1. **vector** semantic search (`similarity_search_with_relevance_scores`; the
     `*_with_score` methods return the raw cosine distance);
  2. **metadata-filtered** search — `{"$and": [{"domain": {"$in": [...]}},
     {"date": {"$gte": "..."}}]}` (MongoDB-style operators);
  3. **hybrid** search — vector + keyword fused with RRF, the keyword ranking
     weighted with `fusion_weights=(1.0, 2.0)`; `hybrid_search_with_details`
     shows each half's rank, and a filter narrows both halves;
  4. **`ef_search`** — pgvector's recall control, a setting of the store;
  5. **RAG** — `store.as_retriever()` plugged into an **LCEL** chain
     (`{context: retriever | format, question: passthrough} | prompt | llm |
     StrOutputParser`) for a cited, grounded answer.

## Notes

- All embeddings/LLM go through OpenAI. Ingest cost scales with `NEWS_LIMIT`
  (pennies); wall-time is embedding-bound.
- Each store object is one configuration over the same chunks: the demo opens a
  `VECTOR` one and a `HYBRID` one with `from_existing_index`, reading the same
  vertices and index.
- Chunk-level retrieval means multiple chunks of one article can appear together;
  that's expected.
