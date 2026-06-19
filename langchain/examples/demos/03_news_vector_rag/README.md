# 03 · News vector RAG (hybrid + metadata filters)

The deep **`AgensgraphVector`** showcase. Streams real news (CC-News) into a
pgvector **HNSW** store configured for **HYBRID** search (vector + a fulltext
keyword index), with per-chunk metadata (`domain`, `date`, `title`, `url`), then
retrieves five ways — culminating in a LangChain **LCEL RAG chain**.

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

## What it demonstrates

- **`ingest.py`** — `AgensgraphVector.from_texts(..., search_type=HYBRID)` builds
  the HNSW vector index **and** a fulltext keyword index over the same `Article`
  nodes, then batched `add_texts` streams the rest. Chunks carry metadata for
  filtering.
- **`rag.py`** — five retrieval modes over one store:
  1. **vector** semantic search (`similarity_search_with_score`);
  2. **metadata-filtered** search — `{"$and": [{"domain": {"$in": [...]}},
     {"date": {"$gte": "..."}}]}` (MongoDB-style operators);
  3. **hybrid** search — vector + keyword fused with RRF, tuned via
     `HybridSearchConfig(keyword_weight=...)`;
  4. **`effective_search_ratio`** — over-fetch ANN candidates for better recall
     when a filter is applied;
  5. **RAG** — `store.as_retriever()` plugged into an **LCEL** chain
     (`{context: retriever | format, question: passthrough} | prompt | llm |
     StrOutputParser`) for a cited, grounded answer.

## Notes

- All embeddings/LLM go through OpenAI. Ingest cost scales with `NEWS_LIMIT`
  (pennies); wall-time is embedding-bound (~50–100 chunks/s).
- **Hybrid + `filter=` can't be combined** (the store raises) — the demo uses a
  `VECTOR`-typed view for filtered search and a `HYBRID` view for fused search,
  both over the same index.
- Chunk-level retrieval means multiple chunks of one article can appear together;
  that's expected.
