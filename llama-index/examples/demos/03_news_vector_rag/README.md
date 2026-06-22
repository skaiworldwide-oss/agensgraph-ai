# 03 · News — vector RAG (semantic, filtered, hybrid, cited)

Use `AgensgraphVectorStore` as a LlamaIndex `VectorStoreIndex`: ingest news
articles with metadata, then query four ways — plain semantic search,
metadata-filtered, hybrid (vector + keyword), and RAG with citations.

## Run

```bash
# from llama-index/
.venv/bin/python examples/demos/03_news_vector_rag/ingest.py   # chunk + embed + store (NEWS_LIMIT, default 100000 chunks)
.venv/bin/python examples/demos/03_news_vector_rag/rag.py
.venv/bin/python examples/demos/03_news_vector_rag/rag.py "your question"

# quick dry-run:
NEWS_LIMIT=3000 NEWS_RESET=1 .venv/bin/python examples/demos/03_news_vector_rag/ingest.py
```

Knobs: `NEWS_LIMIT` (chunks), `NEWS_CHUNK_SIZE` (tokens/chunk), `NEWS_BATCH`,
`EMBED_CONCURRENCY`, `NEWS_RESET=1`.

## The patterns (`ingest.py`)

Chunk, embed and store with metadata; index the keys you'll filter on:

```python
from llama_index.core.node_parser import SentenceSplitter
nodes = SentenceSplitter(chunk_size=256).get_nodes_from_documents(docs)  # metadata: {domain, date, title, url}
for n, v in zip(nodes, embeddings):
    n.embedding = v
await vectors.async_add(nodes)
vectors.create_property_index("domain")   # so filtered search uses an index
vectors.create_property_index("date")
```

## The patterns (`rag.py`)

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters, MetadataFilter, FilterOperator, FilterCondition)
from llama_index.core.query_engine import CitationQueryEngine

index = VectorStoreIndex.from_vector_store(vectors, embed_model=embed)

# semantic
index.as_retriever(similarity_top_k=5).retrieve("AI in business")

# metadata-filtered (IN domain AND date >= ...)
filters = MetadataFilters(condition=FilterCondition.AND, filters=[
    MetadataFilter(key="domain", operator=FilterOperator.IN, value=["techrepublic.com", ...]),
    MetadataFilter(key="date",   operator=FilterOperator.GTE, value="2017-01-01"),
])
index.as_query_engine(filters=filters).query("...")

# hybrid (vector + keyword RRF) — a hybrid_search=True store over the same data
hybrid = AgensgraphVectorStore(url=..., embedding_dimension=1536,
                               graph_name="news", node_label="Article", hybrid_search=True)
VectorStoreIndex.from_vector_store(hybrid).as_retriever(
    vector_store_query_mode="hybrid").retrieve("...")

# RAG with inline [N] citations
CitationQueryEngine.from_args(index, similarity_top_k=5).query("...")
```

## What you get

Semantic, filtered, hybrid and cited retrieval over a large corpus — all from
one AgensGraph graph with an HNSW vector index, a full-text index, and btree
indexes on the metadata you filter on.

## Tips

- **Hybrid and filters are separate paths.** `hybrid_search=True` can't be
  combined with `MetadataFilters` — use a plain store for filtered search and a
  `hybrid_search=True` store (over the same nodes) for hybrid.
- Hybrid search runs the vector and keyword halves each against their own index
  and fuses them with reciprocal rank fusion — fast even at 100k+ chunks.
