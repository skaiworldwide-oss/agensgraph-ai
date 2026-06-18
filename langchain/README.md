# 🦜️🔗 LangChain AgensGraph

LangChain integration for [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), Skai's PostgreSQL-based multi-model graph database. Ships a `GraphStore` and a `VectorStore` (pgvector-backed) plus async variants for RAG workloads.

## What's new in 0.2.0

- **LangChain 1.x compatible.** No dependency on the archived `langchain-community` package; the integration vendors its own `GraphStore`, `GraphDocument`, and `DistanceStrategy`.
- **AgensGraph v2.17 optimized.** Capability probe + automatic use of the `meta` extension (`meta.vertex_labels`, `meta.edge_labels`, etc.) for cheap schema introspection. Falls back to the system catalog on older versions.
- **Async surface** for hot RAG paths: `aquery`, `aadd_texts`, `aadd_embeddings`, `asimilarity_search`, `asimilarity_search_with_score`, `adelete`, `aget_by_ids`, `aclose` on both `AgensGraph` and `AgensgraphVector`.
- **Standard `VectorStore` conformance**: passes the canonical `langchain_tests.integration_tests.VectorStoreIntegrationTests` suite (`add_documents`/`get_by_ids`/`delete` semantics, id round-trip, async parity).
- **Production-grade ingest**: `add_texts(..., batch_size=N, embed_batch_size=M)` chunks both the embedding API call and the Cypher UNWIND insert. `add_graph_documents` is now wrapped in a single transaction — partial failures roll back cleanly.
- **`effective_search_ratio` over-fetch** parameter on similarity-search methods for higher recall when combined with metadata filtering.
- **Bug fixes**: `IVFFLAT` enum value was `"IVFLLAT"` (extra L → pgvector rejected the DDL); two stray `print("DEBUG: ...")` calls in production paths replaced with `logger.debug`; `_format_properties` now escapes apostrophes/backslashes; multi-label nodes are now correctly enumerated by `refresh_schema`.

## Installation

```bash
pip install -U langchain-agensgraph
```

### AgensGraph requirements

AgensGraph 2.17+ is recommended. AgensGraph does **not** bundle the pgvector or `meta` extensions; build and install them against your AgensGraph install's `pg_config`:

```bash
# pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector && PG_CONFIG=/path/to/agens/bin/pg_config make && make install

# meta extension (ships in AgensGraph's contrib/)
cd /path/to/agensgraph/contrib/meta
PG_CONFIG=/path/to/agens/bin/pg_config make USE_PGXS=1 install

# in your AgensGraph database:
CREATE EXTENSION vector;
CREATE EXTENSION meta;
```

The integration works without `meta` (falls back to `ag_label` catalog scans) but `refresh_schema` is much faster with it.

## Usage

### AgensGraph (graph store)

```python
from langchain_agensgraph import AgensGraph

conf = {
    "dbname": "...",
    "user": "...",
    "password": "...",
    "host": "...",
    "port": 5432,
}

graph = AgensGraph(graph_name="my_graph", conf=conf, create=True)
graph.query("MATCH (n) RETURN n LIMIT 1")

# Optional: cache the schema between refreshes (seconds)
graph = AgensGraph(graph_name="my_graph", conf=conf, schema_cache_ttl=60)

# Async
results = await graph.aquery("MATCH (n) RETURN count(n) AS c")
await graph.aclose()
```

### AgensgraphVector (vector store)

```python
from langchain_agensgraph import AgensgraphVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = AgensgraphVector.from_documents(
    docs,
    embeddings,
    url="postgresql://user:pwd@host:5432/db",
)

# Search
docs_with_score = db.similarity_search_with_score("What is LangChain?", k=4)

# Higher recall — fetch 3× candidates from the ANN index, then trim to k
hits = db.similarity_search("...", k=10, effective_search_ratio=3.0)

# Mutation
db.add_texts(["...", "..."], ids=["a", "b"], batch_size=500)
db.delete(["a"])
got = db.get_by_ids(["b"])

# Async
hits = await db.asimilarity_search("...", k=10)
await db.aadd_texts(["..."], batch_size=500)
await db.aclose()
```

## Production tips

- **Connection pooling**: `psycopg-pool` is a runtime dependency. Pass a pool directly via `psycopg.connect(...)` upstream of `AgensGraph`, or set `application_name` in your connection string for visibility in `pg_stat_activity`. A dedicated `AgensEngine` pool object is planned for 0.3.0.
- **PgBouncer transaction mode**: AgensGraph speaks the standard PG wire protocol, so PgBouncer works unchanged. In transaction-pool mode, disable psycopg's server-side prepared-statement cache (`prepare_threshold=None`).
- **HNSW + AgensGraph 2.17**: two June-2026 commits (`e7e1be9`, `47b38ed`) finally make `CREATE PROPERTY INDEX ... USING HNSW (((embedding)::vector(N)) vector_cosine_ops)` actually use an `Index Scan` plan instead of falling back to seq-scan. If you're seeing seq-scan on v2.17 with a small table, that's expected — the planner picks seq-scan when it's cheaper.
- **`auto_gather_graphmeta`**: enable on the database (`ALTER DATABASE x SET auto_gather_graphmeta = on`) for ~30× faster `DETACH DELETE` on large graphs.

## Compatibility

| | Old (`0.1.0`) | New (`0.2.0`) |
|---|---|---|
| `langchain-core` | `>=0.3.34,<1.0.0` | `>=1.0.0,<2.0.0` |
| `langchain-community` | required | not used |
| Python | 3.9–3.12 | 3.10–3.14 |
| Build system | Poetry | hatchling (PEP 621) |
| `Document.id` after retrieval | unset | set to internal `__id__` |
| User metadata key `"id"` | clobbered our system id | round-trips intact |
| `add_graph_documents` | per-statement commit | single transaction |

## License

Apache-2.0.
