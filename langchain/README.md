# 🦜️🔗 LangChain AgensGraph

LangChain integration for [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph), Skai's PostgreSQL-based multi-model graph database. Ships a `GraphStore`, a pgvector-backed `VectorStore`, chat-message history, a LangGraph checkpointer, a LangGraph long-term memory store, an LLM graph transformer, and a connection-pooling engine — with async variants throughout.

## What's new in 0.2.0

A ground-up modernization for LangChain 1.x and AgensGraph 2.17, with a full set of production components.

**Compatibility & packaging**
- Targets `langchain-core` 1.x; **no dependency on the archived `langchain-community`** — the `GraphStore`, `GraphDocument`, and `DistanceStrategy` types are vendored locally.
- Python 3.10–3.14; `uv` + `hatchling` build (PEP 621).

**Graph + vector store**
- `AgensGraph` and `AgensgraphVector` with full sync **and async** surfaces (`aquery`, `asimilarity_search`, `aadd_texts`, `adelete`, `aget_by_ids`, `aclose`, …).
- `delete`, `get_by_ids`, `effective_search_ratio` over-fetch, and `batch_size`/`embed_batch_size` for production ingest.
- The internal system id is stored under `__id__`, so user metadata `"id"` round-trips intact and `Document.id` is populated on retrieval.
- `add_graph_documents` runs in a single transaction — partial failures roll back cleanly, no orphan nodes.
- Passes LangChain's standard `langchain_tests.integration_tests.VectorStoreIntegrationTests` conformance suite.

**New components**
- **`AgensEngine`** — a shareable `psycopg` connection pool (sync + async). Pass `engine=` to share one pool across an `AgensGraph` and multiple `AgensgraphVector` stores, so concurrent requests stop serializing on one connection. Without it, behavior is unchanged.
- **`AgensChatMessageHistory`** — `BaseChatMessageHistory` storing a session's messages as an ordered chain of graph vertices; sync + async; per-session isolation; optional `window`.
- **`AgensSaver` / `AsyncAgensSaver`** — a LangGraph `BaseCheckpointSaver` that persists agent state to the graph so threads resume across restarts. Drop-in `checkpointer=AgensSaver(graph=...)`.
- **`LLMGraphTransformer`** — text→graph extraction via any chat model's `with_structured_output`; feeds straight into `add_graph_documents`.

**AgensGraph 2.17 & ergonomics**
- Schema introspection uses the `meta` extension when present (`meta.vertex_labels`, `meta.edge_labels`, …), falling back to catalog scans on older versions; multi-label nodes and NULL-safe type detection.
- Connection lifecycle: `close()`/`aclose()`, sync & async context managers, and `application_name` tagging for `pg_stat_activity`.
- Query `timeout` (per-instance and per-call) and a `sanitize` flag that strips oversized list properties from results.
- Typed `IndexConfig` (HNSW `m`/`ef_construction`, IVFFlat `lists`) and `HybridSearchConfig` (reciprocal rank fusion `rank_constant` + per-modality weights).
- `enhanced_schema=True` samples example property values into the schema for better Text2Cypher prompting.
- Bug fixes: `IVFFLAT` enum value was `"IVFLLAT"` (extra L → pgvector rejected the DDL); stray `print("DEBUG: ...")` calls replaced with `logger.debug`; `_format_properties` now escapes apostrophes/backslashes.

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

### AgensStore (LangGraph long-term memory)

`AgensSaver` persists a single thread's state; `AgensStore` is the other half — cross-thread
long-term memory, implementing LangGraph's `BaseStore` (`get` / `put` / `search` / `delete` /
`list_namespaces`, sync and async). Because items are ordinary vertices, a memory can be linked
to other memories and to your domain data with ordinary edges — which is the thing LangGraph's
stock `PostgresStore` cannot do.

```python
from langchain_agensgraph import AgensGraph, AgensStore

graph = AgensGraph("memories", conf=conf, create=True)
store = AgensStore(graph=graph)

store.put(("users", "alice", "memories"), "m1", {"text": "prefers tea", "topic": "prefs"})
item = store.get(("users", "alice", "memories"), "m1")

# a namespace search also returns its descendants
hits = store.search(("users", "alice"), filter={"topic": "prefs"}, limit=10)
namespaces = store.list_namespaces(prefix=("users", "*", "memories"))

# async throughout
item = await store.aget(("users", "alice", "memories"), "m1")
```

Semantic search is opt-in and needs `pgvector`:

```python
from langchain_openai import OpenAIEmbeddings

store = AgensStore(
    graph=graph,
    index={"dims": 1536, "embed": OpenAIEmbeddings(), "fields": ["text"]},
)
hits = store.search(("users", "alice"), query="what do they drink?", limit=5)
```

Embeddings are deliberately **not** stored as a property. A vector serialised into the jsonb
property bag pushes the bag out of line, after which every property read on that row pays a
detoast. They live instead in a narrow table in a companion schema (`<graph>_store`), keyed by
graphid with an HNSW index, and a foreign key that cascades — so deleting a memory removes its
embedding with it and searching never touches jsonb until the surviving rows are read back.

Two notes on the storage layout:

- **Namespaces** are flattened to a `.`-joined path. LangGraph already forbids `.` inside a
  namespace label, so nothing is lost and no escaping is needed.
- **`promoted=[...]`** is an opt-in tier that mirrors chosen properties into typed columns, so
  filters and sorts compare in the column's native type rather than as jsonb. It is faster, but
  it stores those values twice for now, and native comparison is not jsonb comparison — so
  filter and sort results can legitimately differ from the default layout. Leave it unset unless
  you have measured a reason to set it.

### Shared connection pool

```python
from langchain_agensgraph import AgensEngine, AgensGraph, AgensgraphVector

engine = AgensEngine.from_url("postgresql://user:pwd@host:5432/db", min_size=2, max_size=20)
graph = AgensGraph("my_graph", conf={...}, engine=engine, create=True)
store = AgensgraphVector(embeddings, graph_name="my_graph", engine=engine)
# ... concurrent requests each borrow their own pooled connection ...
engine.close()
```

## Production tips

- **Connection pooling**: use `AgensEngine` (backed by `psycopg-pool`) and share it across your graph and vector stores so concurrent requests don't serialize on a single connection.
- **PgBouncer transaction mode**: AgensGraph speaks the standard PG wire protocol, so PgBouncer works unchanged. In transaction-pool mode, disable psycopg's server-side prepared-statement cache (`prepare_threshold=None`).
- **HNSW + AgensGraph 2.17**: two June-2026 commits (`e7e1be9`, `47b38ed`) finally make `CREATE PROPERTY INDEX ... USING HNSW (((embedding)::vector(N)) vector_cosine_ops)` use an `Index Scan` plan instead of falling back to seq-scan. If you see seq-scan on v2.17 with a small table, that's expected — the planner picks seq-scan when it's cheaper.
- **`auto_gather_graphmeta`**: enable on the database (`ALTER DATABASE x SET auto_gather_graphmeta = on`) for ~30× faster `DETACH DELETE` on large graphs.

## Compatibility

| | Old (`0.1.0`) | New (`0.2.0`) |
|---|---|---|
| `langchain-core` | `>=0.3.34,<1.0.0` | `>=1.0.0,<2.0.0` |
| `langchain-community` | required | not used |
| `langgraph` | — | `>=1.0.0,<2.0.0` (checkpointer) |
| Python | 3.9–3.12 | 3.10–3.14 |
| Build system | Poetry | hatchling (PEP 621) |
| `Document.id` after retrieval | unset | set to internal `__id__` |
| User metadata key `"id"` | clobbered our system id | round-trips intact |
| `add_graph_documents` | per-statement commit | single transaction |
| Connection model | one connection | optional pooled `AgensEngine` |
| Components | graph + vector | + chat history, checkpointer, transformer |

## License

Apache-2.0.
