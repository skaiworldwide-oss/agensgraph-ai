# LightRAG AgensGraph

A complete [LightRAG](https://github.com/HKUDS/LightRAG) storage backend for
[AgensGraph](https://github.com/skaiworldwide-oss/agensgraph). Because AgensGraph
is PostgreSQL + Cypher + `pgvector`, a single AgensGraph database can serve **all
four** LightRAG storage types:

| LightRAG storage | Class | Backing |
|---|---|---|
| Graph | `AgensgraphStorage` | Cypher graph (`base` / `DIRECTED`) |
| Vector | `AgensgraphVectorStorage` | `pgvector` HNSW tables |
| Key-value | `AgensgraphKVStorage` | JSONB table |
| Doc-status | `AgensgraphDocStatusStorage` | indexed status table |

You can mix and match — use AgensGraph for the graph only, or for everything.

## Requirements

- Python 3.10+
- `lightrag-hku>=1.5.3,<1.6`
- AgensGraph with the `vector` extension (for vector storage). The `meta`
  extension is used for schema introspection when present.

## Install

```bash
pip install lightrag-hku lightrag-agensgraph
```

To build from source (hatchling):

```bash
pip install build && python -m build
```

## Configuration

Connection is read from the environment (one shared, pooled connection per
process is reused across all four stores):

| Variable | Default | Notes |
|---|---|---|
| `AGENSGRAPH_DB` | — (required) | database name |
| `AGENSGRAPH_USER` | — (required) | |
| `AGENSGRAPH_PASSWORD` | — (required) | |
| `AGENSGRAPH_HOST` | `localhost` | |
| `AGENSGRAPH_PORT` | `5432` | |
| `AGENSGRAPH_GRAPHNAME` | `lightrag` | graph name (graph store) |
| `AGENSGRAPH_WORKSPACE` | `""` | tenant isolation for the relational stores |

## Usage

```python
import os
import lightrag_agensgraph  # registers the storages with LightRAG
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

os.environ["AGENSGRAPH_DB"] = "lightrag"
os.environ["AGENSGRAPH_USER"] = "..."
os.environ["AGENSGRAPH_PASSWORD"] = "..."
os.environ["AGENSGRAPH_HOST"] = "localhost"
os.environ["AGENSGRAPH_PORT"] = "5432"

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=...,                 # your LLM
    embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=...),
    graph_storage="AgensgraphStorage",
    vector_storage="AgensgraphVectorStorage",
    kv_storage="AgensgraphKVStorage",
    doc_status_storage="AgensgraphDocStatusStorage",
)

await rag.initialize_storages()
await rag.ainsert("your document text ...")
print(await rag.aquery("your question"))
```

See [examples/](./examples/) and [tests/](./tests/) for runnable code.

## Performance & indexing

The backend is indexed and pooled for its hot paths out of the box:

- **One shared async connection pool** is opened once per process (refcounted)
  and reused by all four stores — not one pool per store, and not reopened per
  query.
- **Graph ingest** is index-backed: nodes MERGE on an indexed `entity_id`;
  `upsert_nodes_batch` / `upsert_edges_batch` use `UNWIND` batching; id-keyed
  lookups, `remove_nodes` / `remove_edges`, `get_triplets`, and the
  `get_knowledge_graph` BFS use OR-of-equalities, which the planner serves via a
  BitmapOr index scan (plain `IN` / `<@` / UNWIND-variable lookups sequential-scan
  on AgensGraph).
- **Vector search** uses an HNSW (`vector_cosine_ops`) index; the column is typed
  `vector(dim)` so the query's `<=>` cast matches the index expression and the
  index is used at scale. Entity/relation deletes are index-backed too
  (`entity_name`, `source_id`, `target_id`), so purging an entity's vectors does
  not scan the table.
- **Doc-status** filtering, counting, and pagination are served by indexes on
  `status` / `file_path` / `content_hash` / `track_id` and on the
  `created_at` / `updated_at` sort keys.

> The vector embedding dimension is fixed when the vector tables are first
> created. To change embedding models (different dimension), drop the
> `LIGHTRAG_VDB_*` tables so they are recreated.
