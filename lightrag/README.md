# LightRAG · AgensGraph

Use [**LightRAG**](https://github.com/HKUDS/LightRAG) — which turns your documents
into a queryable **knowledge graph** and answers questions with graph-aware
retrieval — backed entirely by [**AgensGraph**](https://github.com/skaiworldwide-oss/agensgraph).

Because AgensGraph is PostgreSQL + Cypher + `pgvector`, **one database serves all
four** of LightRAG's storage roles. No separate graph DB, vector DB, key-value
store, and status store to run and keep in sync — just one Postgres-compatible
database:

| LightRAG storage | Class | Stored as |
|---|---|---|
| **Graph** (entities + relationships) | `AgensgraphStorage` | a Cypher graph (`base` / `DIRECTED`) |
| **Vectors** (entity/relation/chunk embeddings) | `AgensgraphVectorStorage` | `pgvector` HNSW tables |
| **Key-value** (documents, chunks, LLM cache) | `AgensgraphKVStorage` | a JSONB table |
| **Doc-status** (the ingestion pipeline) | `AgensgraphDocStatusStorage` | an indexed status table |

Use it for everything, or mix and match (e.g. AgensGraph for the graph only).

## Requirements

- Python 3.10+
- A running **AgensGraph** with the `vector` extension enabled
  (`CREATE EXTENSION vector;`). The `meta` extension is used for schema
  introspection when present.
- `lightrag-hku>=1.5.3,<1.6`

## Install

> **0.2.0 is in development** (not yet on PyPI) — install it from this repo.

```bash
pip install lightrag-hku
pip install -e .          # run from the lightrag/ directory of this repo
# (uv: uv pip install lightrag-hku -e .)
```

## Quickstart

```python
import os, asyncio
import lightrag_agensgraph                       # importing registers the four storages
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# how to reach AgensGraph (the integration reads these from the environment)
os.environ["AGENSGRAPH_DB"] = "lightrag"
os.environ["AGENSGRAPH_USER"] = "postgres"
os.environ["AGENSGRAPH_PASSWORD"] = "postgres"
os.environ["AGENSGRAPH_HOST"] = "localhost"
os.environ["AGENSGRAPH_PORT"] = "5432"
os.environ["OPENAI_API_KEY"] = "sk-..."

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,      # OpenAI gpt-4o-mini
        embedding_func=openai_embed,              # text-embedding-3-small (1536-dim)
        graph_storage="AgensgraphStorage",
        vector_storage="AgensgraphVectorStorage",
        kv_storage="AgensgraphKVStorage",
        doc_status_storage="AgensgraphDocStatusStorage",
    )
    await rag.initialize_storages()               # creates tables/graph on first run
    await initialize_pipeline_status()

    await rag.ainsert("Marie Curie discovered radium and won two Nobel Prizes.")
    print(await rag.aquery("What did Marie Curie discover?", QueryParam(mode="mix")))

    await rag.finalize_storages()

asyncio.run(main())
```

That's the whole loop: `ainsert` documents (LightRAG extracts the knowledge graph
with the LLM), then `aquery`. Tables and the graph are created automatically on
the first `initialize_storages()`.

> Don't have OpenAI handy? Any LightRAG `llm_model_func` / `embedding_func` works
> (Ollama, Azure, etc.) — the OpenAI helpers above are just the quickest start.

## Query modes

LightRAG's strength is **dual-level retrieval**. Pick a mode with `QueryParam(mode=...)`:

| mode | what it does |
|---|---|
| `naive` | vector search over text chunks (classic RAG) |
| `local` | entity-centric: pulls specific entities and their facts from the graph |
| `global` | relationship-centric: pulls cross-document themes from the graph |
| `hybrid` | `local` + `global` combined |
| `mix` | graph retrieval **and** chunks together — the default, most thorough |

## Multi-tenancy

Pass `workspace="tenant_a"` to keep a tenant's data — graph, vectors, documents,
and status — fully isolated from other workspaces **in the same database**. An
empty workspace (the default) uses the default graph, so existing single-tenant
setups are unchanged.

## Demos

A runnable demo suite lives in [`examples/demos/`](./examples/demos) — five
focused, big-data examples on real public datasets (Wikipedia, CC-News), each
with its own README and a **pre-executed notebook**:

| Demo | What it shows |
|---|---|
| [01 · KG modes](./examples/demos/01_kg_modes) | build a KG from Wikipedia, then compare all five query modes |
| [02 · Incremental](./examples/demos/02_incremental) | incremental ingestion, the doc-status pipeline, cross-document entity merging |
| [03 · Explore](./examples/demos/03_kg_explore) | explore the extracted KG (top entities, search, subgraph export) + multi-hop |
| [04 · Curation](./examples/demos/04_curation) | merge, edit, and delete entities / relations / documents |
| [05 · Workspace](./examples/demos/05_workspace) | multi-tenancy — isolated tenants in one database |

Start at [`examples/demos/README.md`](./examples/demos/README.md).

## Configuration

Connection settings are read from the environment (one shared, pooled connection
is reused across all four stores):

| Variable | Default | Notes |
|---|---|---|
| `AGENSGRAPH_DB` | — (required) | database name |
| `AGENSGRAPH_USER` | — (required) | |
| `AGENSGRAPH_PASSWORD` | — (required) | (any value under trust auth) |
| `AGENSGRAPH_HOST` | `localhost` | |
| `AGENSGRAPH_PORT` | `5432` | |
| `AGENSGRAPH_GRAPHNAME` | `lightrag` | base graph name (single-tenant) |
| `AGENSGRAPH_WORKSPACE` | `""` | tenant isolation (graph + relational stores) |

## Under the hood

The backend is indexed and pooled on its hot paths out of the box:

- **One shared async connection pool** per process (refcounted), reused by all
  four stores — not one pool per store, and not reopened per query.
- **Graph ingest** is index-backed: nodes MERGE on an indexed `entity_id`;
  batches use `UNWIND`; id-keyed lookups, deletes, and the `get_knowledge_graph`
  traversal use OR-of-equalities so the planner serves them via a BitmapOr index
  scan (plain `IN` / UNWIND-variable lookups sequential-scan on AgensGraph).
- **Vector search** uses an HNSW (`vector_cosine_ops`) index, with the column
  typed `vector(dim)` so the query's `<=>` cast matches the index and it's used at
  scale. Entity/relation vector deletes are index-backed too.
- **Doc-status** filtering, counting, and pagination are served by indexes on
  `status` / `file_path` / `content_hash` / `track_id` and the sort keys.

> The vector dimension is fixed when the vector tables are first created. To
> switch to an embedding model with a different dimension, drop the
> `LIGHTRAG_VDB_*` tables so they're recreated.

See [`examples/`](./examples/) and [`tests/`](./tests/) for more runnable code.
