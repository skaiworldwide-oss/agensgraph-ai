# LightRAG · AgensGraph

Use [**LightRAG**](https://github.com/HKUDS/LightRAG), which turns documents into a
queryable knowledge graph and answers questions with graph-aware retrieval, on
[**AgensGraph**](https://github.com/skaiworldwide-oss/agensgraph).

AgensGraph is PostgreSQL with Cypher and `pgvector`, so **one database serves all
four** of LightRAG's storage roles. There is no separate graph database, vector
database, key-value store and status store to run and keep in step:

| LightRAG storage | Class | Stored as |
|---|---|---|
| **Graph** (entities and relations) | `AgensgraphStorage` | a property graph: `base` nodes, `DIRECTED` edges |
| **Vectors** (entity, relation and chunk embeddings) | `AgensgraphVectorStorage` | three `pgvector` tables with HNSW indexes |
| **Key-value** (documents, chunks, the LLM cache) | `AgensgraphKVStorage` | one JSONB table |
| **Document status** (the ingestion pipeline) | `AgensgraphDocStatusStorage` | one typed table |

Use all four, or only some of them.

Every statement goes through the
[`agensgraph-python`](https://github.com/skaiworldwide-oss/agensgraph-python) driver:
one connection pool per graph, vertices and edges decoded by the driver, every
value bound as a parameter, embeddings carried in binary.

## Requirements

- Python 3.11 or later
- `lightrag-hku` 1.5.6 (the `>=1.5.6,<1.6` range)
- `agensgraph-python` 2.0
- A running **AgensGraph** 2.17 or later with the `vector` extension available.
  `agensgraph-python` 2.0 refuses an older server at connect.
  The stores create it in the database on first use if the role may; otherwise
  run `CREATE EXTENSION vector;` once.

## Install

> 0.2.0 is in development and not on PyPI yet; install it from this repository.

```bash
pip install lightrag-hku
pip install -e .          # from the lightrag/ directory of this repository
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
        embedding_func=openai_embed,              # text-embedding-3-small (1536 dimensions)
        graph_storage="AgensgraphStorage",
        vector_storage="AgensgraphVectorStorage",
        kv_storage="AgensgraphKVStorage",
        doc_status_storage="AgensgraphDocStatusStorage",
    )
    await rag.initialize_storages()               # creates the graph and the tables on first run
    await initialize_pipeline_status()

    await rag.ainsert("Marie Curie discovered radium and won two Nobel Prizes.")
    print(await rag.aquery("What did Marie Curie discover?", QueryParam(mode="mix")))

    await rag.finalize_storages()

asyncio.run(main())
```

`ainsert` documents (LightRAG extracts the knowledge graph with the LLM), then
`aquery`. Any LightRAG `llm_model_func` and `embedding_func` works; the OpenAI
helpers are only the quickest start.

## Query modes

LightRAG's strength is **dual-level retrieval**. Pick a mode with `QueryParam(mode=...)`:

| mode | what it does |
|---|---|
| `naive` | vector search over text chunks (classic RAG) |
| `local` | entity-centric: pulls specific entities and their facts from the graph |
| `global` | relationship-centric: pulls cross-document themes from the graph |
| `hybrid` | `local` + `global` combined |
| `mix` | graph retrieval **and** chunks together; the default and the most thorough |

## Multi-tenancy

Pass `workspace="tenant_a"` to keep a tenant's data apart from every other
tenant's **in the same database**. The graph store gives each workspace a graph
of its own (`<workspace>_chunk_entity_relation`); the other three stores keep a
`workspace` column. An empty workspace, the default, uses the graph
`chunk_entity_relation`, so a single-tenant setup needs nothing.

## How the data is stored

**Graph.** Every entity is a `base` node whose `entity_id` is the entity name,
under a uniqueness constraint; every relation is a `DIRECTED` edge. LightRAG
treats a relation as undirected, so an edge is stored once per pair of entities,
with its endpoints in sorted order, under a unique index on the pair. Two writers
merging the same entity or the same relation at once make one of it, or an
error the store retries, never two. Reads that involve edges (does this relation
exist, what are its properties, what is this entity's degree, who are its
neighbours) are answered from the edge table by graph id, so they cost the same
on an entity with six hundred relations as on one with two.

**Vectors.** LightRAG hands the vector store one record at a time while it merges
a document, and asks the store to embed it. The store keeps the records until
the document is done (`index_done_callback`), embeds them in batches of
`embedding_batch_num`, and writes each batch with one statement, the vectors in
binary. A record still waiting can be read back, a delete cancels a waiting
write, and a search writes what is waiting first. A nearest search runs on the
HNSW index; the planner would rather scan the table, because it cannot see that
the vectors are stored out of line, so the store turns sequential scans, bitmap
scans and sorts off for the search's own transaction. The threshold LightRAG
sets (`cosine_better_than_threshold`) is applied in the same statement, and the
index walk continues past dropped candidates and as far as `top_k` asks, so a
`top_k` above forty returns what was asked for.

**Key-value.** One table for every namespace (documents, chunks, the LLM cache,
the entity and relation chunk lists), written one statement per batch.

**Document status.** Every field of a status record is a column, so the
pipeline's sweeps, counts and lookups are index reads. This store implements
the scheduling API LightRAG 1.5.6 added: pages of documents in `(created_at,
id)` order with a cursor, strict reads by id, resolution of a source file to the
documents it produced, and listing and repairing sources with more than one
primary document.

### Upgrading from the previous layout

The document status table and the vector tables of the previous development
layout are brought to the current one in place, keeping their rows, the first
time a store starts. The graph is not: the previous layout stored relations in
extraction order and could hold the same relation twice, and the unique index on
the pair refuses such a graph. Drop the graph (`DROP GRAPH chunk_entity_relation
CASCADE;`) and ingest again.

## Configuration

Connection settings are read from the environment. One pool serves every store of
a workspace.

| Variable | Default | Notes |
|---|---|---|
| `AGENSGRAPH_DB` | (required) | database name |
| `AGENSGRAPH_USER` | (required) | |
| `AGENSGRAPH_PASSWORD` | (required, may be empty) | |
| `AGENSGRAPH_HOST` | `localhost` | |
| `AGENSGRAPH_PORT` | `5432` | |
| `AGENSGRAPH_URI` | | a full libpq connection string; when set, the five above are only what LightRAG's own start-up check reads |
| `AGENSGRAPH_GRAPHNAME` | `chunk_entity_relation` | the graph's name (the workspace is prefixed to it) |
| `AGENSGRAPH_WORKSPACE` | `""` | the workspace, when not given to `LightRAG(workspace=...)` |

The vector store reads two more settings from LightRAG's
`vector_db_storage_cls_kwargs`: `hnsw_m` (default 16) and `hnsw_ef_construction`
(default 64), the HNSW build parameters. The tables are created for the width of
the embedding function they first see; a store started with a different width
refuses to run. To change embedding models, drop the `LIGHTRAG_VDB_*` tables.

### Loading many documents at once

Inserting into an HNSW index costs far more than building the index afterwards.
For a large one-off load, wrap the work in `bulk_ingest()` on each vector store:
the index is dropped for the duration and rebuilt at the end with
`maintenance_work_mem` raised (`1GB` by default, settable on the store).

```python
async with rag.entities_vdb.bulk_ingest(), rag.relationships_vdb.bulk_ingest(), rag.chunks_vdb.bulk_ingest():
    await rag.ainsert(documents)
```

## Performance

Every number below was measured, not asserted: the same seeded dataset (1,000
documents, 1,200 chunks, 15,000 entities, 20,000 relations, 36,200 embeddings of
1,536 dimensions, the size of the Wikipedia demo) loaded and read through the
previous stores and through these on one AgensGraph 2.18 server, the two
alternating per repetition, medians of 3 repetitions for the load and 7 for the
rest (the rows from the knowledge graph down were measured again, on a fresh
load, after the edge removal and the label sorts were reworked). The insert rows replay the sequence of calls LightRAG makes while merging a
document (24 coroutines at once); the query rows replay one `mix`-mode query.

| Operation | Previous stores | These stores | Ratio |
|---|---|---|---|
| Load 15,000 nodes (batches of 1,000) | 4.86 s | 9.00 s | 0.5× |
| Load 20,000 edges (batches of 1,000) | 8.18 s | 7.75 s | 1.1× |
| Load 15,000 entity vectors, index kept | 227.5 s | 262.3 s | 0.9× |
| Load 20,000 relation vectors, index kept | 269.1 s | 206.2 s | 1.3× |
| Load 1,200 chunk vectors, index kept | 7.77 s | 6.20 s | 1.3× |
| Load 1,200 chunks into the key-value store | 301.2 ms | 298.3 ms | 1.0× |
| Load 1,000 document status rows | 291.3 ms | 201.6 ms | 1.4× |
| Whole load | 544.0 s | 542.1 s | 1.0× |
| Insert: 300 entities the way LightRAG merges them, 24-way | 1.97 s | 2.06 s | 1.0× |
| Insert: 300 relations the way LightRAG merges them, 24-way | 2.98 s | 2.72 s | 1.1× |
| Insert: 100 chunk records read and written back | 563.7 ms | 442.3 ms | 1.3× |
| Query: entity search, top 40 | 225.0 ms | 10.5 ms | 21.4× |
| Query: relation search, top 40 | 302.4 ms | 12.2 ms | 24.8× |
| Query: chunk search, top 20 | 20.5 ms | 4.8 ms | 4.3× |
| Query: nodes and degrees for 40 entities | 5.5 ms | 3.4 ms | 1.6× |
| Query: neighbours of 40 entities | 5.0 ms | 3.6 ms | 1.4× |
| Query: the edges among them and their degrees | 62.3 ms | 9.8 ms | 6.4× |
| Query: vectors of the chunks found | 35.8 ms | 10.5 ms | 3.4× |
| Query: chunk texts by id | 4.3 ms | 3.4 ms | 1.3× |
| Query: LLM cache write | 12.1 ms | 3.4 ms | 3.6× |
| Knowledge graph, densest 1,000 nodes | 1.27 s | 119.2 ms | 10.6× |
| Knowledge graph, 2 hops from a hub, 1,000 nodes | 160.4 ms | 63.9 ms | 2.5× |
| Knowledge graph, 3 hops from a leaf | 241.9 ms | 67.4 ms | 3.6× |
| Every node and every edge | 907.0 ms | 203.6 ms | 4.5× |
| All 15,000 labels | 109.9 ms | 27.4 ms | 4.0× |
| 300 most connected labels | 128.0 ms | 66.1 ms | 1.9× |
| Label search | 5.7 ms | 17.0 ms | 0.3× |
| 100 has_node | 84.4 ms | 54.2 ms | 1.6× |
| 100 node_degree | 234.0 ms | 182.3 ms | 1.3× |
| Neighbours of an entity with 600 relations | 25.9 ms | 6.4 ms | 4.0× |
| Delete an entity with 600 relations (graph and vectors) | 42.6 ms | 19.5 ms | 2.2× |
| Remove 20 edges | 9.8 ms | 8.4 ms | 1.2× |
| Document status: counts | 2.5 ms | 2.2 ms | 1.1× |
| Document status: a page of 50 | 6.5 ms | 5.7 ms | 1.1× |
| Document status: all processed documents | 14.6 ms | 26.6 ms | 0.5× |
| Document status: 20 lookups by content hash | 22.4 ms | 21.7 ms | 1.0× |
| Key-value: 50 chunks by id | 3.4 ms | 3.9 ms | 0.9× |
| Statements sent, load | 41,714 | 122 | 342× |
| Statements sent, insert + query + management replay | 15,007 | 5,131 | 2.9× |

What the table says:

- **Searches run on the HNSW index** now, which is where the 20-fold gains are.
  The previous stores scanned the vector tables for every search.
- **Edges are answered by graph id.** Anything that asks about a relation between
  two named entities cost what the busier entity's degree cost; it is now one
  index probe. The subgraph for the web view is built the same way.
- **Loading is bound by HNSW insertion**, about 10 ms per vector on this server
  once an index outgrows `shared_buffers` (128 MB here). No client-side change
  moves that, and the load totals are equal within the noise of a shared machine
  (the two node-load medians, 4.9 s and 9.0 s, were 1.4–2.0 s each when measured
  alone). Statements sent, which the client does control, went from 41,714 to
  122. For a large one-off load use `bulk_ingest()`: the same load took 80 s, the vectors written in 14 s and the three
  indexes built afterwards.
- **One operation got slower.** Label search ranks exact matches, then
  prefixes, then shorter names, as LightRAG 1.5.6 asks, which needs one pass
  over every name; reading 15,000 names out of their property maps is what the
  17 ms is, and the previous store's 6 ms came from stopping at the first fifty
  matches. Sorting the labels in byte order, which is also the order LightRAG's
  own file-backed store uses, is what made the list of all labels four times
  faster.

## Demos

A runnable demo suite lives in [`examples/demos/`](./examples/demos): five
focused examples on real public datasets (Wikipedia, CC-News), each with its own
README and an executed notebook.

| Demo | What it shows |
|---|---|
| [01 · KG modes](./examples/demos/01_kg_modes) | build a knowledge graph from Wikipedia, then compare all five query modes |
| [02 · Incremental](./examples/demos/02_incremental) | incremental ingestion, the document status pipeline, cross-document entity merging |
| [03 · Explore](./examples/demos/03_kg_explore) | explore the extracted graph (top entities, search, subgraph export) and a multi-hop question |
| [04 · Curation](./examples/demos/04_curation) | merge, edit and delete entities, relations and documents |
| [05 · Workspace](./examples/demos/05_workspace) | two tenants, isolated, in one database |

Start at [`examples/demos/README.md`](./examples/demos/README.md).

## Tests

The suite needs a server; a run without one refuses to start instead of
reporting success.

```bash
AGENSGRAPH_DB=lightrag_test AGENSGRAPH_USER=postgres AGENSGRAPH_PASSWORD=postgres \
  AGENSGRAPH_HOST=localhost AGENSGRAPH_PORT=5432 pytest
```
