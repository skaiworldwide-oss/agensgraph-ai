# LlamaIndex AgensGraph

This plugin integrates [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph)
with [LlamaIndex](https://www.llamaindex.ai/), persisting graphs and vectors
directly in AgensGraph. It powers `PropertyGraphIndex` and `VectorStoreIndex`,
so you can store and query property graphs and embeddings in one database.

- Property Graph Store: `AgensPropertyGraphStore`
- Vector Store: `AgensgraphVectorStore`
- Connection pool: `AgensEngine` (optional, shared across stores)

## Demos & guides

**Start here:** the [**`examples/demos/`**](./examples/demos) suite — runnable,
end-to-end demos on real datasets (arXiv, Wikipedia, CC-News) that show how to
build with this integration at realistic scale. Its
[README](./examples/demos/README.md) has a quickstart and copy-paste building
blocks.

| Demo | What you build |
|------|----------------|
| [01 · arXiv](./examples/demos/01_arxiv_pg) | a property graph + vector search + GraphRAG in one store |
| [02 · Wikipedia](./examples/demos/02_wikipedia_pgindex) | an LLM-built knowledge graph + natural-language (Text2Cypher) Q&A |
| [03 · News](./examples/demos/03_news_vector_rag) | vector RAG: semantic, metadata-filtered, hybrid, and cited |
| [04 · Router](./examples/demos/04_router) | one `AgensEngine` routing questions to the graph or the vector store |

Each demo folder also ships a **pre-executed notebook** — a narrated, end-to-end
tour with real embedded outputs.

Short, single-feature notebooks:

- [Property graph store](./examples/property_graph/property_graph_agensgraph.ipynb)
- [Vector store](./examples/vector_stores/AgensgraphVectorDemo.ipynb)
- [Vector store — metadata filters](./examples/vector_stores/agensgraph_metadata_filter.ipynb)

## Requirements

- Python 3.10+
- AgensGraph with the `vector` extension (for vector / HNSW search). The `meta`
  extension is used for schema introspection when present, with a catalog
  fallback otherwise.

## What's new in 0.2.0

- **Fixed `AgensPropertyGraphStore.vector_query`.** It previously hard-coded a
  3-dimension embedding cast and ordered by a fixed literal vector, so results
  ignored the query embedding and errored at any other dimension. It now uses
  the configured `vector_dimension` consistently and ranks by the actual query
  embedding, backed by the HNSW index.
- **Metadata-filtered vector search.** Both `AgensPropertyGraphStore.vector_query`
  and `AgensgraphVectorStore.query` honor `MetadataFilters`, translated into a
  fully parameterized (injection-safe) Cypher `WHERE`. All 14 `FilterOperator`
  values are supported — `EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`,
  `CONTAINS`, `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`, `ANY`, `ALL`, `IS_EMPTY` —
  along with `AND`/`OR`/`NOT` conditions and nested filter groups.
- **Hybrid search.** `AgensgraphVectorStore(hybrid_search=True)` fuses HNSW
  semantic search with full-text keyword search by reciprocal rank fusion — each
  modality is queried against its own index (so both stay index-backed) and the
  two rankings are merged.
- **AgensGraph-dialect Text2Cypher.** The property graph store sets a default
  `text_to_cypher_template` that knows the storage model (every node on a single
  `"__Node__"` label, entity type held in a `labels` list) and avoids Neo4j-only
  syntax, so `TextToCypherRetriever` generates runnable Cypher out of the box.
- **Lazy schema introspection.** `AgensPropertyGraphStore(refresh_schema=False)`
  defers the (O(N)) schema scan to the first `get_schema()`/`get_schema_str()`
  call, so opening a large existing graph is instant.
- **Indexed type filter.** Each node also stores its primary type in a
  btree-indexed `__type__` scalar, so a type-scoped query
  (`WHERE n.__type__ = 'X'`) is an index scan rather than a `'X' IN n.labels`
  jsonb membership scan.
- **Correctness fixes.** Entity embeddings are persisted on `upsert_nodes` even
  when the entity has no source chunk; `get(ids=[])` returns nothing (instead of
  the whole graph); and depth-1 `get_rel_map` uses a fixed pattern (AgensGraph's
  variable-length edges are far slower).
- **Modern vector-store node management.** `AgensgraphVectorStore` implements
  `get_nodes(node_ids, filters)`, `delete_nodes(node_ids, filters)` and `clear()`
  (plus async `aget_nodes` / `adelete_nodes` / `aclear`).
- **Richer enhanced schema.** With `enhanced_schema=True`, numeric properties get
  `min` / `max` / `distinct_count`, list properties get `min_size` / `max_size`,
  and other properties get example values + `distinct_count` (computed
  exhaustively under a row threshold, sampled above it).
- **Performance.** Ingest is index-backed and near-linear (a btree index on the
  `id` MERGE key; bulk `add`/`upsert` batched). Id-keyed lookups (`get`,
  `get_nodes`, `get_triplets`, `get_rel_map`, `delete_nodes`) and the vector
  store's `delete(ref_doc_id)` are index-backed rather than sequential scans,
  relation upserts are UNWIND-batched per type, and schema introspection no
  longer materializes every distinct property value. Metadata-filter keys can be
  indexed with `create_property_index(...)`. See
  [Performance & indexing](#performance--indexing).
- **True async + connection pooling.** See [Async & connection pooling](#async--connection-pooling).
- **Breaking change.** The deprecated triplet `AgensGraphStore` (Knowledge Graph
  Store) has been removed. Use `AgensPropertyGraphStore` with `PropertyGraphIndex`.

## Installation

```shell
pip install llama-index llama-index-agensgraph
```

## Usage

### Property Graph Store

```python
import os
import urllib.request
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore

os.environ[
    "OPENAI_API_KEY"
] = "<YOUR_API_KEY>"  # Replace with your OpenAI API key

os.makedirs("data/paul_graham/", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = "data/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

nest_asyncio.apply()

with open(output_path, "r", encoding="utf-8") as file:
    content = file.read()

modified_content = content.replace("'", "\\'")

with open(output_path, "w", encoding="utf-8") as file:
    file.write(modified_content)

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# Setup AgensGraph connection (ensure AgensGraph is running)
conf = {
    "dbname": "",
    "user": "",
    "password": "",
    "host": "",
    "port": 5432,
}

# Pass vector_dimension to enable the HNSW vector index (match your embedding
# model's dimension, e.g. 1536 for text-embedding-3-small). Without it, vector
# search still works but is unindexed.
graph_store = AgensPropertyGraphStore(
    graph_name="graph",
    conf=conf,
    vector_dimension=1536,
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-4o-mini", temperature=0.0),
            # strict=True can yield zero triplets with some models; strict=False is more forgiving
            strict=False,
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")
print("\nDetailed Query Response:")
print(str(response))
```

### Natural-language queries (Text2Cypher)

`TextToCypherRetriever` turns a question into AgensGraph Cypher using the store's
built-in dialect prompt — no custom prompt needed:

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

retriever = TextToCypherRetriever(
    graph_store=graph_store, llm=OpenAI(model="gpt-4o-mini")
)
nodes = retriever.retrieve("How many entities of each type are there?")
print(nodes[0].node.text)  # the generated Cypher and its result
```

### Vector Store
```python
import os
import urllib.request
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"  # Replace with your key

# Download example data
os.makedirs("data/paul_graham/", exist_ok=True)
url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = "data/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

# Load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# Setup AgensGraph connection (ensure AgensGraph is running)
url = "postgresql://username:password@host:port/database_name"
embed_dim = 1536

# Initialize vector store
vector_store = AgensgraphVectorStore(url=url, embedding_dimension=embed_dim)

# Build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What happened at Interleaf?")
print("\nQuery Response:")
print(str(response))
```

For hybrid (vector + keyword) search, build the store with `hybrid_search=True`
and query with a `query_str`:

```python
hybrid_store = AgensgraphVectorStore(
    url=url, embedding_dimension=embed_dim, hybrid_search=True
)
index = VectorStoreIndex.from_vector_store(hybrid_store)
index.as_retriever(vector_store_query_mode="hybrid").retrieve(
    "What happened at Interleaf?"
)
```

## Async & connection pooling

By default each store opens a single dedicated connection. For concurrent
workloads, share an `AgensEngine` (a `psycopg` connection pool) across stores so
each request checks out its own connection instead of serializing on one:

```python
from llama_index_agensgraph.engine import AgensEngine
from llama_index_agensgraph.graph_stores.agensgraph import AgensPropertyGraphStore
from llama_index_agensgraph.vector_stores.agensgraph import AgensgraphVectorStore

engine = AgensEngine.from_url(
    "postgresql://user:pwd@host:5432/db", min_size=2, max_size=20
)

graph_store = AgensPropertyGraphStore(graph_name="graph", conf=conf, engine=engine)
vector_store = AgensgraphVectorStore(
    url="postgresql://user:pwd@host:5432/db",
    embedding_dimension=1536,
    engine=engine,
)

# ... use the stores ...
engine.close()  # await engine.aclose() if you used the async pool
```

The stores also provide true-async hot paths backed by `psycopg.AsyncConnection`
(no thread-pool wrapping):

- Vector store: `async_add`, `aquery`, `adelete`
- Property graph store: `aupsert_nodes`, `aupsert_relations`, `aget`,
  `avector_query`, `astructured_query`

These work with or without an `AgensEngine`; with one, they draw from the async
pool.

## Performance & indexing

The stores are indexed for their hot paths out of the box:

- **Ingest** (`MERGE`-by-`id`) is backed by a btree index on `id`, so bulk
  `upsert`/`add` stays near-linear rather than O(N²).
- **Vector search** uses the HNSW index on the embedding.
- **Lookups by id** (`get` / `get_nodes` / `delete_nodes`) and the vector
  store's **`delete(ref_doc_id)`** are index-backed.
- **Relation upserts** are UNWIND-batched per relationship type (not one query
  per relation).

**Metadata-filtered vector search.** A metadata filter cannot use the HNSW index
for the filter itself, so a filter on an *un-indexed* property degrades to a
sequential scan over the embedded nodes. Index the keys you filter on to keep it
fast:

```python
# Property graph store
graph_store.create_property_index("country")

# Vector store
vector_store.create_property_index("topic")
```

With the index present, the planner pre-selects matching rows via an index/bitmap
scan and then ranks them — instead of scanning every node.

**Counting and type filters.** Prefer `count(*)` over `count(n)` in aggregations:
`count(n)` materializes each matched node (including its embedding), so it is much
slower on a graph that stores embeddings. For "all nodes of type X", filter on the
indexed `n.__type__` scalar rather than `'X' IN n.labels`.