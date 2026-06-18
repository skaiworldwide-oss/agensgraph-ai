# LlamaIndex AgensGraph

This plugin integrates [AgensGraph](https://github.com/skaiworldwide-oss/agensgraph)
with [LlamaIndex](https://www.llamaindex.ai/), persisting graphs and vectors
directly in AgensGraph. It powers `PropertyGraphIndex` and `VectorStoreIndex`,
so you can store and query property graphs and embeddings in one database.

- Property Graph Store: `AgensPropertyGraphStore`
- Vector Store: `AgensgraphVectorStore`
- Connection pool: `AgensEngine` (optional, shared across stores)

See the associated guides below:

- [Agens Property Graph Store](./examples//property_graph/property_graph_agensgraph.ipynb)
- [Agensgraph Vector Store](./examples/vector_stores/AgensgraphVectorDemo.ipynb)

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
- **Performance.** The vector store now creates a btree index on its `id` MERGE
  key (ingest was O(N²) without it), bulk `add` is batched, and schema
  introspection no longer materializes every distinct property value (it would
  ship full embedding vectors on every refresh).
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
    "port": "",
}

# Pass vector_dimension to enable the HNSW vector index (match your embedding
# model's dimension, e.g. 1536 for text-embedding-ada-002). Without it, vector
# search still works but is unindexed.
graph_store = AgensPropertyGraphStore(
    graph_name="graph",
    conf=conf,
    vector_dimension=1536,
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-ada-002"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0),
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