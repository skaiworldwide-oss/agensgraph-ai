# Cognee AgensGraph adapters

AgensGraph **graph** and **vector** database adapters for the Cognee framework.
Because AgensGraph is PostgreSQL + Cypher + pgvector, a single AgensGraph
database can serve as both cognee's graph store and its vector store — or you
can use just the graph adapter and keep vectors elsewhere.

## What's new in 0.2.0

- **Fixed an O(N²) ingest bug.** Every node lookup and MERGE keys on the `id`
  property, but the only index was on a different label (`base`) and a different
  property (`entity_id`) — so ingest was quadratic and every lookup did a
  sequential scan. Node ingest, edge ingest, and id lookups are now backed by a
  `base_id_idx ON "__Node__" (id)` index (e.g. 5000 nodes + 5000 edges ingest in
  ~1.2s).
- **New dedicated vector adapter.** `AgensgraphVectorAdapter` implements cognee's
  `VectorDBInterface` over pgvector (HNSW cosine), selectable with
  `VECTOR_DB_PROVIDER=agensgraph`, so AgensGraph can be cognee's vector store too.
- **End-to-end `cognify` with AgensGraph vectors.** The vector adapter now
  implements cognee's indexing hooks (`create_vector_index` / `index_data_points`),
  so a full `add → cognify → search` runs with `VECTOR_DB_PROVIDER=agensgraph`
  (previously `cognify` raised `AttributeError` on the missing methods).
- **Batched `add_edges`.** Edges are UNWIND-batched per relationship type instead
  of one query per edge.
- **`name` is indexed** and nodeset retrieval uses it (was a sequential scan).
- **Shared async connection pool.** One refcounted pool is opened once per
  process and reused by both adapters; `graph_path` is reapplied per checkout
  instead of on every query.
- **Correctness fixes.** `add_node`, `has_node`, `has_edge`, and `get_neighbors`
  were broken (bad parameter binding / never awaited / undefined variables / a
  call to a non-existent method) and now work.
- **Modernized packaging.** Poetry → hatchling, Python ≥3.10, version 0.2.0.

## Demos

A runnable demo suite lives in [`examples/demos/`](./examples/demos) — five
focused examples on real public datasets (Wikipedia, CC-News, a Python repo),
each with a README and a pre-executed notebook:

| Demo | What it shows |
|---|---|
| [01 · Search modes](./examples/demos/01_search_modes) | build a KG from Wikipedia, then query it via GRAPH_COMPLETION / INSIGHTS / RAG / CHUNKS / SUMMARIES / chain-of-thought |
| [02 · Typed](./examples/demos/02_typed) | ontology-guided extraction — make the graph follow your domain vocabulary |
| [03 · Memory](./examples/demos/03_memory) | a multi-dataset memory layer — isolated datasets, `node_set`, incremental builds |
| [04 · Code graph](./examples/demos/04_code_graph) | turn a Python repo into a code knowledge graph; `SearchType.CODE` + visualize |
| [05 · Explore](./examples/demos/05_explore) | inspect the AgensGraph-backed graph — metrics, raw Cypher, HTML visualization |

Start at [`examples/demos/README.md`](./examples/demos/README.md).

## Installation

```bash
pip install cognee-agensgraph
```

## Usage

```python
import asyncio
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
import pathlib
import os
import pprint
import cognee_agensgraph

async def main():
    # Set up agensgraph credentials in .env file and get the values from environment variables
    agensgraph_url = os.getenv("GRAPH_DATABASE_URL")

    # Configure agensgraph as the graph database provider
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": agensgraph_url,  # agensgraph connection DSN
            "graph_database_provider": "agensgraph",  # Specify agensgraph as provider
        }
    )
    
    # Optional: Set custom data and system directories
    system_path = pathlib.Path(__file__).parent
    cognee.config.system_root_directory(os.path.join(system_path, ".cognee_system"))
    cognee.config.data_root_directory(os.path.join(system_path, ".data_storage"))
    
    # Sample data to add to the knowledge graph
    sample_data = [
        "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
        "Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Natural language processing enables computers to understand and process human language.",
        "Computer vision allows machines to interpret and make decisions based on visual information."
    ]
    
    try:
        print("Adding data to Cognee...")
        await cognee.add(sample_data, "ai_knowledge")
        
        print("Processing data with Cognee...")
        await cognee.cognify(["ai_knowledge"])
        
        print("Searching for insights...")
        search_results = await cognee.search(
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            query_text="artificial intelligence"
        )
        
        print(f"Found {len(search_results)} insights:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result}")
            
        print("\nSearching with Chain of Thought reasoning...")
        await cognee.search(
            query_type=cognee.SearchType.GRAPH_COMPLETION_COT,
            query_text="How does machine learning relate to artificial intelligence and what are its applications?"
        )

        print("\nYou can get the graph data directly, or visualize it in an HTML file like below:")
        
        # Get graph data directly
        graph_engine = await get_graph_engine()
        graph_data = await graph_engine.get_graph_data()
        
        print("\nDirect graph data:")
        pprint.pprint(graph_data)

        # Or visualize it in HTML
        print("\nVisualizing the graph...")
        await cognee.visualize_graph(system_path / "graph.html")
        print(f"Graph visualization saved to {system_path / 'graph.html'}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure AgensGraph is running and your DSN is correct.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Requirements

- Python >= 3.10, <= 3.13
- AgensGraph database instance
- psycopg >= 3.1.0 with extras(binary and pool)

## Configuration

The adapter requires the following configuration using the `set_graph_db_config()` method:

```python
cognee.config.set_graph_db_config({
    "graph_database_url": "postgresql://username:password@host:port/dbname",
    "graph_database_provider": "agensgraph",
})
```

To use AgensGraph as the **vector store** as well (pgvector HNSW), point the
vector config at the same database:

```python
cognee.config.set_vector_db_config({
    "vector_db_url": "postgresql://username:password@host:port/dbname",
    "vector_db_provider": "agensgraph",
})
```

### Environment Variables

Set the following environment variables or pass them directly in the config:

```bash
export GRAPH_DATABASE_URL="postgresql://username:password@host:port/dbname"
export GRAPH_DATABASE_PROVIDER="agensgraph"
# Optional: AgensGraph as the vector store too
export VECTOR_DB_URL="postgresql://username:password@host:port/dbname"
export VECTOR_DB_PROVIDER="agensgraph"
```

**Alternative:** You can also use the [`.env.template`](https://github.com/topoteretes/cognee/blob/main/.env.template) file from the main cognee repository. Copy it to your project directory, rename it to `.env`, and fill in your AgensGraph configuration values.

### Optional Configuration

You can also set custom directories for system and data storage:

```python
cognee.config.system_root_directory("/path/to/system")
cognee.config.data_root_directory("/path/to/data")
```

## Features

- **Graph adapter** (`GRAPH_DATABASE_PROVIDER=agensgraph`): full GraphDBInterface
  support over AgensGraph's Cypher, async, with graph-completion and
  Chain-of-Thought search, direct `get_graph_engine()` access, and HTML
  visualization.
- **Vector adapter** (`VECTOR_DB_PROVIDER=agensgraph`): cognee VectorDBInterface
  over pgvector with an HNSW cosine index.

## Performance & indexing

- A **single async connection pool** is shared (refcounted) across the graph and
  vector adapters and opened once per process; `graph_path` is reapplied per
  checkout, not per query.
- **Graph ingest and lookups are index-backed**: nodes MERGE/lookup on an
  indexed `id` (the previous build indexed the wrong property, making ingest
  O(N²) and every lookup a sequential scan); `add_edges` is UNWIND-batched per
  relationship type.
- **Vector search** uses an HNSW (`vector_cosine_ops`) index; the column is typed
  `vector(dim)` so the query's `<=>` cast matches the index and it is used at
  scale.

> The vector embedding dimension is fixed when a collection's table is first
> created; to change embedding models, drop the affected collection tables.

## Example

See `examples` for a complete working example that demonstrates:
- Setting up the AgensGraph adapter
- Adding comprehensive AI/ML knowledge to the graph
- Processing data with cognee
- Searching with graph completion
- Chain of Thought reasoning searches
- Direct graph data access and inspection
- Comprehensive error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
