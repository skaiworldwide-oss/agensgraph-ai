# Cognee AgensGraph adapters

[Cognee](https://github.com/topoteretes/cognee) is an **AI-memory** framework: you
`add` your data, `cognify` it into a **knowledge graph + embeddings**, then
`search` that memory many ways. This package lets **one AgensGraph database back
both** of cognee's stores at once.

AgensGraph is PostgreSQL + Cypher + `pgvector`, so a single database can be cognee's
**graph store** *and* its **vector store**:

- **Graph adapter** (`GRAPH_DATABASE_PROVIDER=agensgraph`) — the knowledge graph
  (entities + relationships) as a Cypher property graph.
- **Vector adapter** (`VECTOR_DB_PROVIDER=agensgraph`) — the embeddings as
  `pgvector` HNSW tables.

Use both for one-database simplicity, or use just the graph adapter and keep your
vectors elsewhere. (cognee's small bookkeeping — datasets, users — stays in a local
SQLite file by default.)

Both adapters talk to the server through
[agensgraph-python](https://github.com/skaiworldwide-oss/agensgraph-python), the
AgensGraph driver.

## Requirements

- Python >= 3.11, < 3.14
- cognee 0.2 (the `>=0.2.1,<0.3` range)
- AgensGraph 2.17 or later, with the `vector` extension available for the vector adapter
- agensgraph-python >= 2.0.0 (installed with the package)

## Installation

```bash
pip install cognee-agensgraph
```

Then activate the adapters by importing the package once at startup:

```python
import cognee_agensgraph   # registers the agensgraph graph + vector providers
```

## Quickstart

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

## How the graph is stored

Every cognee class (`Entity`, `EntityType`, `DocumentChunk`, `TextDocument`,
`TextSummary`, ...) is a vertex label of its own, and all of them are children of
`__node__`, so `MATCH (n:Entity)` reads the entities and `MATCH (n:__Node__)` reads
every node. Label names are lower case: Cypher folds an unquoted identifier to lower
case, so `MATCH (n:Entity)`, `MATCH (n:entity)` and `MATCH (n:"entity")` all find the
label, which is what makes the natural-language search's generated Cypher work. The
class name as written stays in the `type` property. Each label has a uniqueness
constraint on `id` and an index on `name`. cognee derives a node's id from its name, so
an `Entity` and an `EntityType` with the same name share an id; such a node is stored
once, on the label of the class that first wrote it, and its `type` is the class written
last — the same one node per id that cognee's own adapters keep. Every relationship is an edge label named
after it (`contains`, `is_a`, `is_part_of`, ...), and every edge carries
`source_node_id`, `target_node_id` and `relationship_name` as properties. The adapter
creates nothing outside the graph.

A class name longer than 63 bytes is shortened and given a hash of the full name,
since the server truncates a longer label name.

Vector collections are tables in the `public` schema named `<Class>_<field>`, for
example `Entity_name` and `DocumentChunk_text`, with an HNSW cosine index.

> **Upgrading from 0.2.0:** graphs written by 0.2.0 kept every node on one label
> with the class in a `labels` property and are not readable by this layout. Drop
> the graph (`cognee.prune.prune_system()`) and run `cognify` again. Vector
> collections need no change.

## `query()` is read-only by default

`graph_engine.query(cypher)` is what `SearchType.CYPHER` and `SearchType.NATURAL_LANGUAGE`
run — a statement written by a user or by a language model. It takes one statement at
a time and runs it in a read-only transaction. To write through it:

```python
graph_engine = await get_graph_engine()
graph_engine.query_read_only = False
```

A read-only transaction does not stop a superuser from running a program on the
server, so the driver refuses one for such a role unless told to go ahead. Development
setups run as a superuser, so `query_allow_server_programs` defaults to `True`; set it
to `False` on a deployment where the application role is not a superuser.

`query()` also accepts Neo4j's edge shorthand — `(a)--(b)`, `(a)-->(b)`, `(a)<--(b)` —
which language models write in most statements. In AgensGraph `--` starts a comment,
so such a statement would otherwise lose its second half; between two pattern nodes it
is rewritten to `-[]-`, `-[]->` and `<-[]-`. Everything else is sent as written.

## Search scores

`search()` returns the cosine distance as the score: 0 for an identical vector, larger
for less similar ones. cognee adds the distances of a triplet's two nodes and its
edge across collections, so the scores are left as distances rather than rescaled
per collection.

## Performance

Measured on 1,000 documents cognified into 26,564 nodes, 72,944 edges and 26,000
1,536-dimension embeddings, on AgensGraph 2.18, the same machine for both columns.

| operation | before | after | |
|---|---:|---:|---:|
| full ingest (nodes, edges, embeddings, per-batch whole-graph read) | 459.3 s | 253.2 s | 1.8× |
|   `add_nodes` | 126.2 s | 9.9 s | 12.7× |
|   `add_edges` | 8.6 s | 8.7 s | 0.99× (slower) |
|   embeddings into HNSW collections | 265.6 s | 216.2 s | 1.2× |
|   whole-graph read after each batch (sum of 20) | 58.0 s | 17.5 s | 3.3× |
| `get_graph_data` (whole graph) | 3.0 s | 1.2 s | 2.5× |
| graph search projection (`project_graph_from_db`) | 4.4 s | 2.2 s | 2.0× |
| `get_nodeset_subgraph`, 10 names → 3,716 nodes | 2.7 s | 283.8 ms | 9.4× |
| node-set search projection | 3.2 s | 828.1 ms | 3.8× |
| `get_graph_metrics` (10 keys, was 5) | 5.9 s | 435.3 ms | 13.6× |
| `get_disconnected_nodes` | 1.4 s | 41.7 ms | 33.9× |
| `get_degree_one_nodes` | 633.8 ms | 232.2 ms | 2.7× |
| `get_document_subgraph` | 12.7 ms | 5.0 ms | 2.5× |
| `has_edges`, 100 edges (old returned none) | 19.3 s | 11.2 ms | 1718.8× |
| `get_connections`, 1,400 edges | 109.9 ms | 77.0 ms | 1.4× |
| `get_predecessors`, 465 nodes | 23.1 ms | 17.3 ms | 1.3× |
| `get_neighbors` | 3.8 ms | 1.6 ms | 2.3× |
| `get_edges` | 3.6 ms | 1.7 ms | 2.1× |
| `has_node` | 1.3 ms | 0.6 ms | 2.1× |
| `get_node` | 1.5 ms | 0.8 ms | 1.8× |
| `has_edge` | 2.1 ms | 0.9 ms | 2.4× |
| `get_nodes`, 50 ids | 3.3 ms | 4.8 ms | 0.69× (slower) |
| 4 × `search(limit=0)` (every row of 4 collections) | 1.0 s | 887.8 ms | 1.2× |
| `search` top 15, 13k-row collection | 102.5 ms | 4.9 ms | 20.7× |
| `search` top 15, 6k-row collection | 54.1 ms | 5.2 ms | 10.5× |
| `batch_search`, 8 queries | 463.0 ms | 45.3 ms | 10.2× |
| `has_collection` | 2.2 ms | 0.0 ms | 447.2× |

Reads are medians of 7 warm repetitions, the two versions alternating per repetition in two processes on the same machine; ingest is one run each at the same load. `get_nodes` is slower because a statement that matches nodes from a bound list runs in a transaction that turns sequential scans off, which is what keeps it from joining every node's property map once the graph has more than a few thousand nodes.

Where the time went before, and what changed:

- **Writes.** Every `add_nodes` merged through one label and a list property
  maintained by a trigger and plpgsql functions; every statement paid a `SET
  graph_path` and two commits. Now each class merges on its own label in one
  statement per 1,000 rows, connections run in autocommit mode with the graph bound
  once per connection, and a statement that matches nodes from a bound list runs
  with sequential scans off for its transaction: past a few thousand nodes the
  planner otherwise joins the list by hashing every node's property map instead of
  probing the unique index.
- **Reads.** cognee reads the whole graph after every cognify batch and before every
  graph search (`get_graph_data`). The edge read joined both endpoints to return ids
  it then discarded, and the node read called a plpgsql function per vertex; both are
  gone, and the rows are taken without conversion. `get_nodeset_subgraph` reads nodes
  and edges by graphid through the tables' own indexes.
- **Vectors.** Embeddings go in binary, one statement per batch. A search for the
  nearest rows runs with sequential scans off: a 1,536-dimension vector is stored out
  of line and the planner does not see those reads, so it chose a sequential scan over
  the HNSW index on every collection. A search for every row (`limit=0`, how cognee
  reads a collection) keeps the sequential scan.
- **Metrics.** All ten keys, from the label tables and the edge list; the optional
  ones (self-loops, diameter, average shortest path, clustering) only when asked for.

> The vector embedding dimension is fixed when a collection's table is first
> created; to change embedding models, drop the affected collection tables.

## Demos

The fastest way to see what this enables is the runnable demo suite in
[`examples/demos/`](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos) — five focused examples on real public
datasets (Wikipedia, CC-News, a Python repo), each with its own README and a
**pre-executed notebook** you can read without running anything:

| Demo | What it shows |
|---|---|
| [01 · Search modes](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/01_search_modes) | Build a knowledge graph from Wikipedia, then query it ten ways — `GRAPH_COMPLETION` (+ summary / chain-of-thought / context-extension variants), `RAG_COMPLETION`, `INSIGHTS`, `CHUNKS`, `SUMMARIES`, `NATURAL_LANGUAGE`, and raw `CYPHER` |
| [02 · Typed](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/02_typed) | Ontology-guided extraction — make the graph follow *your* domain vocabulary |
| [03 · Memory](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/03_memory) | A multi-dataset memory layer — named datasets, `node_set` tags, incremental builds |
| [04 · Code graph](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/04_code_graph) | Turn a Python repo into a code knowledge graph; `SearchType.CODE` + visualize |
| [05 · Explore](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/05_explore) | Inspect the AgensGraph-backed graph — metrics, traversal, raw Cypher, HTML visualization |

Start at [`examples/demos/README.md`](https://github.com/skaiworldwide-oss/agensgraph-ai/tree/main/cognee/examples/demos/README.md).

## Contributing

Contributions are welcome! Please open an issue or submit a Pull Request.

## License

Apache-2.0.
